"""Bounded terminal search with a declared WP/Point opponent mixture.

History uses strict CPU opponent policies; simulated continuations use separately
identified frozen batched policies. Neither opponent identity nor hidden cards
from the real match are supplied. This adapter is experimental, not deployed.
"""

import hashlib
import math
from dataclasses import asdict

import numpy as np
import torch

from koikoi.agents import EndgameAgent, solved_koi_action
from koikoi.layout import BANKRUPT, PHASE, PLAYER, ROUNDS, TERMINAL
from koikoi.published_actor import PublishedActor
from koikoi.search import sample_hidden_batch
from koikoi.upstream import PublishedAgent
from scripts.enumerate_history_support import context
from scripts.history_conditioned_worlds import (
    enumerate_support,
    joint_probabilities,
    sample_joint_worlds,
)
from scripts.published_action_search import PublishedActionSearch


class MixtureContinuation:
    """Keep a particle's opponent model fixed across branches and round changes."""

    names = ("point", "wp")

    def __init__(self, parent_actor, device):
        self.device = torch.device(device)
        self.actors = [
            PublishedActor.from_published(name, device).eval().requires_grad_(False)
            for name in self.names
        ] + [parent_actor.eval().requires_grad_(False)]
        self.labels = None

    def prepare(self, root_player, model_indices):
        labels = torch.as_tensor(model_indices, device=self.device, dtype=torch.long)
        if root_player not in (0, 1) or labels.ndim != 1 or not len(labels):
            raise ValueError("Expected a seat and one model index per particle")
        if not ((labels >= 0) & (labels < 2)).all():
            raise ValueError("Unknown opponent model index")
        self.root_player = int(root_player)
        self.labels = labels.repeat_interleave(2)

    @torch.no_grad()
    def __call__(self, env):
        if self.labels is None or len(self.labels) != env.count:
            raise ValueError("Continuation labels do not match the paired workspace")
        features, legal = env.published_observe()
        solved = env.solved_actions()
        legal &= (solved[:, None] < 0) | (
            torch.arange(50, device=env.device)[None] == solved[:, None]
        )
        active = env.state[:, PHASE] != TERMINAL
        routes = torch.where(env.state[:, PLAYER] == self.root_player, 2, self.labels)
        actions = torch.full((env.count,), -1, device=env.device, dtype=torch.int64)
        for route, actor in enumerate(self.actors):
            selected = torch.where(active & (routes == route))[0]
            if len(selected):
                actions[selected] = actor(
                    features[selected], env.state[selected, PHASE], legal[selected]
                ).argmax(-1)
        return actions


class HistoryConditionedSearch(PublishedActionSearch):
    """Compare all legal actions at publicly bounded roots, or keep the parent.

    mode='uniform' is the unconditioned hand/model control with the same public
    eligibility cap and mixture continuation. Both modes start each root from an
    equal WP/Point model prior; no earlier same-round posterior is fed back.
    """

    def __init__(self, checkpoint, *, mode="history", max_hands=100, **kwargs):
        if mode not in ("history", "uniform"):
            raise ValueError("mode must be history or uniform")
        if type(max_hands) is not int or max_hands < 1:
            raise ValueError("max_hands must be a positive integer")
        super().__init__(checkpoint, **kwargs)
        self.mode, self.max_hands = mode, max_hands
        self.continuation = MixtureContinuation(self.continuation.actor, self.device)
        self.opponents = {
            name: EndgameAgent(PublishedAgent(name)) for name in self.continuation.names
        }
        self.history_work = {}
        self.refusals = {"too_many_hands": 0, "empty_support": 0}
        self.last_context = None
        self.last_sampling = {}
        self.support = None

    def act(self, reference):
        if reference.state[ROUNDS] != 8 or not reference.state[BANKRUPT]:
            raise ValueError("Search supports paper8 only")
        if reference.state[PHASE] == TERMINAL:
            raise ValueError("Search requires an active root")
        self.last_decision = self.last_context = self.support = None
        self.last_sampling = {}
        if solved_koi_action(reference) is not None or reference.legal().sum() == 1:
            return super().act(reference)
        unknown, count, _ = context(reference.state)
        candidates = math.comb(len(unknown), count)
        self.last_context = {"mode": self.mode, "candidate_hands": candidates}
        if candidates > self.max_hands:
            self.last_context["status"] = "too_many_hands"
            self.refusals["too_many_hands"] += 1
            return int(self.parent.act(reference))
        if self.mode == "history":
            self.support = enumerate_support(reference, self.opponents, self.max_hands)
            if self.support.status != "complete":
                raise RuntimeError("Public eligibility and enumeration disagree")
            for key, value in self.support.counts.items():
                self.history_work[key] = self.history_work.get(key, 0) + value
            compatible = np.asarray(self.support.compatible, dtype=bool)
            self.last_context.update(
                valid_hands=self.support.valid_hands,
                invalid_hands=self.support.invalid_hands,
                model_names=list(self.support.model_names),
                supported_model_counts=compatible.sum(0).tolist(),
                replay_work=self.support.counts.copy(),
            )
            if not compatible.any():
                self.last_context["status"] = "empty_support"
                self.refusals["empty_support"] += 1
                return int(self.parent.act(reference))
            self.last_context["model_posterior"] = (
                joint_probabilities(self.support).sum(0).tolist()
            )
        self.last_context["status"] = "searched"
        return super().act(reference)

    @torch.no_grad()
    def _compare_actions(
        self, reference, base, alternatives, count, rng, confirmation=False
    ):
        if self.mode == "history":
            if self.support is None:
                raise ValueError("Enumerate this root before conditioned comparison")
            worlds, labels = sample_joint_worlds(reference, self.support, rng, count)
            if self.support.model_names != self.continuation.names:
                raise ValueError("Historical and rollout model indices disagree")
        else:
            worlds = sample_hidden_batch(reference, rng, count)
            labels = rng.integers(0, 2, count, dtype=np.int32)
        self.last_sampling["confirmation" if confirmation else "proposal"] = {
            "worlds_sha256": hashlib.sha256(worlds.tobytes()).hexdigest(),
            "model_indices": labels.tolist(),
        }
        self.continuation.prepare(int(reference.state[PLAYER]), labels)
        states = torch.from_numpy(worlds).to(self.device)
        records, previous_base = [], None
        for alternative in alternatives:
            proposals = torch.tensor([base, alternative], device=self.device).expand(
                count, 2
            )
            wins, points, work = self.workspaces[count](
                states, proposals, self.continuation
            )
            for key, value in asdict(work).items():
                setattr(self.work, key, getattr(self.work, key) + value)
                if confirmation:
                    setattr(
                        self.confirmation_work,
                        key,
                        getattr(self.confirmation_work, key) + value,
                    )
            point_rows = points.cpu().numpy().copy()
            if previous_base is not None:
                np.testing.assert_array_equal(previous_base, point_rows[:, 0])
            previous_base = point_rows[:, 0].copy()
            differences = (wins[:, 1] - wins[:, 0]).double().cpu().numpy() / 2
            records.append(
                {
                    "alternative": int(alternative),
                    "win_change": float(differences.mean()),
                    "lower_margin": self._lower(differences),
                    "point_change": float(np.diff(point_rows, axis=1).mean()),
                    "paired_points": point_rows.tolist(),
                    "work": asdict(work),
                }
            )
        return records
