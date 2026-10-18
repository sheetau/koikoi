"""Experimental terminal-rollout search over an adapted published koi policy.

Root fallback is the unchanged singleton CPU policy. Continuations use a separate
frozen, batched actor and can differ numerically. Hidden worlds are sampled from
the root observation, not copied from the actual opponent hand or future stock.
Uniform proposals do not condition on the likelihood of past opponent actions.
"""

import math
from dataclasses import asdict

import numpy as np
import torch

from koikoi.agents import NeuralAgent, solved_koi_action
from koikoi.layout import BANKRUPT, KOI, PHASE, ROUNDS
from koikoi.published_actor import PublishedActor
from koikoi.search import sample_hidden_batch
from scripts.collect_reusable_paired_targets import PairedWorkspace
from scripts.paired_rollout_targets import PublicContinuation, RolloutWork


class PublishedKoiSearch:
    """Compare stop and continue on paired terminal outcomes; preserve card play.

    Confidence is a standard-error margin, not a distribution-free safety bound.
    Optional fresh particles confirm only the already selected alternative.
    Call reset_seed explicitly for each duplicate deal; the production evaluator
    does not yet recognize this experimental adapter.
    """

    def __init__(
        self,
        checkpoint,
        *,
        particles=32,
        confirmation_particles=32,
        confidence=2.5,
        seed=25100001,
        device="cuda",
    ):
        for count in (particles, confirmation_particles):
            if type(count) is not int or count < 0:
                raise ValueError("Particle counts must be nonnegative integers")
        if particles < 2 or confirmation_particles == 1:
            raise ValueError("Use at least two particles, or zero confirmation")
        if not math.isfinite(confidence) or confidence < 0:
            raise ValueError("Confidence must be finite and nonnegative")
        if torch.get_num_threads() != 2:
            raise ValueError("Strict root inference requires two CPU threads")
        self.parent = NeuralAgent(checkpoint, device="cpu", deterministic=True)
        if self.parent.published_env is None or self.parent.profile != "paper8":
            raise ValueError("Expected a paper8 adapted published parent")
        # Do not freeze or move the root model: either can change near-tie actions.
        self.continuation = PublicContinuation(
            PublishedActor(self.parent.model.actor.checkpoint(), device)
        )
        self.device = torch.device(device)
        self.particles = particles
        self.confirmation_particles = confirmation_particles
        self.confidence = confidence
        self.checkpoint_sha256 = self.parent.checkpoint_sha256
        self.workspaces = {
            count: PairedWorkspace(count, self.device)
            for count in {particles, confirmation_particles} - {0}
        }
        self.work = RolloutWork()
        self.confirmation_work = RolloutWork()
        self.searched_decisions = self.proposed_decisions = self.changed_decisions = 0
        self.confirmation_decisions = self.solved_decisions = 0
        self.last_decision = None
        self.reset_seed(seed)

    def reset_seed(self, seed):
        self.rng = np.random.default_rng(seed)
        self.confirmation_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))

    def _lower(self, differences):
        return float(
            differences.mean()
            - self.confidence * differences.std(ddof=1) / math.sqrt(len(differences))
        )

    @torch.no_grad()
    def _compare(self, reference, base, alternative, count, rng, confirmation=False):
        worlds = sample_hidden_batch(reference, rng, count)
        states = torch.from_numpy(worlds).to(self.device)
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
        # Signed terminal return maps to draw-adjusted win score by (R + 1) / 2.
        differences = (wins[:, 1] - wins[:, 0]).cpu().numpy().astype(np.float64) / 2
        return {
            "win_change": float(differences.mean()),
            "lower_margin": self._lower(differences),
            "point_change": float((points[:, 1] - points[:, 0]).double().mean()),
        }

    def act(self, reference):
        if int(reference.state[ROUNDS]) != 8 or not reference.state[BANKRUPT]:
            raise ValueError("Search supports paper8 only")
        self.last_decision = None
        solved = solved_koi_action(reference)
        if solved is not None:
            self.solved_decisions += 1
            return solved
        base = self.parent.act(reference)
        if reference.state[PHASE] != KOI:
            return base
        legal = np.flatnonzero(reference.legal())
        if len(legal) == 1:
            return base
        if legal.tolist() != [48, 49] or base not in legal:
            raise ValueError("Unexpected koi action set")
        alternative = 97 - base
        proposal = self._compare(reference, base, alternative, self.particles, self.rng)
        self.searched_decisions += 1
        accept = proposal["lower_margin"] > 0
        self.proposed_decisions += int(accept)
        confirmation = None
        if accept and self.confirmation_particles:
            confirmation = self._compare(
                reference,
                base,
                alternative,
                self.confirmation_particles,
                self.confirmation_rng,
                confirmation=True,
            )
            self.confirmation_decisions += 1
            accept = confirmation["lower_margin"] > 0
        self.changed_decisions += int(accept)
        self.last_decision = {
            "base": base,
            "alternative": alternative,
            "proposal": proposal,
            "confirmation": confirmation,
            "changed": bool(accept),
        }
        return alternative if accept else base
