"""First-round fourth-hand discard search with exact pruned history support."""

import math

import numpy as np

from koikoi.agents import solved_koi_action
from koikoi.layout import BANKRUPT, DISCARD, PHASE, ROUND, ROUNDS, TERMINAL, TURN
from scripts.enumerate_history_support import context
from scripts.history_conditioned_worlds import joint_probabilities
from scripts.pruned_history_support import enumerate_pruned_support
from scripts.published_action_search import PublishedActionSearch
from scripts.round_value_search import RoundValueSearch


def is_midround(reference):
    """One potential intervention per seat: its fifth discard in round one."""
    state = reference.state
    return bool(state[ROUND] == 0 and state[TURN] in (8, 9) and state[PHASE] == DISCARD)


class MidroundRoundSearch(RoundValueSearch):
    """At most one opportunity per match; equal model prior before round one.

    Modes differ only in conditioning the hand/model distribution on public
    opponent actions. No true opponent identity is supplied. Other decisions
    use the strict parent and existing public endgame guard.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("max_hands", 5000)
        super().__init__(*args, **kwargs)
        self.outside_midround = 0

    def act(self, reference):
        state = reference.state
        if state[ROUNDS] != 8 or not state[BANKRUPT] or state[PHASE] == TERMINAL:
            raise ValueError("Midround search requires a live paper8 state")
        self.last_decision = self.support = None
        self.last_sampling = {}
        if not is_midround(reference):
            self.outside_midround += 1
            self.last_context = {"status": "outside_midround"}
            solved = solved_koi_action(reference)
            if solved is not None:
                self.solved_decisions += 1
                return solved
            return int(self.parent.act(reference))
        if reference.legal().sum() != 4:
            raise ValueError("Expected four own cards at the fifth discard")
        unknown, count, _ = context(state)
        candidates = math.comb(len(unknown), count)
        self.last_context = {"mode": self.mode, "candidate_hands": candidates}
        if candidates > self.max_hands:
            self.last_context["status"] = "too_many_hands"
            self.refusals["too_many_hands"] += 1
            return int(self.parent.act(reference))
        if self.mode == "history":
            self.support = enumerate_pruned_support(reference, self.opponents, self.max_hands)
            assert self.support.status == "complete"
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
            self.last_context["model_posterior"] = joint_probabilities(self.support).sum(0).tolist()
        self.last_context["status"] = "searched"
        return PublishedActionSearch.act(self, reference)
