"""Search only the opening dealer discard, with a declared uniform-deal prior.

No opponent has acted at this root. Uniform valid-hand sampling therefore does
not omit an opponent-action likelihood in the ideal shuffled-deal model. This
does not assert exact conditioning of the native finite PRNG's seed space.
"""

import math

import numpy as np

from koikoi.agents import solved_koi_action
from koikoi.layout import (
    BANKRUPT,
    DEALER,
    DISCARD,
    PHASE,
    PLAYER,
    ROUND,
    ROUNDS,
    TERMINAL,
    TURN,
)
from scripts.history_conditioned_search import HistoryConditionedSearch
from scripts.published_action_search import PublishedActionSearch
from scripts.round_value_search import RoundValueSearch


def is_opening(reference):
    state = reference.state
    return bool(
        state[ROUND] == 0
        and state[TURN] == 0
        and state[PHASE] == DISCARD
        and state[PLAYER] == state[DEALER]
    )


class OpeningOnly:
    def __init__(self, *args, **kwargs):
        if kwargs.get("mode", "uniform") != "uniform":
            raise ValueError("Opening search has no opponent-action likelihood")
        kwargs["mode"] = "uniform"
        super().__init__(*args, **kwargs)
        self.outside_opening = 0

    def act(self, reference):
        state = reference.state
        if state[ROUNDS] != 8 or not state[BANKRUPT] or state[PHASE] == TERMINAL:
            raise ValueError("Opening search requires a live paper8 state")
        self.last_decision = self.support = None
        self.last_sampling = {}
        if not is_opening(reference):
            self.outside_opening += 1
            self.last_context = {"status": "outside_opening"}
            solved = solved_koi_action(reference)
            if solved is not None:
                self.solved_decisions += 1
                return solved
            return int(self.parent.act(reference))
        observation = reference.observe()
        unknown = int(np.count_nonzero(observation[:528].reshape(48, 11)[:, 4]))
        hand_count = round(float(observation[541]) * 8)
        if unknown != 32 or hand_count != 8 or reference.legal().sum() != 8:
            raise ValueError("Malformed opening public card counts")
        self.last_context = {
            "status": "opening_prior",
            "candidate_hands_before_validity": math.comb(unknown, hand_count),
            "model_names": list(self.continuation.names),
            "model_prior": [0.5, 0.5],
            "history_likelihood_calls": 0,
        }
        # Deliberately bypass the bounded-history eligibility test, not legality,
        # paired worlds, candidate selection or independent confirmation.
        return PublishedActionSearch.act(self, reference)


class OpeningTerminalSearch(OpeningOnly, HistoryConditionedSearch):
    pass


class OpeningRoundSearch(OpeningOnly, RoundValueSearch):
    pass
