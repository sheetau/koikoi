"""Final-round card search with uniform hidden proposals and fixed64 inference.

The equal WP/Point mixture is an approximation: it ignores opponent-action
likelihoods both within this round and in earlier rounds. It is not a posterior.
This experimental adapter is not wired into human play or a strength protocol.
"""

from koikoi.agents import solved_koi_action
from koikoi.layout import BANKRUPT, PHASE, ROUND, ROUNDS, TERMINAL
from scripts.fixed_microbatch_continuation import FixedMicrobatchContinuation
from scripts.history_conditioned_search import HistoryConditionedSearch
from scripts.published_action_search import PublishedActionSearch


def is_final_round_card(reference):
    return bool(reference.state[ROUND] == 7 and reference.state[PHASE] in (0, 1, 2))


class FinalRoundSearch(HistoryConditionedSearch):
    """Compare final-round cards; retain the strict parent at all other roots."""

    def __init__(self, checkpoint, **kwargs):
        if kwargs.get("mode", "uniform") != "uniform":
            raise ValueError(
                "Final-round search currently supports uniform proposals only"
            )
        kwargs["mode"] = "uniform"
        super().__init__(checkpoint, **kwargs)
        self.continuation = FixedMicrobatchContinuation(
            self.continuation.actors[2], self.device, microbatch=64
        )
        self.outside_card_gate = 0

    def act(self, reference):
        state = reference.state
        if state[ROUNDS] != 8 or not state[BANKRUPT] or state[PHASE] == TERMINAL:
            raise ValueError("Final-round search requires a live paper8 state")
        self.last_decision = self.support = None
        self.last_sampling = {}
        if not is_final_round_card(reference):
            self.outside_card_gate += 1
            self.last_context = {"status": "outside_final_round_cards"}
            solved = solved_koi_action(reference)
            if solved is not None:
                self.solved_decisions += 1
                return solved
            return int(self.parent.act(reference))
        self.last_context = {
            "status": "final_round_uniform",
            "model_names": list(self.continuation.names),
            "model_prior": [0.5, 0.5],
            "history_likelihood_calls": 0,
        }
        # Uniform sampling does not enumerate hands. Retain all-action selection,
        # paired terminal simulation and fresh selected-action confirmation.
        return PublishedActionSearch.act(self, reference)
