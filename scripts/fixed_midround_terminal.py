"""Bounded midround search with exact history and fixed64 terminal continuations."""

from scripts.fixed_microbatch_continuation import FixedMicrobatchContinuation
from scripts.history_conditioned_search import HistoryConditionedSearch
from scripts.midround_search import MidroundRoundSearch


class FixedMidroundTerminal(HistoryConditionedSearch):
    """Reuse the public gate and pruned support; no boundary-value table is loaded."""

    # This gate calls PublishedActionSearch.act directly and does not access the
    # round estimator. Keep the already tested eligibility/fallback logic intact.
    act = MidroundRoundSearch.act

    def __init__(self, checkpoint, **kwargs):
        kwargs.setdefault("max_hands", 5000)
        super().__init__(checkpoint, **kwargs)
        self.outside_midround = 0
        self.continuation = FixedMicrobatchContinuation(
            self.continuation.actors[2],
            self.device,
            microbatch=64,
        )
