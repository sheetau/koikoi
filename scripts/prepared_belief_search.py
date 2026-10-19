"""Separate opponent identity and current-hand evidence in prepared-root search.

Diagnostic adapter: exact support must already be computed from public history.
It is not a human-play adapter and does not hide the cost of support inference.
"""

import math

import numpy as np

from scripts.diagnose_sampled_history_support import eligible
from scripts.enumerate_history_support import context
from scripts.fixed_microbatch_continuation import FixedMicrobatchContinuation
from scripts.history_conditioned_search import HistoryConditionedSearch
from scripts.history_conditioned_worlds import HistorySupport, observation_key
from scripts.published_action_search import PublishedActionSearch

KINDS = ("uniform", "identified_uniform", "history", "identified_history")


def make_support(batch, kind, identified_model=None):
    """Use one common legal-hand prior; vary only public evidence conditioning."""
    if kind not in KINDS:
        raise ValueError("Unknown belief condition")
    if (
        batch.mode != "enumerated"
        or batch.model_names != ("point", "wp")
        or batch.candidate_hands != batch.proposed_draws
        or len(batch.hands) + batch.invalid_draws != batch.candidate_hands
        or len(batch.hands) != len(set(batch.hands))
        or len(batch.compatible) != len(batch.hands)
        or not batch.hands
        or any(
            len(row) != 2 or any(type(v) is not bool for v in row)
            for row in batch.compatible
        )
    ):
        raise ValueError("Complete exact two-model table required")
    identified = kind in ("identified_uniform", "identified_history")
    if identified and identified_model not in batch.model_names:
        raise ValueError("A publicly identified model is required")
    use_history = kind in ("history", "identified_history")
    compatible = tuple(
        tuple(
            (not identified or model == identified_model)
            and (not use_history or row[index])
            for index, model in enumerate(batch.model_names)
        )
        for row in batch.compatible
    )
    return HistorySupport(
        batch.root_key,
        batch.model_names,
        "complete",
        batch.candidate_hands,
        len(batch.hands),
        batch.invalid_draws,
        batch.hands,
        compatible,
        {},
    )


class PreparedBeliefSearch(HistoryConditionedSearch):
    """All-action search over frozen exact hand tables,with explicit public prior."""

    def __init__(self, checkpoint, **kwargs):
        super().__init__(checkpoint, mode="history", max_hands=6000, **kwargs)
        self.continuation = FixedMicrobatchContinuation(
            self.continuation.actors[2], self.device, microbatch=64
        )
        self.prepared_context = None

    def prepare(self, reference, batch, kind, identification, decision, seat):
        # The caller supplies the earlier audited public event,not an opponent
        # label from the match scheduler. Every control shares this root gate.
        if type(decision) is not int or not eligible(
            reference, seat, identification, decision
        ):
            raise ValueError("Eligible public identification and root required")
        unknown, count, _ = context(reference.state)
        if batch.root_key != observation_key(reference):
            raise ValueError("Support belongs to another public observation")
        if batch.candidate_hands != math.comb(len(unknown), count):
            raise ValueError("Hand table does not cover this public space")
        self.support = make_support(batch, kind, identification["model"])
        self.prepared_context = {
            "kind": kind,
            "root_key": batch.root_key,
            "public_identification": dict(identification),
            "decision": decision,
            "candidate_hands": batch.candidate_hands,
            "valid_hands": len(batch.hands),
            "historical_inference_reused": True,
            "new_historical_neural_calls": 0,
            "rollout_numerics": "fixed64 GPU/CPU execution; not strict singleton likelihood policy",
        }

    def act(self, reference):
        if self.support is None or self.prepared_context is None:
            raise ValueError("Prepare this public root before searching")
        if observation_key(reference) != self.support.root_key:
            raise ValueError("Prepared support belongs to another public observation")
        self.last_sampling = {}
        self.last_context = dict(self.prepared_context)
        if not np.asarray(self.support.compatible).any():
            self.last_context["status"] = "empty_support"
            self.last_decision = None
            self.refusals["empty_support"] += 1
            return int(self.parent.act(reference))
        self.last_context["status"] = "searched"
        return PublishedActionSearch.act(self, reference)
