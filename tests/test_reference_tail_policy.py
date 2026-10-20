import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.env import BatchEnv
from koikoi.layout import KOI, PHASE, PICK_DRAW, ROUND, TURN
from koikoi.upstream import PublishedAgent
from scripts.belief_tail import KoiTailDynamics
from scripts.check_earlier_belief_tail import collect
from scripts.deferred_belief_tail import DeferredBeliefPlanner, deferred_worlds
from scripts.history_conditioned_worlds import enumerate_support
from scripts.reference_tail_policy import verify_deferred_policy_values
from scripts.revealed_opponent_models import StrictPublishedPrediction
from scripts.run_arbitration_study import PARENTS
from tests.test_belief_tail import swap_seats
from tests.test_history_conditioned_worlds import Extremal, late_root


@pytest.mark.parametrize("swap", [False, True])
def test_independent_weighted_replay_matches_all_root_policies(swap):
    torch.set_num_threads(2)
    root = late_root()
    root.state[ROUND] = 7
    if swap:
        root = swap_seats(root)
    actor = Extremal()
    support = enumerate_support(root, {"first": actor})
    worlds = deferred_worlds(root, support, {"first": 3})
    dynamics = KoiTailDynamics(4096, worlds.model_names, {"first": actor}, actor)
    for optimize in (False, True):
        plan = DeferredBeliefPlanner(dynamics).solve(worlds, optimize)
        replay = verify_deferred_policy_values(worlds, plan, dynamics.policy_cache)
        assert replay["values"] == {str(a): str(q) for a, q in plan.values.items()}
        assert replay["physical_work"]["decisions"] <= plan.work["decisions"]
        if not optimize:
            for key, count in replay["physical_work"].items():
                assert count == plan.work[key]
            for key, count in replay["represented_work"].items():
                assert count == plan.work[f"represented_{key}"]
        assert replay["represented_work"]["completions"] == (
            int(worlds.weights.sum()) * len(plan.values)
        )
    # The oracle must detect a corrupted value, even if the policy itself is valid.
    first = next(iter(plan.values))
    corrupt = replace(
        plan, values={**plan.values, first: plan.values[first] + Fraction(1, 100)}
    )
    with pytest.raises(AssertionError, match="root-value mismatch"):
        verify_deferred_policy_values(worlds, corrupt, dynamics.policy_cache)


def test_earlier_gate_preserves_full_source_and_uses_prior_public_identification():
    torch.set_num_threads(2)
    # Replay an already closed source; this consumes no fresh study stream.
    source = json.loads(
        Path("runs/belief_tail_capability/sources.json").read_text(encoding="utf-8")
    )[7]
    case = {
        key: source[key] for key in ("parent", "opponent", "seat", "replicate", "seed")
    }
    parent = NeuralAgent(PARENTS[case["parent"]][0], "cpu", deterministic=True)
    published = {name: PublishedAgent(name) for name in ("wp", "point")}
    predictors = {
        name: StrictPublishedPrediction(agent) for name, agent in published.items()
    }
    opponent = EndgameAgent(published[case["opponent"]])
    native = BatchEnv(1, "cpu", debug=True)
    for cap in (3, 5):
        result = collect(case, parent, opponent, predictors, native, max_draws=cap)
        for key in (
            "actions",
            "final",
            "work",
            "identification_events",
            "identification_calls",
        ):
            assert result[key] == source[key]
        selected = result["selected_root"]
        assert (
            selected["public_identification"]
            == source["selected_root"]["public_identification"]
        )
        assert selected["public_identification"]["round"] < 7
        state = selected["state"]
        draws = 16 - state[TURN] - int(state[PHASE] in (PICK_DRAW, KOI))
        if cap == 3:
            assert selected == source["selected_root"]
        else:
            assert selected["decision"] < source["selected_root"]["decision"]
            assert 3 < draws <= 5
