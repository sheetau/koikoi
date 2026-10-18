from dataclasses import replace

import numpy as np
import pytest
import torch

from koikoi.agents import EndgameAgent
from koikoi.reference import ReferenceEnv
from koikoi.search import sample_hidden
from koikoi.upstream import PublishedAgent
from scripts.diagnose_history_support import read, reference
from scripts.history_conditioned_worlds import (
    enumerate_support,
    joint_probabilities,
    sample_joint_worlds,
)
from scripts.pruned_history_support import enumerate_pruned_support
from tests.test_history_conditioned_worlds import Extremal, late_root


def compare(root, opponents, cap):
    before = root.state.copy()
    full = enumerate_support(root, opponents, cap)
    pruned = enumerate_pruned_support(root, opponents, cap)
    assert replace(pruned, counts=full.counts) == full
    np.testing.assert_array_equal(root.state, before)
    assert pruned.counts["replay_decisions"] <= full.counts["replay_decisions"]
    assert pruned.counts["model_calls"] <= full.counts["opponent_decisions"]
    assert pruned.counts["completed_hands"] + pruned.counts["pruned_hands"] == full.valid_hands
    assert pruned.counts["rejected_models"] == np.logical_not(full.compatible).sum()
    assert pruned.counts["model_calls"] == (
        pruned.counts["guarded_model_calls"] + pruned.counts["unguarded_model_calls"]
    )
    np.testing.assert_array_equal(joint_probabilities(full), joint_probabilities(pruned))
    for seed in (1, 19):
        a = sample_joint_worlds(root, full, np.random.default_rng(seed), 64)
        b = sample_joint_worlds(root, pruned, np.random.default_rng(seed), 64)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
    return pruned


def test_exact_support_pruning_and_hidden_invariance():
    root = late_root()
    opponents = {"min": Extremal(), "max": Extremal(True)}
    pruned = compare(root, opponents, 100)
    assert pruned.counts["pruned_hands"] > 0
    assert pruned.counts["completed_hands"] > 0
    alternate = reference(sample_hidden(root, np.random.default_rng(71)))
    assert enumerate_pruned_support(alternate, opponents, 100) == pruned


@pytest.mark.parametrize("index", [7, 32])
def test_strict_published_models_equal_complete_replay(index):
    torch.set_num_threads(2)
    root = reference(read("runs/rollout_opponent_roots.json")["roots"][index]["state"])
    opponents = {name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")}
    support = compare(root, opponents, 1000)
    assert support.counts["model_calls"] > 0


def test_cap_precedes_inference_and_validation():
    root = ReferenceEnv(31010001, "paper8")
    opponent = Extremal()
    a = enumerate_support(root, {"a": opponent}, 100)
    b = enumerate_pruned_support(root, {"a": opponent}, 100)
    assert a == b and b.status == "too_many_hands" and opponent.calls == 0
    for cap in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            enumerate_pruned_support(root, {"a": opponent}, cap)
    with pytest.raises(ValueError, match="named"):
        enumerate_pruned_support(root, {})
