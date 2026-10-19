"""Factorial belief controls and public-only sampling invariance."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from koikoi.reference import ReferenceEnv
from scripts.complete_sampled_history_exact import batch_from_record
from scripts.enumerate_history_support import canonical_world
from scripts.history_conditioned_worlds import joint_probabilities, sample_joint_worlds
from scripts.prepared_belief_search import KINDS, make_support
from scripts.run_arbitration_study import read
from scripts.sampled_history_support import HistoryProposalBatch


def test_four_conditions_match_hand_computed_joint_masses():
    batch = HistoryProposalBatch(
        "public",
        ("point", "wp"),
        "enumerated",
        2,
        2,
        0,
        ((0,), (1,)),
        ((True, False), (True, True)),
        {},
    )
    expected = {
        "uniform": [[0.25, 0.25], [0.25, 0.25]],
        "identified_uniform": [[0, 0.5], [0, 0.5]],
        "history": [[1 / 3, 0], [1 / 3, 1 / 3]],
        "identified_history": [[0, 0], [0, 1]],
    }
    for kind in KINDS:
        support = make_support(batch, kind, "wp")
        np.testing.assert_allclose(
            joint_probabilities(support), expected[kind], rtol=0, atol=1e-15
        )
        assert support.hands == batch.hands and support.counts == {}


def test_incomplete_tables_and_missing_identity_are_refused():
    batch = HistoryProposalBatch(
        "public",
        ("point", "wp"),
        "enumerated",
        2,
        2,
        0,
        ((0,), (1,)),
        ((True, False), (True, True)),
        {},
    )
    for bad in (
        replace(batch, mode="sampled"),
        replace(batch, hands=((0,), (0,))),
        replace(batch, candidate_hands=3),
        replace(batch, compatible=((True,),) * 2),
    ):
        with pytest.raises(ValueError):
            make_support(bad, "history")
    for kind in ("identified_uniform", "identified_history"):
        with pytest.raises(ValueError, match="publicly identified"):
            make_support(batch, kind)


def test_all_four_world_distributions_ignore_real_hidden_hand_and_chance():
    original = next(
        r
        for r in map(
            json.loads,
            Path("runs/sampled_history_diagnostic/records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines(),
        )
        if (r["parent"], r["case_index"]) == (1, 0)
    )
    source = json.loads(
        Path("runs/final_round_training_study/parent1/source.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    env = ReferenceEnv(source["seed"], "paper8")
    for action in source["actions"][: original["decision"]]:
        env.step(action)
    batch = batch_from_record(
        read("runs/sampled_history_exact_check/parent1_wp_support.json")
    )
    alternate = copy.deepcopy(env)
    alternate.state[:] = canonical_world(env.state, batch.hands[1])
    np.testing.assert_array_equal(alternate.observe(), env.observe())
    for kind in KINDS:
        support = make_support(batch, kind, original["public_identification"]["model"])
        worlds, labels = sample_joint_worlds(
            env, support, np.random.default_rng(555), 32
        )
        other, other_labels = sample_joint_worlds(
            alternate, support, np.random.default_rng(555), 32
        )
        np.testing.assert_array_equal(worlds, other)
        np.testing.assert_array_equal(labels, other_labels)
        if kind.startswith("identified"):
            assert set(labels) == {
                batch.model_names.index(original["public_identification"]["model"])
            }
    # Root mismatch is rejected before a world can be used for search.
    with pytest.raises(ValueError, match="different public"):
        sample_joint_worlds(
            env, replace(support, root_key="wrong"), np.random.default_rng(555), 1
        )
