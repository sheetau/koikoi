"""Independent complete replay,proposal multiplicity and hidden-state checks."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from koikoi.reference import ReferenceEnv
from koikoi.search import sample_hidden
from scripts.diagnose_history_support import reference, replay
from scripts.enumerate_history_support import canonical_world, context
from scripts.history_conditioned_worlds import enumerate_support, joint_probabilities
from scripts.sampled_history_support import (
    HistoryProposalBatch,
    marginal_summary,
    proposal_probabilities,
    propose_history_support,
)
from tests.test_history_conditioned_worlds import Extremal, late_root


def test_small_spaces_equal_independent_exhaustive_posterior():
    root = late_root()
    models = {"min": Extremal(), "max": Extremal(True)}
    exact = enumerate_support(root, models, max_hands=100)
    sampled = propose_history_support(
        root, models, np.random.default_rng(4), max_draws=100
    )
    assert sampled.mode == "enumerated"
    assert sampled.proposed_draws == exact.candidate_hands
    assert sampled.invalid_draws == exact.invalid_hands
    assert sampled.hands == exact.hands and sampled.compatible == exact.compatible
    prior = {"min": 1, "max": 3}
    weights = joint_probabilities(exact, prior)
    np.testing.assert_array_equal(proposal_probabilities(sampled, prior), weights)
    expected_cards = [
        sum(weights[i].sum() for i, hand in enumerate(exact.hands) if card in hand)
        for card in range(48)
    ]
    summary = marginal_summary(sampled, prior)
    np.testing.assert_allclose(summary["card_probabilities"], expected_cards)
    assert summary["exact_enumeration"]


def test_sampled_budget_full_replay_and_hidden_world_invariance():
    root = late_root()
    original = root.state.copy()
    models = {"min": Extremal(), "max": Extremal(True)}
    result = propose_history_support(
        root, models, np.random.default_rng(19), max_draws=7
    )
    assert result.mode == "sampled" and result.candidate_hands > 7
    unknown, count, _ = context(root.state)
    rng = np.random.default_rng(19)
    expected, invalid = [], 0
    for _ in range(7):
        hand = tuple(sorted(int(c) for c in rng.choice(unknown, count, replace=False)))
        world = canonical_world(root.state, hand)
        if world is None:
            invalid += 1
        else:
            matched = tuple(
                replay(world, models[name])["compatible"] for name in sorted(models)
            )
            expected.append((hand, matched))
    assert list(zip(result.hands, result.compatible)) == expected
    assert result.proposed_draws == len(expected) + invalid == 7
    assert result.invalid_draws == invalid
    alternate = reference(sample_hidden(root, np.random.default_rng(71)))
    other = propose_history_support(
        alternate, models, np.random.default_rng(19), max_draws=7
    )
    assert result == other
    np.testing.assert_array_equal(root.state, original)


def test_duplicate_proposals_keep_mass_and_zero_mass_does_not_become_uniform():
    batch = HistoryProposalBatch(
        "public-key",
        ("known",),
        "sampled",
        100,
        3,
        0,
        ((1,), (1,), (2,)),
        ((True,), (True,), (True,)),
        {},
    )
    probabilities = proposal_probabilities(batch)
    np.testing.assert_allclose(probabilities[:, 0], [1 / 3] * 3)
    summary = marginal_summary(batch)
    assert summary["card_probabilities"][1] == 2 / 3
    assert summary["card_probabilities"][2] == 1 / 3
    assert summary["effective_proposals"] == 3
    assert summary["distinct_supported_hands"] == 2
    assert summary["supported_proposal_occurrences"] == 3
    assert not summary["exact_enumeration"]
    with pytest.raises(ValueError, match="Zero compatible"):
        proposal_probabilities(replace(batch, compatible=((False,),) * 3))
    for prior in ({"wrong": 1}, {"known": 0}, {"known": np.nan}):
        with pytest.raises(ValueError):
            proposal_probabilities(batch, prior)


def test_invalid_initial_hands_exhaust_budget_without_retries_or_calls():
    root = ReferenceEnv(31010001, "paper8")
    unknown, size, _ = context(root.state)
    month = next(
        set(range(start, start + 4))
        for start in range(0, 48, 4)
        if set(range(start, start + 4)) <= set(unknown)
    )
    invalid = sorted(month) + sorted(set(unknown) - month)[: size - 4]

    class FixedInvalid:
        calls = 0

        def choice(self, values, count, replace):
            self.calls += 1
            assert count == size and not replace and set(invalid) <= set(values)
            return np.array(invalid)

    rng, model = FixedInvalid(), Extremal()
    result = propose_history_support(root, {"known": model}, rng, max_draws=8)
    assert result.proposed_draws == result.invalid_draws == rng.calls == 8
    assert not result.hands and model.calls == 0
    with pytest.raises(ValueError, match="Zero compatible"):
        proposal_probabilities(result)
    valid_rng = np.random.default_rng(6)
    before = copy.deepcopy(valid_rng.bit_generator.state)
    for budget in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            propose_history_support(root, {"known": model}, valid_rng, max_draws=budget)
    assert valid_rng.bit_generator.state == before and model.calls == 0
