import itertools
from fractions import Fraction as F

import numpy as np
import pytest

from scripts.directional_return_probe import (
    directional_scores,
    estimate_advantages,
    seat_contributions,
)


def test_score_probes_match_independent_log_probability_derivatives():
    weights = np.zeros((6, 50))
    weights[:3, :3] = [0.2, 0.3, 0.5]
    weights[3, 48:] = [0.8, 0.2]
    weights[4, 49] = 1
    weights[5] = 1  # Padded terminal sampling weights are deliberately ignored.
    legal = weights > 0
    legal[5] = False
    phases = np.array([0, 1, 2, 3, 3, 4])
    actions = np.array([0, 1, 2, 49, 49, -1])
    valid = np.array([True] * 5 + [False])
    actual = directional_scores(weights, actions, phases, legal, valid)
    expected = np.zeros((6, 5))
    for row in range(5):
        base = weights[row] / weights[row].sum()
        for probe in range(5):
            feature = np.zeros(50)
            if probe == 0 and phases[row] == 3:
                feature[49] = 1
            elif probe == phases[row] + 1:
                feature = base.copy()

            def logp(theta, base=base, feature=feature, action=int(actions[row])):
                shifted = base * np.exp(theta * feature)
                return np.log(shifted[action] / shifted.sum())

            expected[row, probe] = (logp(1e-6) - logp(-1e-6)) / 2e-6
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=0)
    np.testing.assert_array_equal(actual[4:], 0)
    for action in (48, 49):
        actions[3] = action
        score = directional_scores(weights, actions, phases, legal, valid)[3]
        if action == 48:
            mean_score = 0.8 * score
        else:
            mean_score += 0.2 * score
    np.testing.assert_allclose(mean_score, 0, atol=1e-15)
    weights[0, 49] = 0.01
    with pytest.raises(ValueError, match="legal actions only"):
        directional_scores(weights, actions, phases, legal, valid)


def test_exact_shared_parameter_counterexample_and_nonterminal_cutoff():
    # Independently enumerate the exact finite example already used in the paper.
    means = ((F(1, 5), F(21, 100)), (F(1), F(-1)))
    atoms = list(itertools.product((0, 1), (0, 1), (-1, 1)))
    probability, value, reward, score = (
        [],
        np.zeros((2, 8)),
        np.zeros((2, 8)),
        np.zeros((2, 8, 1)),
    )
    for i, (signal, action, payoff) in enumerate(atoms):
        probability.append(float((1 + payoff * means[signal][action]) / 8))
        value[0, i] = float(sum(means[signal]) / 2)
        value[1, i] = -float((means[0][action] + means[1][action]) / 2)
        reward[1, i] = payoff
        score[0, i, 0] = float(
            (F(action) - F(1, 2)) * (1 if signal == 0 else F(-1, 100))
        )
    player = np.array([[0] * 8, [1] * 8])
    valid = np.ones((2, 8), bool)
    done = np.array([[False] * 8, [True] * 8])
    estimates = estimate_advantages(
        reward, value, player, valid, done, np.zeros(8, int)
    )
    grad = {
        name: np.dot(probability, seat_contributions(score, a, player, valid)[:, 0, 0])
        for name, a in estimates.items()
    }
    assert grad["mc"] == pytest.approx(float(F(3, 800)), abs=1e-15)
    assert grad["gae95_full"] == pytest.approx(float(F(-8301, 3200000)), abs=1e-15)
    cutoff = estimate_advantages(reward, value, player, valid, done, np.full(8, 127))
    for name in ("gae95_window", "gae1_window"):
        actual = np.dot(
            probability, seat_contributions(score, cutoff[name], player, valid)[:, 0, 0]
        )
        assert actual == pytest.approx(float(F(-19701, 160000)), abs=1e-15)
    # The alternate seat has no policy-dependent action in this example.
    np.testing.assert_array_equal(
        seat_contributions(score, estimates["mc"], player, valid)[:, 1], 0
    )


def test_variable_lengths_same_player_turns_cutoffs_and_padding():
    rng = np.random.default_rng(9981)
    t_count, count, horizon = 11, 7, 4
    lengths = np.arange(3, 10)
    valid = np.arange(t_count)[:, None] < lengths
    done = np.arange(t_count)[:, None] == lengths - 1
    player = rng.integers(0, 2, (t_count, count), dtype=np.uint8)
    value = rng.normal(size=(t_count, count))
    rewards = np.zeros((t_count, count))
    outcomes = rng.choice([-1, 0, 1], count)
    rewards[lengths - 1, np.arange(count)] = outcomes
    offsets = np.arange(count) % horizon
    result = estimate_advantages(rewards, value, player, valid, done, offsets, horizon)
    # Direct per-root weighted TD expansion, using root-relative payoff units.
    for name, lam, cuts in (
        ("gae95_full", 0.95, False),
        ("gae95_window", 0.95, True),
        ("gae1_window", 1.0, True),
    ):
        expected = np.zeros((t_count, count))
        for col in range(count):
            for root in range(lengths[col]):
                for t in range(root, lengths[col]):
                    root_sign = (-1) ** int(player[root, col])
                    current = (-1) ** int(player[t, col]) * value[t, col]
                    next_value = (
                        0
                        if done[t, col]
                        else (-1) ** int(player[t + 1, col]) * value[t + 1, col]
                    )
                    expected[root, col] += (
                        lam ** (t - root)
                        * root_sign
                        * (rewards[t, col] + next_value - current)
                    )
                    if cuts and (t + offsets[col]) % horizon == horizon - 1:
                        break
        np.testing.assert_allclose(result[name], expected, atol=2e-15, rtol=0)
    scores = rng.normal(size=(t_count, count, 5))
    got = seat_contributions(scores, result["mc"], player, valid)
    assert got.shape == (count, 2, 5)
    expected = np.zeros_like(got)
    for col in range(count):
        for t in range(lengths[col]):
            expected[col, player[t, col]] += scores[t, col] * result["mc"][t, col]
    np.testing.assert_allclose(got, expected, atol=1e-15, rtol=0)
    broken = done.copy()
    broken[0, 0] = True
    with pytest.raises(ValueError, match="terminate exactly"):
        estimate_advantages(rewards, value, player, valid, broken, offsets, horizon)
