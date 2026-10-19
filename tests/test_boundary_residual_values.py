"""Exact finite expectations and independent trajectory checks."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest
import torch

from koikoi.agents import NeuralAgent
from koikoi.layout import PHASE, POINTS, TERMINAL
from koikoi.published_actor import PublishedActor
from scripts.boundary_residual_values import (
    BoundaryResidualWorkspace,
    corrected_estimate,
)
from scripts.diagnose_history_support import reference
from scripts.fixed_microbatch_continuation import FixedMicrobatchContinuation
from scripts.history_conditioned_worlds import sample_joint_worlds
from scripts.prepared_belief_search import make_support
from scripts.run_arbitration_study import PARENTS
from scripts.run_prepared_belief_actions import load_roots


def test_exact_unbiased_expectation_for_biased_surrogate():
    # Uniform two-point world: true mean 0, deliberately biased proxy mean 3/8.
    high = (Fraction(-1), Fraction(1))
    proxy = (Fraction(1, 4), Fraction(1, 2))
    for alpha in (Fraction(0), Fraction(1), Fraction(3, 2)):
        estimates = [
            alpha * proxy[i] + high[j] - alpha * proxy[j]
            for i, j in product(range(2), repeat=2)
        ]
        assert sum(estimates) / 4 == 0
        exact_variance = sum(e**2 for e in estimates) / 4
        mean_proxy = sum(proxy) / 2
        residual = [high[j] - alpha * proxy[j] for j in range(2)]
        mean_residual = sum(residual) / 2
        formula = (
            alpha**2 * sum((y - mean_proxy) ** 2 for y in proxy) / 2
            + sum((r - mean_residual) ** 2 for r in residual) / 2
        )
        assert exact_variance == formula


def test_estimator_arithmetic_and_no_automatic_variance_gain():
    row = corrected_estimate([-1, 1], [-1, 1], [1, -1])
    assert row["mean"] == 0 and row["standard_error"] == pytest.approx(np.sqrt(5))
    assert row["residual_variance"] == 8 > row["high_variance"]
    zero = corrected_estimate([-1, 1], [-1, 1], [1, -1], alpha=0)
    assert zero["standard_error"] == 1
    with pytest.raises(ValueError):
        corrected_estimate([0], [0, 1], [0, 1])


class PublicPointTable:
    def lookup(self, rounds, points0, dealers, terminal, seat, labels, table):
        value = np.clip((np.asarray(points0) - 30) / 30, -1, 1) * (1 - 2 * seat)
        value[terminal] = np.sign(np.asarray(points0)[terminal] - 30) * (1 - 2 * seat)
        return value, value[None]


@pytest.mark.parametrize("case", [0, 256])
def test_cuda_boundary_and_tail_match_reference_and_identical_cheap_prefix(case):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    row, root, batch = next(
        (r, e, b)
        for r, e, b in load_roots()
        if r["parent"] == 1 and r["case_index"] == case
    )
    parent = NeuralAgent(PARENTS[0][0], device="cpu", deterministic=True)
    continuation = FixedMicrobatchContinuation(
        PublishedActor(parent.model.actor.checkpoint(), "cuda"), "cuda", 64
    )
    support = make_support(
        batch, "identified_history", row["public_identification"]["model"]
    )
    worlds, labels = sample_joint_worlds(root, support, np.random.default_rng(443), 2)
    base = int(parent.act(root))
    alternative = next(int(a) for a in np.flatnonzero(root.legal()) if a != base)
    expected, first_done = [], []
    counts = {"decisions": 0, "round_endings": 0, "terminal_endings": 0}

    def step(env, act):
        assert env.legal()[act]
        result = env.step(act)
        counts["decisions"] += 1
        counts["round_endings"] += int(result[1])
        counts["terminal_endings"] += int(result[2])
        return bool(result[1])

    for world in worlds:
        for act in (base, alternative):
            env = reference(world)
            first_done.append(step(env, act))
            expected.append(env)

    class Checked:
        prepare = continuation.prepare

        def __call__(self, env):
            states = env.state.cpu().numpy()
            for i, (state, ref) in enumerate(zip(states, expected)):
                compare = ref.state.copy()
                if first_done[i] and state[PHASE] == TERMINAL:
                    compare[PHASE] = TERMINAL  # RoundWorkspace freezes ended lanes.
                np.testing.assert_array_equal(state, compare)
            actions = continuation(env)
            for i, (act, ref) in enumerate(zip(actions.cpu().tolist(), expected)):
                if states[i, PHASE] != TERMINAL:
                    first_done[i] |= step(ref, act)
            return actions

    workspace = BoundaryResidualWorkspace(2)
    states = torch.from_numpy(worlds).cuda()
    high = workspace.run(
        states, base, alternative, Checked(), row["seat"], labels, PublicPointTable()
    )
    assert high["work"] == counts and counts["terminal_endings"] == 4
    for state, ref in zip(workspace.rounds.env.state.cpu().numpy(), expected):
        np.testing.assert_array_equal(state, ref.state)
    expected_points = [
        (int(e.state[POINTS]) - int(e.state[POINTS + 1])) * (1 - 2 * row["seat"])
        for e in expected
    ]
    np.testing.assert_array_equal(
        high["paired_points"], np.array(expected_points).reshape(2, 2)
    )
    cheap = workspace.run(
        states,
        base,
        alternative,
        continuation,
        row["seat"],
        labels,
        PublicPointTable(),
        terminal=False,
    )
    for key in (
        "proxy",
        "paired_boundary_values",
        "boundary_fields",
        "first_round_work",
    ):
        assert cheap[key] == high[key]
    np.testing.assert_array_equal(states.cpu().numpy(), worlds)
