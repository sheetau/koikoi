import numpy as np
import pytest
import torch

from koikoi.layout import KOIS, PHASE, PLAYER, POINTS, ROUND, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.rule_variants import (
    AUTO_ROUNDS,
    COMPLETED_ROUNDS,
    VARIANTS,
    VariantEnv,
    initial_yaku,
    solved_variant_action,
)
from koikoi.variant_native import VariantBackend, VariantBatch


@pytest.fixture(scope="module", params=["cpu", "cuda"])
def backend(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param, VariantBackend(request.param)


@pytest.mark.parametrize("variant", tuple(VARIANTS))
def test_complete_matches_python_cpu_cuda_differential(backend, variant):
    device, native = backend
    seeds = list(range(810000, 810016))
    refs = [VariantEnv(seed, variant) for seed in seeds]
    batch = VariantBatch(seeds, variant, device, native)
    rng = np.random.default_rng(49031)
    np.testing.assert_array_equal(batch.state.cpu().numpy(), [e.state for e in refs])
    decisions = 0
    for _ in range(600):
        if all(e.state[PHASE] == TERMINAL for e in refs):
            break
        actions, expected = [], []
        np.testing.assert_array_equal(
            batch.solved_actions().cpu().numpy(),
            [
                solved_variant_action(e)
                if e.state[PHASE] != TERMINAL and solved_variant_action(e) is not None
                else -1
                for e in refs
            ],
        )
        for env in refs:
            if env.state[PHASE] == TERMINAL:
                actions.append(0)
                expected.append(np.zeros(4, dtype=np.int32))
            else:
                action = int(rng.choice(np.flatnonzero(env.legal())))
                actions.append(action)
                expected.append(env.step(action))
                decisions += 1
        np.testing.assert_array_equal(batch.step(actions).cpu().numpy(), expected)
        np.testing.assert_array_equal(
            batch.state.cpu().numpy(), [e.state for e in refs]
        )
    assert all(e.state[PHASE] == TERMINAL for e in refs)
    assert decisions > 0
    assert all(1 <= e.state[COMPLETED_ROUNDS] <= 8 for e in refs)


def test_control_is_exactly_original_except_new_accounting_slots():
    for seed in range(55, 65):
        old, new = ReferenceEnv(seed), VariantEnv(seed)
        rng = np.random.default_rng(seed)
        while old.state[PHASE] != TERMINAL:
            copy = new.state.copy()
            copy[112:115] = 0
            np.testing.assert_array_equal(copy, old.state)
            action = int(rng.choice(np.flatnonzero(old.legal())))
            np.testing.assert_array_equal(new.step(action), old.step(action))


def test_opening_patterns_and_symmetric_payments():
    assert initial_yaku([0, 1, 2, 3, 8, 12, 16, 20])
    assert initial_yaku([0, 1, 8, 9, 16, 17, 24, 25])
    assert not initial_yaku([0, 1, 8, 9, 16, 17, 24, 28])
    for variant in VARIANTS:
        env = VariantEnv(1, variant)
        env.state[:48] = -1
        env.state[[0, 8, 28, 44]] = 3  # eight-point four lights
        env.state[[1, 5, 9]] = 4  # five-point red ribbons
        env.state[KOIS : KOIS + 2] = 1
        value = env.payment(0)
        swapped = env.clone()
        swapped.state[:48] = np.where(
            env.state[:48] == 3, 4, np.where(env.state[:48] == 4, 3, env.state[:48])
        )
        assert swapped.payment(1) == value
        if variant == "paper_control":
            assert value == 9
        if variant == "seven_double":
            assert value == 18
        if variant == "winner_takes_yaku":
            assert value == 15
    env = VariantEnv(1, "opening_six")
    hand = [0, 1, 2, 3, 8, 12, 16, 20]
    other = [4, 5, 9, 13, 17, 21, 24, 28]
    field = [6, 10, 14, 18, 22, 25, 29, 32]
    deck = hand + other + field
    deck += sorted(set(range(48)) - set(deck))
    env.state[ROUND] = 7
    before = env.state[AUTO_ROUNDS]
    env.deal(deck)
    assert env.state[PHASE] == TERMINAL
    assert env.state[AUTO_ROUNDS] == before + 1
    assert env.state[POINTS] - env.state[POINTS + 1] == 12


def test_assistance_uses_variant_payment():
    env = VariantEnv(1, "seven_double")
    env.state[:48] = -1
    env.state[[0, 8, 28, 44]] = 3
    env.state[PLAYER], env.state[PHASE], env.state[ROUND] = 0, 3, 7
    env.state[POINTS : POINTS + 2] = [20, 40]
    assert env.yaku(0) == 8 and env.payment(0) == 16
    assert solved_variant_action(env) == 48
