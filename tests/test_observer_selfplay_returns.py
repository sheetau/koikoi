"""Independent forward returns, exact gradients and live observation isolation."""

from fractions import Fraction as F

import numpy as np
import pytest
import torch

from koikoi.env import BatchEnv
from koikoi.layout import EPISODE, PHASE, PLAYER, SEED
from koikoi.reference import ReferenceEnv
from scripts.observer_selfplay_returns import (
    ObserverFeatures,
    observer_gae,
    policy_advantages,
)

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]


def forward_oracle(reward, value, done, tail, lam):
    result = np.zeros_like(value)
    for root in range(len(reward)):
        for lane in range(reward.shape[1]):
            for seat in (0, 1):
                for end in range(root, len(reward)):
                    future = (
                        0
                        if done[end, lane]
                        else tail[lane, seat]
                        if end + 1 == len(reward)
                        else value[end + 1, lane, seat]
                    )
                    result[root, lane, seat] += lam ** (end - root) * (
                        (1 - 2 * seat) * reward[end, lane]
                        + future
                        - value[end, lane, seat]
                    )
                    if done[end, lane]:
                        break
    return result, result + value


@pytest.fixture(params=DEVICES)
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    return request.param


def test_random_forward_expansion_and_observer_isolation(device):
    rng = np.random.default_rng(26010001)
    shape = (31, 5)
    reward = rng.normal(size=shape)
    value = rng.normal(size=(*shape, 2))
    done = rng.random(shape) < 0.16
    tail = rng.normal(size=(5, 2))
    arrays = [torch.as_tensor(x, device=device) for x in (reward, value, done, tail)]
    snapshots = [x.clone() for x in arrays]
    for lam in (0, 0.37, 0.95, 1):
        actual = observer_gae(*arrays, lam)
        for x, y in zip(actual, forward_oracle(reward, value, done, tail, lam)):
            np.testing.assert_allclose(x.cpu(), y, rtol=2e-13, atol=2e-13)
        # An opponent-observation prediction cannot enter this seat's return.
        for seat in (0, 1):
            changed = [x.clone() for x in arrays]
            changed[1][:, :, 1 - seat] = float("nan")
            changed[3][:, 1 - seat] = float("nan")
            other = observer_gae(*changed, lam)
            for x, y in zip(actual, other):
                torch.testing.assert_close(x[..., seat], y[..., seat], rtol=0, atol=0)
    for before, after in zip(snapshots, arrays):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_terminal_reset_nan_tail_and_known_payoffs(device):
    reward = torch.tensor([[0.0], [1], [0], [-1]], device=device)
    values = torch.zeros(4, 1, 2, device=device)
    done = torch.tensor([[False], [True], [False], [True]], device=device)
    tail = torch.full((1, 2), float("nan"), device=device)
    adv, ret = observer_gae(reward, values, done, tail, 0.5)
    expected = torch.tensor([[0.5, -0.5], [1, -1], [-0.5, 0.5], [-1, 1]], device=device)
    torch.testing.assert_close(adv[:, 0], expected, rtol=0, atol=0)
    torch.testing.assert_close(ret, adv, rtol=0, atol=0)
    reward[2:] *= 17
    changed = observer_gae(reward, values, done, tail, 0.5)[0]
    torch.testing.assert_close(adv[:2], changed[:2], rtol=0, atol=0)


def test_shared_scalar_counterexample_and_lambda_one_equivalence(device):
    # All eight outcome atoms, with exact probability/score weights. At the
    # opponent step, seat0 still knows X,A; seat1 knows only public A.
    means = ((F(1, 5), F(21, 100)), (F(1), F(-1)))
    reward, value, probability, score = [], [], [], []
    for x in (0, 1):
        for action in (0, 1):
            for outcome in (-1, 1):
                probability.append(float((1 + outcome * means[x][action]) / 8))
                score.append(
                    float((F(action) - F(1, 2)) * (1 if x == 0 else F(-1, 100)))
                )
                reward.append([0, outcome])
                value.append(
                    [
                        [float(sum(means[x]) / 2), -0.1025],
                        [
                            float(means[x][action]),
                            -float((means[0][action] + means[1][action]) / 2),
                        ],
                    ]
                )
    r = torch.tensor(reward, dtype=torch.float64, device=device).T
    v = torch.tensor(value, dtype=torch.float64, device=device).transpose(0, 1)
    d = torch.tensor([[False] * 8, [True] * 8], device=device)
    p = torch.tensor([[0] * 8, [1] * 8], device=device)
    tail = torch.zeros(8, 2, dtype=torch.float64, device=device)
    next_players = torch.zeros(8, dtype=torch.int32, device=device)
    weight = torch.tensor(np.array(probability) * score, device=device)
    gradients = {}
    for mode in ("observer", "signed"):
        advantage = policy_advantages(mode, r, v, d, p, tail, next_players)
        gradients[mode] = float((weight * advantage[0]).sum())
    assert gradients["observer"] == pytest.approx(float(F(3, 800)), abs=1e-15)
    # The unchanged historical function computes its eligibility factor in
    # float32, including when the value fixture uses float64.
    assert gradients["signed"] == pytest.approx(float(F(-8301, 3200000)), abs=2e-9)
    # For a terminal horizon lambda=1 telescopes away intermediate predictions.
    fixed = policy_advantages("observer", r, v, d, p, tail, next_players, 1)
    signed = policy_advantages("signed", r, v, d, p, tail, next_players, 1)
    torch.testing.assert_close(fixed, signed, rtol=0, atol=1e-15)


def test_real_observations_both_seats_all_phases_and_hidden_worlds(device):
    env = BatchEnv(4, device, "paper8", seed=26020001)
    extractor = ObserverFeatures(env)
    rng = np.random.default_rng(26020002)
    refs = [ReferenceEnv(26020001 + i, "paper8") for i in range(4)]
    phases = set()
    rounds = matches = 0
    for _ in range(256):
        before = env.state.clone()
        features = extractor(env)
        torch.testing.assert_close(env.state, before, rtol=0, atol=0)
        native, _ = env.observe()
        for lane, ref in enumerate(refs):
            actor = int(ref.state[PLAYER])
            phases.add(int(ref.state[PHASE]))
            for seat in (0, 1):
                projection = ReferenceEnv.__new__(ReferenceEnv)
                projection.state = ref.state.copy()
                projection.state[PLAYER] = seat
                expected = np.r_[projection.observe(), float(actor == seat)].astype(
                    "float32"
                )
                np.testing.assert_array_equal(features[lane, seat].cpu(), expected)
            torch.testing.assert_close(
                features[lane, actor, :560], native[lane], rtol=0, atol=0
            )
        if len(phases) == 4:
            # Per-view invariance. Change only what THAT viewer cannot observe.
            for seat in (0, 1):
                for row in env.state:
                    hidden = torch.where((row[:48] == 1 - seat) | (row[:48] == -1))[0]
                    row[hidden] = row[hidden].flip(0)
                    row[72:96] = row[72:96].flip(0)
                    row[SEED] += 100
                other = extractor(env)
                torch.testing.assert_close(
                    features[:, seat], other[:, seat], rtol=0, atol=0
                )
                env.state.copy_(before)
        actions = [rng.choice(np.flatnonzero(ref.legal())) for ref in refs]
        expected = np.array([ref.step(a) for ref, a in zip(refs, actions)])
        actual = env.step(torch.tensor(actions, device=device))
        np.testing.assert_array_equal(actual.cpu(), expected)
        np.testing.assert_array_equal(
            env.state.cpu(), np.stack([ref.state for ref in refs])
        )
        rounds += int(expected[:, 1].sum())
        matches += int(expected[:, 2].sum())
        for lane, ref in enumerate(refs):
            if ref.state[PHASE] == 4:
                refs[lane] = ReferenceEnv(
                    int(ref.state[SEED]), "paper8", int(ref.state[EPISODE]) + 1
                )
        env.reset_done()
    assert phases == {0, 1, 2, 3}
    assert rounds > matches > 0


def test_validation_and_detached_targets():
    reward = torch.zeros(2, 1, requires_grad=True)
    value = torch.zeros(2, 1, 2, requires_grad=True)
    done = torch.zeros(2, 1, dtype=torch.bool)
    tail = torch.zeros(1, 2, requires_grad=True)
    adv, targets = observer_gae(reward, value, done, tail)
    assert not adv.requires_grad and not targets.requires_grad
    for lam in (-1, float("nan"), 1.1):
        with pytest.raises(ValueError):
            observer_gae(reward, value, done, tail, lam)
    with pytest.raises(ValueError):
        observer_gae(reward, value[..., 0], done, tail)
    with pytest.raises(ValueError):
        observer_gae(reward, value, done.float(), tail)
    with pytest.raises(ValueError):
        policy_advantages(
            "observer",
            reward,
            value,
            done,
            torch.full((2, 1), 2),
            tail,
            torch.zeros(1, dtype=torch.int64),
        )
