"""Independent interval expansion, boundary fixtures and actual game traces."""

import numpy as np
import pytest
import torch

from koikoi.env import BatchEnv
from koikoi.layout import EPISODE, PHASE, PLAYER, SEED
from koikoi.reference import ReferenceEnv
from koikoi.train import signed_gae
from scripts.learner_returns import learner_gae

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]


def forward_oracle(rewards, values, dones, learner, tail, tail_learner, lam):
    """Expand TD residuals forward from each own action, without a reverse scan."""
    h, n = rewards.shape
    adv, ret = np.zeros((h, n)), np.zeros((h, n))
    valid = np.zeros((h, n), dtype=bool)
    for lane in range(n):
        for root in np.flatnonzero(learner[:, lane]):
            cursor, weight, total = root, 1.0, 0.0
            while cursor < h:
                reward, endpoint, terminal = 0.0, cursor, False
                while endpoint < h:
                    reward += rewards[endpoint, lane]
                    terminal = dones[endpoint, lane]
                    endpoint += 1
                    if terminal or endpoint == h or learner[endpoint, lane]:
                        break
                known = terminal or endpoint < h or tail_learner[lane]
                if not known:
                    break
                valid[root, lane] = True
                bootstrap = (
                    0.0
                    if terminal
                    else values[endpoint, lane]
                    if endpoint < h
                    else tail[lane]
                )
                total += weight * (reward + bootstrap - values[cursor, lane])
                if terminal or endpoint == h:
                    break
                weight *= lam ** (endpoint - cursor)
                cursor = endpoint
            if valid[root, lane]:
                adv[root, lane] = total
                ret[root, lane] = total + values[root, lane]
    return adv, ret, valid


def check(device, arrays, lam):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    tensors = [torch.as_tensor(x, device=device) for x in arrays]
    snapshots = [x.clone() for x in tensors]
    expected = forward_oracle(*arrays, lam)
    actual = learner_gae(*tensors, lam)
    for x, y in zip(actual, expected):
        np.testing.assert_allclose(x.cpu(), y, rtol=2e-6, atol=2e-6)
    for x, y in zip(tensors, snapshots):
        torch.testing.assert_close(x, y, rtol=0, atol=0, equal_nan=True)
    return actual


@pytest.mark.parametrize("device", DEVICES)
def test_random_interval_expansion_and_ignored_opponent_values(device):
    rng = np.random.default_rng(20291001)
    shape = (71, 17)
    rewards, values = [rng.normal(size=shape).astype("float32") for _ in range(2)]
    dones = rng.random(shape) < 0.12
    learner = rng.random(shape) < 0.47
    learner[:, 0] = False  # No own decisions in this lane.
    tail = rng.normal(size=shape[1]).astype("float32")
    tail_learner = rng.random(shape[1]) < 0.5
    arrays = [rewards, values, dones, learner, tail, tail_learner]
    for lam in (0.0, 0.37, 0.95, 1.0):
        baseline = check(device, arrays, lam)
        changed = [x.copy() for x in arrays]
        changed[1][~learner] = np.nan
        changed[4][~tail_learner] = np.nan
        ignored = check(device, changed, lam)
        for a, b in zip(baseline, ignored):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_terminal_reset_tail_and_primitive_gap(device):
    # Own rows0,1,4,6. Opponent row3 ends the first episode. Row6 is an
    # unfinished final action: exclude it, but use V6 to bootstrap row4.
    rewards = np.array([0.1, 0.0, 0.2, 1.0, -0.3, 0.4, 0.7, 0.8])[:, None]
    values = np.array([0.1, 0.2, 99, 99, -0.2, 99, 0.6, 99.0])[:, None]
    dones = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=bool)[:, None]
    learner = np.array([1, 1, 0, 0, 1, 0, 1, 0], dtype=bool)[:, None]
    args = [rewards, values, dones, learner, np.array([99.0]), np.array([False])]
    adv, ret, valid = check(device, args, 0.5)
    np.testing.assert_allclose(adv.cpu().numpy().ravel(), [0.7, 1, 0, 0, 0.9, 0, 0, 0])
    assert valid.cpu().numpy().ravel().tolist() == [
        True,
        True,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    # Change rewards after terminal: earlier-episode targets must remain equal.
    changed = [x.copy() for x in args]
    changed[0][4:] += 10
    other = check(device, changed, 0.5)
    torch.testing.assert_close(ret[:4], other[1][:4], rtol=0, atol=0)
    # Exposing a tail learner value now completes the final own transition.
    args[-2:] = [np.array([-0.4]), np.array([True])]
    tail_adv, _, tail_valid = check(device, args, 0.5)
    assert tail_valid[6].item()
    assert tail_adv[6].item() == pytest.approx(0.5)
    assert tail_adv[4].item() == pytest.approx(0.9 + 0.5**2 * 0.5)


@pytest.mark.parametrize("device", DEVICES)
def test_all_learner_reduces_to_ordinary_gae(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    generator = torch.Generator().manual_seed(97)
    reward, value = [
        torch.randn(29, 3, generator=generator).to(device) for _ in range(2)
    ]
    dones = (torch.rand(29, 3, generator=generator) < 0.12).to(device)
    tail = torch.tensor([0.2, -0.3, 0.9], device=device)
    for lam in (0.0, 0.95, 1.0):
        adv, ret, valid = learner_gae(
            reward,
            value,
            dones,
            torch.ones_like(dones),
            tail,
            torch.ones_like(tail, dtype=torch.bool),
            lam,
        )
        expected = signed_gae(
            reward,
            value,
            dones,
            torch.zeros_like(dones, dtype=torch.int32),
            tail,
            torch.zeros_like(tail, dtype=torch.int32),
            lam,
        )
        assert valid.all()
        for a, b in zip((adv, ret), expected):
            torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_real_match_traces_rewards_before_seat_reset(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    env = BatchEnv(4, device=device, profile="paper8", seed=20292001)
    refs = []
    for row in env.state.cpu().numpy():
        ref = ReferenceEnv.__new__(ReferenceEnv)
        ref.state = row.copy()
        refs.append(ref)
    rewards, values, dones, learner = [], [], [], []
    rounds = matches = opponent_terminals = 0
    for _ in range(512):
        obs, mask = env.observe()
        roles, seats = [], []
        for lane, ref in enumerate(refs):
            seat = (lane + int(ref.state[EPISODE])) % 2
            seats.append(seat)
            roles.append(int(ref.state[PLAYER]) == seat)
        # Deliberately simple legal public policy, not a trained strength test.
        actions = mask.long().argmax(-1)
        expected = np.array([r.step(a) for r, a in zip(refs, actions.cpu().tolist())])
        actual = env.step(actions)
        np.testing.assert_array_equal(actual.cpu(), expected)
        np.testing.assert_array_equal(
            env.state.cpu(), np.stack([r.state for r in refs])
        )
        rewards.append(expected[:, 0] * (1 - 2 * np.array(seats)))
        values.append(obs[:, :48].mean(-1).cpu().numpy())
        dones.append(expected[:, 2].astype(bool))
        learner.append(roles)
        rounds += int(expected[:, 1].sum())
        matches += int(expected[:, 2].sum())
        opponent_terminals += int((expected[:, 2] * ~np.array(roles)).sum())
        env.reset_done()
        for lane, ref in enumerate(refs):
            if ref.state[PHASE] == 4:
                refs[lane] = ReferenceEnv(
                    int(ref.state[SEED]), "paper8", int(ref.state[EPISODE]) + 1
                )
    tail = env.observe()[0][:, :48].mean(-1).cpu().numpy()
    tail_learner = np.array(
        [
            int(r.state[PLAYER]) == (lane + int(r.state[EPISODE])) % 2
            for lane, r in enumerate(refs)
        ]
    )
    arrays = [
        np.asarray(rewards, dtype="float32"),
        np.asarray(values),
        np.asarray(dones),
        np.asarray(learner),
        tail,
        tail_learner,
    ]
    assert rounds > matches > 0 and opponent_terminals > 0
    for lam in (0.0, 0.95, 1.0):
        check(device, arrays, lam)


def test_contract_and_detached_targets():
    reward = torch.zeros(2, 1, requires_grad=True)
    value = torch.ones_like(reward, requires_grad=True)
    flags = torch.ones_like(reward, dtype=torch.bool)
    args = [reward, value, flags, flags, value[0], flags[0]]
    assert all(not x.requires_grad for x in learner_gae(*args))
    for lam in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="Lambda"):
            learner_gae(*args, lam)
    for index, replacement, message in [
        (1, value[:, :0], "shapes"),
        (4, value, "Tail"),
        (2, flags.int(), "boolean"),
        (0, reward.int(), "floating"),
    ]:
        bad = args.copy()
        bad[index] = replacement
        with pytest.raises(ValueError, match=message):
            learner_gae(*bad)
