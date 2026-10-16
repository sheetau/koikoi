import pytest
import torch

from koikoi.env import BatchEnv
from koikoi.estimators import critic_input, q_boost
from koikoi.layout import PLAYER
from koikoi.train import Config, Trainer, signed_gae


def test_q_boost_eliminates_exact_future_action_noise():
    # Matching pennies: P0's action is hidden from P1. P1 mixes uniformly.
    # For either future action, exact centralized Q yields zero noise at P0.
    estimates = []
    gae_estimates = []
    for payoff in (-1.0, 1.0):
        rewards = torch.tensor([[0.0], [-payoff]])
        q_selected = torch.tensor([[0.0], [-payoff]])
        expected = torch.zeros(2, 1)
        dones = torch.tensor([[False], [True]])
        players = torch.tensor([[0], [1]])
        adv, _ = q_boost(
            rewards,
            q_selected,
            expected,
            dones,
            players,
            torch.zeros(1),
            torch.zeros(1, dtype=torch.int64),
            lam=1,
        )
        estimates.append(float(adv[0]))
        gae, _ = signed_gae(
            rewards,
            expected,
            dones,
            players,
            torch.zeros(1),
            torch.zeros(1, dtype=torch.int64),
            lam=1,
        )
        gae_estimates.append(float(gae[0]))
    assert estimates == [0.0, 0.0]
    assert gae_estimates == [-1.0, 1.0]


def test_q_boost_same_player_and_terminal():
    rewards = torch.tensor([[0.0], [1.0]])
    q = torch.tensor([[1.0], [1.0]])
    v = torch.tensor([[0.5], [0.7]])
    adv, target = q_boost(
        rewards,
        q,
        v,
        torch.tensor([[False], [True]]),
        torch.zeros(2, 1),
        torch.tensor([99.0]),
        torch.zeros(1),
        lam=1,
    )
    torch.testing.assert_close(target, torch.tensor([[0.7], [1.0]]))
    torch.testing.assert_close(adv, torch.tensor([[0.2], [0.3]]))


def test_private_critic_input_is_separate_from_actor():
    env = BatchEnv(2)
    obs, _ = env.observe()
    before = critic_input(obs, env.state)
    for state in env.state:
        p = int(state[PLAYER])
        hand = torch.where(state[:48] == 1 - p)[0][0]
        stock = torch.where(state[:48] == -1)[0][0]
        state[hand], state[stock] = -1, 1 - p
    other_obs, _ = env.observe()
    after = critic_input(other_obs, env.state)
    assert torch.equal(obs, other_obs)
    assert not torch.equal(before, after)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_vrpo_checkpoint_continuation(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = Config(
        algorithm="vrpo",
        device=device,
        envs=8,
        horizon=8,
        width=32,
        critic_width=32,
        epochs=1,
        critic_epochs=1,
        minibatch=64,
    )
    trainer = Trainer(cfg)
    trainer.update()
    path = tmp_path / "q.pt"
    trainer.save(path)
    original = trainer.update()
    resumed = Trainer.restore(path)
    restored = resumed.update()
    assert original == restored
    assert torch.equal(trainer.env.state, resumed.env.state)
    for k, v in trainer.critic.state_dict().items():
        assert torch.equal(v, resumed.critic.state_dict()[k])


@pytest.mark.cuda
def test_bfloat16_training_step():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = Config(
        device="cuda",
        kind="transformer",
        width=32,
        layers=1,
        precision="bf16",
        envs=8,
        horizon=8,
        epochs=1,
        minibatch=64,
    )
    trainer = Trainer(config)
    metrics = trainer.update()
    assert all(torch.isfinite(torch.tensor(list(metrics.values()))))
