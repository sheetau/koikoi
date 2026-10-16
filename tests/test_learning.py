import torch

from koikoi.model import PolicyValue, sample_action
from koikoi.train import Config, Trainer, signed_gae


def test_player_relative_returns():
    # Player 0 chooses discard and capture before player 1 wins the match.
    reward = torch.tensor([[0.], [0.], [1.]])
    players = torch.tensor([[0], [0], [1]])
    dones = torch.tensor([[False], [False], [True]])
    adv, ret = signed_gae(reward, torch.zeros_like(reward), dones, players,
                          torch.tensor([100.]), torch.tensor([0]), lam=1)
    torch.testing.assert_close(ret, torch.tensor([[-1.], [-1.], [1.]]))
    torch.testing.assert_close(adv, ret)


def test_bootstrap_and_no_round_reset():
    reward = torch.zeros(2, 1)
    _, ret = signed_gae(reward, reward, torch.zeros(2, 1, dtype=torch.bool),
                        torch.tensor([[0], [1]]), torch.tensor([0.7]), torch.tensor([1]), lam=1)
    torch.testing.assert_close(ret, torch.tensor([[-0.7], [0.7]]))


def test_both_models_and_legal_sampling():
    for kind in ("mlp", "transformer"):
        model = PolicyValue(kind, width=32, layers=1)
        obs = torch.randn(4, 560)
        mask = torch.zeros(4, 50, dtype=torch.bool)
        mask[:, 7] = True
        logits, value, belief = model(obs, mask)
        actions, logp = sample_action(logits)
        assert (actions == 7).all()
        assert value.shape == (4,) and belief.shape == (4, 48)
        (value.mean() + belief.mean() + logp.mean()).backward()


def test_checkpoint_resumes_exactly(tmp_path):
    cfg = Config(device="cpu", envs=8, horizon=8, width=32, epochs=1, minibatch=64)
    original = Trainer(cfg)
    original.update()
    path = tmp_path / "state.pt"
    original.save(path)
    metrics = original.update()
    state = original.env.state.clone()
    weights = {k: v.clone() for k, v in original.model.state_dict().items()}
    resumed = Trainer.restore(path)
    resumed_metrics = resumed.update()
    assert metrics == resumed_metrics
    assert torch.equal(state, resumed.env.state)
    for key, val in weights.items():
        assert torch.equal(val, resumed.model.state_dict()[key])
