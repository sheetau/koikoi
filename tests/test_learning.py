import pytest
import torch

from koikoi.model import PolicyValue, sample_action
from koikoi.train import Config, Trainer, signed_gae


def test_player_relative_returns():
    # Player 0 chooses discard and capture before player 1 wins the match.
    reward = torch.tensor([[0.0], [0.0], [1.0]])
    players = torch.tensor([[0], [0], [1]])
    dones = torch.tensor([[False], [False], [True]])
    adv, ret = signed_gae(
        reward,
        torch.zeros_like(reward),
        dones,
        players,
        torch.tensor([100.0]),
        torch.tensor([0]),
        lam=1,
    )
    torch.testing.assert_close(ret, torch.tensor([[-1.0], [-1.0], [1.0]]))
    torch.testing.assert_close(adv, ret)


def test_bootstrap_and_no_round_reset():
    reward = torch.zeros(2, 1)
    _, ret = signed_gae(
        reward,
        reward,
        torch.zeros(2, 1, dtype=torch.bool),
        torch.tensor([[0], [1]]),
        torch.tensor([0.7]),
        torch.tensor([1]),
        lam=1,
    )
    torch.testing.assert_close(ret, torch.tensor([[-0.7], [0.7]]))


def test_both_models_and_legal_sampling():
    for kind in ("mlp", "features", "relational", "transformer"):
        model = PolicyValue(kind, width=32, layers=1)
        obs = torch.randn(4, 560)
        mask = torch.zeros(4, 50, dtype=torch.bool)
        mask[:, 7] = True
        logits, value, belief = model(obs, mask)
        actions, logp = sample_action(logits)
        assert (actions == 7).all()
        assert value.shape == (4,) and belief.shape == (4, 48)
        (value.mean() + belief.mean() + logp.mean()).backward()


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("assisted", [False, True])
def test_checkpoint_resumes_exactly(tmp_path, device, assisted):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = Config(
        device=device,
        envs=8,
        horizon=8,
        width=32,
        epochs=1,
        minibatch=64,
        endgame=assisted,
    )
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


def test_transfer_resets_training_budget_and_preserves_lineage(tmp_path):
    cfg = {
        "device": "cpu",
        "envs": 8,
        "horizon": 8,
        "width": 32,
        "epochs": 1,
        "minibatch": 64,
    }
    original = Trainer(Config(**cfg, profile="paper8"))
    original.update()
    path = tmp_path / "paper8.pt"
    original.save(path)
    fresh = Trainer(Config(**cfg, profile="research12", seed=13))
    state = fresh.env.state.clone()
    rng = torch.get_rng_state().clone()
    fresh.initialize_weights(path)
    assert not any(fresh.counters.values())
    assert not fresh.optimizer.state
    assert torch.equal(state, fresh.env.state)
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in original.model.state_dict().items():
        assert torch.equal(value, fresh.model.state_dict()[key])
    lineage = fresh.provenance["initialized_from"]
    assert lineage["profile"] == "paper8"
    assert lineage["counters"] == original.counters
    fresh.update()
    transferred = tmp_path / "research12.pt"
    fresh.save(transferred)
    resumed = Trainer.restore(transferred)
    assert resumed.provenance["initialized_from"] == lineage
    assert resumed.counters["updates"] == 1
    with pytest.raises(ValueError, match="fresh trainer"):
        resumed.initialize_weights(path)


def test_transfer_rejects_incompatible_model(tmp_path):
    path = tmp_path / "mlp.pt"
    Trainer(Config(device="cpu", envs=2, width=32)).save(path)
    target = Trainer(Config(device="cpu", envs=2, kind="relational", width=32))
    with pytest.raises(ValueError, match="architecture mismatch"):
        target.initialize_weights(path)
