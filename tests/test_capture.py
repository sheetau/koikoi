import pytest
import torch

from koikoi.capture import CapturedMinibatch
from koikoi.train import Config, Trainer


@pytest.mark.cuda
@pytest.mark.parametrize(
    "kind,precision",
    [("mlp", "float32"), ("relational", "bf16"), ("transformer", "bf16")],
)
def test_capture_does_not_add_updates_and_matches_eager(kind, precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = Config(
        envs=8,
        horizon=8,
        width=32,
        layers=1,
        epochs=1,
        minibatch=64,
        kind=kind,
        precision=precision,
        cuda_graph=True,
    )
    source = Trainer(config)
    data = source.collect()
    eager, captured = Trainer(config), Trainer(config)
    rng = torch.cuda.get_rng_state()
    graph = CapturedMinibatch(
        captured.minibatch_step, captured.model, captured.optimizer, data
    )
    assert torch.equal(rng, torch.cuda.get_rng_state())
    for key, value in eager.model.state_dict().items():
        assert torch.equal(value, captured.model.state_dict()[key])
    assert all(int(state["step"]) == 0 for state in captured.optimizer.state.values())
    for i in range(3):
        batch = {key: value.roll(i, 0) for key, value in data.items()}
        batch["return"] = batch["return"] + i / 10
        expected, actual = eager.minibatch_step(batch), graph(batch)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for key, value in eager.model.state_dict().items():
            torch.testing.assert_close(
                value, captured.model.state_dict()[key], rtol=0, atol=0
            )
        assert all(
            int(state["step"]) == i + 1 for state in captured.optimizer.state.values()
        )


@pytest.mark.cuda
@pytest.mark.parametrize("precision", ["float32", "bf16"])
def test_graph_checkpoint_restores_exact_next_update(tmp_path, precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = Trainer(
        Config(
            envs=8,
            horizon=8,
            width=32,
            epochs=1,
            minibatch=32,
            precision=precision,
            endgame=True,
            cuda_graph=True,
        )
    )
    trainer.update()
    path = tmp_path / "graph.pt"
    trainer.save(path)
    expected = trainer.update()
    state = trainer.env.state.clone()
    weights = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    restored = Trainer.restore(path)
    assert restored.update() == expected
    assert torch.equal(restored.env.state, state)
    for key, value in weights.items():
        assert torch.equal(value, restored.model.state_dict()[key])


def test_graph_rejects_dynamic_batches_and_cpu():
    with pytest.raises(ValueError, match="CUDA PPO"):
        Config(device="cpu", cuda_graph=True)
    with pytest.raises(ValueError, match="same size"):
        Config(envs=3, horizon=3, minibatch=4, cuda_graph=True)
