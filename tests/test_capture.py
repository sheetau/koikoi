import pytest
import torch

from koikoi.capture import CapturedMinibatch, CapturedRollout
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
@pytest.mark.parametrize("rollout", [False, True])
def test_graph_checkpoint_restores_exact_next_update(tmp_path, precision, rollout):
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
            cuda_rollout=rollout,
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
    with pytest.raises(ValueError, match="CUDA PPO"):
        Config(device="cpu", cuda_rollout=True)
    with pytest.raises(ValueError, match="CUDA PPO"):
        Config(algorithm="vrpo", cuda_rollout=True)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "kind,precision,endgame",
    [("mlp", "float32", False), ("relational", "bf16", True)],
)
def test_rollout_capture_preserves_games_rng_and_observes_changed_weights(
    kind,
    precision,
    endgame,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = Config(
        envs=8,
        horizon=256,
        width=32,
        profile="paper8",
        kind=kind,
        precision=precision,
        endgame=endgame,
        cuda_rollout=True,
    )
    eager, captured = Trainer(config), Trainer(config)
    initial_state = captured.env.state.clone()
    initial_rng = torch.cuda.get_rng_state()
    graph = CapturedRollout(captured.collect_device, captured.env.state)
    assert torch.equal(captured.env.state, initial_state)
    assert torch.equal(torch.cuda.get_rng_state(), initial_rng)
    assert not any(captured.counters.values())
    previous_state = initial_state
    for i in range(3):
        with torch.no_grad():
            # Updated policy tensors must be read on each replay, including
            # BF16 casts; stale inference-weight caching would fail here.
            for trainer in (eager, captured):
                trainer.model.koi.bias.add_(torch.tensor([0.1, -0.1], device="cuda"))
        before = torch.cuda.get_rng_state()
        expected, expected_totals = eager.collect_device()
        after = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(before)
        actual, actual_totals = graph()
        assert torch.equal(torch.cuda.get_rng_state(), after)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        assert torch.equal(actual_totals, expected_totals)
        assert int(actual_totals[1]) > 0  # Complete matches and resets occurred.
        assert torch.equal(captured.env.state, eager.env.state)
        assert not torch.equal(captured.env.state, previous_state)
        previous_state = captured.env.state.clone()
        actual["advantage"] = torch.zeros_like(actual["advantage"])


@pytest.mark.cuda
@pytest.mark.parametrize(
    "kind,precision",
    [("mlp", "float32"), ("relational", "bf16"), ("transformer", "bf16")],
)
def test_complete_updates_match_with_and_without_rollout_capture(kind, precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from dataclasses import replace

    config = Config(
        envs=8,
        horizon=32,
        width=32,
        kind=kind,
        precision=precision,
        epochs=1,
        minibatch=64,
        cuda_graph=True,
        endgame=True,
    )
    eager = Trainer(config)
    captured = Trainer(replace(config, cuda_rollout=True))
    for _ in range(3):
        rng = torch.cuda.get_rng_state()
        expected = eager.update()
        after = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(rng)
        assert captured.update() == expected
        assert torch.equal(torch.cuda.get_rng_state(), after)
        assert captured.counters == eager.counters
        assert torch.equal(captured.env.state, eager.env.state)
        for key, value in eager.model.state_dict().items():
            assert torch.equal(value, captured.model.state_dict()[key])


@pytest.mark.cuda
@pytest.mark.parametrize("precision", ["float32", "bf16"])
@pytest.mark.parametrize("minibatch_graph", [False, True])
def test_enabling_rollout_capture_on_resume_keeps_next_update(
    tmp_path,
    precision,
    minibatch_graph,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = Trainer(
        Config(
            envs=8,
            horizon=32,
            width=32,
            epochs=1,
            minibatch=64,
            precision=precision,
            cuda_graph=minibatch_graph,
        )
    )
    trainer.update()
    path = tmp_path / "before_rollout_capture.pt"
    trainer.save(path)
    expected = trainer.update()
    rng = torch.cuda.get_rng_state()
    restored = Trainer.restore(path)
    restored.enable_rollout_capture()
    assert restored.provenance["execution_change"]["retains_training_state"]
    assert restored.update() == expected
    assert restored.counters == trainer.counters
    assert torch.equal(restored.env.state, trainer.env.state)
    assert torch.equal(torch.cuda.get_rng_state(), rng)
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(restored.model.state_dict()[key], value)
    restored.save(path)
    again = Trainer.restore(path)
    assert (
        again.provenance["execution_change"] == restored.provenance["execution_change"]
    )
    again.enable_rollout_capture()
    assert again.provenance["execution_change"]["cuda_rollout"]["from"] is False
