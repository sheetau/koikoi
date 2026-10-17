"""Complete-match targets against actual reference trajectories and restart."""

import copy

import numpy as np
import pytest
import torch

from koikoi.layout import EPISODE, PHASE, PLAYER, POINTS, ROUND
from koikoi.reference import ReferenceEnv
from koikoi.train import Config, Trainer


def config(device="cpu", **extra):
    return Config(
        **(
            {
                "device": device,
                "profile": "paper8",
                "envs": 8,
                "horizon": 512,
                "width": 32,
                "epochs": 1,
                "minibatch": 128,
                "cpu_threads": 2,
                "endgame": True,
                "complete_matches": True,
                "gae_lambda": 1.0,
            }
            | extra
        )
    )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("lam", [0.95, 1.0])
def test_complete_reference_trajectories_and_returns(device, lam, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = Trainer(config(device, gae_lambda=lam))
    refs = []
    for state in trainer.env.state.cpu().numpy():
        ref = ReferenceEnv.__new__(ReferenceEnv)
        ref.state = state.copy()
        refs.append(ref)
    native_step = trainer.env.step
    trace = []
    rounds = 0

    def checked_step(actions, freeze_done=False):
        nonlocal rounds
        assert freeze_done
        before = trainer.env.state.cpu().numpy().copy()
        expected = []
        for index, (ref, action) in enumerate(zip(refs, actions.cpu().tolist())):
            np.testing.assert_array_equal(before[index], ref.state)
            if ref.state[PHASE] == 4:
                expected.append([0, 0, 0, 0])
                continue
            player = int(ref.state[PLAYER])
            obs = ref.observe()
            result = ref.step(action)
            trace.append((index, player, obs, result))
            expected.append(result)
            rounds += int(result[1])
        result = native_step(actions, freeze_done=True)
        np.testing.assert_array_equal(result.cpu(), expected)
        np.testing.assert_array_equal(
            trainer.env.state.cpu(), np.stack([r.state for r in refs])
        )
        return result

    monkeypatch.setattr(trainer.env, "step", checked_step)
    batch = trainer.collect()
    assert len(trace) == len(batch["action"]) == trainer.counters["decisions"]
    assert trainer.counters["matches"] == 8 and trainer.counters["rounds"] == rounds
    assert trainer.counters["collection_rows"] >= len(trace)
    assert (trainer.env.state[:, EPISODE] == 1).all()
    assert (trainer.env.state[:, ROUND] == 0).all()
    assert (trainer.env.state[:, PHASE] == 0).all()
    assert batch["done"].sum() == 8
    np.testing.assert_array_equal(batch["obs"].cpu(), np.stack([r[2] for r in trace]))
    assert batch["mask"].gather(1, batch["action"][:, None]).all()
    for index, ref in enumerate(refs):
        selection = [i for i, row in enumerate(trace) if row[0] == index]
        sign = np.array([1 - 2 * trace[i][1] for i in selection])
        outcome = int(np.sign(ref.state[POINTS] - ref.state[POINTS + 1]))
        values = batch["value"][selection].cpu().numpy().astype(float) * sign
        # Independent forward lambda-return mixture, seat-zero coordinates.
        expected = []
        for time in range(len(values)):
            future = values[time + 1 :]
            weights = (1 - lam) * lam ** np.arange(len(future))
            expected.append(
                float(np.dot(weights, future)) + lam ** len(future) * outcome
            )
        actual = batch["return"][selection].cpu().numpy()
        np.testing.assert_allclose(actual, np.array(expected) * sign, rtol=0, atol=2e-6)
    torch.testing.assert_close(
        batch["advantage"], batch["return"] - batch["value"], rtol=0, atol=2e-6
    )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_complete_restart_and_optimizer_exact(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = Trainer(config(device))
    trainer.update()
    checkpoint = tmp_path / "full.pt"
    trainer.save(checkpoint)
    expected = trainer.update()
    state = trainer.env.state.clone()
    model = copy.deepcopy(trainer.model.state_dict())
    optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    restored = Trainer.restore(checkpoint)
    assert restored.config.complete_matches
    assert restored.update() == expected
    assert restored.counters == trainer.counters
    assert torch.equal(state, restored.env.state)
    assert torch.equal(rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    for key, value in model.items():
        assert torch.equal(value, restored.model.state_dict()[key])
    actual = restored.optimizer.state_dict()
    assert optimizer["param_groups"] == actual["param_groups"]
    for key, values in optimizer["state"].items():
        for name, value in values.items():
            assert torch.equal(value, actual["state"][key][name])


def test_complete_bound_failure_never_bootstraps():
    trainer = Trainer(config(horizon=1))
    before = copy.deepcopy(trainer.model.state_dict())
    with pytest.raises(RuntimeError, match="horizon exhausted"):
        trainer.collect()
    assert not any(trainer.counters.values())
    assert not trainer.optimizer.state
    assert all(
        torch.equal(value, trainer.model.state_dict()[key])
        for key, value in before.items()
    )
    with pytest.raises(ValueError, match="fresh match"):
        trainer.collect()


@pytest.mark.parametrize(
    "extra",
    [
        {"cuda_graph": True},
        {"cuda_rollout": True},
        {"opponent_fraction": 0.5},
        {"ppo_value": "observation"},
        {"algorithm": "vrpo"},
        {"reward": "round"},
        {"policy_scope": "koi"},
    ],
)
def test_complete_rejects_unsupported_modes(extra):
    with pytest.raises(ValueError, match="Complete matches require"):
        config("cuda", **extra)


def test_complete_torch_returns_and_twelve_rounds():
    trainer = Trainer(
        config(profile="research12", horizon=768, returns_backend="torch")
    )
    batch = trainer.collect()
    assert trainer.counters["rounds"] == 12 * 8
    assert trainer.counters["matches"] == 8
    torch.testing.assert_close(
        batch["return"], batch["return"].round(), atol=2e-6, rtol=0
    )
