"""Public-only targets, true termination, reference replay and exact restart."""

import copy
import hashlib
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from koikoi.complete_matches import collect_complete_matches
from koikoi.layout import DEALER, PHASE, PLAYER, POINTS, ROUND
from koikoi.reference import ReferenceEnv
from koikoi.round_continuation import RoundContinuation
from koikoi.train import Config, Trainer


def spec(parent="0" * 64):
    return {
        "coefficients": [[0.0, 0.1]] + [[2.0 + r, 0.2] for r in range(1, 8)],
        "source": {
            "parent_sha256": parent,
            "report_sha256": "1" * 64,
            "data_sha256": "2" * 64,
            "audit_sha256": "3" * 64,
        },
    }


def make_trainer(tmp_path, device="cpu", **extra):
    base = {
        "device": device,
        "profile": "paper8",
        "envs": 8,
        "horizon": 512,
        "width": 32,
        "epochs": 1,
        "minibatch": 128,
        "cpu_threads": 2,
        "complete_matches": True,
        "gae_lambda": 1.0,
        "endgame": True,
    }
    parent = tmp_path / "parent.pt"
    Trainer(Config(**base)).save(parent)
    metadata = spec(hashlib.sha256(parent.read_bytes()).hexdigest())
    trainer = Trainer(Config(**(base | {"round_continuation": metadata} | extra)))
    trainer.initialize_weights(parent)
    return trainer


def test_known_round_targets_keep_actual_terminals_and_player_signs():
    # Three games: multi-round win, immediate bankruptcy, multi-round draw.
    state = torch.zeros((3, 208), dtype=torch.int32)
    state[:, POINTS : POINTS + 2] = 30
    state[:, PLAYER] = torch.tensor([0, 1, 0])
    state[:, DEALER] = state[:, PLAYER]
    env = SimpleNamespace(state=state, count=3, device=torch.device("cpu"))
    step_index = 0

    def step(actions, freeze_done):
        nonlocal step_index
        assert freeze_done
        if step_index == 0:
            result = torch.tensor(
                [[0, 1, 0, 6], [-1, 1, 1, -30], [0, 1, 0, 0]], dtype=torch.int32
            )
            state[:, ROUND] = torch.tensor([1, 0, 1])
            state[:, PLAYER] = torch.tensor([1, 1, 0])
            state[:, DEALER] = state[:, PLAYER]
            state[0, POINTS : POINTS + 2] = torch.tensor([36, 24])
            state[1, POINTS : POINTS + 2] = torch.tensor([0, 60])
            state[1, PHASE] = 4
        elif step_index == 1:
            result = torch.tensor(
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]], dtype=torch.int32
            )
            state[2, PHASE] = 4
        else:
            result = torch.tensor(
                [[1, 1, 1, 4], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.int32
            )
            state[0, POINTS : POINTS + 2] = torch.tensor([40, 20])
            state[0, PHASE] = 4
        step_index += 1
        return result

    env.step, env.reset_done = step, lambda: None
    trainer = SimpleNamespace(
        env=env,
        config=SimpleNamespace(horizon=4, gae_lambda=1.0, returns_backend="torch"),
        round_continuation=RoundContinuation(spec(), "cpu"),
        observe=lambda labels: (
            torch.zeros(3, 2),
            torch.ones(3, 1, dtype=torch.bool),
            torch.zeros(3, 2),
        ),
        actor=lambda obs, mask: (
            torch.zeros(3, 1),
            torch.tensor([0.7, -0.4, 0.2]),
            None,
        ),
    )
    batch, totals, rows = collect_complete_matches(trainer)
    expected = [math.tanh(0.4 / 2), 1, math.tanh(0.2 / 2), -1, 0, -1]
    torch.testing.assert_close(
        batch["return"], torch.tensor(expected), atol=2e-7, rtol=0
    )
    assert totals.tolist() == [5, 3] and rows == 9
    assert batch["done"].sum() == 3 and batch["learning_done"].sum() == 5
    assert batch["reward"].tolist() == [0, 1, 0, 0, 0, -1]


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_reference_public_round_targets_and_noninterference(
    tmp_path, device, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = make_trainer(tmp_path, device)
    refs = []
    for state in trainer.env.state.cpu().numpy():
        ref = ReferenceEnv.__new__(ReferenceEnv)
        ref.state = state.copy()
        refs.append(ref)
    step = trainer.env.step
    trace, endpoints = [], {}

    def checked_step(actions, freeze_done=False):
        expected = []
        for i, (ref, action) in enumerate(zip(refs, actions.cpu().tolist())):
            if ref.state[PHASE] == 4:
                expected.append([0, 0, 0, 0])
                continue
            player, round_index = int(ref.state[PLAYER]), int(ref.state[ROUND])
            trace.append((i, round_index, player))
            result = ref.step(action)
            expected.append(result)
            if result[1]:
                if result[2]:
                    value = result[0]
                else:
                    s = ref.state
                    a, b = trainer.config.round_continuation["coefficients"][s[ROUND]]
                    value = math.tanh(
                        (a * (s[POINTS] - 30) / 30 + b * (1 - 2 * s[DEALER])) / 2
                    )
                endpoints[(i, round_index)] = value
        actual = step(actions, freeze_done=freeze_done)
        np.testing.assert_array_equal(actual.cpu(), expected)
        np.testing.assert_array_equal(
            trainer.env.state.cpu(), np.stack([r.state for r in refs])
        )
        return actual

    monkeypatch.setattr(trainer.env, "step", checked_step)
    batch = trainer.collect()
    truth = [endpoints[(i, r)] * (1 - 2 * player) for i, r, player in trace]
    np.testing.assert_allclose(batch["return"].cpu(), truth, rtol=0, atol=2e-6)
    assert batch["done"].sum() == trainer.counters["matches"] == 8
    assert batch["learning_done"].sum() == trainer.counters["rounds"] == len(endpoints)
    assert len(trace) == trainer.counters["decisions"]
    # The target API accepts only these three public arrays, not packed states.
    public = (
        torch.arange(8, device=device),
        torch.arange(8, device=device) + 20,
        torch.arange(8, device=device) % 2,
    )
    value = trainer.round_continuation.value(*public)
    swapped = trainer.round_continuation.value(public[0], 60 - public[1], 1 - public[2])
    torch.testing.assert_close(value, -swapped, rtol=0, atol=2e-7)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_embedded_table_resume_exact_without_calibration_files(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = make_trainer(tmp_path, device)
    trainer.update()
    checkpoint = tmp_path / "round.pt"
    trainer.save(checkpoint)
    expected = trainer.update()
    model, optimizer = (
        copy.deepcopy(trainer.model.state_dict()),
        copy.deepcopy(trainer.optimizer.state_dict()),
    )
    state, rng = trainer.env.state.clone(), torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    (tmp_path / "parent.pt").unlink()
    restored = Trainer.restore(checkpoint)
    assert restored.update() == expected
    assert restored.counters == trainer.counters
    assert torch.equal(restored.env.state, state) and torch.equal(
        torch.get_rng_state(), rng
    )
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    for key, value in model.items():
        assert torch.equal(restored.model.state_dict()[key], value)
    actual = restored.optimizer.state_dict()
    assert actual["param_groups"] == optimizer["param_groups"]
    for key, items in optimizer["state"].items():
        for name, value in items.items():
            assert torch.equal(actual["state"][key][name], value)


@pytest.mark.parametrize("mode", ["config", "identity", "ancestry", "missing"])
def test_rejects_checkpoint_table_drift(tmp_path, mode):
    trainer = make_trainer(tmp_path)
    path = tmp_path / "checkpoint.pt"
    trainer.save(path)
    payload = torch.load(path, weights_only=False)
    if mode == "config":
        payload["config"]["round_continuation"]["coefficients"][1][0] += 0.1
    elif mode == "identity":
        payload["provenance"]["round_continuation_sha256"] = "f" * 64
    elif mode == "ancestry":
        payload["provenance"]["initialized_from"]["sha256"] = "f" * 64
    else:
        del payload["provenance"]["round_continuation_sha256"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Round continuation"):
        Trainer.restore(path)


def test_rejects_live_config_drift_and_uncalibrated_parent(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.config.round_continuation["coefficients"][0][1] += 1
    with pytest.raises(ValueError, match="specification changed"):
        trainer.collect()
    with pytest.raises(ValueError, match="specification changed"):
        trainer.save(tmp_path / "bad.pt")
    fresh = Trainer(
        Config(
            device="cpu",
            profile="paper8",
            complete_matches=True,
            gae_lambda=1,
            round_continuation=spec(),
        )
    )
    with pytest.raises(ValueError, match="calibrated parent"):
        fresh.collect()
    with pytest.raises(ValueError, match="calibrated parent"):
        fresh.initialize_weights(tmp_path / "parent.pt")


@pytest.mark.parametrize(
    "extra",
    [{"complete_matches": False}, {"profile": "research12"}, {"gae_lambda": 0.95}],
)
def test_rejects_unsupported_context(extra):
    with pytest.raises(ValueError, match="Round continuation"):
        Config(
            **(
                {
                    "complete_matches": True,
                    "profile": "paper8",
                    "gae_lambda": 1,
                    "round_continuation": spec(),
                }
                | extra
            )
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e300, "1", True])
def test_rejects_bad_coefficients(value):
    data = spec()
    data["coefficients"][1][0] = value
    with pytest.raises(ValueError, match="finite pairs"):
        RoundContinuation(data, "cpu")
