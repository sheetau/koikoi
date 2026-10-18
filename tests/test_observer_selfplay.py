"""Matched warmup, actual trajectories, parent identity and exact restart."""

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from koikoi.layout import EPISODE, PHASE, PLAYER, SEED
from koikoi.published_train import PublishedConfig, PublishedTrainer
from koikoi.reference import ReferenceEnv
from koikoi.upstream import UPSTREAM_PATH
from scripts.observer_selfplay import ObserverConfig, ObserverTrainer
from tests.test_observer_selfplay_returns import forward_oracle

pytestmark = [
    pytest.mark.upstream,
    pytest.mark.skipif(not UPSTREAM_PATH.exists(), reason="Upstream not fetched"),
]


def config(**changes):
    return ObserverConfig(
        **(
            {
                "device": "cpu",
                "envs": 4,
                "horizon": 32,
                "minibatch": 64,
                "epochs": 1,
                "critic_width": 32,
                "critic_warmup_updates": 2,
                "seed": 26030001,
            }
            | changes
        )
    )


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


def test_matched_warmup_and_exact_resume(device, tmp_path):
    trained = []
    for mode in ("observer", "signed"):
        trainer = ObserverTrainer(config(device=device, policy_returns=mode))
        initial = trainer.model.actor.checkpoint()["current"]["parameters_sha256"]
        for _ in range(2):
            assert trainer.update()["critic_only"]
        assert (
            trainer.model.actor.checkpoint()["current"]["parameters_sha256"] == initial
        )
        trained.append(
            (
                copy.deepcopy(trainer.model.state_dict()),
                copy.deepcopy(trainer.optimizer.state_dict()),
                trainer.env.state.clone(),
                trainer.counters.copy(),
            )
        )
        path = tmp_path / f"{mode}.pt"
        trainer.save(path)
        metrics = trainer.update()
        assert not metrics["critic_only"]
        assert (
            trainer.model.actor.checkpoint()["current"]["parameters_sha256"] != initial
        )
        expected = (
            copy.deepcopy(trainer.model.state_dict()),
            copy.deepcopy(trainer.optimizer.state_dict()),
            trainer.env.state.clone(),
            trainer.counters.copy(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all(),
        )
        restored = ObserverTrainer.restore(path)
        assert restored.update() == metrics
        actual = (
            restored.model.state_dict(),
            restored.optimizer.state_dict(),
            restored.env.state,
            restored.counters,
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all(),
        )
        equal(expected, actual)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["provenance"]["observer_sources"]["observer_selfplay_returns.py"] = (
            "tampered"
        )
        bad = tmp_path / f"{mode}_bad.pt"
        torch.save(payload, bad)
        with pytest.raises(ValueError, match="identity mismatch"):
            ObserverTrainer.restore(bad)
    # Policy advantages differ, but the actor is frozen during warmup and the
    # full parameter/Adam/environment/counter states must be exactly identical.
    equal(*trained)


def test_collector_reference_and_independent_return_expansion(device, monkeypatch):
    trainer = ObserverTrainer(config(device=device, horizon=192))
    # Nonzero values ensure the return/seat tests exercise intermediate values.
    with torch.no_grad():
        trainer.model.value[-1].weight.fill_(0.02)
    refs = []
    for row in trainer.env.state.cpu().numpy():
        ref = ReferenceEnv.__new__(ReferenceEnv)
        ref.state = row.copy()
        refs.append(ref)
    rows = []
    native_step = trainer.env.step

    def checked(actions):
        players = [int(ref.state[PLAYER]) for ref in refs]
        expected = np.array(
            [ref.step(action) for ref, action in zip(refs, actions.cpu().tolist())]
        )
        actual = native_step(actions)
        np.testing.assert_array_equal(actual.cpu(), expected)
        np.testing.assert_array_equal(
            trainer.env.state.cpu(), np.stack([ref.state for ref in refs])
        )
        rows.append((players, expected))
        for lane, ref in enumerate(refs):
            if ref.state[PHASE] == 4:
                refs[lane] = ReferenceEnv(
                    int(ref.state[SEED]), "paper8", int(ref.state[EPISODE]) + 1
                )
        return actual

    monkeypatch.setattr(trainer.env, "step", checked)
    batch = trainer.collect()
    players = np.array([row[0] for row in rows])
    results = np.array([row[1] for row in rows])
    np.testing.assert_array_equal(batch["reward"].cpu(), results[:, :, 0].ravel())
    np.testing.assert_array_equal(batch["player"].cpu(), players.ravel())
    np.testing.assert_array_equal(
        batch["done"].cpu(), results[:, :, 2].ravel().astype(bool)
    )
    with torch.no_grad():
        tail = (
            trainer.model.value(trainer.observer_features(trainer.env))
            .squeeze(-1)
            .cpu()
            .numpy()
        )
    advantage, targets = forward_oracle(
        results[:, :, 0],
        batch["value"].reshape(192, 4, 2).cpu().numpy(),
        results[:, :, 2].astype(bool),
        tail,
        0.95,
    )
    chosen = np.take_along_axis(advantage, players[:, :, None], axis=-1).ravel()
    np.testing.assert_allclose(batch["advantage"].cpu(), chosen, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        batch["return"].cpu(), targets.reshape(-1, 2), rtol=2e-5, atol=2e-6
    )
    assert trainer.counters["decisions"] == 768
    assert trainer.counters["value_rows"] == 1536
    assert trainer.counters["matches"] == int(results[:, :, 2].sum()) > 0
    assert (
        trainer.counters["rounds"]
        == int(results[:, :, 1].sum())
        > trainer.counters["matches"]
    )


def test_parent_actor_and_initialization_rng():
    path = Path("runs/cold_ppo_screen/seed6710001_warm/update_00000520.pt")
    if not path.exists():
        pytest.skip("Retained parent unavailable")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    c = config(critic_width=256)
    trainer = ObserverTrainer.from_parent(c, path)
    rng = torch.get_rng_state()
    equal(trainer.model.actor.checkpoint(), payload["actor"])
    assert not trainer.optimizer.state
    for key, value in payload["value"].items():
        actual = trainer.model.value.state_dict()[key]
        equal(actual[:, :560] if key == "0.weight" else actual, value)
    assert not trainer.model.value[0].weight[:, 560].any()
    original_config = PublishedConfig(
        **{key: value for key, value in vars(c).items() if key != "policy_returns"}
    )
    baseline = PublishedTrainer(original_config, payload["actor"])
    baseline.model.value.load_state_dict(payload["value"])
    equal(rng, torch.get_rng_state())
    equal(trainer.env.state, baseline.env.state)
    phases = set()
    for _ in range(96):
        obs, mask = trainer.observe()
        with torch.no_grad():
            new = trainer.actor(obs, mask)
            old = baseline.actor(obs, mask)
        equal(new[0], old[0])
        torch.testing.assert_close(new[1], old[1], rtol=1e-6, atol=1e-6)
        phases.update(obs[:, 560].int().tolist())
        trainer.env.step(new[0].argmax(-1))
        trainer.env.reset_done()
    assert phases == {0, 1, 2, 3}


def test_configuration_rejects_unsupported_controls():
    for change in (
        {"policy_returns": "bad"},
        {"returns_backend": "native"},
        {"opponent_fraction": 0.5},
    ):
        with pytest.raises(ValueError):
            replace(config(), **change)
