"""Learner-decision PPO integration, loss exclusion and exact restart."""

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from koikoi.agents import NeuralAgent
from koikoi.layout import EPISODE, PHASE, PLAYER, SEED
from koikoi.reference import ReferenceEnv
from koikoi.upstream import UPSTREAM_PATH
from scripts.learner_response import LearnerAgent, LearnerConfig, LearnerTrainer
from scripts.learner_returns import learner_gae

pytestmark = [
    pytest.mark.upstream,
    pytest.mark.skipif(not UPSTREAM_PATH.exists(), reason="Upstream not fetched"),
]


def config(**changes):
    return LearnerConfig(
        **(
            {
                "device": "cpu",
                "envs": 4,
                "horizon": 64,
                "minibatch": 64,
                "epochs": 1,
                "critic_width": 32,
                "critic_warmup_updates": 1,
                "seed": 20293001,
            }
            | changes
        )
    )


def equal_optimizer(left, right):
    assert left["param_groups"] == right["param_groups"]
    assert left["state"].keys() == right["state"].keys()
    for key, values in left["state"].items():
        for name, value in values.items():
            assert torch.equal(value, right["state"][key][name])


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_collector_reference_actions_rewards_returns_counters(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = LearnerTrainer(config(device=device, horizon=192))
    refs = []
    for row in trainer.env.state.cpu().numpy():
        ref = ReferenceEnv.__new__(ReferenceEnv)
        ref.state = row.copy()
        refs.append(ref)
    rows = []
    opponent_ends = 0
    step = trainer.env.step

    def checked(actions):
        nonlocal opponent_ends
        seats = np.array(
            [(lane + int(ref.state[EPISODE])) % 2 for lane, ref in enumerate(refs)]
        )
        players = np.array([int(ref.state[PLAYER]) for ref in refs])
        own = players == seats
        ids = np.array(
            [
                (lane // 2 + int(ref.state[EPISODE]) // 2) % 2
                for lane, ref in enumerate(refs)
            ]
        )
        for index, opponent in enumerate(trainer.frozen.agents.values()):
            expected_actions = opponent.agent.act(trainer.env).cpu().numpy()
            active = (ids == index) & ~own
            np.testing.assert_array_equal(
                actions.cpu().numpy()[active], expected_actions[active]
            )
        expected = np.array([r.step(a) for r, a in zip(refs, actions.cpu().tolist())])
        actual = step(actions)
        np.testing.assert_array_equal(actual.cpu(), expected)
        np.testing.assert_array_equal(
            trainer.env.state.cpu(), np.stack([r.state for r in refs])
        )
        rows.append(
            (own, expected[:, 0] * (1 - 2 * seats), expected[:, 2], ids, expected[:, 1])
        )
        opponent_ends += int((expected[:, 2] * ~own).sum())
        for lane, ref in enumerate(refs):
            if ref.state[PHASE] == 4:
                refs[lane] = ReferenceEnv(
                    int(ref.state[SEED]), "paper8", int(ref.state[EPISODE]) + 1
                )
        return actual

    monkeypatch.setattr(trainer.env, "step", checked)
    batch = trainer.collect()
    own, reward, done, ids, rounds = [
        np.stack([row[i] for row in rows]) for i in range(5)
    ]
    np.testing.assert_array_equal(batch["is_learner"].cpu(), own.ravel())
    np.testing.assert_array_equal(batch["reward"].cpu(), reward.ravel())
    np.testing.assert_array_equal(batch["done"].cpu(), done.ravel())
    with torch.no_grad():
        obs, mask = trainer.observe()
        tail = trainer.actor(obs, mask)[1]
    args = [
        torch.as_tensor(x, device=device)
        for x in (reward.astype("float32"), done.astype(bool), own)
    ]
    expected = learner_gae(
        args[0],
        batch["value"].reshape(192, 4),
        args[1],
        args[2],
        tail,
        trainer.learner_rows(),
        trainer.config.gae_lambda,
    )
    for key, value in zip(("advantage", "return", "learner"), expected):
        torch.testing.assert_close(
            batch[key], value.flatten().to(batch[key].dtype), rtol=0, atol=0
        )
    c = trainer.counters
    assert c["decisions"] == 768 == c["learner_decisions"] + c["published_decisions"]
    assert (
        c["learner_decisions"]
        == int(own.sum())
        == c["valid_learner_decisions"] + c["dropped_learner_tails"]
    )
    assert 0 <= c["dropped_learner_tails"] <= 4
    assert c["rounds"] == c["published_rounds"] == int(rounds.sum())
    assert c["matches"] == c["published_matches"] == int(done.sum()) > 0
    assert opponent_ends > 0
    for index, name in enumerate(trainer.frozen.agents):
        chosen = ids == index
        assert trainer.opponent_work[name] == {
            "decisions": int((chosen & ~own).sum()),
            "rounds": int((rounds * chosen).sum()),
            "matches": int((done * chosen).sum()),
        }
    assert trainer.opponent_value is None
    assert len(trainer.optimizer.param_groups) == 2


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_warmup_restart_and_frozen_weights(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = LearnerTrainer(config(device=device))
    original = trainer.model.actor.checkpoint()["current"]["parameters_sha256"]
    opponents = copy.deepcopy(trainer.frozen.checkpoint())
    assert trainer.update()["critic_only"]
    assert trainer.model.actor.checkpoint()["current"]["parameters_sha256"] == original
    checkpoint = tmp_path / "learner.pt"
    trainer.save(checkpoint)
    metrics = trainer.update()
    assert not metrics["critic_only"]
    assert trainer.model.actor.checkpoint()["current"]["parameters_sha256"] != original
    model = copy.deepcopy(trainer.model.state_dict())
    optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    state, rng = trainer.env.state.clone(), torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all()
    restored = LearnerTrainer.restore(checkpoint)
    assert restored.update() == metrics
    assert trainer.counters == restored.counters
    assert trainer.opponent_work == restored.opponent_work
    assert torch.equal(state, restored.env.state)
    assert torch.equal(rng, torch.get_rng_state())
    for expected, actual in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True):
        assert torch.equal(expected, actual)
    for name, value in model.items():
        assert torch.equal(value, restored.model.state_dict()[name])
    equal_optimizer(optimizer, restored.optimizer.state_dict())
    for name, saved in opponents.items():
        now = restored.frozen.checkpoint()[name]
        assert saved["agent"]["parameters_sha256"] == now["agent"]["parameters_sha256"]
        assert all(
            not p.requires_grad
            for m in restored.frozen.agents[name].agent.models.values()
            for p in m.parameters()
        )
    payload = torch.load(checkpoint, weights_only=False)
    payload["provenance"]["learner_sources"]["learner_returns.py"] = "tampered"
    torch.save(payload, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="identity mismatch"):
        LearnerTrainer.restore(tmp_path / "bad.pt")


def test_invalid_rows_and_empty_minibatch_cannot_move_adam(tmp_path):
    trainer = LearnerTrainer(config(critic_warmup_updates=0))
    trainer.update()  # Ensure Adam has nonzero momentum before testing exclusion.
    batch = trainer.collect()
    invalid = ~batch["learner"].bool()
    assert invalid.any() and (~invalid).any()
    checkpoint = tmp_path / "momentum.pt"
    trainer.save(checkpoint)
    metrics = trainer.minibatch_step(batch)
    model = copy.deepcopy(trainer.model.state_dict())
    optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    restored = LearnerTrainer.restore(checkpoint)
    contaminated = {name: value.clone() for name, value in batch.items()}
    for name in ("obs", "return", "advantage", "old_logp", "target"):
        contaminated[name][invalid] = float("nan")
    torch.testing.assert_close(
        restored.minibatch_step(contaminated), metrics, rtol=0, atol=0
    )
    for name, value in model.items():
        assert torch.equal(value, restored.model.state_dict()[name])
    equal_optimizer(optimizer, restored.optimizer.state_dict())
    counters = restored.counters.copy()
    contaminated["learner"].zero_()
    assert not restored.minibatch_step(contaminated).any()
    counters["skipped_empty_minibatches"] += 1
    assert restored.counters == counters
    equal_optimizer(optimizer, restored.optimizer.state_dict())
    for name, value in model.items():
        assert torch.equal(value, restored.model.state_dict()[name])


def test_parent_initial_actions_and_policy_only_deployment(tmp_path):
    path = Path("runs/cold_ppo_screen/seed6710001_warm/update_00000520.pt")
    if not path.exists():
        pytest.skip("Retained research parent unavailable")
    trainer = LearnerTrainer.from_parent(config(critic_width=256), path)
    assert not trainer.optimizer.state
    checkpoint = tmp_path / "initial.pt"
    trainer.save(checkpoint)
    agent, parent = LearnerAgent(checkpoint), NeuralAgent(path, deterministic=True)
    env = ReferenceEnv(20294001, "paper8")
    phases = set()
    for _ in range(256):
        phases.add(int(env.state[PHASE]))
        action = agent.act(env)
        assert action == parent.act(env)
        env.step(action)
        if env.state[PHASE] == 4:
            env = ReferenceEnv(20294001, "paper8", int(env.state[EPISODE]) + 1)
    assert phases == {0, 1, 2, 3}
    restored = LearnerTrainer.restore(checkpoint)
    assert (
        restored.provenance["weight_initialization"]
        == trainer.provenance["weight_initialization"]
    )
    assert all(value == 0 for value in restored.counters.values())


def test_rejects_unsupported_modes():
    for changes in (
        {"opponent_fraction": 0.5},
        {"returns_backend": "native"},
        {"reward": "round"},
        {"cuda_graph": True},
    ):
        with pytest.raises(ValueError):
            replace(config(), **changes)
