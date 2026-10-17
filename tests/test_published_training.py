"""Actual published-opponent trajectories, loss exclusion and exact restart."""

import copy

import numpy as np
import pytest
import torch

from koikoi.agents import EndgameAgent
from koikoi.layout import EPISODE, PLAYER, SEED
from koikoi.reference import ReferenceEnv
from koikoi.train import Config, Trainer
from koikoi.upstream import UPSTREAM_PATH, PublishedAgent
from koikoi.upstream_batch import parameter_digest

pytestmark = [
    pytest.mark.upstream,
    pytest.mark.skipif(not UPSTREAM_PATH.exists(), reason="Upstream not fetched"),
]


def config(device="cpu", **extra):
    values = {
        "device": device,
        "profile": "paper8",
        "envs": 4,
        "horizon": 192,
        "width": 32,
        "epochs": 1,
        "minibatch": 128,
        "cpu_threads": 2,
        "opponent_fraction": 0.5,
        "published_opponent": "wp",
        "endgame": True,
    }
    return Config(**(values | extra))


def equal_state(first, second):
    assert first.keys() == second.keys()
    for key in first:
        assert torch.equal(first[key], second[key]), key


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_actual_teacher_trajectory_returns_and_counts(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = Trainer(config(device, gae_lambda=1))
    original = EndgameAgent(PublishedAgent("wp"))
    refs = []
    for state in trainer.env.state.cpu().numpy():
        ref = ReferenceEnv.__new__(ReferenceEnv)
        ref.state = state.copy()
        refs.append(ref)
    expected = {
        "matches": 0,
        "rounds": 0,
        "decisions": 0,
        "learner_decisions": 0,
        "published_decisions": 0,
        "published_rounds": 0,
        "published_matches": 0,
    }
    native_step = trainer.env.step

    def checked_step(actions):
        before = trainer.env.state.cpu().numpy().copy()
        actions_cpu = actions.cpu().tolist()
        outputs = []
        for row, (ref, action) in enumerate(zip(refs, actions_cpu)):
            np.testing.assert_array_equal(before[row], ref.state)
            learner = row >= 2 or ref.state[PLAYER] == (ref.state[EPISODE] + row) % 2
            expected["learner_decisions"] += int(learner)
            if not learner:
                assert action == original.act(ref)
                expected["published_decisions"] += 1
            output = ref.step(action)
            outputs.append(output)
            expected["matches"] += int(output[2])
            expected["rounds"] += int(output[1])
            expected["decisions"] += 1
            if row < 2:
                expected["published_matches"] += int(output[2])
                expected["published_rounds"] += int(output[1])
        result = native_step(actions)
        np.testing.assert_array_equal(result.cpu(), outputs)
        np.testing.assert_array_equal(
            trainer.env.state.cpu(), np.stack([r.state for r in refs])
        )
        for row, output in enumerate(outputs):
            if output[2]:
                s = refs[row].state
                refs[row] = ReferenceEnv(int(s[SEED]), "paper8", int(s[EPISODE]) + 1)
        return result

    monkeypatch.setattr(trainer.env, "step", checked_step)
    batch = trainer.collect()
    assert {k: trainer.counters[k] for k in expected} == expected
    assert expected["matches"] > 0
    h, n = trainer.config.horizon, trainer.config.envs
    done, player, reward, returns = [
        batch[k].reshape(h, n).cpu() for k in ("done", "player", "reward", "return")
    ]
    for column in range(n):
        start = 0
        for end in done[:, column].nonzero().flatten().tolist():
            outcome_p0 = reward[end, column] * (1 - 2 * player[end, column])
            outcome = outcome_p0 * (1 - 2 * player[start : end + 1, column])
            torch.testing.assert_close(
                returns[start : end + 1, column], outcome, atol=2e-6, rtol=0
            )
            start = end + 1
        with torch.no_grad():
            obs, mask = trainer.observe()
            bootstrap = torch.where(
                trainer.learner_rows(),
                trainer.actor(obs, mask)[1],
                trainer.role_value(obs),
            )[column].cpu()
        signs = (1 - 2 * int(trainer.env.state[column, PLAYER])) * (
            1 - 2 * player[start:, column]
        )
        torch.testing.assert_close(
            returns[start:, column], bootstrap * signs, atol=2e-6, rtol=0
        )


def test_published_rows_only_train_role_critic():
    trainer = Trainer(config(horizon=32, belief_weight=0.3))
    batch = trainer.collect()
    selected = ~batch["learner"].bool()
    assert selected.any()
    data = {key: value[selected] for key, value in batch.items()}
    data["advantage"].fill_(1000)
    data["old_logp"].fill_(-10000)
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    critic_before = {
        k: v.clone() for k, v in trainer.opponent_value.state_dict().items()
    }
    teacher_before = trainer.frozen.agent.checkpoint()["parameters_sha256"]
    metrics = trainer.minibatch_step(data)
    assert torch.isfinite(metrics).all()
    assert (metrics[[0, 2, 3, 4]] == 0).all()
    equal_state(before, trainer.model.state_dict())
    assert any(
        not torch.equal(v, trainer.opponent_value.state_dict()[k])
        for k, v in critic_before.items()
    )
    assert trainer.frozen.agent.checkpoint()["parameters_sha256"] == teacher_before
    assert all(
        p.grad is None
        for m in trainer.frozen.agent.models.values()
        for p in m.parameters()
    )


@pytest.mark.parametrize(
    "device,capture",
    [
        ("cpu", False),
        pytest.param("cuda", False, marks=pytest.mark.cuda),
        pytest.param("cuda", True, marks=pytest.mark.cuda),
    ],
)
def test_published_resume_embeds_weights_and_repeats_update(
    tmp_path, monkeypatch, device, capture
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = Trainer(config(device, horizon=32, cuda_graph=capture))
    trainer.update()
    path = tmp_path / "published.pt"
    trainer.save(path)
    weights = trainer.frozen.agent.checkpoint()["parameters_sha256"]
    expected = trainer.update()
    rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None

    def no_pickle_reload(*args, **kwargs):
        raise AssertionError("Resume must use embedded teacher tensors")

    monkeypatch.setattr("koikoi.upstream_batch.PublishedAgent", no_pickle_reload)
    restored = Trainer.restore(path)
    assert restored.update() == expected
    assert torch.equal(rng, torch.get_rng_state())
    if gpu_rng is not None:
        assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
    assert restored.counters == trainer.counters
    assert (
        restored.provenance["frozen_opponent"] == trainer.provenance["frozen_opponent"]
    )
    assert torch.equal(restored.env.state, trainer.env.state)
    equal_state(restored.trainable.state_dict(), trainer.trainable.state_dict())
    for a, b in zip(
        restored.optimizer.state.values(), trainer.optimizer.state.values()
    ):
        equal_state(a, b)
    assert restored.frozen.agent.checkpoint()["parameters_sha256"] == weights


def test_reject_invalid_modes_and_changed_runtime(tmp_path):
    for extra in [
        {"profile": "research12"},
        {"opponent_fraction": 0},
        {"cuda_rollout": True},
        {"reward": "round"},
        {"published_opponent": "other"},
    ]:
        with pytest.raises(ValueError):
            config(**extra)
    trainer = Trainer(config(horizon=1))
    with pytest.raises(ValueError, match="eager collection"):
        trainer.enable_rollout_capture()
    with pytest.raises(ValueError, match="fresh trainer"):
        trainer.set_frozen_opponent(tmp_path / "unused.pt")
    torch.backends.cudnn.allow_tf32 = True
    try:
        with pytest.raises(ValueError, match="runtime changed"):
            trainer.collect()
        with pytest.raises(ValueError, match="runtime changed"):
            trainer.save(tmp_path / "invalid.pt")
    finally:
        torch.backends.cudnn.allow_tf32 = False


@pytest.mark.parametrize("tamper", ["weights", "identity", "runtime", "missing"])
def test_tampered_published_resume_rejected(tmp_path, tamper):
    trainer = Trainer(config(horizon=1))
    path = tmp_path / "state.pt"
    trainer.save(path)
    payload = torch.load(path, weights_only=False)
    if tamper == "weights":
        state = payload["published_opponent"]["agent"]["weights"][0]
        next(iter(state.values())).view(-1)[0] += 1
    elif tamper == "identity":
        payload["published_opponent"]["agent"]["model_hashes"]["discard"] = "incorrect"
    elif tamper == "runtime":
        payload["published_opponent"]["runtime"]["torch"] = "different"
    else:
        del payload["published_opponent"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Published"):
        Trainer.restore(path)


@pytest.mark.parametrize("variant", ["wp", "point"])
def test_teacher_calls_preserve_rng_and_learner_rows(variant):
    trainer = Trainer(config(horizon=1, published_opponent=variant))
    state = trainer.env.state.clone()
    rng = torch.get_rng_state().clone()
    learner = trainer.learner_rows()
    action = trainer.frozen.actions(trainer.env, learner)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(state, trainer.env.state)
    assert trainer.env.observe()[1][~learner, action[~learner]].all()
    assert (action[learner] == 0).all()
    payload = trainer.frozen.agent.checkpoint()
    assert (
        parameter_digest(copy.deepcopy(payload["weights"]))
        == payload["parameters_sha256"]
    )
