"""Complete multi-round returns against an independent transition oracle."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from koikoi.layout import PHASE, PLAYER, POINTS, ROUND, TERMINAL
from koikoi.published_train import packed_observation
from koikoi.reference import ReferenceEnv
from scripts.public_memory_continuation import (
    ContinuationConfig,
    ContinuationTrainer,
    validate_continuation_bank,
)
from tests.test_final_round_actual_match import StopFirst
from tests.test_final_round_training import PARENT, assert_nested_equal


def bank():
    env = ReferenceEnv(31010001, "paper8")
    actor = StopFirst()
    states = []
    while env.state[PHASE] != TERMINAL:
        previous = int(env.state[ROUND])
        env.step(actor.act(env))
        if int(env.state[ROUND]) != previous and env.state[ROUND] in (1, 3, 5, 7):
            states.append(env.state.copy())
    assert len(states) == 4
    return (
        torch.from_numpy(np.stack(states)),
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 1, 0]),
    )


def test_boundary_validation():
    data = bank()
    validate_continuation_bank(*data)
    bad = data[0].clone()
    bad[0, ROUND] = 0
    with pytest.raises(ValueError, match="rounds2--8"):
        validate_continuation_bank(bad, *data[1:])
    bad = data[0].clone()
    bad[0, POINTS] = 0
    with pytest.raises(ValueError, match="Bankrupt"):
        validate_continuation_bank(bad, *data[1:])
    with pytest.raises(ValueError, match="public identification"):
        validate_continuation_bank(*data[:3], torch.tensor([-1, 0, 0, 0]))
    with pytest.raises(ValueError, match="448"):
        ContinuationConfig(horizon=64)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_reference_returns_matched_warmup_and_restart(tmp_path, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    data = bank()
    config = ContinuationConfig(
        device=device,
        envs=8,
        minibatch=64,
        epochs=1,
        critic_width=256,
        critic_warmup_updates=1,
        seed=9210001,
    )
    trainer = ContinuationTrainer.from_parent(config, PARENT, *data)
    initial = copy.deepcopy(trainer.model.state_dict())
    prefix = copy.deepcopy(trainer.prefix_actor)
    frozen = {n: copy.deepcopy(a.state_dict()) for n, a in trainer.opponents.items()}
    original_step, original_observe, original_collect = (
        trainer.env.step,
        trainer.observe,
        trainer.collect,
    )
    refs, ended_on_opponent = [], []
    seen_rounds = set()

    def observe(labels=False):
        result = original_observe(labels)
        plain, _ = packed_observation(trainer.env)
        obs = result[0].clone()
        features = obs[:, 561:].reshape(-1, 300, 50)
        indices = trainer.last_root_indices
        own = trainer.env.state[:, PLAYER] == data[1][indices].to(device)
        for bit in (0, 1):
            expected = (data[3][indices].to(device) == bit) & own
            assert torch.equal(features[:, 2 + bit, 0].bool(), expected)
        features[:, 2:4] = 0
        assert torch.equal(obs, plain)
        return result

    def step(actions, freeze_done=False):
        assert freeze_done
        if not refs:
            for state in trainer.env.state.cpu().numpy():
                ref = ReferenceEnv.__new__(ReferenceEnv)
                ref.state = state.copy()
                refs.append(ref)
        seats = data[1][trainer.last_root_indices].numpy()
        expected = np.zeros((len(refs), 4), np.int32)
        for i, (ref, action) in enumerate(zip(refs, actions.cpu().tolist())):
            if ref.state[PHASE] != TERMINAL:
                seen_rounds.add(int(ref.state[ROUND]))
                opponent = ref.state[PLAYER] != seats[i]
                expected[i] = ref.step(action)
                ended_on_opponent.append(bool(opponent and expected[i, 2]))
        actual = original_step(actions, freeze_done=True)
        np.testing.assert_array_equal(actual.cpu(), expected)
        np.testing.assert_array_equal(
            trainer.env.state.cpu(), np.stack([r.state for r in refs])
        )
        return actual

    def collect():
        batch = original_collect()
        seats = data[1][trainer.last_root_indices].numpy()
        rewards = torch.tensor(
            [
                np.sign(ref.state[POINTS + seat] - ref.state[POINTS + 1 - seat])
                for ref, seat in zip(refs, seats)
            ],
            dtype=torch.float32,
            device=device,
        )
        torch.testing.assert_close(
            trainer.last_terminal_rewards, rewards, rtol=0, atol=0
        )
        torch.testing.assert_close(
            batch["return"], rewards[batch["lane"]], rtol=0, atol=0
        )
        assert int(trainer.last_round_endings.sum()) > config.envs
        return batch

    monkeypatch.setattr(trainer, "observe", observe)
    monkeypatch.setattr(trainer.env, "step", step)
    monkeypatch.setattr(trainer, "collect", collect)
    trainer.update()
    assert len(seen_rounds) >= 3 and any(ended_on_opponent)
    for key, value in trainer.model.state_dict().items():
        if key.startswith("actor."):
            assert_nested_equal(initial[key], value)
    warm = copy.deepcopy(trainer.model.state_dict())
    counters, indices = trainer.counters.copy(), trainer.last_root_indices.clone()
    monkeypatch.setattr(trainer, "observe", original_observe)
    monkeypatch.setattr(trainer.env, "step", original_step)
    monkeypatch.setattr(trainer, "collect", original_collect)
    saved = tmp_path / "continuation.pt"
    trainer.save(saved)
    metrics = trainer.update()
    weights, optimizer = (
        copy.deepcopy(trainer.model.state_dict()),
        copy.deepcopy(trainer.optimizer.state_dict()),
    )
    resumed = ContinuationTrainer.restore(saved)
    assert resumed.update() == metrics
    assert_nested_equal(resumed.model.state_dict(), weights)
    assert_nested_equal(resumed.optimizer.state_dict(), optimizer)
    assert_nested_equal(prefix, resumed.prefix_actor)
    for name, actor in resumed.opponents.items():
        assert_nested_equal(frozen[name], actor.state_dict())
    assert resumed.counters["matches"] == resumed.counters["rounds"] == 0
    assert resumed.counters["conditional_tail_completions"] == 2 * config.envs
    masked = ContinuationTrainer.from_parent(
        replace(config, public_memory=False), PARENT, *data
    )
    assert_nested_equal(initial, masked.model.state_dict())
    masked.update()
    assert_nested_equal(warm, masked.model.state_dict())
    assert counters == masked.counters and torch.equal(
        indices, masked.last_root_indices
    )
    masked.update()
    assert all(
        not bool(m.encoder_block.f1.weight[:, 2:4].any())
        for m in masked.model.actor.models.values()
    )
    assert any(
        bool(m.encoder_block.f1.weight[:, 2:4].any())
        for m in resumed.model.actor.models.values()
    )
    resumed.memory[0] = 1 - resumed.memory[0]
    with pytest.raises(ValueError, match="memory root bank changed"):
        resumed.collect()
