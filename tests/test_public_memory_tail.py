"""Public information boundaries, matched initialization and exact restart."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from koikoi.agents import NeuralAgent
from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.published_train import packed_observation
from koikoi.reference import ReferenceEnv
from koikoi.upstream import PublishedAgent
from scripts.public_memory_tail import (
    MemoryTailAgent,
    MemoryTailConfig,
    MemoryTailTrainer,
    candidate_code,
    condition_observation,
    load_public_memory,
)
from scripts.revealed_opponent_inputs import public_snapshot
from scripts.revealed_opponent_models import StrictPublishedPrediction
from tests.test_final_round_training import PARENT, assert_nested_equal, natural_bank


def config(device):
    return MemoryTailConfig(
        device=device,
        envs=8,
        minibatch=64,
        epochs=1,
        critic_width=256,
        critic_warmup_updates=1,
        seed=9010001,
    )


def test_public_codes_and_reserved_feature_boundary():
    assert [candidate_code(x) for x in (set(), {"wp", "point"}, {"wp"}, {"point"})] == [
        -1,
        -1,
        0,
        1,
    ]
    obs = torch.zeros(3, 15561)
    obs[:, :561] = torch.arange(561)
    codes = torch.tensor([-1, 0, 1])
    actual = condition_observation(obs, codes)
    assert torch.equal(actual[:, :561], obs[:, :561])
    rows = actual[:, 561:].reshape(3, 300, 50)
    assert rows[0].sum() == 0 and rows[1, 2].sum() == rows[2, 3].sum() == 50
    assert rows.sum() == 100
    assert torch.equal(condition_observation(obs, codes, False), obs)
    with pytest.raises(ValueError, match="not empty"):
        condition_observation(actual, codes)
    with pytest.raises(ValueError, match="candidate codes"):
        condition_observation(obs, torch.tensor([-1, 0, 2]))
    with pytest.raises(ValueError, match="candidate class"):
        candidate_code({"human"})


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_matched_initialization_reference_returns_and_restart(
    tmp_path, device, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    bank, memory = natural_bank(), torch.tensor([0, 1, -1, 1])
    trainer = MemoryTailTrainer.from_parent(config(device), PARENT, *bank, memory)
    initial = copy.deepcopy(trainer.model.state_dict())
    prefix = copy.deepcopy(trainer.prefix_actor)
    frozen = {n: copy.deepcopy(a.state_dict()) for n, a in trainer.opponents.items()}
    # Building an independent comparison model initializes its unused critic;
    # preserve the training RNG so the two warmup arms have identical draws.
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    parent = NeuralAgent(PARENT, device=device, deterministic=True)
    trainer.env.state.copy_(bank[0].repeat(2, 1).to(device))
    obs, mask = packed_observation(trainer.env)
    codes = memory.repeat(2).to(device)
    # Zero-initialized columns preserve scores and values, for every code.
    with torch.no_grad():
        expected = parent.model(obs, mask)
        actual = trainer.model(condition_observation(obs, codes), mask)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.set_rng_state(cpu_rng)
    if cuda_rng:
        torch.cuda.set_rng_state_all(cuda_rng)
    original_step = trainer.env.step
    original_observe = trainer.observe
    refs, opponent_endings = [], []

    def observe(labels=False):
        result = original_observe(labels)
        plain, plain_mask = packed_observation(trainer.env)
        features = result[0][:, 561:].reshape(-1, 300, 50)
        selected = trainer.last_root_indices
        own = trainer.env.state[:, PLAYER].cpu() == bank[1][selected]
        assert not bool(features[~own.to(device), 2:4].any())
        # Value and original actor inputs are untouched outside the two rows.
        check = result[0].clone()
        check[:, 561:].reshape(-1, 300, 50)[:, 2:4] = 0
        assert torch.equal(check, plain)
        expected_codes = memory[selected].to(device)
        for bit in range(2):
            assert torch.equal(
                features[:, 2 + bit, 0].bool(), (expected_codes == bit) & own.to(device)
            )
        assert bool((result[1] <= plain_mask).all())
        return result

    def step(actions, freeze_done=False):
        assert freeze_done
        if not refs:
            for state in trainer.env.state.cpu().numpy():
                ref = ReferenceEnv.__new__(ReferenceEnv)
                ref.state = state.copy()
                refs.append(ref)
        seats = bank[1][trainer.last_root_indices].numpy()
        expected = np.zeros((len(refs), 4), np.int32)
        for i, (ref, action) in enumerate(zip(refs, actions.cpu().tolist())):
            if ref.state[PHASE] != TERMINAL:
                opponent = ref.state[PLAYER] != seats[i]
                expected[i] = ref.step(action)
                opponent_endings.append(bool(opponent and expected[i, 2]))
        actual = original_step(actions, freeze_done=True)
        np.testing.assert_array_equal(actual.cpu(), expected)
        np.testing.assert_array_equal(
            trainer.env.state.cpu(), np.stack([r.state for r in refs])
        )
        return actual

    monkeypatch.setattr(trainer, "observe", observe)
    monkeypatch.setattr(trainer.env, "step", step)
    # Independent reference transitions check a complete critic-warmup batch.
    trainer.update()
    assert any(opponent_endings)
    seats = bank[1][trainer.last_root_indices].numpy()
    rewards = [
        np.sign(r.state[POINTS + s] - r.state[POINTS + 1 - s])
        for r, s in zip(refs, seats)
    ]
    np.testing.assert_array_equal(trainer.last_terminal_rewards.cpu(), rewards)
    assert_nested_equal(
        initial["actor.models.0.encoder_block.f1.weight"],
        trainer.model.state_dict()["actor.models.0.encoder_block.f1.weight"],
    )
    warm = copy.deepcopy(trainer.model.state_dict())
    indices = trainer.last_root_indices.clone()
    counters = trainer.counters.copy()
    monkeypatch.setattr(trainer, "observe", original_observe)
    monkeypatch.setattr(trainer.env, "step", original_step)
    saved = tmp_path / "memory.pt"
    trainer.save(saved)
    metrics = trainer.update()
    weights = copy.deepcopy(trainer.model.state_dict())
    optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    resumed = MemoryTailTrainer.restore(saved)
    assert resumed.update() == metrics
    assert_nested_equal(resumed.model.state_dict(), weights)
    assert_nested_equal(resumed.optimizer.state_dict(), optimizer)
    assert_nested_equal(prefix, resumed.prefix_actor)
    for name, actor in resumed.opponents.items():
        assert_nested_equal(frozen[name], actor.state_dict())
    assert any(
        bool(m.encoder_block.f1.weight[:, 2:4].any())
        for m in resumed.model.actor.models.values()
    )
    assert resumed.counters["matches"] == resumed.counters["rounds"] == 0
    # Matched masked control has the same initial parameters, roots and warmup.
    masked = MemoryTailTrainer.from_parent(
        replace(config(device), public_memory=False), PARENT, *bank, memory
    )
    assert_nested_equal(initial, masked.model.state_dict())
    masked.update()
    assert_nested_equal(warm, masked.model.state_dict())
    assert torch.equal(indices, masked.last_root_indices)
    assert counters == masked.counters
    masked.update()
    assert all(
        not bool(m.encoder_block.f1.weight[:, 2:4].any())
        for m in masked.model.actor.models.values()
    )
    resumed.memory[0] = -1
    with pytest.raises(ValueError, match="memory root bank changed"):
        resumed.collect()


def test_audited_training_join_and_online_prefix_memory(tmp_path):
    base = Path("runs/final_round_training_study")
    diagnostic = Path("runs/revealed_opponent_identification")
    memory = load_public_memory(base, diagnostic, 1)
    assert len(memory) == 938 and int((memory >= 0).sum()) == 797
    bank = torch.load(base / "parent1/bank.pt", weights_only=False)
    trainer = MemoryTailTrainer.from_parent(config("cpu"), PARENT, *bank, memory)
    path = tmp_path / "agent.pt"
    trainer.save(path)
    predictors = {
        n: StrictPublishedPrediction(PublishedAgent(n)) for n in ("wp", "point")
    }
    agent = MemoryTailAgent(path, predictors)
    rows = [
        json.loads(x)
        for x in (base / "parent1/source.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    row = next(
        r for r in rows if r["reached_final_round"] and memory[r["root_index"]] >= 0
    )
    agent.start_match("online-fixture", row["seat"])
    env = ReferenceEnv(row["seed"], "paper8")
    for decision, action in enumerate(row["actions"]):
        opponent = int(env.state[PLAYER]) != row["seat"]
        if opponent:
            snapshot = public_snapshot(
                env.observe(),
                int(env.state[PLAYER]),
                decision,
                match_key="online-fixture",
            )
        else:
            assert agent.act(env) == action
        env.step(action)
        if opponent:
            agent.observe_opponent(snapshot, action)
    assert candidate_code(agent.tracker.candidates) == memory[row["root_index"]]
    np.testing.assert_array_equal(env.state, bank[0][row["root_index"]])
    agent.start_match("next-match", row["seat"])
    assert agent.tail_code is None and agent.tracker.candidates == {"wp", "point"}
