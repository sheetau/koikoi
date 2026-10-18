"""Natural prefixes, complete-tail rewards and exact optimizer restart."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from koikoi.agents import NeuralAgent
from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER, POINTS, ROUND, TERMINAL
from koikoi.reference import ReferenceEnv
from scripts.final_round_training import (
    TailAgent,
    TailConfig,
    TailTrainer,
    prefix_to_final_round,
    validate_bank,
)
from tests.test_final_round_actual_match import ContinueFirst, StopFirst

PARENT = Path("runs/cold_ppo_screen/seed6710001_warm/update_00000520.pt")


def natural_bank():
    native = BatchEnv(1, "cpu", "paper8")
    states, seats, labels = [], [], []
    for i in range(4):
        state, record = prefix_to_final_round(
            {"seed": 31010001 + i, "seat": i % 2},
            StopFirst(),
            StopFirst(),
            native,
        )
        assert state is not None and record["reached_final_round"]
        states.append(state)
        seats.append(i % 2)
        labels.append(i // 2)
    return (
        torch.from_numpy(np.stack(states)),
        torch.tensor(seats),
        torch.tensor(labels),
    )


def config(device="cpu"):
    return TailConfig(
        device=device,
        envs=8,
        minibatch=64,
        epochs=1,
        critic_width=256,
        critic_warmup_updates=1,
        seed=8810001,
    )


def test_prefix_exact_boundary_and_early_completion():
    native = BatchEnv(1, "cpu", "paper8")
    for seed, actor, reaches in (
        (31010001, StopFirst(), True),
        (19, ContinueFirst(), False),
    ):
        for seat in (0, 1):
            root, row = prefix_to_final_round(
                {"seed": seed, "seat": seat}, actor, actor, native
            )
            ref = ReferenceEnv(seed, "paper8")
            rounds = ends = 0
            for action in row["actions"]:
                assert ref.state[ROUND] < 7 and ref.state[PHASE] != TERMINAL
                assert action == actor.act(ref)
                result = ref.step(action)
                rounds += int(result[1])
                ends += int(result[2])
            np.testing.assert_array_equal(row["endpoint"], ref.state)
            assert row["work"] == {
                "decisions": len(row["actions"]),
                "round_endings": rounds,
                "completed_matches": ends,
            }
            assert row["reached_final_round"] == reaches
            if reaches:
                np.testing.assert_array_equal(root, ref.state)
                assert rounds == 7 and ends == 0
            else:
                assert root is None and ends == 1 and rounds < 8
    bank = natural_bank()
    validate_bank(*bank)
    bad = bank[0].clone()
    bad[0, ROUND] = 6
    with pytest.raises(ValueError, match="final-round start"):
        validate_bank(bad, *bank[1:])
    with pytest.raises(ValueError, match="seats"):
        validate_bank(bank[0], torch.full_like(bank[1], 2), bank[2])


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_tail_reference_rewards_masks_accounting(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = TailTrainer(config(device), *natural_bank())
    original = trainer.env.step
    refs = []
    actions_seen, own_seen, opponent_terminal = [], [], []

    def checked(actions, freeze_done=False):
        assert freeze_done
        states = trainer.env.state.cpu().numpy()
        if not refs:
            for state in states:
                ref = ReferenceEnv.__new__(ReferenceEnv)
                ref.state = state.copy()
                refs.append(ref)
        seats = trainer.bank[1][trainer.last_root_indices].numpy()
        active = states[:, PHASE] != TERMINAL
        own = active & (states[:, PLAYER] == seats)
        expected = np.zeros((len(refs), 4), dtype=np.int32)
        for lane, (ref, action) in enumerate(zip(refs, actions.cpu().tolist())):
            if active[lane]:
                assert ref.legal()[action]
                expected[lane] = ref.step(action)
        actual = original(actions, freeze_done=True)
        np.testing.assert_array_equal(actual.cpu(), expected)
        np.testing.assert_array_equal(
            trainer.env.state.cpu(), np.stack([r.state for r in refs])
        )
        actions_seen.append(int(active.sum()))
        own_seen.append(int(own.sum()))
        opponent_terminal.append(int((expected[:, 2].astype(bool) & ~own).sum()))
        return actual

    monkeypatch.setattr(trainer.env, "step", checked)
    batch = trainer.collect()
    seats = trainer.bank[1][trainer.last_root_indices].numpy()
    expected = np.array(
        [
            np.sign(r.state[POINTS + s] - r.state[POINTS + 1 - s])
            for r, s in zip(refs, seats)
        ]
    )
    np.testing.assert_array_equal(batch["return"].cpu(), expected[batch["lane"].cpu()])
    torch.testing.assert_close(
        batch["advantage"], batch["return"] - batch["value"], rtol=0, atol=0
    )
    assert sum(opponent_terminal) > 0
    assert batch["obs"].shape[1] == 15561
    assert not bool(batch["target"].any())
    assert bool(batch["mask"].gather(1, batch["action"][:, None]).all())
    assert bool(torch.isfinite(batch["old_logp"]).all())
    assert trainer.counters["decisions"] == sum(actions_seen)
    assert trainer.counters["learner_decisions"] == sum(own_seen) == len(batch["obs"])
    assert trainer.counters["opponent_decisions"] == sum(actions_seen) - sum(own_seen)
    assert trainer.counters["conditional_tail_completions"] == trainer.env.count
    assert trainer.counters["conditional_round_endings"] == trainer.env.count
    assert trainer.counters["matches"] == trainer.counters["rounds"] == 0


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_parent_freeze_warmup_update_and_exact_resume(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = TailTrainer.from_parent(config(device), PARENT, *natural_bank())
    before = copy.deepcopy(trainer.model.actor.state_dict())
    frozen = {n: copy.deepcopy(a.state_dict()) for n, a in trainer.opponents.items()}
    prefix = copy.deepcopy(trainer.prefix_actor)
    trainer.update()
    assert_nested_equal(before, trainer.model.actor.state_dict())
    path = tmp_path / "tail.pt"
    trainer.save(path)
    expected_metrics = trainer.update()
    expected_weights = copy.deepcopy(trainer.model.state_dict())
    expected_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    expected_indices = trainer.last_root_indices.clone()
    expected_counters = trainer.counters.copy()
    resumed = TailTrainer.restore(path)
    assert resumed.update() == expected_metrics
    assert_nested_equal(expected_weights, resumed.model.state_dict())
    assert_nested_equal(expected_optimizer, resumed.optimizer.state_dict())
    assert_nested_equal(expected_counters, resumed.counters)
    assert torch.equal(expected_indices, resumed.last_root_indices)
    assert any(
        not torch.equal(before[k], resumed.model.actor.state_dict()[k]) for k in before
    )
    assert_nested_equal(prefix, resumed.prefix_actor)
    for name, actor in resumed.opponents.items():
        assert_nested_equal(frozen[name], actor.state_dict())
        assert not any(p.requires_grad for p in actor.parameters())
    resumed.save(path)
    agent = TailAgent(path)
    parent = NeuralAgent(PARENT, deterministic=True)
    env = ReferenceEnv(31010001, "paper8")
    count = 0
    while env.state[ROUND] < 7 and env.state[PHASE] != TERMINAL:
        action = parent.act(env)
        assert agent.act(env) == action
        env.step(action)
        count += 1
    assert count > 0 and env.state[ROUND] == 7
    assert agent.act(env) == agent.tail.act(env)
    # A changed root bank is rejected instead of silently changing the target.
    resumed.bank[0][0, POINTS] += 1
    with pytest.raises(ValueError, match="Root bank changed"):
        resumed.collect()
