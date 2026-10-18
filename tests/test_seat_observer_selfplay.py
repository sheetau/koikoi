"""Public-seat information, fixed actor initialization and real CPU restart."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from koikoi.env import BatchEnv
from koikoi.layout import EPISODE, PHASE, PLAYER, SEED
from koikoi.published_train import PublishedConfig, PublishedTrainer, packed_observation
from koikoi.reference import ReferenceEnv
from scripts.diagnose_observer_seat_information import relabel
from scripts.observer_selfplay import ObserverConfig, ObserverTrainer
from scripts.seat_observer_selfplay import (
    SeatObserverConfig,
    SeatObserverFeatures,
    SeatObserverTrainer,
    load_policy,
)
from tests.test_observer_selfplay import equal


@pytest.fixture(scope="module")
def states():
    ref = ReferenceEnv(26090001, "paper8")
    rng = np.random.default_rng(26090002)
    rows = []
    rounds = matches = 0
    for _ in range(256):
        rows.append(ref.state.copy())
        outcome = ref.step(int(rng.choice(np.flatnonzero(ref.legal()))))
        rounds += int(outcome[1])
        matches += int(outcome[2])
        if ref.state[PHASE] == 4:
            ref = ReferenceEnv(
                int(ref.state[SEED]), "paper8", int(ref.state[EPISODE]) + 1
            )
    assert set(np.stack(rows)[:, PHASE]) == {0, 1, 2, 3}
    print(
        {
            "feature_source_work": {
                "decisions": 256,
                "rounds": rounds,
                "matches": matches,
            }
        }
    )
    return np.stack(rows)


def config(**extra):
    return SeatObserverConfig(
        **(
            {
                "device": "cpu",
                "envs": 4,
                "horizon": 32,
                "minibatch": 64,
                "epochs": 1,
                "critic_width": 32,
                "critic_warmup_updates": 2,
                "seed": 26091001,
            }
            | extra
        )
    )


def test_public_features_and_hidden_noninterference(states):
    env = BatchEnv(len(states), "cpu", "paper8")
    env.state.copy_(torch.from_numpy(states))
    extractor = SeatObserverFeatures(env, "public")
    features = extractor(env)
    assert features.shape == (256, 2, 562)
    for i, state in enumerate(states):
        for seat in (0, 1):
            ref = ReferenceEnv.__new__(ReferenceEnv)
            ref.state = state.copy()
            ref.state[PLAYER] = seat
            expected = np.r_[
                ref.observe(), float(state[PLAYER] == seat), float(seat)
            ].astype("float32")
            np.testing.assert_array_equal(features[i, seat], expected)
    assert torch.equal(env.state, torch.from_numpy(states))
    env.state.copy_(torch.from_numpy(np.stack([relabel(s) for s in states])))
    renamed = extractor(env).flip(1)
    assert torch.equal(features[..., :561], renamed[..., :561])
    assert torch.equal(features[..., 561], 1 - renamed[..., 561])
    for seat in (0, 1):
        env.state.copy_(torch.from_numpy(states))
        for row in env.state:
            hidden = torch.where((row[:48] == 1 - seat) | (row[:48] == -1))[0]
            row[hidden] = row[hidden].flip(0)
            row[72:96] = row[72:96].flip(0)
            row[SEED] += 123
        assert torch.equal(features[:, seat], extractor(env)[:, seat])
    env.state.copy_(torch.from_numpy(states))
    masked = SeatObserverFeatures(env, "masked")(env)
    assert torch.equal(masked[..., :561], features[..., :561])
    assert not masked[..., 561].any()


@torch.no_grad()
def test_initial_parent_actor_value_and_active_seat(states, tmp_path):
    parent = PublishedTrainer(PublishedConfig(device="cpu", critic_width=32))
    parent.model.value[-1].weight.fill_(0.02)
    path = tmp_path / "parent.pt"
    parent.save(path)
    env = BatchEnv(len(states), "cpu", "paper8")
    env.state.copy_(torch.from_numpy(states))
    obs, mask = packed_observation(env)
    expected = parent.model(obs, mask)
    initial = []
    for context in ("public", "masked"):
        trainer = SeatObserverTrainer.from_parent(config(seat_context=context), path)
        state = torch.get_rng_state().clone()
        actual = trainer.model(obs, mask)
        assert torch.equal(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-6, atol=1e-7)
        features = SeatObserverFeatures(env, context)(env)
        assert torch.equal(
            trainer.model.active_features(obs),
            features[torch.arange(len(states)), env.state[:, PLAYER].long()],
        )
        assert not trainer.model.value[0].weight[:, 560:].any()
        initial.append(
            (
                copy.deepcopy(trainer.model.state_dict()),
                trainer.env.state.clone(),
                state,
            )
        )
        output = tmp_path / f"{context}.pt"
        trainer.save(output)
        loaded = load_policy(output)
        equal(trainer.model(obs, mask), loaded(obs, mask))
        with pytest.raises(ValueError, match="explicit observer"):
            trainer.model.value(obs[:, :560])
    equal(*initial)


@pytest.mark.parametrize("context", ["public", "masked"])
def test_matched_return_warmup_and_exact_cpu_restart(context, tmp_path):
    warmups = []
    for mode in ("observer", "signed"):
        trainer = SeatObserverTrainer(config(seat_context=context, policy_returns=mode))
        # A zero critic before the first match end has zero TD targets. Give
        # both modes the same nonzero fixture so these short rollouts exercise
        # actual critic gradients, including the added public-seat column.
        with torch.no_grad():
            trainer.model.value[-1].weight.fill_(0.02)
        initial_actor = trainer.model.actor.checkpoint()["current"]["parameters_sha256"]
        for _ in range(2):
            assert trainer.update()["critic_only"]
        assert (
            trainer.model.actor.checkpoint()["current"]["parameters_sha256"]
            == initial_actor
        )
        warmups.append(
            (
                copy.deepcopy(trainer.model.state_dict()),
                copy.deepcopy(trainer.optimizer.state_dict()),
                trainer.env.state.clone(),
                trainer.counters.copy(),
                torch.get_rng_state(),
            )
        )
        if context == "public":
            assert trainer.model.value[0].weight[:, 561].abs().sum() > 0
        else:
            assert not trainer.model.value[0].weight[:, 561].any()
        path = tmp_path / f"{context}_{mode}.pt"
        trainer.save(path)
        metrics = trainer.update()
        assert not metrics["critic_only"]
        expected = (
            copy.deepcopy(trainer.model.state_dict()),
            copy.deepcopy(trainer.optimizer.state_dict()),
            trainer.env.state.clone(),
            trainer.counters.copy(),
            torch.get_rng_state(),
        )
        restored = SeatObserverTrainer.restore(path)
        assert restored.update() == metrics
        equal(
            expected,
            (
                restored.model.state_dict(),
                restored.optimizer.state_dict(),
                restored.env.state,
                restored.counters,
                torch.get_rng_state(),
            ),
        )
        assert (
            trainer.model.actor.checkpoint()["current"]["parameters_sha256"]
            != initial_actor
        )
        # Restored continuation executes again and is counted separately.
        warm = warmups[-1][3]
        print(
            {
                "case": [context, mode],
                "executed_work": {
                    k: 2 * trainer.counters[k] - warm[k]
                    for k in ("decisions", "rounds", "matches", "value_rows")
                },
            }
        )
        payload = torch.load(path, weights_only=False)
        payload["provenance"]["seat_observer_source"] = "tampered"
        bad = tmp_path / f"bad_{context}_{mode}.pt"
        torch.save(payload, bad)
        with pytest.raises(ValueError, match="identity mismatch"):
            SeatObserverTrainer.restore(bad)
    equal(*warmups)


def test_row_independence_and_no_implicit_legacy_loading(tmp_path):
    trainer = SeatObserverTrainer(config())
    features = trainer.observer_features(trainer.env)
    with torch.no_grad():
        trainer.model.value[-1].weight.fill_(0.02)
        trainer.model.value[0].weight[:, 561].fill_(0.2)
        before = trainer.model.value(features)
        other = features.clone()
        other[:, 1, :560] = torch.randn_like(other[:, 1, :560])
        assert torch.equal(before[:, 0], trainer.model.value(other)[:, 0])
        flipped = features.clone()
        flipped[..., 561] = 1 - flipped[..., 561]
        assert not torch.equal(before, trainer.model.value(flipped))
    old = ObserverTrainer(ObserverConfig(device="cpu", critic_width=32))
    path = tmp_path / "old.pt"
    old.save(path)
    with pytest.raises(ValueError, match="Unsupported"):
        SeatObserverTrainer.restore(path)
    with pytest.raises(ValueError, match="Unsupported"):
        load_policy(path)
    with pytest.raises(ValueError, match="seat context"):
        replace(config(), seat_context="hidden")
