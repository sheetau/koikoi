from dataclasses import replace

import pytest
import torch

from koikoi.budget import lineage_budget
from koikoi.layout import EPISODE, PLAYER, ROUND, STATE_SIZE
from koikoi.opponents import learner_turn
from koikoi.train import Config, Trainer


def checkpoint(tmp_path, *, endgame=False, profile="paper8", kind="mlp"):
    path = tmp_path / "opponent.pt"
    Trainer(
        Config(
            device="cpu", envs=4, width=32, endgame=endgame, profile=profile, kind=kind
        )
    ).save(path)
    return path


def config(device="cpu", **extra):
    return Config(
        device=device,
        profile="paper8",
        envs=8,
        horizon=256,
        width=32,
        epochs=1,
        minibatch=256,
        opponent_fraction=0.5,
        **extra,
    )


def assert_tensors(first, second):
    assert first.keys() == second.keys()
    for key in first:
        assert torch.equal(first[key], second[key]), key


def test_seats_alternate_only_on_match_reset():
    state = torch.zeros(8, STATE_SIZE, dtype=torch.int32)
    rows = torch.arange(8)
    assert (
        learner_turn(state, rows, 4).tolist() == [True, False, True, False] + [True] * 4
    )
    state[:, ROUND] = 7
    assert (
        learner_turn(state, rows, 4).tolist() == [True, False, True, False] + [True] * 4
    )
    state[:, EPISODE] += 1
    assert (
        learner_turn(state, rows, 4).tolist() == [False, True, False, True] + [True] * 4
    )
    state[:, PLAYER] = 1
    assert (
        learner_turn(state, rows, 4).tolist() == [True, False, True, False] + [True] * 4
    )


def test_roles_actions_and_complete_match_returns(tmp_path):
    trainer = Trainer(config(gae_lambda=1))
    trainer.set_frozen_opponent(checkpoint(tmp_path))

    # Make opponent action selection unmistakably different, retaining legality.
    def last_legal(obs, mask):
        assert obs.shape[1] == 560  # No packed/private simulator input.
        return torch.arange(50).float().expand_as(mask).masked_fill(~mask, -1e9) * 100

    trainer.frozen.logits = last_legal
    batch = trainer.collect()
    h, n = trainer.config.horizon, trainer.config.envs
    frozen = ~batch["learner"].bool()
    expected = (batch["mask"].long() * torch.arange(50)).max(-1).values
    assert torch.equal(batch["action"][frozen], expected[frozen])
    assert batch["mask"].gather(1, batch["action"][:, None]).all()
    assert batch["learner"].reshape(h, n)[:, 4:].all()
    assert trainer.counters["learner_decisions"] == int(batch["learner"].sum())
    assert 0 < trainer.counters["learner_decisions"] < trainer.counters["decisions"]
    players = batch["player"].reshape(h, n)
    done = batch["done"].reshape(h, n)
    reward = batch["reward"].reshape(h, n)
    returns = batch["return"].reshape(h, n)
    checked = 0
    for column in range(n):
        start = 0
        for end in done[:, column].nonzero().flatten().tolist():
            # Independent complete-match outcome, without using value estimates.
            result_p0 = reward[end, column] * (1 - 2 * players[end, column])
            expected = result_p0 * (1 - 2 * players[start : end + 1, column])
            torch.testing.assert_close(
                returns[start : end + 1, column], expected, atol=2e-6, rtol=0
            )
            start = end + 1
            checked += 1
    assert checked > 5

    # The unfinished tail bootstraps the value for the actual next actor's role.
    with torch.no_grad():
        obs, mask = trainer.observe()
        actor_value = trainer.actor(obs, mask)[1]
        other_value = trainer.role_value(obs)
    for column in range(n):
        endings = done[:, column].nonzero().flatten().tolist()
        start = endings[-1] + 1 if endings else 0
        state = trainer.env.state[column]
        next_player = int(state[PLAYER])
        learner_seat = (int(state[EPISODE]) + column) % 2
        bootstrap = (
            other_value[column]
            if column < 4 and next_player != learner_seat
            else actor_value[column]
        )
        signs = (1 - 2 * next_player) * (1 - 2 * players[start:, column])
        torch.testing.assert_close(
            returns[start:, column], bootstrap * signs, atol=2e-6, rtol=0
        )


def test_opponent_rows_have_no_actor_gradient(tmp_path):
    trainer = Trainer(config(belief_weight=0.3))
    trainer.set_frozen_opponent(checkpoint(tmp_path))
    batch = trainer.collect()
    frozen = ~batch["learner"].bool()
    data = {key: value[frozen] for key, value in batch.items()}
    data["advantage"].fill_(1000)
    data["old_logp"].fill_(-10000)  # Exclusion must also avoid exp overflow.
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    frozen_before = {k: v.clone() for k, v in trainer.frozen.model.state_dict().items()}
    aux_before = {k: v.clone() for k, v in trainer.opponent_value.state_dict().items()}
    metrics = trainer.minibatch_step(data)
    assert torch.isfinite(metrics).all()
    assert metrics[0] == 0 and metrics[2] == 0 and metrics[3] == 0 and metrics[4] == 0
    assert all(
        p.grad is None or torch.count_nonzero(p.grad) == 0
        for p in trainer.model.parameters()
    )
    assert all(p.grad is None for p in trainer.frozen.model.parameters())
    assert_tensors(before, trainer.model.state_dict())
    assert_tensors(frozen_before, trainer.frozen.model.state_dict())
    assert any(
        not torch.equal(v, trainer.opponent_value.state_dict()[k])
        for k, v in aux_before.items()
    )


def test_advantage_normalization_ignores_opponent_rows(tmp_path, monkeypatch):
    trainer = Trainer(config())
    trainer.set_frozen_opponent(checkpoint(tmp_path))
    batch = trainer.collect()
    size = len(batch["advantage"])
    batch["obs"][:, 0] = torch.arange(size)
    mask = batch["learner"].bool()
    batch["advantage"][~mask] = 1000000
    recorded = torch.empty(size)

    def capture(data):
        recorded[data["obs"][:, 0].long()] = data["advantage"]
        return torch.zeros(6)

    monkeypatch.setattr(trainer, "collect", lambda: batch)
    monkeypatch.setattr(trainer, "minibatch_step", capture)
    trainer.update()
    assert torch.count_nonzero(recorded[~mask]) == 0
    assert abs(float(recorded[mask].mean())) < 1e-6
    assert abs(float(recorded[mask].std(unbiased=False)) - 1) < 1e-5


@pytest.mark.parametrize(
    "device,capture,precision",
    [
        ("cpu", False, "float32"),
        pytest.param("cuda", False, "float32", marks=pytest.mark.cuda),
        pytest.param("cuda", True, "float32", marks=pytest.mark.cuda),
        pytest.param("cuda", True, "bf16", marks=pytest.mark.cuda),
    ],
)
def test_frozen_resume_embeds_opponent_and_preserves_full_state(
    tmp_path, device, capture, precision
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    opponent = checkpoint(tmp_path)
    trainer = Trainer(
        config(device, cuda_graph=capture, cuda_rollout=capture, precision=precision)
    )
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
    trainer.set_frozen_opponent(opponent)
    assert torch.equal(rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    frozen_weights = {
        k: v.clone() for k, v in trainer.frozen.model.state_dict().items()
    }
    trainer.update()
    path = tmp_path / "mixed.pt"
    trainer.save(path)
    opponent.unlink()  # Resume must not depend on the original file.
    expected = trainer.update()
    after_rng = (
        torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    )
    restored = Trainer.restore(path)
    assert restored.update() == expected
    actual_rng = (
        torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    )
    assert torch.equal(after_rng, actual_rng)
    assert restored.counters == trainer.counters
    assert torch.equal(restored.env.state, trainer.env.state)
    assert_tensors(restored.trainable.state_dict(), trainer.trainable.state_dict())
    assert_tensors(restored.frozen.model.state_dict(), frozen_weights)
    assert (
        restored.provenance["frozen_opponent"] == trainer.provenance["frozen_opponent"]
    )
    for a, b in zip(
        restored.optimizer.state.values(), trainer.optimizer.state.values()
    ):
        assert_tensors(a, b)


@pytest.mark.cuda
@pytest.mark.parametrize("kind,precision", [("mlp", "float32"), ("relational", "bf16")])
def test_frozen_full_updates_match_eager_and_captured(tmp_path, kind, precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    opponent = checkpoint(tmp_path, kind=kind)
    c = config("cuda", cuda_graph=True, precision=precision, kind=kind)
    eager = Trainer(c)
    eager.set_frozen_opponent(opponent)
    captured = Trainer(replace(c, cuda_rollout=True))
    captured.set_frozen_opponent(opponent)
    for _ in range(2):
        before = torch.cuda.get_rng_state()
        expected = eager.update()
        after = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(before)
        assert captured.update() == expected
        assert torch.equal(after, torch.cuda.get_rng_state())
        assert captured.counters == eager.counters
        assert torch.equal(captured.env.state, eager.env.state)
        assert_tensors(captured.trainable.state_dict(), eager.trainable.state_dict())


def test_invalid_opponent_configuration_and_late_replacement(tmp_path):
    for extra in [
        {"opponent_fraction": -0.1},
        {"opponent_fraction": 1.1},
        {"opponent_fraction": 0.2, "envs": 1},
        {"opponent_fraction": 0.2, "algorithm": "vrpo"},
    ]:
        with pytest.raises(ValueError):
            Config(**extra)
    trainer = Trainer(config())
    with pytest.raises(ValueError, match="Load a frozen"):
        trainer.collect()
    with pytest.raises(ValueError, match="profile mismatch"):
        trainer.set_frozen_opponent(checkpoint(tmp_path, profile="research12"))
    with pytest.raises(ValueError, match="endgame setting"):
        trainer.set_frozen_opponent(checkpoint(tmp_path, endgame=True))
    path = checkpoint(tmp_path)
    trainer.set_frozen_opponent(path)
    with pytest.raises(ValueError, match="fresh trainer"):
        trainer.set_frozen_opponent(path)


def test_budget_separates_learner_decisions_and_excludes_opponent_training():
    counters = {
        "matches": 10,
        "rounds": 80,
        "decisions": 1000,
        "updates": 1,
        "learner_decisions": 750,
    }
    payload = {
        "config": {"profile": "paper8"},
        "counters": counters,
        "provenance": {"frozen_opponent": {"counters": {"matches": 1000000}}},
    }
    result = lineage_budget(payload, "learner")
    assert result["by_profile"]["paper8"] == counters
    payload["counters"]["learner_decisions"] = 1001
    with pytest.raises(ValueError, match="cannot exceed"):
        lineage_budget(payload, "learner")
