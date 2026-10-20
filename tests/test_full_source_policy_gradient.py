import numpy as np
import pytest
import torch

from koikoi.agents import solved_koi_action
from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.reference import ReferenceEnv
from scripts.branch_policy_gradient import collect_roots
from scripts.full_source_policy_gradient import (
    Config,
    Trainer,
    collect_full_source,
    full_return_loss,
)
from scripts.run_arbitration_study import PARENTS
from scripts.train_branch_policy_gradient import Config as OldConfig
from scripts.train_branch_policy_gradient import Trainer as OldTrainer
from tests.test_priority_branch import equal


def test_variable_length_full_gradient_and_root_conditional_variance():
    theta = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
    p = torch.stack([theta, -theta]).softmax(0)
    objective = 2 * p[0] + p[1] * (-p[0] + 4 * p[1])
    expectation = theta * 0
    for chance, actions, reward in (
        (p[0], [0], 2),
        (p[1] * p[0], [1, 0], -1),
        (p[1] * p[1], [1, 1], 4),
    ):
        n = len(actions)
        returns = torch.full((n,), reward, dtype=torch.float64, requires_grad=True)
        loss = full_return_loss(p.expand(n, -1), torch.tensor(actions), returns, 1)
        assert (
            torch.autograd.grad(loss, returns, allow_unused=True, retain_graph=True)[0]
            is None
        )
        expectation -= chance.detach() * loss
        single = torch.stack(
            [
                torch.autograd.grad(p[a].log() * reward, theta, retain_graph=True)[0]
                for a in actions
            ]
        )
        full = torch.autograd.grad(-loss, theta, retain_graph=True)[0]
        torch.testing.assert_close(full, (n * single).mean())
        # The root-selection contribution is nonnegative for every path.
        assert float((n * single).square().mean() - full.square()) >= -1e-12
    estimated = torch.autograd.grad(expectation, theta, retain_graph=True)[0]
    exact = torch.autograd.grad(objective, theta)[0]
    torch.testing.assert_close(estimated, exact, atol=1e-12, rtol=1e-12)


def test_chunk_accumulation_preserves_match_normalization():
    weights = torch.tensor([0.2, -0.4], dtype=torch.float64, requires_grad=True)
    features = torch.tensor(
        [[1, 2], [3, -1], [2, 4], [-2, 1], [1, -3]], dtype=torch.float64
    )
    actions = torch.tensor([0, 1, 1, 0, 1])
    returns = torch.tensor([1, -1, 0, 1, -1], dtype=torch.float64)

    def loss(start, end):
        scores = features[start:end] * weights
        return full_return_loss(
            scores.softmax(-1), actions[start:end], returns[start:end], 2
        )

    whole = torch.autograd.grad(loss(0, 5), weights)[0]
    chunks = sum(
        torch.autograd.grad(loss(a, b), weights)[0] for a, b in ((0, 2), (2, 4), (4, 5))
    )
    torch.testing.assert_close(chunks, whole, atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError):
        full_return_loss(torch.ones(1, 2) / 2, torch.tensor([0]), torch.ones(1), 0)


class FixtureActor(torch.nn.Module):
    def forward(self, features, phases, legal):
        return (features[:, :3].sum(1) * 0.01).masked_fill(~legal, -torch.inf)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_all_source_states_actions_returns_and_old_collection_compatibility(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)

    def collect(fn):
        return fn(
            BatchEnv(4, device, seed=119010001, debug=True),
            FixtureActor().to(device),
            torch.Generator(device=device).manual_seed(31),
            torch.Generator(device=device).manual_seed(37),
        )

    old, data = collect(collect_roots), collect(collect_full_source)
    for key in old:
        equal(old[key], data[key])
    states = data["decision_states"].cpu().numpy()
    lanes = data["decision_lanes"].cpu().numpy()
    steps = data["decision_steps"].cpu().numpy()
    actions = data["decision_actions"].cpu().numpy()
    expected_rows = []
    games = []
    for initial in data["initial"].cpu().numpy():
        env = ReferenceEnv()
        env.state = initial.copy()
        games.append(env)
    for step, row in enumerate(data["actions"].numpy()):
        for lane, (env, action) in enumerate(zip(games, row)):
            if env.state[PHASE] == TERMINAL:
                continue
            legal = env.legal()
            forced = solved_koi_action(env)
            if sum(legal) > 1 and forced is None:
                expected_rows.append((step, lane, env.state.copy(), int(action)))
            env.step(int(action))
    assert len(expected_rows) == int(data["eligible_decisions"].sum()) == len(states)
    for index, (step, lane, state, action) in enumerate(expected_rows):
        assert (
            steps[index] == step and lanes[index] == lane and actions[index] == action
        )
        np.testing.assert_array_equal(states[index], state)
        points = games[lane].state[POINTS : POINTS + 2]
        expected = np.sign((points[0] - points[1]) * (1 - 2 * state[PLAYER]))
        assert int(data["decision_returns"][index]) == expected
    assert set(states[:, PLAYER]) == {0, 1}
    for lane, env in enumerate(games):
        np.testing.assert_array_equal(env.state, data["final"][lane].cpu().numpy())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_full_update_exact_restart(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = Config(
        "sampled_return", 119020001, matches=4, device=device, gradient_batch=16
    )
    trainer = Trainer(PARENTS[0][0], config)
    before = trainer.actor.checkpoint()["current"]["parameters_sha256"]
    metrics, data = trainer.update()
    assert before != trainer.actor.checkpoint()["current"]["parameters_sha256"]
    assert metrics["gradient_rows"] == int(data["source"]["eligible_decisions"].sum())
    assert metrics["gradient_rows"] > 4
    assert metrics["source_work"]["completions"] == 4
    assert data["branch"] is None and not any(trainer.counters["branches"].values())
    path = tmp_path / "full.pt"
    trainer.save(path)
    expected_metrics, expected_data = trainer.update()
    restored = Trainer.restore(path)
    actual_metrics, actual_data = restored.update()
    equal(expected_metrics, actual_metrics)
    equal(expected_data, actual_data)
    equal(trainer.actor.checkpoint(), restored.actor.checkpoint())
    equal(trainer.optimizer.state_dict(), restored.optimizer.state_dict())
    equal(trainer.env.state, restored.env.state)
    equal(trainer.counters, restored.counters)
    for key in trainer.generators:
        equal(trainer.generators[key].get_state(), restored.generators[key].get_state())


@pytest.mark.parametrize("mode", ["sampled_return", "shared_actions"])
def test_root_controls_retain_original_update(mode):
    kwargs = {"mode": mode, "seed": 119030001, "matches": 4, "device": "cpu"}
    old = OldTrainer(PARENTS[0][0], OldConfig(**kwargs))
    new = Trainer(PARENTS[0][0], Config(**kwargs, decision_scope="root"))
    old_metrics, old_data = old.update()
    metrics, data = new.update()
    for key in old_metrics:
        equal(old_metrics[key], metrics[key])
    equal(old_data, data)
    equal(old.actor.checkpoint(), new.actor.checkpoint())
    equal(old.optimizer.state_dict(), new.optimizer.state_dict())
