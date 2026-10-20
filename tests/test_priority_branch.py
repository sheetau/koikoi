import math

import pytest
import torch

from koikoi.env import BatchEnv
from scripts.branch_policy_gradient import collect_roots as uniform_roots
from scripts.branch_policy_gradient import expected_return_loss
from scripts.priority_branch_roots import collect_roots, reservoir_step
from scripts.priority_branch_trainer import Config, Trainer
from scripts.run_arbitration_study import PARENTS
from scripts.train_branch_policy_gradient import Config as OldConfig
from scripts.train_branch_policy_gradient import Trainer as OldTrainer


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            equal(left, right)
    else:
        assert a == b


def test_weighted_reservoir_exact_partition_and_zero_weights():
    total = torch.zeros(600, dtype=torch.float64)
    labels = torch.full((600,), -1)
    uniforms = [
        torch.zeros(600),
        ((torch.arange(60) + 0.5) / 60).repeat_interleave(10),
        ((torch.arange(10) + 0.5) / 10).repeat(60),
    ]
    for index, weight in enumerate((1.0, 2.0, 7.0)):
        total, chosen = reservoir_step(
            total, torch.full_like(total, weight), uniforms[index]
        )
        labels[chosen] = index
    assert torch.bincount(labels).tolist() == [60, 120, 420]
    after, chosen = reservoir_step(total, torch.zeros_like(total), torch.zeros(600))
    assert torch.equal(after, total) and not chosen.any()
    with pytest.raises(ValueError):
        reservoir_step(total, -torch.ones_like(total), torch.zeros(600))


def test_weighted_variable_length_gradient_keeps_the_original_objective():
    theta = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
    p = torch.stack([theta, -theta]).softmax(0)
    last = torch.tensor([-1.0, 4.0], dtype=torch.float64)
    objective = 2 * p[0] + p[1] * (p * last).sum()
    exact = torch.autograd.grad(objective, theta, retain_graph=True)[0]
    first = torch.stack([theta.new_tensor(2), (p * last).sum()])
    weights = [1 + 2 * p[0], 2 + p[1]]
    estimate = theta * 0
    for path, values in (
        (p[0], [first]),
        (p[1] * p[0], [first, last]),
        (p[1] * p[1], [first, last]),
    ):
        total = sum(weights[: len(values)])
        for index, q in enumerate(values):
            chance = (weights[index] / total).detach()
            corrected = (total / weights[index]).reshape(1)
            estimate -= (
                path.detach()
                * chance
                * expected_return_loss(p[None], q[None], corrected)
            )
    actual = torch.autograd.grad(estimate, theta)[0]
    torch.testing.assert_close(actual, exact, rtol=1e-12, atol=1e-12)


class FixtureActor(torch.nn.Module):
    def forward(self, features, phases, legal):
        return (features[:, :3].sum(1) * 0.01).masked_fill(~legal, -torch.inf)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_uniform_compatibility_and_priority_preserves_source_actions(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    actor = FixtureActor().to(device)

    def generate(fn, **kwargs):
        env = BatchEnv(4, device, seed=114010001, debug=True)
        return fn(
            env,
            actor,
            torch.Generator(device=device).manual_seed(17),
            torch.Generator(device=device).manual_seed(23),
            **kwargs,
        )

    original = generate(uniform_roots)
    uniform = generate(collect_roots, sampler="uniform")
    for key in original:
        equal(original[key], uniform[key])
    assert torch.equal(uniform["importance"].long(), original["eligible_decisions"])
    priority = generate(collect_roots, sampler="sensitivity")
    for key in ("initial", "final", "actions", "work", "eligible_decisions"):
        equal(priority[key], original[key])
    for lane in range(4):
        weights = priority["priorities"][:, lane].tolist()
        total = 0.0
        selected = -1
        for step, (weight, uniform) in enumerate(
            zip(weights, priority["selection_uniforms"][:, lane].tolist())
        ):
            total += weight
            if weight > 0 and uniform < weight / total:
                selected = step
        assert selected == int(priority["selected_step"][lane])
        expected = math.fsum(weights) / weights[selected]
        assert math.isclose(
            float(priority["importance"][lane]), expected, rel_tol=1e-12
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_original_uniform_update_and_weighted_checkpoint_resume(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    kwargs = {
        "mode": "shared_actions",
        "seed": 114020001,
        "matches": 4,
        "device": device,
    }
    old = OldTrainer(PARENTS[0][0], OldConfig(**kwargs))
    uniform = Trainer(PARENTS[0][0], Config(**kwargs, sampler="uniform"))
    old_metrics, old_data = old.update()
    new_metrics, new_data = uniform.update()
    for key in old_metrics:
        equal(old_metrics[key], new_metrics[key])
    for key in old_data["source"]:
        equal(old_data["source"][key], new_data["source"][key])
    equal(old_data["branch"], new_data["branch"])
    equal(old.actor.checkpoint(), uniform.actor.checkpoint())
    equal(old.optimizer.state_dict(), uniform.optimizer.state_dict())

    weighted = Trainer(PARENTS[0][0], Config(**kwargs, sampler="sensitivity", lr=1e-4))
    weighted.update()
    checkpoint = tmp_path / "weighted.pt"
    weighted.save(checkpoint)
    expected_metrics, expected_data = weighted.update()
    restored = Trainer.restore(checkpoint)
    actual_metrics, actual_data = restored.update()
    equal(expected_metrics, actual_metrics)
    equal(expected_data, actual_data)
    equal(weighted.actor.checkpoint(), restored.actor.checkpoint())
    equal(weighted.optimizer.state_dict(), restored.optimizer.state_dict())
    equal(weighted.counters, restored.counters)
    for key in weighted.generators:
        equal(
            weighted.generators[key].get_state(), restored.generators[key].get_state()
        )
