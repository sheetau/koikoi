import pytest
import torch

from scripts.run_arbitration_study import PARENTS
from scripts.train_branch_policy_gradient import Config, Trainer, sampled_return_loss


def equal(left, right):
    assert type(left) is type(right)
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


def test_sampled_loss_enumeration_equals_expected_objective():
    theta = torch.tensor([0.4, -0.3, 0.1], dtype=torch.float64, requires_grad=True)
    p = theta.softmax(0)
    returns = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64)
    expectation = sum(
        p[a].detach()
        * sampled_return_loss(
            p[None], torch.tensor([a]), returns[a : a + 1], torch.tensor([7])
        )
        for a in range(3)
    )
    estimated = torch.autograd.grad(expectation, theta, retain_graph=True)[0]
    exact = torch.autograd.grad(-7 * (p * returns).sum(), theta)[0]
    torch.testing.assert_close(estimated, exact, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "mode", ["shared_actions", "independent_actions", "sampled_return"]
)
def test_actual_update_exact_restart_and_accounting(tmp_path, device, mode):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    trainer = Trainer(PARENTS[0][0], Config(mode, 110030001, matches=4, device=device))
    before = trainer.actor.checkpoint()["current"]["parameters_sha256"]
    metrics, evidence = trainer.update()
    after = trainer.actor.checkpoint()["current"]["parameters_sha256"]
    if before == after:
        # A tiny shared-noise batch can have identical returns for every action.
        # Floating-point centering can still leave a negligible gradient.
        branch = evidence["branch"]
        assert branch is not None
        assert all(
            row[mask].unique().numel() == 1
            for row, mask in zip(branch["q"], branch["legal"])
        )
        assert metrics["gradient_norm_before_clip"] < 1e-6
    assert metrics["source_work"]["completions"] == 4
    assert evidence["source"]["work"] == metrics["source_work"]
    if mode == "sampled_return":
        assert metrics["branch_work"] is None
        assert not any(trainer.counters["branches"].values())
    else:
        assert metrics["branch_work"]["completions"] == int(
            evidence["branch"]["legal"].sum()
        )
    saved = tmp_path / "resume.pt"
    trainer.save(saved)
    expected_metrics, expected_data = trainer.update()
    resumed = Trainer.restore(saved)
    actual_metrics, actual_data = resumed.update()
    equal(expected_metrics, actual_metrics)
    equal(expected_data, actual_data)
    equal(trainer.actor.checkpoint(), resumed.actor.checkpoint())
    equal(trainer.optimizer.state_dict(), resumed.optimizer.state_dict())
    equal(trainer.env.state, resumed.env.state)
    equal(trainer.counters, resumed.counters)
    for key in trainer.generators:
        assert torch.equal(
            trainer.generators[key].get_state(), resumed.generators[key].get_state()
        )
