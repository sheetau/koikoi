import numpy as np
import pytest
import torch

from koikoi.reference import ReferenceEnv
from koikoi.search import sample_hidden_batch
from scripts.diagnose_arbitration import proposal_mask
from scripts.paired_rollout_targets import match_split
from scripts.run_arbitration_study import PARENTS
from scripts.train_paired_arbitration import collect
from scripts.train_score_distribution import (
    ScoreDistributionAgent,
    ScoreSelector,
    decode_returns,
    fit,
    target_indices,
    validate_roots,
)


def roots(leads):
    public = torch.zeros(len(leads), 562)
    public[:, 535] = 8 / 12
    public[:, 543] = 1
    public[:, 536] = (60 + leads) / 128
    public[:, 537] = (60 - leads) / 128
    public[:, 560:] = torch.tensor([3, 9])
    return public


def test_tail_bins_preserve_win_draw_loss_across_all_active_leads():
    torch.set_num_threads(2)
    leads, increments = torch.meshgrid(
        torch.arange(-58, 60, 2),
        torch.cat(
            (
                torch.tensor([-1000, -128, -66]),
                torch.arange(-64, 66, 2),
                torch.tensor([66, 128, 1000]),
            )
        ),
        indexing="ij",
    )
    leads, increments = leads.flatten(), increments.flatten()
    public = roots(leads)
    final = torch.stack((leads + increments, leads - increments), dim=1)
    ids, unbounded = target_indices(public, final)
    support = torch.arange(-64, 65, 2)
    assert torch.equal((support[ids] + leads[:, None]).sign(), final.sign())
    assert unbounded.abs().max() == 1000
    assert ids.min() == 0 and ids.max() == 64
    with pytest.raises(ValueError, match="even"):
        target_indices(public, final + 1)
    bad = public.clone()
    bad[:, 536] = 0
    with pytest.raises(ValueError, match="total score"):
        validate_roots(bad)
    bad = public.clone()
    bad[:, 543] = 0
    with pytest.raises(ValueError, match="bankruptcy"):
        validate_roots(bad)


def test_win_decoder_changes_risk_preference_at_score_threshold():
    # A loses4 points certainly. B gains or loses20 with equal probability.
    logits = torch.full((3, 2, 65), -1e9)
    logits[:, 0, (-4 + 64) // 2] = 0
    logits[:, 1, (-20 + 64) // 2] = 0
    logits[:, 1, (20 + 64) // 2] = 0
    values = decode_returns(logits, torch.tensor([10, -10, 4]))
    torch.testing.assert_close(
        values, torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]]), rtol=0, atol=0
    )
    # Expected points prefer B at all scores; wins prefer A when leading.
    assert values[0].argmax() == 0 and values[1].argmax() == 1


def test_preference_loss_draw_weights_ties_and_common_offsets():
    from scripts.train_score_distribution import loss_for

    class FixedScores:
        kind = "preference"

        def __init__(self, scores):
            self.scores = scores

        def branch_outputs(self, public):
            return self.scores.unsqueeze(-1)

    public = roots(torch.tensor([0, 4, -8, 2, -2]))
    indices, increments = target_indices(
        public, torch.tensor([[-2, 2], [0, 6], [4, -6], [2, 0], [8, 4]])
    )
    scores = torch.zeros(5, 2, requires_grad=True)
    loss = loss_for(FixedScores(scores), public, indices, increments)
    # Win-loss changes weigh1; draw transitions weigh1/2; equal outcomes weigh0.
    assert loss.item() == pytest.approx(3 * np.log(2) / 5)
    loss.backward()
    expected = torch.tensor(
        [[0.1, -0.1], [0.05, -0.05], [-0.1, 0.1], [-0.05, 0.05], [0, 0]]
    )
    torch.testing.assert_close(scores.grad, expected)
    shifted = scores.detach() + torch.tensor([4, -3, 2, 1, -1])[:, None]
    torch.testing.assert_close(
        loss_for(FixedScores(shifted), public, indices, increments), loss
    )
    equal_indices, equal_increments = target_indices(public, torch.full((5, 2), 2))
    tied = torch.randn(5, 2, requires_grad=True)
    zero = loss_for(FixedScores(tied), public, equal_indices, equal_increments)
    zero.backward()
    assert zero.item() == 0 and torch.count_nonzero(tied.grad) == 0


def test_preference_weights_recover_expected_win_choice_not_majority_label():
    from scripts.train_score_distribution import loss_for

    class ConstantDifference:
        kind = "preference"

        def __init__(self, value):
            self.value = value

        def branch_outputs(self, public):
            return torch.stack((self.value * 0, self.value)).expand(10, 2)[..., None]

    public = roots(torch.zeros(10))
    final = torch.tensor([[0, 2]] * 6 + [[2, -2]] * 4)
    indices, increments = target_indices(public, final)
    # Point wins six labels, but its expected win-score change is negative.
    assert (final.sign()[:, 1] - final.sign()[:, 0]).float().mean() < 0
    optimum = torch.tensor(np.log(3 / 4), dtype=torch.float64, requires_grad=True)
    loss = loss_for(ConstantDifference(optimum), public, indices, increments)
    loss.backward()
    assert abs(optimum.grad.item()) < 1e-8
    assert optimum.item() < 0


@pytest.mark.parametrize("kind", ["paired_win", "paired_mean"])
def test_paired_loss_signs_scaling_and_common_offset(kind):
    from scripts.train_score_distribution import loss_for

    class FixedScores(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.kind = kind
            self.outputs = torch.nn.Parameter(values.clone())

        def branch_outputs(self, public):
            return self.outputs.unsqueeze(-1)

    public = roots(torch.tensor([0, 4, -8]))
    indices, increments = target_indices(
        public, torch.tensor([[6, -4], [-2, 10], [0, -80]])
    )
    outputs = torch.tensor([[0.2, -0.1], [0.1, 0.4], [0.5, -0.3]])
    target = (
        torch.tensor([-2.0, 2.0, -1.0])
        if kind == "paired_win"
        else torch.tensor([-10.0, 12.0, -80.0]) / 64
    )
    model = FixedScores(outputs)
    actual = loss_for(model, public, indices, increments)
    errors = outputs[:, 1] - outputs[:, 0] - target
    torch.testing.assert_close(actual, errors.square().mean())
    actual.backward()
    torch.testing.assert_close(
        model.outputs.grad, torch.stack((-errors, errors), dim=1) * (2 / 3)
    )
    translated = FixedScores(outputs + torch.tensor([4.0, -3.0, 2.0])[:, None])
    torch.testing.assert_close(
        loss_for(translated, public, indices, increments), actual, rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("kind", ["categorical", "mean", "win"])
def test_shared_action_scores_swap_and_duplicate_consistently(kind):
    torch.manual_seed(31)
    model = ScoreSelector(kind, width=16)
    torch.nn.init.normal_(model.head.weight)
    public = roots(torch.tensor([0, 4, -8]))
    public[1, 560:] = 3
    mask = proposal_mask(public)
    before = model(public, mask)[0]
    swapped = public.clone()
    swapped[:, 560:] = public[:, 560:].flip(-1)
    torch.testing.assert_close(before, model(swapped, mask)[0], rtol=0, atol=0)
    assert mask.gather(1, before.argmax(-1)[:, None]).all()
    assert (before[~mask] == -1e9).all()
    branches = model.values(public)
    assert branches[1, 0] == branches[1, 1]


def test_real_paired_data_fit_reload_and_hidden_invariance(tmp_path):
    dataset = tmp_path / "targets.pt"
    data = collect(
        PARENTS[0][0], dataset, seed=18960001, roots=32, batch=16, device="cpu"
    )
    split = match_split(data["match_groups"])
    initial_bodies, order_seeds = set(), set()
    for kind in (
        "categorical",
        "mean",
        "win",
        "paired_win",
        "paired_mean",
        "preference",
    ):
        target = tmp_path / (kind + ".pt")
        payload = fit(dataset, target, kind=kind, seed=18970001, epochs=2)
        initial_bodies.add(payload["training_config"]["initial_shared_body_sha256"])
        order_seeds.add(payload["training_config"]["minibatch_order_seed"])
        assert payload["counters"]["validation_roots"] == int(split.sum())
        assert payload["counters"]["new_simulation_decisions"] == 0
        assert payload["counters"]["reused_branch_work"] == data["branch_work"]
        assert len(payload["history"]) == 2
        assert all(np.isfinite(row["validation_loss"]) for row in payload["history"])
        agent = ScoreDistributionAgent(target)
        env = ReferenceEnv(18980001, "paper8")
        action = agent.act(env)
        assert env.legal()[action]
        original, legal = agent.bank.observe(agent.env)
        worlds = sample_hidden_batch(env, np.random.default_rng(5), 1)
        agent.env.state.copy_(torch.tensor(worlds))
        changed, changed_legal = agent.bank.observe(agent.env)
        torch.testing.assert_close(original, changed, rtol=0, atol=0)
        torch.testing.assert_close(legal, changed_legal, rtol=0, atol=0)
        assert agent.batch_act(agent.env).item() == action
        with pytest.raises(ValueError, match="fresh"):
            fit(dataset, target, kind=kind, epochs=2)
    assert len(initial_bodies) == len(order_seeds) == 1
    explicit_default = fit(
        dataset,
        tmp_path / "explicit_default.pt",
        kind="preference",
        seed=18970001,
        epochs=2,
        weight_decay=1e-4,
    )
    assert all(
        torch.equal(value, explicit_default["model"][name])
        for name, value in payload["model"].items()
    )
    unregularized = fit(
        dataset,
        tmp_path / "unregularized.pt",
        kind="preference",
        seed=18970001,
        epochs=2,
        weight_decay=0,
    )
    assert unregularized["training_config"]["weight_decay"] == 0
    assert any(
        not torch.equal(value, unregularized["model"][name])
        for name, value in explicit_default["model"].items()
    )


@pytest.mark.cuda
def test_cpu_cuda_shared_scorer_losses_and_gradients():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from scripts.train_score_distribution import loss_for

    torch.backends.cuda.matmul.allow_tf32 = False
    public = roots(torch.tensor([0, 4, -8]))
    indices, increments = target_indices(
        public, torch.tensor([[6, -4], [-2, 10], [0, -80]])
    )
    for kind in (
        "categorical",
        "mean",
        "win",
        "paired_win",
        "paired_mean",
        "preference",
    ):
        cpu = ScoreSelector(kind, width=16)
        torch.nn.init.normal_(cpu.head.weight, std=0.1)
        gpu = ScoreSelector(kind, width=16).cuda()
        gpu.load_state_dict(cpu.state_dict())
        a = loss_for(cpu, public, indices, increments)
        b = loss_for(gpu, public.cuda(), indices.cuda(), increments.cuda())
        torch.testing.assert_close(a, b.cpu(), rtol=1e-5, atol=1e-6)
        a.backward()
        b.backward()
        for left, right in zip(cpu.parameters(), gpu.parameters(), strict=True):
            torch.testing.assert_close(
                left.grad, right.grad.cpu(), rtol=1e-4, atol=1e-6
            )
