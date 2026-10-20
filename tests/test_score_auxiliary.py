import copy
import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.published_actor import PublishedActor
from koikoi.reference import ReferenceEnv
from koikoi.upstream import UPSTREAM_PATH
from scripts.score_auxiliary import (
    ScoreAuxiliary,
    margin_classes,
    outcome_losses,
    policy_kl,
)


def test_likelihood_decomposition_and_within_result_information():
    logits = torch.tensor([[0.02 * i for i in range(65)]] * 3, dtype=torch.float64)
    logits.requires_grad_()
    targets = torch.tensor([0, 32, 64])
    coarse, within = outcome_losses(logits, targets)
    probabilities = logits.detach().softmax(-1).tolist()
    expected_coarse = []
    expected_within = []
    for row, target in zip(probabilities, targets.tolist(), strict=True):
        group = row[:32] if target < 32 else row[33:] if target > 32 else row[32:33]
        mass = sum(group)
        expected_coarse.append(-math.log(mass))
        expected_within.append(-math.log(row[target] / mass))
    torch.testing.assert_close(
        coarse, torch.tensor(expected_coarse, dtype=torch.float64)
    )
    torch.testing.assert_close(
        within, torch.tensor(expected_within, dtype=torch.float64)
    )
    torch.testing.assert_close(
        coarse + within, F.cross_entropy(logits, targets, reduction="none")
    )
    assert within[1].item() == 0
    within.sum().backward()
    assert torch.equal(logits.grad[1], torch.zeros(65, dtype=torch.float64))
    assert logits.grad[0, 32:].count_nonzero() == 0
    assert logits.grad[2, :33].count_nonzero() == 0
    torch.testing.assert_close(logits.grad.sum(-1), torch.zeros(3, dtype=torch.float64))
    # This does NOT preserve predicted win mass after a finite gradient step.
    before = logits.detach()[2].softmax(-1)[33:].sum()
    after = (logits.detach() - logits.grad)[2].softmax(-1)[33:].sum()
    assert abs(float(before - after)) > 1e-5


def test_terminal_margin_support_seat_symmetry_and_rejections():
    margins = torch.arange(-1024, 1026, 2, dtype=torch.float32)
    targets = margin_classes(margins)
    assert torch.equal((targets - 32).sign(), margins.sign().long())
    assert torch.equal(margin_classes(-margins), 64 - targets)
    for bad in (torch.tensor([1.0]), torch.tensor([float("nan")])):
        with pytest.raises(ValueError):
            margin_classes(bad)
    for bad in (torch.tensor([-1]), torch.tensor([65]), torch.tensor([1.0])):
        with pytest.raises(ValueError):
            outcome_losses(torch.zeros(1, 65), bad)


def test_masked_kl_scalar_fixture_and_gradients():
    legal = torch.tensor([[True, False, True], [False, True, False]])
    reference = torch.tensor(
        [[math.log(0.25), -torch.inf, math.log(0.75)], [-torch.inf, 0, -torch.inf]]
    )
    scores = torch.tensor(
        [[math.log(0.6), -torch.inf, math.log(0.4)], [-torch.inf, 2, -torch.inf]],
        requires_grad=True,
    )
    result = policy_kl(reference, scores, legal, temperature=1)
    expected = 0.25 * math.log(0.25 / 0.6) + 0.75 * math.log(0.75 / 0.4)
    torch.testing.assert_close(result, torch.tensor([expected, 0]))
    result.sum().backward()
    assert torch.isfinite(scores.grad).all()
    assert scores.grad[~legal].count_nonzero() == 0
    assert scores.grad[1].count_nonzero() == 0
    with pytest.raises(ValueError):
        policy_kl(reference + 1, scores, legal)


@pytest.mark.upstream
@pytest.mark.skipif(not UPSTREAM_PATH.exists(), reason="Upstream not fetched")
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_actual_public_trajectories_scores_labels_and_auxiliary_gradients(
    device, record_property
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    torch.manual_seed(105010001)
    actor = PublishedActor.from_published("point", device)
    rng = torch.get_rng_state().clone()
    model = ScoreAuxiliary(actor)
    assert torch.equal(torch.get_rng_state(), rng)
    env = BatchEnv(2, device, "paper8", 105010001)
    refs = []
    for state in env.state.cpu().numpy():
        ref = ReferenceEnv.__new__(ReferenceEnv)
        ref.state = state.copy()
        refs.append(ref)
    retained = {}
    decisions, rounds = 0, 0
    for _ in range(448):
        active = env.state[:, PHASE] != TERMINAL
        if not active.any():
            break
        features, legal = env.published_observe()
        phases = env.state[:, PHASE].clone()
        solved = env.solved_actions()
        legal &= (solved[:, None] < 0) | (
            torch.arange(50, device=device)[None] == solved[:, None]
        )
        ids = torch.where(active)[0]
        with torch.no_grad():
            expected = actor(features[ids], phases[ids], legal[ids])
            actual, aux = model(features[ids], phases[ids], legal[ids])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        coarse, _ = outcome_losses(aux, torch.full((len(ids),), 32, device=device))
        torch.testing.assert_close(coarse, torch.full_like(coarse, math.log(3)))
        for index in ids.tolist():
            phase = int(phases[index])
            if phase not in retained:
                retained[phase] = (
                    features[index].clone(),
                    phases[index].clone(),
                    legal[index].clone(),
                    int(env.state[index, PLAYER]),
                    index,
                )
        actions = torch.zeros(2, device=device, dtype=torch.int32)
        actions[ids] = actual.argmax(-1).int()
        old_active = active.cpu().tolist()
        results = env.step(actions, freeze_done=True)
        for index, ref in enumerate(refs):
            if old_active[index]:
                expected_result = ref.step(int(actions[index]))
                np.testing.assert_array_equal(
                    results[index].cpu().numpy(), expected_result
                )
        np.testing.assert_array_equal(
            env.state.cpu().numpy(), np.stack([r.state for r in refs])
        )
        decisions += len(ids)
        rounds += int(results[ids, 1].sum())
    else:
        pytest.fail("Capability matches exceeded fixed bound")
    assert len(retained) == 4
    record_property("decisions", decisions)
    record_property("round_endings", rounds)
    record_property("complete_matches", 2)
    rows = list(retained.values())
    features, phases, legal = (torch.stack([r[i] for r in rows]) for i in range(3))
    margins = torch.tensor(
        [
            int(refs[r[4]].state[POINTS + r[3]])
            - int(refs[r[4]].state[POINTS + 1 - r[3]])
            for r in rows
        ],
        device=device,
        dtype=torch.float32,
    )
    targets = margin_classes(margins)
    for margin, row in zip(margins.tolist(), rows, strict=True):
        assert np.sign(margin) == np.sign(
            refs[row[4]].state[POINTS + row[3]]
            - refs[row[4]].state[POINTS + 1 - row[3]]
        )
    with torch.no_grad():
        reference = (actor(features, phases, legal) / 0.05).log_softmax(-1)
    initial = copy.deepcopy(model.state_dict())
    for kind in ("win", "score"):
        model.load_state_dict(initial)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-6, eps=1e-5)
        for _ in range(2):
            # Actor scores remain exactly the inherited formula with autograd on.
            scores, _ = model(features, phases, legal)
            torch.testing.assert_close(
                scores, actor(features, phases, legal), rtol=0, atol=0
            )
            loss, metrics = model.auxiliary_loss(
                features, phases, legal, targets, reference, kind
            )
            assert torch.isfinite(loss)
            assert all(torch.isfinite(v) for v in metrics.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            optimizer.step()
        assert any(
            p.grad is not None and p.grad.count_nonzero()
            for net in actor.models.values()
            for p in net.encoder_block.parameters()
        )
    record_property("auxiliary_optimizer_steps", 4)
    record_property("auxiliary_example_presentations", 16)
