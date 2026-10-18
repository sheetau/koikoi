from dataclasses import asdict

import numpy as np
import pytest
import torch

from koikoi.agents import GreedyAgent, NeuralAgent, solved_koi_action
from koikoi.layout import KOI, PHASE, PLAYER, POINTS, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.search import sample_hidden
from scripts.published_koi_search import PublishedKoiSearch
from scripts.run_arbitration_study import PARENTS


def roots():
    phases, koi = {}, {}
    source_decisions = 0
    for seed in range(25110001, 25110101):
        env = ReferenceEnv(seed, "paper8")
        while env.state[PHASE] != TERMINAL:
            phase, player = int(env.state[PHASE]), int(env.state[PLAYER])
            phases.setdefault(phase, env.clone())
            if phase == KOI and solved_koi_action(env) is None:
                koi.setdefault(player, env.clone())
            if len(phases) == 4 and len(koi) == 2:
                return phases, koi, source_decisions
            env.step(GreedyAgent().act(env))
            source_decisions += 1
    raise AssertionError("Missing phase/seat fixtures")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_search_reference_trajectories_hidden_invariance_and_fallback(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    search = PublishedKoiSearch(
        PARENTS[0][0],
        device=device,
        particles=2,
        confirmation_particles=0,
    )
    original = NeuralAgent(PARENTS[0][0], device="cpu", deterministic=True)
    phases, koi, _ = roots()
    for phase, root in phases.items():
        if phase != KOI:
            assert search.act(root) == original.act(root)
            assert search.last_decision is None
    assert all(p.requires_grad for p in search.parent.model.actor.parameters())
    assert not any(p.requires_grad for p in search.continuation.actor.parameters())

    # Replay every actual batched action through independent Python transitions.
    real_continuation = search.continuation
    workspace = search.workspaces[2]
    for player, root in koi.items():
        base = original.act(root)
        seed = 25112000 + player
        from koikoi.search import sample_hidden_batch

        worlds = sample_hidden_batch(root, np.random.default_rng(seed), 2)
        references, totals = (
            [],
            {"decisions": 0, "round_endings": 0, "completed_branches": 0},
        )
        for world in worlds:
            for first in (base, 97 - base):
                env = ReferenceEnv(1, "paper8")
                env.state[:] = world
                result = env.step(first)
                totals["decisions"] += 1
                totals["round_endings"] += int(result[1])
                totals["completed_branches"] += int(result[2])
                references.append(env)

        def checked_continuation(env, references=references, totals=totals):
            packed = env.state.cpu().numpy()
            for actual, expected in zip(packed, references):
                np.testing.assert_array_equal(actual, expected.state)
            actions = real_continuation(env)
            for action, expected in zip(actions.cpu().tolist(), references):
                if expected.state[PHASE] != TERMINAL:
                    result = expected.step(action)
                    totals["decisions"] += 1
                    totals["round_endings"] += int(result[1])
                    totals["completed_branches"] += int(result[2])
            return actions

        search.continuation = checked_continuation
        before_work = asdict(search.work)
        unchanged = root.state.copy()
        search.reset_seed(seed)
        action = search.act(root)
        record = search.last_decision
        assert action in (48, 49)
        np.testing.assert_array_equal(root.state, unchanged)
        for actual, expected in zip(workspace.env.state.cpu().numpy(), references):
            np.testing.assert_array_equal(actual, expected.state)
            assert expected.state[PHASE] == TERMINAL
        assert {k: v - before_work[k] for k, v in asdict(search.work).items()} == totals
        point_rows = np.array(
            [
                int(e.state[POINTS] - e.state[POINTS + 1]) * (1 - 2 * player)
                for e in references
            ]
        ).reshape(2, 2)
        assert record["proposal"]["point_change"] == np.diff(point_rows, axis=1).mean()
        assert (
            record["proposal"]["win_change"]
            == np.diff(np.sign(point_rows), axis=1).mean() / 2
        )

        alternate = root.clone()
        alternate.state = sample_hidden(root, np.random.default_rng(77))
        np.testing.assert_array_equal(root.observe(), alternate.observe())
        search.continuation = real_continuation
        search.reset_seed(seed)
        assert search.act(alternate) == action
        assert search.last_decision == record
    assert search.work.completed_branches == 16


@pytest.mark.parametrize("confirmation", [1.0, -1.0, 0.0])
def test_fresh_confirmation_can_only_accept_or_reject(monkeypatch, confirmation):
    torch.set_num_threads(2)
    search = PublishedKoiSearch(
        PARENTS[0][0], device="cpu", particles=2, confirmation_particles=4
    )
    root = roots()[1][0]
    calls = []

    def compare(reference, base, alternative, count, rng, confirmation=False):
        calls.append((base, alternative, count, rng, confirmation))
        margin = 1.0 if len(calls) == 1 else test_margin
        return {"win_change": margin, "lower_margin": margin, "point_change": 0.0}

    test_margin = confirmation
    monkeypatch.setattr(search, "_compare", compare)
    base = search.parent.act(root)
    assert search.act(root) == (97 - base if confirmation > 0 else base)
    assert calls[0] == (base, 97 - base, 2, search.rng, False)
    assert calls[1] == (base, 97 - base, 4, search.confirmation_rng, True)
    assert search.confirmation_decisions == search.proposed_decisions == 1
    assert search.changed_decisions == int(confirmation > 0)


@pytest.mark.parametrize(
    "options",
    [
        {"particles": 1},
        {"particles": 2.5},
        {"confirmation_particles": -1},
        {"confirmation_particles": 1},
        {"confidence": float("nan")},
        {"confidence": float("inf")},
        {"confidence": -1},
    ],
)
def test_invalid_budgets_fail_before_loading(options):
    with pytest.raises(ValueError):
        PublishedKoiSearch("nonexistent.pt", **options)
