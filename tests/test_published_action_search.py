from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.search import sample_hidden, sample_hidden_batch
from scripts.published_action_search import PublishedActionSearch
from scripts.run_arbitration_study import PARENTS


def retained_roots():
    path = Path("runs/observer_seat_information_v2/parent1_states.npy")
    states = np.load(path, allow_pickle=False)
    roots = {}
    for state in states:
        key = (int(state[PHASE]), int(state[PLAYER]))
        env = ReferenceEnv(1, "paper8")
        env.state[:] = state
        if key[0] < 3 and env.legal().sum() > 1:
            roots.setdefault(key, env)
    assert len(roots) == 6
    return roots


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_card_branches_match_reference_for_all_phases_and_seats(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    search = PublishedActionSearch(
        PARENTS[0][0], device=device, particles=2, confirmation_particles=0
    )
    real_continuation = search.continuation
    for (phase, player), root in retained_roots().items():
        base = int(search.parent.act(root))
        alternative = next(int(a) for a in np.flatnonzero(root.legal()) if a != base)
        seed = 29110000 + phase * 2 + player
        worlds = sample_hidden_batch(root, np.random.default_rng(seed), 2)
        references = []
        totals = {"decisions": 0, "round_endings": 0, "completed_branches": 0}

        def step(env, action, totals=totals):
            result = env.step(action)
            totals["decisions"] += 1
            totals["round_endings"] += int(result[1])
            totals["completed_branches"] += int(result[2])

        for world in worlds:
            for action in (base, alternative):
                env = ReferenceEnv(1, "paper8")
                env.state[:] = world
                step(env, action)
                references.append(env)

        def checked(env, references=references, step=step):
            for actual, expected in zip(env.state.cpu().numpy(), references):
                np.testing.assert_array_equal(actual, expected.state)
            actions = real_continuation(env)
            for action, expected in zip(actions.cpu().tolist(), references):
                if expected.state[PHASE] != TERMINAL:
                    step(expected, action)
            return actions

        before = root.state.copy()
        search.continuation = checked
        row = search._compare_actions(
            root, base, [alternative], 2, np.random.default_rng(seed)
        )[0]
        assert row["work"] == totals
        assert row["work"]["completed_branches"] == 4
        expected_points = []
        for actual, expected in zip(
            search.workspaces[2].env.state.cpu().numpy(), references
        ):
            np.testing.assert_array_equal(actual, expected.state)
            assert expected.state[PHASE] == TERMINAL
            expected_points.append(
                int(expected.state[POINTS] - expected.state[POINTS + 1])
                * (1 - 2 * player)
            )
        np.testing.assert_array_equal(
            row["paired_points"], np.array(expected_points).reshape(2, 2)
        )
        np.testing.assert_array_equal(root.state, before)


def test_hidden_world_invariance_and_all_alternatives():
    torch.set_num_threads(2)
    search = PublishedActionSearch(
        PARENTS[0][0], device="cpu", particles=2, confirmation_particles=0
    )
    root = retained_roots()[(0, 0)]
    search.reset_seed(29120001)
    action = search.act(root)
    record = search.last_decision
    alternate = root.clone()
    alternate.state = sample_hidden(root, np.random.default_rng(19))
    np.testing.assert_array_equal(root.observe(), alternate.observe())
    search.reset_seed(29120001)
    assert search.act(alternate) == action
    assert search.last_decision == record
    assert {row["alternative"] for row in record["alternatives"]} == (
        set(np.flatnonzero(root.legal())) - {record["base"]}
    )
    assert asdict(search.work)["completed_branches"] == 8 * (
        int(root.legal().sum()) - 1
    )


@pytest.mark.parametrize("confirmation_margin", [0.2, 0.0, -0.2])
def test_confirmation_checks_selected_action_only(monkeypatch, confirmation_margin):
    torch.set_num_threads(2)
    search = PublishedActionSearch(
        PARENTS[0][0], device="cpu", particles=2, confirmation_particles=4
    )
    root = retained_roots()[(0, 0)]
    base = int(search.parent.act(root))
    alternatives = [int(a) for a in np.flatnonzero(root.legal()) if a != base]
    calls = []

    def compare(reference, fallback, candidates, count, rng, confirmation=False):
        calls.append((candidates, count, rng, confirmation))
        return [
            {
                "alternative": a,
                "win_change": i / 10 + 0.1,
                "lower_margin": confirmation_margin if confirmation else 0.01,
            }
            for i, a in enumerate(candidates)
        ]

    monkeypatch.setattr(search, "_compare_actions", compare)
    assert search.act(root) == (alternatives[-1] if confirmation_margin > 0 else base)
    assert calls == [
        (alternatives, 2, search.rng, False),
        ([alternatives[-1]], 4, search.confirmation_rng, True),
    ]
    assert search.confirmation_decisions == 1
    assert search.changed_decisions == int(confirmation_margin > 0)
