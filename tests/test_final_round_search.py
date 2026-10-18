import copy
from dataclasses import asdict

import numpy as np
import pytest
import torch

from koikoi.agents import solved_koi_action
from koikoi.layout import PHASE, PLAYER, POINTS, ROUND, STOP, TERMINAL, TURN
from koikoi.reference import ReferenceEnv
from koikoi.search import sample_hidden, sample_hidden_batch
from scripts.diagnose_history_support import reference
from scripts.final_round_search import FinalRoundSearch, is_final_round_card
from scripts.run_arbitration_study import PARENTS


def roots():
    """Artificial last-round fixtures, not newly collected strength positions."""
    states = np.load(
        "runs/observer_seat_information_v2/parent1_states.npy", allow_pickle=False
    )
    selected = {}
    for state in states:
        key = (int(state[PHASE]), int(state[PLAYER]))
        env = reference(state)
        if (
            key[0] < 3
            and env.legal().sum() > 1
            and (key not in selected or state[TURN] > selected[key].state[TURN])
        ):
            selected[key] = env
    assert len(selected) == 6
    for env in selected.values():
        env.state[ROUND] = 7
    return selected


@pytest.fixture
def agent():
    torch.set_num_threads(2)
    return FinalRoundSearch(
        PARENTS[0][0], device="cpu", particles=2, confirmation_particles=2
    )


def test_final_round_all_card_phases_confirmation_invariance_and_fallback(
    agent, monkeypatch
):
    monkeypatch.setattr(agent, "_lower", lambda values: 1.0)
    for root in roots().values():
        assert is_final_round_card(root)
        before = root.state.copy()
        work_before = copy.deepcopy(agent.work)
        agent.reset_seed(86110001)
        action = agent.act(root)
        assert root.legal()[action]
        decision, context, sampling = copy.deepcopy(
            (agent.last_decision, agent.last_context, agent.last_sampling)
        )
        assert decision["confirmation"] is not None
        assert decision["confirmation"]["alternative"] == decision["alternative"]
        assert {r["alternative"] for r in decision["alternatives"]} == set(
            np.flatnonzero(root.legal())
        ) - {decision["base"]}
        assert (
            sampling["proposal"]["worlds_sha256"]
            != sampling["confirmation"]["worlds_sha256"]
        )
        alternate = reference(sample_hidden(root, np.random.default_rng(86110002)))
        agent.reset_seed(86110001)
        assert agent.act(alternate) == action
        assert (agent.last_decision, agent.last_context, agent.last_sampling) == (
            decision,
            context,
            sampling,
        )
        branches = 8 * int(root.legal().sum())
        assert (
            agent.work.completed_branches - work_before.completed_branches == branches
        )
        assert agent.work.round_endings - work_before.round_endings == branches
        np.testing.assert_array_equal(root.state, before)
        earlier = root.clone()
        earlier.state[ROUND] = 6
        assert not is_final_round_card(earlier)
        current = copy.deepcopy(agent.work)
        assert agent.act(earlier) == agent.parent.act(earlier)
        assert agent.work == current and agent.last_decision is None
        assert agent.last_sampling == {} and agent.support is None
    assert agent.continuation.microbatch == 64 and not agent.history_work
    initial = ReferenceEnv(86110003, "paper8")
    assert agent.act(initial) == agent.parent.act(initial)
    invalid = ReferenceEnv(86110003, "research12")
    with pytest.raises(ValueError, match="paper8"):
        agent.act(invalid)
    with pytest.raises(ValueError, match="uniform"):
        FinalRoundSearch(PARENTS[0][0], mode="history")


def test_final_round_paired_trajectories_match_python_reference(agent):
    actual_continuation = agent.continuation
    for (phase, seat), root in roots().items():
        seed = 86120001 + 2 * phase + seat
        worlds = sample_hidden_batch(root, np.random.default_rng(seed), 2)
        base = int(agent.parent.act(root))
        other = next(int(a) for a in np.flatnonzero(root.legal()) if a != base)
        expected = []
        counts = dict.fromkeys(asdict(agent.work), 0)

        def step(env, action, counts=counts):
            result = env.step(action)
            counts["decisions"] += 1
            counts["round_endings"] += int(result[1])
            counts["completed_branches"] += int(result[2])

        for world in worlds:
            for action in (base, other):
                env = reference(world)
                step(env, action)
                expected.append(env)

        class Checked:
            names = actual_continuation.names
            prepare = actual_continuation.prepare

            def __call__(self, env, expected=expected, step=step):
                for state, ref in zip(env.state.numpy(), expected):
                    np.testing.assert_array_equal(state, ref.state)
                actions = actual_continuation(env)
                for action, ref in zip(actions.tolist(), expected):
                    if ref.state[PHASE] != TERMINAL:
                        step(ref, action)
                return actions

        agent.continuation = Checked()
        row = agent._compare_actions(
            root, base, [other], 2, np.random.default_rng(seed)
        )[0]
        assert row["work"] == counts
        assert counts["round_endings"] == counts["completed_branches"] == 4
        points = []
        for state, ref in zip(agent.workspaces[2].env.state.numpy(), expected):
            np.testing.assert_array_equal(state, ref.state)
            assert ref.state[PHASE] == TERMINAL
            points.append(
                int(ref.state[POINTS] - ref.state[POINTS + 1]) * (1 - 2 * seat)
            )
        np.testing.assert_array_equal(
            row["paired_points"], np.array(points).reshape(2, 2)
        )


def test_final_round_koi_guard_forced_cards_and_terminal_rejection(agent):
    states = np.load(
        "runs/observer_seat_information_v2/parent1_states.npy", allow_pickle=False
    )
    checked_seats, forced = set(), False
    before = copy.deepcopy(agent.work)
    for state in states:
        root = reference(state)
        root.state[ROUND] = 7
        seat = int(root.state[PLAYER])
        if root.state[PHASE] == 3 and seat not in checked_seats:
            root.state[POINTS + seat] = 31
            root.state[POINTS + 1 - seat] = 29
            assert solved_koi_action(root) == STOP
            assert agent.act(root) == agent.parent.act(root) == STOP
            assert agent.last_context["status"] == "outside_final_round_cards"
            checked_seats.add(seat)
        elif is_final_round_card(root) and root.legal().sum() == 1 and not forced:
            assert agent.act(root) == agent.parent.act(root)
            assert agent.last_decision is None and agent.last_sampling == {}
            forced = True
    assert checked_seats == {0, 1} and forced
    assert agent.work == before and agent.searched_decisions == 0
    root.state[PHASE] = TERMINAL
    with pytest.raises(ValueError, match="live paper8"):
        agent.act(root)
