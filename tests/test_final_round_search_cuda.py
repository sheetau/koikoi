import numpy as np
import pytest
import torch

from koikoi.layout import PHASE, POINTS, TERMINAL
from koikoi.search import sample_hidden_batch
from scripts.diagnose_history_support import reference
from scripts.final_round_search import FinalRoundSearch
from scripts.run_arbitration_study import PARENTS
from tests.test_final_round_search import (
    roots,
)
from tests.test_final_round_search import (
    test_final_round_all_card_phases_confirmation_invariance_and_fallback as check_invariance,
)
from tests.test_final_round_search import (
    test_final_round_koi_guard_forced_cards_and_terminal_rejection as check_guard,
)


@pytest.fixture
def agent():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return FinalRoundSearch(
        PARENTS[0][0], device="cuda", particles=2, confirmation_particles=2
    )


def test_cuda_final_round_choices_confirmation_and_information_invariance(
    agent, monkeypatch
):
    check_invariance(agent, monkeypatch)


def test_cuda_final_round_guard_and_fallback(agent):
    check_guard(agent)


def test_cuda_final_round_all_phases_and_seats_match_reference(agent):
    continuation = agent.continuation
    for (phase, seat), root in roots().items():
        seed = 86120001 + 2 * phase + seat
        worlds = sample_hidden_batch(root, np.random.default_rng(seed), 2)
        base = int(agent.parent.act(root))
        other = next(int(a) for a in np.flatnonzero(root.legal()) if a != base)
        expected = []
        counts = {"decisions": 0, "round_endings": 0, "completed_branches": 0}

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
            names = continuation.names
            prepare = continuation.prepare

            def __call__(self, env, expected=expected, step=step):
                for state, ref in zip(env.state.cpu().numpy(), expected):
                    np.testing.assert_array_equal(state, ref.state)
                actions = continuation(env)
                for action, ref in zip(actions.cpu().tolist(), expected):
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
        for state, ref in zip(agent.workspaces[2].env.state.cpu().numpy(), expected):
            np.testing.assert_array_equal(state, ref.state)
            assert ref.state[PHASE] == TERMINAL
            points.append(
                int(ref.state[POINTS] - ref.state[POINTS + 1]) * (1 - 2 * seat)
            )
        np.testing.assert_array_equal(
            row["paired_points"], np.array(points).reshape(2, 2)
        )
