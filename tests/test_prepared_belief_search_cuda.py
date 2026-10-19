"""Independent reference transition check for the prepared-belief CUDA path."""

import numpy as np
import pytest
import torch

from koikoi.layout import PHASE, POINTS, TERMINAL
from scripts.diagnose_history_support import reference
from scripts.history_conditioned_worlds import sample_joint_worlds
from scripts.prepared_belief_search import KINDS, PreparedBeliefSearch
from scripts.run_arbitration_study import PARENTS
from scripts.run_prepared_belief_actions import load_roots


def test_cuda_prepared_beliefs_match_reference_through_terminal():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    agent = PreparedBeliefSearch(PARENTS[0][0], particles=2, confirmation_particles=2)
    row, root, batch = next(load_roots())
    base = int(agent.parent.act(root))
    alternative = next(int(a) for a in np.flatnonzero(root.legal()) if a != base)
    original = agent.continuation
    for kind in KINDS:
        agent.prepare(
            root,
            batch,
            kind,
            row["public_identification"],
            row["decision"],
            row["seat"],
        )
        worlds, _ = sample_joint_worlds(
            root, agent.support, np.random.default_rng(777), 2
        )
        expected, counts = [], {"decisions": 0, "round_endings": 0, "completed_branches": 0}

        def step(env, action, counts=counts):
            assert env.legal()[action]
            result = env.step(action)
            counts["decisions"] += 1
            counts["round_endings"] += int(result[1])
            counts["completed_branches"] += int(result[2])

        for world in worlds:
            for action in (base, alternative):
                env = reference(world)
                step(env, action)
                expected.append(env)

        class Checked:
            names = original.names
            prepare = original.prepare

            def __call__(self, env, expected=expected, step=step):
                for state, ref in zip(env.state.cpu().numpy(), expected):
                    np.testing.assert_array_equal(state, ref.state)
                actions = original(env)
                for action, ref in zip(actions.cpu().tolist(), expected):
                    if ref.state[PHASE] != TERMINAL:
                        step(ref, action)
                return actions

        agent.continuation = Checked()
        result = agent._compare_actions(
            root, base, [alternative], 2, np.random.default_rng(777)
        )[0]
        assert result["work"] == counts
        assert counts["completed_branches"] == 4
        points = []
        for state, ref in zip(agent.workspaces[2].env.state.cpu().numpy(), expected):
            np.testing.assert_array_equal(state, ref.state)
            assert ref.state[PHASE] == TERMINAL
            points.append(
                int(ref.state[POINTS] - ref.state[POINTS + 1]) * (1 - 2 * row["seat"])
            )
        paired = np.array(points).reshape(2, 2)
        np.testing.assert_array_equal(result["paired_points"], paired)
        difference = np.diff(np.sign(paired), axis=1).ravel() / 2
        assert result["win_change"] == float(difference.mean())
        assert result["lower_margin"] == pytest.approx(
            float(difference.mean() - 2.5 * difference.std(ddof=1) / np.sqrt(2))
        )
    agent.continuation = original
