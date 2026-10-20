import numpy as np
import pytest
import torch

from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER, POINTS, ROUND, TERMINAL
from koikoi.reference import ReferenceEnv
from scripts.evaluate_branch_policy_gradient import contrasts, play_chunk, report


class OrderedActor(torch.nn.Module):
    def __init__(self, sign):
        super().__init__()
        self.sign = sign

    def forward(self, features, phases, legal):
        scores = torch.arange(50, device=legal.device).float() * self.sign
        return scores.expand_as(legal).masked_fill(~legal, -torch.inf)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_seat_pairs_initialization_roles_and_independent_replay(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    env = BatchEnv(4, device, debug=True)
    data = play_chunk(env, OrderedActor(1), OrderedActor(-1), 110040001)
    work = dict.fromkeys(data["work"], 0)
    for lane in range(4):
        game = ReferenceEnv(110040001 + lane // 2)
        assert np.array_equal(game.state, data["initial"][lane].numpy())
        legal = np.flatnonzero(game.legal())
        expected_first = legal[-1] if game.state[PLAYER] == lane % 2 else legal[0]
        assert int(data["actions"][0, lane]) == expected_first
        for action in data["actions"][:, lane]:
            if game.state[PHASE] == TERMINAL:
                continue
            old_round = int(game.state[ROUND])
            assert game.legal()[int(action)]
            game.step(int(action))
            work["decisions"] += 1
            work["round_endings"] += (
                int(game.state[ROUND]) - old_round + int(game.state[PHASE] == TERMINAL)
            )
            work["completions"] += int(game.state[PHASE] == TERMINAL)
        assert np.array_equal(game.state, data["final"][lane].numpy())
        margin = (game.state[POINTS] - game.state[POINTS + 1]) * (1 - 2 * (lane % 2))
        assert int(data["points"].reshape(-1)[lane]) == margin
    assert work == data["work"]


def test_primary_axes_draw_adjustment_and_simultaneous_null():
    order = [
        f"parent{parent}{suffix}"
        for parent in (1, 2)
        for suffix in ("", "_shared_actions", "_independent_actions", "_sampled_return")
    ]
    points = np.zeros((8, 2, 5, 2), dtype=np.int32)
    settings = {"bootstrap_seed": 42, "bootstrap_replicates": 100}
    null = report(points, order, settings)
    assert not null["progression_criterion"]
    assert all(r["simultaneous_95"] == [0, 0] for r in null["primary"])
    points[order.index("parent1_shared_actions"), 0, 0] = [2, -2]
    points[order.index("parent1_shared_actions"), 1, 0] = [4, 0]
    points[order.index("parent2_shared_actions"), :, 3] = [-2, 0]
    _, rows = contrasts(points, order)
    expected = np.zeros((5, 6))
    expected[0, :3] = 0.125
    expected[3, 3:] = -0.25
    np.testing.assert_array_equal(rows, expected)
    result = report(points, order, settings)
    assert result["primary"][0]["win_change"] == 0.025
    assert result["primary"][3]["win_change"] == -0.05
    assert not result["progression_criterion"]
