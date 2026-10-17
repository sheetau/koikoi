import numpy as np
import pytest
import torch

from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.reference import ReferenceEnv
from scripts.analyze_arbitration_study import tensor_equal
from scripts.analyze_paired_arbitration_study import data_report
from scripts.cache_paired_expert_features import replay
from scripts.collect_reusable_paired_targets import PairedWorkspace, collect
from scripts.run_arbitration_study import PARENTS
from scripts.train_paired_arbitration import collect as original_collect


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reused_collection_matches_original_and_replays(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    kwargs = {"seed": 18964001, "roots": 32, "batch": 8, "device": device}
    old = original_collect(PARENTS[0][0], tmp_path / "old.pt", **kwargs)
    path = tmp_path / "parent1" / "targets.pt"
    new = collect(PARENTS[0][0], path, **kwargs)
    for field in (
        "public_observations",
        "match_groups",
        "outcomes",
        "point_differences",
        "experts",
        "parent_value",
        "parent",
        "source_work",
        "branch_work",
    ):
        assert tensor_equal(old[field], new[field]), field
    protocol = {
        "collection": {"roots_per_parent": 32, "seeds": [kwargs["seed"]]},
        "parents": [{"sha256": PARENTS[0][1]}],
        "source": new["provenance"],
    }
    _, audit = data_report(tmp_path, 1, protocol)
    assert audit["roots"] == 32
    cache = replay(path, tmp_path / "features.pt")
    assert cache["verified_roots"] == 32
    assert cache["new_source_replay_work"] == new["source_work"]
    with pytest.raises(ValueError, match="fresh"):
        collect(PARENTS[0][0], path, **kwargs)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_workspace_reuses_backend_and_overwrites_terminal_state(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    game = ReferenceEnv(18965001, "paper8")
    while game.state[PHASE] != TERMINAL:
        before = game.state.copy()
        action = int(np.flatnonzero(game.legal())[0])
        game.step(action)
    expected = int(game.state[POINTS] - game.state[POINTS + 1]) * (
        1 - 2 * int(before[PLAYER])
    )
    states = torch.tensor(before[None], device=device)
    proposals = torch.tensor([[action, action]], device=device)
    workspace = PairedWorkspace(1, device)
    backend = workspace.env.backend

    def unexpected_continuation(env):
        raise AssertionError("Final action must terminate both branches")

    # Above the observed Windows failure threshold of334 registrations.
    for _ in range(512):
        wins, points, work = workspace(states, proposals, unexpected_continuation)
        assert workspace.env.backend is backend
        assert points.tolist() == [[expected, expected]]
        assert wins.tolist() == [[int(np.sign(expected)), int(np.sign(expected))]]
        assert work.completed_branches == work.decisions == 2
    assert np.array_equal(states.cpu().numpy()[0], before)
    with pytest.raises(ValueError, match="root count"):
        workspace(states.repeat(2, 1), proposals.repeat(2, 1), unexpected_continuation)
