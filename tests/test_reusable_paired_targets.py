import numpy as np
import pytest
import torch

from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.reference import ReferenceEnv
from scripts.analyze_arbitration_study import tensor_equal
from scripts.analyze_paired_arbitration_study import data_report
from scripts.cache_paired_expert_features import replay
from scripts.collect_reusable_paired_targets import PairedWorkspace, collect
from scripts.replay_reusable_paired_features import replay as reusable_replay
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
    reused_cache = reusable_replay(path, tmp_path / "reused_features.pt")
    for field in (
        "embeddings",
        "scores",
        "public_dealer",
        "new_source_replay_work",
        "reused_branch_work",
        "verified_roots",
    ):
        assert tensor_equal(cache[field], reused_cache[field]), field
    from scripts.audit_paired_feature_cache import inspect

    inspected = inspect(path, tmp_path / "reused_features.pt")
    assert inspected["roots"] == 32
    assert inspected["new_branch_decisions"] == 0
    reused_cache["scores"][0, 0, 0] += 1
    corrupt = tmp_path / "bad_scores.pt"
    torch.save(reused_cache, corrupt)
    with pytest.raises(AssertionError):
        inspect(path, corrupt)
    reused_cache["dataset_sha256"] = "wrong"
    torch.save(reused_cache, corrupt)
    with pytest.raises(ValueError, match="identity"):
        inspect(path, corrupt)
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


def test_recovery_reuses_completed_dataset_and_finishes_other_parent(
    tmp_path, monkeypatch
):
    import copy
    import json

    from scripts import complete_scaled_paired_targets as recovery
    from scripts.run_arbitration_study import write

    predecessor, target = tmp_path / "previous", tmp_path / "recovered"
    plan = recovery.make_plan()
    plan["collection"]["roots_per_parent"] = 32
    plan["collection"]["batch"] = 16
    plan["collection"]["device"] = "cpu"
    monkeypatch.setattr(recovery, "make_plan", lambda: copy.deepcopy(plan))
    write(predecessor / "protocol.json", plan)
    first = predecessor / "data" / "parent1" / "targets.pt"
    collect(
        PARENTS[0][0],
        first,
        plan["collection"]["seeds"][0],
        roots=32,
        batch=16,
        device="cpu",
    )
    _, report = data_report(predecessor / "data", 1, plan)
    write(first.parent / "dataset_audit.json", {"status": "passed", **report})
    write(
        predecessor / "process.json",
        {
            "status": "failed",
            "error": "FileNotFoundError(2, 'The filename or extension is too long')",
        },
    )
    calls = []

    def cpu_collect(parent, output, **kwargs):
        calls.append(parent)
        kwargs["device"] = "cpu"
        return collect(parent, output, **kwargs)

    monkeypatch.setattr(recovery, "collect", cpu_collect)
    recovery.run(target, predecessor)
    assert calls == [PARENTS[1][0]]
    assert (
        first.read_bytes() == (target / "data" / "parent1" / "targets.pt").read_bytes()
    )
    summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
    assert summary["1"]["reused_predecessor_dataset"]
    assert summary["1"]["new_collection_branch_work"] is None
    assert not summary["2"]["reused_predecessor_dataset"]
    assert summary["2"]["new_collection_branch_work"]["completed_branches"] == 64
    assert all(row["verified_replay_roots"] == 32 for row in summary.values())
    assert (
        json.loads((target / "process.json").read_text(encoding="utf-8"))["status"]
        == "complete"
    )
    with pytest.raises(ValueError, match="fresh"):
        recovery.run(target, predecessor)
    plan["collection"]["roots_per_parent"] = 64
    with pytest.raises(ValueError, match="protocol mismatch"):
        recovery.recovery_plan(predecessor)
