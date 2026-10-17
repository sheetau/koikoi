import pytest
import torch

from koikoi.arbitration import read_checkpoint
from scripts.analyze_arbitration_study import tensor_equal
from scripts.audit_paired_feature_cache import inspect
from scripts.collect_opponent_paired_targets import CONDITIONS, collect
from scripts.run_arbitration_study import PARENTS
from scripts.train_score_distribution import fit


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_common_roots_inline_features_and_actual_fitting(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    base = tmp_path / "collected"
    report = collect(
        PARENTS[0][0], base, 18971001, roots=64, batch=16, envs=16, device=device
    )
    datasets, caches = [], []
    for condition in CONDITIONS:
        folder = base / condition
        dataset, cache_path = folder / "targets.pt", folder / "features.pt"
        data = read_checkpoint(dataset)[0]
        cache = read_checkpoint(cache_path)[0]
        audited = inspect(dataset, cache_path)
        assert audited["roots"] == 64
        assert audited["extraction"] == "at_collection"
        assert not any(audited["source_replay_work"].values())
        assert (
            report["conditions"][condition]["branch_work"]["completed_branches"] == 128
        )
        assert torch.equal(data["outcomes"], data["point_differences"].sign().float())
        assert "states" not in data and "states" not in cache
        assert data["source_work"] == report["shared_source_work"]
        fitted = fit(
            dataset,
            folder / "fitted.pt",
            kind="paired_win",
            epochs=2,
            device=device,
            feature_cache=cache_path,
        )
        assert fitted["feature_cache"]["extraction"] == "at_collection"
        assert not any(fitted["feature_cache"]["source_replay_work"].values())
        assert fitted["counters"]["new_simulation_decisions"] == 0
        datasets.append(data)
        caches.append(cache)
    for field in (
        "public_observations",
        "match_groups",
        "opponent_ids",
        "source_work",
        "experts",
    ):
        assert tensor_equal(datasets[0][field], datasets[1][field])
    for field in ("embeddings", "scores", "public_dealer", "aligned_source_work"):
        assert tensor_equal(caches[0][field], caches[1][field])
    assert caches[0]["dataset_sha256"] != caches[1]["dataset_sha256"]
    work = report["shared_source_work"]
    assert (
        work["wp_completed_matches"] + work["point_completed_matches"]
        == work["completed_matches"]
    )
    # Actual cached-vector corruption must be detected before a fit is accepted.
    cache = caches[0]
    cache["scores"][0, 0, 0] += 1
    corrupt = tmp_path / "corrupt.pt"
    torch.save(cache, corrupt)
    with pytest.raises(AssertionError):
        inspect(base / CONDITIONS[0] / "targets.pt", corrupt)
    cache["aligned_source_work"] = dict.fromkeys(work, 0)
    torch.save(cache, corrupt)
    with pytest.raises(ValueError, match="does not match"):
        fit(
            base / CONDITIONS[0] / "targets.pt",
            tmp_path / "rejected.pt",
            feature_cache=corrupt,
        )
    with pytest.raises(ValueError, match="fresh"):
        collect(
            PARENTS[0][0], base, 18971001, roots=64, batch=16, envs=16, device=device
        )


def test_fixed_study_collects_fits_and_cross_evaluates(tmp_path, monkeypatch):
    import copy
    import json

    from scripts import run_opponent_target_study as study

    plan = study.make_plan()
    plan["parents"] = plan["parents"][:1]
    plan["collection"].update(roots=64, batch=16, envs=16, device="cpu")
    plan["fitting"]["epochs"] = 2
    monkeypatch.setattr(study, "make_plan", lambda: copy.deepcopy(plan))
    base = tmp_path / "study"
    study.run(base)
    assert (
        json.loads((base / "process.json").read_text(encoding="utf-8"))["status"]
        == "complete"
    )
    result = json.loads((base / "results.json").read_text(encoding="utf-8"))["1"]
    assert len(result["analysis"]["endpoints"]) == 4
    assert len(result["analysis"]["opponent_target_minus_parent_target"]) == 2
    assert result["collection"]["roots"] == 64
    assert (
        sum(
            row["branch_work"]["completed_branches"]
            for row in result["data_audit"].values()
        )
        == 256
    )
    assert all(
        row["training_evaluation_groups_disjoint"]
        for row in result["analysis"]["endpoints"].values()
    )
    with pytest.raises(ValueError, match="fresh"):
        study.run(base)
    bad = copy.deepcopy(plan)
    bad["inputs"]["scripts/opponent_paired_roots.py"] = "wrong"
    with pytest.raises(ValueError, match="input changed"):
        study.verify_inputs(bad)
