import json

import pytest

from scripts.run_arbitration_study import sha
from scripts.run_score_fit_screen import make_plan


def test_custom_data_budget_and_audit_binding(tmp_path):
    data, features = tmp_path / "data", tmp_path / "features"
    features.mkdir()
    parents = {}
    for parent in (1, 2):
        folder = data / f"parent{parent}"
        folder.mkdir(parents=True)
        dataset = folder / "targets.pt"
        cache = features / f"parent{parent}.pt"
        dataset.write_bytes(f"dataset{parent}".encode())
        cache.write_bytes(f"cache{parent}".encode())
        parents[str(parent)] = {
            "dataset_sha256": sha(dataset),
            "cache_sha256": sha(cache),
        }
    audit = features / "audit.json"
    audit.write_text(
        json.dumps({"status": "passed", "parents": parents}), encoding="utf-8"
    )
    plan = make_plan(("paired_win", "preference"), data, features, 80)
    assert len(plan["cases"]) == 8 and plan["epochs"] == 80
    assert all(str(data) in case["dataset"] for case in plan["cases"])
    assert {case["seed"] for case in plan["cases"]} == {19410001, 19420001}
    assert plan["weight_decay"] == 1e-4
    assert (
        make_plan(("preference",), data, features, weight_decay=0)["weight_decay"] == 0
    )
    for bad_decay in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="weight decay"):
            make_plan(("preference",), data, features, weight_decay=bad_decay)
    with pytest.raises(ValueError, match="positive"):
        make_plan(("preference",), data, features, 0)
    (features / "parent1.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not bind"):
        make_plan(("preference",), data, features, 20)
