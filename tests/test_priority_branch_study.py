import json

import pytest
import torch

from scripts.priority_branch_study import make_plan, run, verify_reservoir


def test_complete_reduced_study_and_independent_audits(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable for the fixed development execution path")
    plan = make_plan()
    plan["run_kind"] = "capability"
    plan["updates"] = 2
    plan["save_interval"] = 1
    for case in plan["cases"].values():
        case["config"]["matches"] = 4
        case["config"]["seed"] = 114030001 + case["parent_index"] * 10000
    plan["evaluation"].update(
        {
            "pairs": 2,
            "batch": 4,
            "bootstrap_replicates": 30,
            "wp_seed": 114050001,
            "point_seed": 114060001,
            "bootstrap_seed": 114070001,
        }
    )
    base = tmp_path / "study"
    run(base, plan)
    finished = json.loads((base / "finished.json").read_text(encoding="utf-8"))
    assert finished["status"] == "complete"
    train_audit = json.loads((base / "audit.json").read_text(encoding="utf-8"))
    eval_audit = json.loads(
        (base / "evaluation" / "audit.json").read_text(encoding="utf-8")
    )
    assert train_audit["status"] == eval_audit["status"] == "passed"
    assert eval_audit["replayed_work"]["completions"] == 80
    assert (
        len(json.loads((base / "complete.json").read_text(encoding="utf-8"))["cases"])
        == 8
    )
    with pytest.raises(ValueError, match="fresh"):
        run(base, plan)
    data = torch.load(
        base / "parent1_priority_high" / "evidence_0002.pt",
        map_location="cpu",
        weights_only=False,
    )
    data["source"]["importance"][0] *= 1.01
    with pytest.raises(AssertionError):
        verify_reservoir(data["source"])
