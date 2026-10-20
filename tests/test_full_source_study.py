import json

import pytest
import torch

from scripts.full_source_study import FullReplay, make_plan, run


def test_complete_twelve_arm_pipeline_and_full_state_corruption(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable for the fixed evaluation path")
    plan = make_plan()
    plan["run_kind"] = "capability"
    plan["updates"] = 2
    plan["save_interval"] = 1
    for case in plan["cases"].values():
        case["config"]["matches"] = 4
        case["config"]["gradient_batch"] = 16
        case["config"]["seed"] = 119040001 + case["parent_index"] * 10000
    plan["evaluation"].update(
        pairs=2,
        batch=4,
        bootstrap_replicates=30,
        wp_seed=119060001,
        point_seed=119070001,
        bootstrap_seed=119080001,
    )
    base = tmp_path / "study"
    run(base, plan)
    finished = json.loads((base / "finished.json").read_text(encoding="utf-8"))
    assert finished["status"] == "complete"
    complete = json.loads((base / "complete.json").read_text(encoding="utf-8"))
    summary = json.loads((base / "evaluation/summary.json").read_text(encoding="utf-8"))
    assert len(complete["cases"]) == 12
    assert len(summary["primary"]) == 6 and len(summary["model_order"]) == 14
    for path in ("audit.json", "evaluation/audit.json"):
        assert (
            json.loads((base / path).read_text(encoding="utf-8"))["status"] == "passed"
        )
    assert summary["work"]["completions"] == 112
    with pytest.raises(ValueError, match="fresh"):
        run(base, plan)
    data = torch.load(
        base / "parent1_full_high/evidence_0002.pt",
        map_location="cpu",
        weights_only=False,
    )["source"]
    data["decision_returns"][0] += 0.25
    with pytest.raises(AssertionError):
        FullReplay()(data["initial"], data, source=True)
