"""Final confirmation must qualify first and preserve every declared result."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.audit_renewed_confirmation as auditor
import scripts.renewed_confirmation_analysis as analysis
import scripts.run_renewed_confirmation as runner
from koikoi.train import provenance
from scripts.audit_public_continuation_study import replay_match
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.summarize_renewed_confirmation import summarize


def test_unfinished_development_cannot_create_output(tmp_path):
    study, output = tmp_path / "study", tmp_path / "confirmation"
    study.mkdir()
    write(study / "process.json", {"status": "running"})
    with pytest.raises(ValueError, match="Completed renewed study"):
        runner.run(output, study)
    assert not output.exists()


def test_failed_primary_cannot_read_models_or_create_output(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "check", lambda _: {"qualified": False})
    # No receipts, checkpoints or native environment are needed to refuse.
    with pytest.raises(ValueError, match="Replicated primary"):
        runner.run(tmp_path / "confirmation", tmp_path / "study")
    assert not list(tmp_path.iterdir())


def test_predeclaration_tamper_is_refused(tmp_path, monkeypatch):
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "check", lambda _: {"qualified": True})
    monkeypatch.setattr(runner, "PREDECLARATION", receipt)
    with pytest.raises(ValueError, match="Changed prospective"):
        runner.make_plan(tmp_path)


def test_qualified_plan_binds_models_and_independent_checker_rejects_drift(
    tmp_path, monkeypatch
):
    # Qualification is synthetic here; this tests plan construction, not the
    # already-tested strength gate. Dummy new endpoints are never loaded.
    study = tmp_path / "synthetic_development"
    inputs = {}
    for index, (path, digest) in enumerate(PARENTS, 1):
        inputs[path] = digest
        fixed = runner.FIXED / f"parent{index}_public/final.pt"
        inputs[str(fixed)] = sha(fixed)
        fresh = study / f"parent{index}_public/final.pt"
        fresh.parent.mkdir(parents=True)
        fresh.write_bytes(b"synthetic endpoint; do not load")
        inputs[str(fresh)] = sha(fresh)
    write(study / "independent_audit.json", {"inputs": inputs})
    for name in ("summary_receipt.json", "process.json"):
        write(study / name, {"fixture": True})
    (study / "summary.md").write_text("synthetic fixture\n", encoding="utf-8")
    predecl = tmp_path / "predeclaration.json"
    write(predecl, {"inputs": inputs})
    xml = tmp_path / "tests.xml"
    xml.write_text("<testsuite>" + "<testcase/>" * 8 + "</testsuite>", encoding="utf-8")
    qualification = {"qualified": True, "development_claims": {"fixture": True}}
    monkeypatch.setattr(runner, "check", lambda _: qualification)
    monkeypatch.setattr(auditor, "check", lambda _: qualification)
    monkeypatch.setattr(runner, "PREDECLARATION", predecl)
    monkeypatch.setattr(runner, "PREDECLARATION_SHA", sha(predecl))
    monkeypatch.setattr(runner, "TEST_XML", xml)
    plan = runner.make_plan(study)
    auditor.verify_plan(plan)
    assert plan["matches"] == 98304 and plan["distinct_deals"] == 8192
    for key, value in (("workers", 3), ("deal_start", 95110001), ("matches", 8192)):
        with pytest.raises(AssertionError):
            auditor.verify_plan({**plan, key: value})
    bad = copy.deepcopy(plan)
    bad["parents"][1]["models"]["renewed_public"] = PARENTS[1][0]
    with pytest.raises(AssertionError):
        auditor.verify_plan(bad)
    (study / "parent2_public/final.pt").write_bytes(b"changed endpoint")
    with pytest.raises(AssertionError):
        auditor.verify_plan(plan)


def test_audit_refuses_unfinished_or_existing_receipt(tmp_path):
    write(tmp_path / "process.json", {"status": "running"})
    with pytest.raises(AssertionError):
        auditor.audit(tmp_path)
    write(tmp_path / "independent_audit.json", {"preserve": True})
    with pytest.raises(ValueError, match="overwrite"):
        auditor.audit(tmp_path)
    assert read(tmp_path / "independent_audit.json") == {"preserve": True}


def test_independent_identity_rejects_wrong_or_missing_cells():
    # Identifiers only: no final deals are constructed.
    for index, policy in enumerate(("parent", "fixed_public", "renewed_public")):
        row = {
            "parent": 2,
            "opponent": "point",
            "policy": policy,
            "seat": 1,
            "replicate": 4095,
            "seed": 101114096,
            "terminal_points": [28, 32],
            "memory": None if index == 0 else {},
        }
        assert auditor.verify_identity(row, 2, "point") == (index, 4095, 1)
        for key, value in (
            ("parent", True),
            ("seat", True),
            ("replicate", 4096),
            ("seed", 95120001),
            ("terminal_points", [30, 31]),
            ("terminal_points", [30.0, 30.0]),
            ("memory", {} if index == 0 else None),
        ):
            with pytest.raises(AssertionError):
                auditor.verify_identity({**row, key: value}, 2, "point")
        with pytest.raises(ValueError):
            auditor.verify_identity({**row, "policy": "renewed_masked"}, 2, "point")


def test_scalar_analysis_matches_all_variable_contrasts_and_absolutes(monkeypatch):
    # 129 resamples cross a chunk boundary at the full4,096-pair shape.
    # No final deal is constructed; all outcomes are artificial integer scores.
    monkeypatch.setattr(auditor, "BOOTSTRAP_DRAWS", 129)
    monkeypatch.setattr(analysis, "BOOTSTRAP_DRAWS", 129)
    points = np.random.default_rng(634).integers(-9, 10, (2, 3, 2, 4096, 2))
    result = auditor.independent_analysis(points)
    assert result == analysis.analyse(points)
    assert len(result["contrasts"]) == len(result["absolute"]) == 18
    assert not result["confirmed_strength_gain"]
    # Decisive constant outcomes use the complete production10,000-draw budget.
    monkeypatch.setattr(auditor, "BOOTSTRAP_DRAWS", 10000)
    monkeypatch.setattr(analysis, "BOOTSTRAP_DRAWS", 10000)
    points[:] = -1
    points[:, 2] = 1
    result = auditor.independent_analysis(points)
    assert result == analysis.analyse(points) and result["confirmed_strength_gain"]
    points[1, 2] = -1
    assert not auditor.independent_analysis(points)["confirmed_strength_gain"]


def test_cell_audit_rejects_empty_and_duplicate_records(tmp_path, monkeypatch):
    path = tmp_path / "cell.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(AssertionError):
        auditor.audit_cell(path, 1, "wp")
    row = {
        "parent": 1,
        "opponent": "wp",
        "policy": "parent",
        "seat": 0,
        "replicate": 0,
        "seed": 101100001,
        "terminal_points": [30, 30],
        "memory": None,
    }
    path.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
    calls = []

    def synthetic_replay(record):
        calls.append(record)
        return [], {"decisions": 0, "round_endings": 0, "completed_matches": 1}, 0

    monkeypatch.setattr(auditor, "replay_match", synthetic_replay)
    with pytest.raises(AssertionError):
        auditor.audit_cell(path, 1, "wp")
    assert len(calls) == 1  # Duplicate rejected before replay; no simulated deal.


def summary_fixture(root):
    result = auditor.independent_analysis(np.zeros((2, 3, 2, 4096, 2)))
    write(root / "process.json", {"status": "complete"})
    write(root / "protocol.json", {"fixture": True})
    write(root / "analysis.json", {"status": "complete", **result})
    audit = {
        "status": "passed",
        **result,
        "inputs": {str(root / "analysis.json"): sha(root / "analysis.json")},
        "evaluation_replay_work": {
            "decisions": 0,
            "round_endings": 0,
            "completed_matches": 98304,
        },
        "distinct_evaluation_deals": 8192,
        "strict_input_checks": 0,
        "development_mechanism_claims": {
            "replicated_strength_gain": True,
            "replicated_public_memory_gain": False,
            "replicated_positive_interaction": False,
        },
    }
    write(root / "independent_audit.json", audit)
    return audit


def test_summary_preserves_negative_primary_and_all_cells(tmp_path):
    summary_fixture(tmp_path)
    text = summarize(tmp_path).read_text(encoding="utf-8")
    assert "**False**" in text and text.count("| 1 |") == text.count("| 2 |") == 18
    assert "does not retest either mechanism" in text
    receipt = read(tmp_path / "summary_receipt.json")
    assert not receipt["confirmed_strength_gain"] and not receipt["automatic_promotion"]
    with pytest.raises(ValueError, match="overwrite"):
        summarize(tmp_path)


def test_summary_rejects_changed_or_inconsistent_audit(tmp_path):
    audit = summary_fixture(tmp_path)
    broken = copy.deepcopy(audit)
    broken["confirmed_strength_gain"] = True
    write(tmp_path / "independent_audit.json", broken)
    with pytest.raises(ValueError, match="results disagree"):
        summarize(tmp_path)
    write(tmp_path / "independent_audit.json", audit)
    with (tmp_path / "analysis.json").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Changed audited input"):
        summarize(tmp_path)
    assert not (tmp_path / "summary.md").exists()


def test_worker_loads_three_formats_and_replays_old_deal_both_seats(
    tmp_path, monkeypatch
):
    # Only initial models and one already-consumed88M development deal.
    # No trained ongoing endpoint or reserved101M trajectory is inspected.
    parent = {
        "index": 1,
        "models": {
            "parent": PARENTS[0][0],
            "fixed_public": "runs/public_continuation_study/parent1_public/initial.pt",
            "renewed_public": "runs/renewed_continuation_study/parent1_public/initial.pt",
        },
    }
    write(
        tmp_path / "protocol.json",
        {
            "parents": [parent],
            "inputs": {path: sha(path) for path in parent["models"].values()},
            "core_source_sha256": provenance()["source_sha256"],
        },
    )
    monkeypatch.setattr(runner, "configure", lambda: torch.set_num_threads(2))
    monkeypatch.setattr(
        runner,
        "cases",
        lambda index, opponent: [
            {
                "parent": index,
                "opponent": opponent,
                "seat": seat,
                "replicate": 0,
                "seed": 88110001,
            }
            for seat in (0, 1)
        ],
    )
    path = Path(runner.evaluate_cell(str(tmp_path), parent, "wp"))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["policy"] for r in rows] == [
        "parent",
        "parent",
        "fixed_public",
        "fixed_public",
        "renewed_public",
        "renewed_public",
    ]
    work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    checks = 0
    for row in rows:
        seat = row["seat"]
        assert row["actions"] == rows[seat]["actions"]
        if row["memory"] is not None:
            assert row["memory"] == rows[2 + seat]["memory"]
        _, counts, checked = replay_match(row)
        checks += checked
        for key, value in counts.items():
            work[key] += value
    assert work["completed_matches"] == 6
    write(
        Path("runs/renewed_confirmation_worker_fixture.json"),
        {
            "kind": "old_development_deal_initial_models_correctness_only",
            "inputs": {path: sha(path) for path in parent["models"].values()},
            "actual_work": work,
            "replay_work": work,
            "strict_input_checks": checks,
            "shared_previously_consumed_deals": 1,
            "fresh_confirmation_deals": 0,
            "prediction_calls": {
                name: sum(
                    row["memory"]["prediction_calls"][name]
                    for row in rows
                    if row["memory"] is not None
                )
                for name in ("wp", "point")
            },
        },
    )
