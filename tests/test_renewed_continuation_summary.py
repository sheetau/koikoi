"""Completed-only reporting and separation of strength/mechanism claims."""

import numpy as np
import pytest

from scripts.renewed_continuation_analysis import analyse, decisions
from scripts.run_arbitration_study import sha, write
from scripts.summarize_renewed_continuation_study import render, summarize


def outcomes():
    points = np.empty((2, 5, 2, 1024, 2))
    points[..., 0], points[..., 1] = 2, -2
    points[:, 3:5] = 4
    return analyse(points)


def test_refuses_incomplete_failed_audit_and_existing_summary(tmp_path):
    with pytest.raises(ValueError, match="Complete producer"):
        summarize(tmp_path)
    write(tmp_path / "process.json", {"status": "complete"})
    with pytest.raises(ValueError, match="Passed independent audit"):
        summarize(tmp_path)
    write(tmp_path / "independent_audit.json", {"status": "failed"})
    with pytest.raises(ValueError, match="Passed independent audit"):
        summarize(tmp_path)
    assert not (tmp_path / "summary.md").exists()
    assert not (tmp_path / "summary_receipt.json").exists()
    (tmp_path / "summary.md").write_text("original\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite"):
        summarize(tmp_path)
    assert (tmp_path / "summary.md").read_text(encoding="utf-8") == "original\n"


def test_changed_audited_input_prevents_any_report(tmp_path):
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("original\n", encoding="utf-8")
    write(tmp_path / "process.json", {"status": "complete"})
    write(
        tmp_path / "independent_audit.json",
        {"status": "passed", "inputs": {str(evidence): sha(evidence)}},
    )
    evidence.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Changed audited input"):
        summarize(tmp_path)
    assert not (tmp_path / "summary.md").exists()


def test_inconsistent_claim_flags_prevent_report(tmp_path):
    contrasts = outcomes()
    write(tmp_path / "process.json", {"status": "complete"})
    write(
        tmp_path / "protocol.json",
        {
            "kind": "renewed_continuation_study_v1",
            "updates": 520,
            "pairs_per_opponent": 1024,
            "bootstrap_seed": 95310001,
            "bootstrap_replicates": 10000,
        },
    )
    expected = decisions(contrasts)
    wrong = {**expected, "replicated_public_memory_gain": True}
    write(
        tmp_path / "independent_audit.json",
        {
            "status": "passed",
            "inputs": {},
            "contrasts": contrasts,
            "decisions": expected,
        },
    )
    write(
        tmp_path / "analysis.json",
        {"status": "complete", "contrasts": contrasts, "decisions": wrong},
    )
    with pytest.raises(AssertionError):
        summarize(tmp_path)
    assert not (tmp_path / "summary.md").exists()


def test_report_retains_all_rows_and_separates_strength_from_memory():
    contrasts = outcomes()
    evidence = {
        "contrasts": contrasts,
        "decisions": decisions(contrasts),
        "evaluation_replay_work": {
            "decisions": 5000000,
            "round_endings": 300000,
            "completed_matches": 40960,
        },
        "root_replay_work": {
            "decisions": 212212,
            "round_endings": 14101,
            "completed_matches": 192,
        },
        "strict_input_checks": 1234,
    }
    training = {
        "decisions": 20000000,
        "learner_decisions": 9000000,
        "opponent_decisions": 11000000,
        "conditional_tail_completions": 266240,
        "conditional_round_endings": 1300000,
    }
    text = render(
        evidence, training, {"wp": 12, "point": 34}, "protocol-fixture", "audit-fixture"
    )
    lines = text.splitlines()
    assert next(line for line in lines if line.startswith("| Primary:")).endswith(
        "| Pass |"
    )
    assert all(
        line.endswith("| Fail |") for line in lines if line.startswith("| Secondary:")
    )
    rows = [line for line in lines if line.startswith(("| 1 |", "| 2 |"))]
    assert len(rows) == 66
    assert len(set(rows)) == 66
    assert (
        "| 1 | renewed_public-minus-parent | mixture | +50.000 [+50.000, +50.000] | +4.000 [+4.000, +4.000] |"
        in rows
    )
    assert "266,240" in text and "133,120 distinct chance keys" in text
    assert (
        "40,960 full evaluation matches" in text
        and "2,048 distinct evaluation deals" in text
    )
    assert "46 strict predictions (12 WP / 34 Point)" in text
    assert "zero full training matches" in text
    assert "zero neural predictions or independent deals" in text
