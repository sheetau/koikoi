"""Check the new scalar verifier against a previously verified exact table."""

import copy
import json
from pathlib import Path

import pytest

from scripts.audit_complete_sampled_history_exact import audit, verify_row
from scripts.run_arbitration_study import read, write


def fixture():
    base = Path("runs/sampled_history_exact_check")
    row = copy.deepcopy(read(base / "report.json")["results"][0])
    original = next(
        r
        for r in map(
            json.loads,
            Path("runs/sampled_history_diagnostic/records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines(),
        )
        if (r["parent"], r["case_index"]) == (row["parent"], row["case_index"])
    )
    batch = read(base / row["support_file"])
    row["public_root_key"] = batch["root_key"]
    for name in row["comparison"]:
        row["comparison"][name]["status"] = "estimated"
        row["comparison"][name]["sampled_status"] = original["beliefs"][name]["status"]
    return row, original, batch


def test_previous_exact_candidate_set_and_all_scalar_estimates_reproduce():
    row, original, batch = fixture()
    values, steps, valid, invalid = verify_row(row, original, batch)
    assert (valid, invalid) == (5985, 0)
    assert steps == row["decision"]
    assert values["identified"] < values["joint"] < values["uniform"]


def test_changed_probability_or_missing_candidate_is_rejected():
    row, original, batch = fixture()
    bad = copy.deepcopy(row)
    bad["exact_beliefs"]["joint"]["card_probabilities"][0] += 0.01
    with pytest.raises(AssertionError):
        verify_row(bad, original, batch)
    bad_batch = copy.deepcopy(batch)
    bad_batch["hands"].pop()
    with pytest.raises(AssertionError):
        verify_row(row, original, bad_batch)


def test_incomplete_producer_cannot_create_audit(tmp_path):
    write(tmp_path / "process.json", {"status": "running"})
    with pytest.raises(AssertionError):
        audit(tmp_path)
    assert not (tmp_path / "arithmetic_verification.json").exists()
