"""Primary qualification cannot be replaced by favorable secondary findings."""

import pytest

import scripts.renewed_confirmation_gate as gate
from scripts.renewed_confirmation_gate import check, qualification
from scripts.renewed_continuation_analysis import NAMES, SCOPES
from scripts.run_arbitration_study import sha, write


def cells():
    return [
        {
            "parent": p,
            "comparison": name,
            "scope": scope,
            "win": {"change": 0.0, "interval95": [-0.01, 0.01]},
            "point": {"change": 3.0, "interval95": [2.0, 4.0]},
        }
        for p in (1, 2)
        for scope, _ in SCOPES
        for name in NAMES
    ]


def test_primary_qualifies_without_upgrading_mechanism_claims():
    rows = cells()
    for row in rows:
        if row["scope"] == "mixture" and row["comparison"] in (
            "renewed_public-minus-parent",
            "renewed_public-minus-fixed_public",
        ):
            row["win"] = {"change": 0.02, "interval95": [0.01, 0.03]}
    result = qualification(rows)
    assert result["qualified"]
    assert not result["development_claims"]["replicated_public_memory_gain"]
    assert not result["development_claims"]["replicated_positive_interaction"]
    assert result["parents"] == [1, 2] and result["endpoint_update"] == 520
    assert not result["automatic_promotion"]
    next(
        r
        for r in rows
        if r["parent"] == 2
        and r["scope"] == "mixture"
        and r["comparison"] == "renewed_public-minus-fixed_public"
    )["win"]["interval95"][0] = 0
    assert not qualification(rows)["qualified"]


def test_secondary_wins_and_points_do_not_qualify():
    rows = cells()
    for row in rows:
        if row["scope"] != "mixture" or row["comparison"] in (
            "renewed_public-minus-renewed_masked",
            "chance_by_memory",
        ):
            row["win"] = {"change": 0.02, "interval95": [0.01, 0.03]}
    result = qualification(rows)
    assert not result["qualified"]
    assert result["development_claims"]["replicated_public_memory_gain"]
    assert result["development_claims"]["replicated_positive_interaction"]
    with pytest.raises(ValueError, match="Incomplete"):
        qualification(rows[:-1])


def test_incomplete_unaudited_or_unsummarized_studies_are_refused(tmp_path):
    with pytest.raises(ValueError, match="Completed renewed study"):
        check(tmp_path)
    write(tmp_path / "process.json", {"status": "complete"})
    with pytest.raises(ValueError, match="Passed independent audit"):
        check(tmp_path)
    write(tmp_path / "independent_audit.json", {"status": "passed"})
    with pytest.raises(ValueError, match="Completed study summary"):
        check(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "independent_audit.json",
        "process.json",
    ]


def test_changed_audited_artifact_cannot_reach_qualification(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol.json"
    write(protocol, {"fixture": True})
    monkeypatch.setattr(gate, "PROTOCOL_SHA", sha(protocol))
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("original", encoding="utf-8")
    write(tmp_path / "process.json", {"status": "complete"})
    write(
        tmp_path / "independent_audit.json",
        {
            "status": "passed",
            "inputs": {str(evidence): sha(evidence)},
        },
    )
    write(
        tmp_path / "summary_receipt.json",
        {
            "status": "complete",
            "protocol_sha256": sha(protocol),
        },
    )
    evidence.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Changed audited input"):
        check(tmp_path)
