from copy import deepcopy
from fractions import Fraction

import pytest

from scripts import export_uniform_paper_evidence as paper
from scripts.run_arbitration_study import read, sha, write
from scripts.uniform_controller_design import POPULATIONS, summarize
from scripts.uniform_controller_study import report_text


def fixture_study(tmp_path, monkeypatch):
    base = tmp_path / "study"
    base.mkdir()
    cases = [{"index": i, "population": name} for i, name in enumerate(POPULATIONS)]
    protocol = {
        "cases": cases,
        "design": {
            "source_matches": 4,
            "sources_per_population": 1,
            "chunk_size": 4,
        },
    }
    write(base / "protocol.json", protocol)
    digest = sha(base / "protocol.json")
    monkeypatch.setattr(paper, "PROTOCOL_SHA", digest)
    monkeypatch.setattr(paper, "check", lambda _: protocol)
    rows = [{"case": case, "status": "missing_root"} for case in cases]
    rows[0].update(
        status="evaluated",
        selection={"gain": "1/100"},
        assessment={
            "outcomes": {"parent": {"margin": 1}, "controller": {"margin": -1}},
        },
    )
    monkeypatch.setattr(paper, "chunk_records", lambda *_: deepcopy(rows))
    folder = base / "chunk_0000"
    folder.mkdir()
    write(folder / "complete.json", {"fixture": True})
    audit = {
        "status": "passed",
        "protocol_sha256": digest,
        "complete_sha256": sha(folder / "complete.json"),
        "fresh_external_chance_draws": 0,
        "gains": [
            {"case": r["case"], "gain": r.get("selection", {}).get("gain", "0")}
            for r in rows
        ],
        "actual_differences": ["-1", "0", "0", "0"],
    }
    write(folder / "audit.json", audit)
    summary = {
        "audit_status": "passed",
        "protocol_sha256": digest,
        "sources": 4,
        "status_counts": {"evaluated": 1, "missing_root": 3},
        "bindings": {
            str(folder / n): sha(folder / n) for n in ("audit.json", "complete.json")
        },
        "inference": summarize(
            {
                n: [Fraction("1/100") if i == 0 else Fraction(0)]
                for i, n in enumerate(POPULATIONS)
            },
            expected_n=1,
        ),
        "actual_mean_difference_descriptive": dict(
            zip(POPULATIONS, ("-1", "0", "0", "0"))
        ),
        "work": {},
    }
    write(base / "queue.json", {"status": "complete"})
    bind_summary(base, summary)
    return base, summary


def bind_summary(base, summary):
    write(base / "summary.json", summary)
    (base / "results.rst").write_bytes(report_text(summary).encode("utf-8"))
    write(
        base / "export.json",
        {
            "status": "passed",
            "protocol_sha256": sha(base / "protocol.json"),
            "summary_sha256": sha(base / "summary.json"),
            "report_sha256": sha(base / "results.rst"),
            "source_matches": summary["sources"],
            "criterion_met": summary["inference"]["criterion_met"],
        },
    )


def test_complete_fixture_exports_separate_tables_and_bound_evidence(
    tmp_path, monkeypatch
):
    base, summary = fixture_study(tmp_path, monkeypatch)
    destination = tmp_path / "paper"
    ledger = read(paper.export(base, destination))
    assert ledger["summary"] == summary
    for name, digest in ledger["outputs"].items():
        assert sha(name) == digest
    primary = (destination / "uniform_primary_table.tex").read_text(encoding="utf-8")
    actual = (destination / "uniform_actual_table.tex").read_text(encoding="utf-8")
    assert "1 & WP & 1 & 1.0000" in primary
    assert "1 & WP & -100.0000" in actual
    assert "2 & Point" in primary and "Equal mixture" in primary
    with pytest.raises(FileExistsError):
        paper.export(base, destination)


def test_running_queue_creates_no_paper_artifacts(tmp_path, monkeypatch):
    base, _ = fixture_study(tmp_path, monkeypatch)
    write(base / "queue.json", {"status": "running"})
    destination = tmp_path / "paper"
    with pytest.raises(ValueError, match="interim"):
        paper.export(base, destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "change", ["mean", "actual", "denominator", "bindings", "criterion"]
)
def test_self_consistently_rehashed_bad_summary_is_rejected(
    tmp_path, monkeypatch, change
):
    base, summary = fixture_study(tmp_path, monkeypatch)
    if change == "mean":
        summary["inference"]["populations"]["0:wp"]["mean"] = "1/2"
    elif change == "actual":
        summary["actual_mean_difference_descriptive"]["0:wp"] = "1"
    elif change == "denominator":
        summary["sources"] = 3
    elif change == "bindings":
        summary["bindings"].clear()
    else:
        summary["inference"]["criterion_met"] = True
    bind_summary(base, summary)
    with pytest.raises(ValueError):
        paper.checked_result(base)


def test_changed_audit_and_missing_audit_are_rejected(tmp_path, monkeypatch):
    base, _ = fixture_study(tmp_path, monkeypatch)
    path = base / "chunk_0000/audit.json"
    audit = read(path)
    audit["gains"][0]["gain"] = "1/2"
    write(path, audit)
    with pytest.raises(ValueError, match="gain"):
        paper.checked_result(base)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        paper.checked_result(base)


def test_percentage_display_preserves_bound_direction_and_signed_actuals():
    assert paper.pp("1000001/1000000000", lower=True) == "0.1000"
    assert paper.pp("999999/1000000000", lower=True) == "0.0999"
    assert paper.pp("999999/1000000000") == "0.1000"
    assert paper.pp("-1/256") == "-0.3906"
    for value in (Fraction(0), Fraction(1, 3), Fraction(1, 7), Fraction(1)):
        displayed = Fraction(paper.pp(value, lower=True)) / 100
        assert displayed <= value < displayed + Fraction(1, 1_000_000)
