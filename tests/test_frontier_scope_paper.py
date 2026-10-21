"""Check scientific transcription against real audited small-cohort ledgers."""

import re
from fractions import Fraction

import pytest

from scripts import export_frontier_scope_paper as paper
from scripts import frontier_scope_study as study
from scripts.export_frontier_scope import export as study_export
from scripts.run_arbitration_study import read, sha, write
from tests.test_frontier_scope_ledger import complete_zero_chunk  # noqa: F401


@pytest.fixture
def complete_export(complete_zero_chunk, monkeypatch):  # noqa: F811
    base, _ = complete_zero_chunk
    write(base / "queue.json", {"status": "complete"})
    study.summarize_study(base)
    study_export(base)
    # Only the size/pin of this deterministic test cohort differs. The actual
    # collection, independent audit and complete reduction all execute.
    monkeypatch.setattr(paper, "PROTOCOL_SHA", sha(base / "protocol.json"))
    monkeypatch.setattr(paper, "SOURCE_MATCHES", 4)
    monkeypatch.setattr(paper, "RETAINED_SOURCES", 0)
    return base


def test_failed_criterion_still_exports_every_source_and_separate_work(complete_export):
    base = complete_export
    destination = base / "paper"
    ledger = read(paper.export(base, destination))
    assert ledger["status"] == "passed"
    assert not ledger["summary"]["inference"]["criterion_met"]
    assert ledger["summary"]["sources"] == 4
    assert ledger["summary"]["status_counts"] == {"missing_root": 4, "new_v2": 4}
    assert ledger["summary"]["work"]["sources"]["completions"] == 4
    assert ledger["summary"]["work"]["audit_source_replay"]["completions"] == 4
    assert ledger["acquisition"] == {
        "retained_before_correction": 0,
        "new_after_correction": 4,
    }
    table = (destination / "frontier_increment_primary_table.tex").read_text(
        encoding="utf-8"
    )
    for parent in (1, 2):
        for opponent in ("WP", "Point"):
            assert f"{parent} & {opponent} & 1 & 0 & 0" in table
        assert f"{parent} & Equal mixture & 2 & 0 & 0" in table
    constants = (destination / "frontier_increment_constants.tex").read_text(
        encoding="utf-8"
    )
    assert r"\FrontierIncrementCriterion}{not met}" in constants
    assert all(sha(path) == digest for path, digest in ledger["outputs"].items())
    with pytest.raises(FileExistsError):
        paper.export(base, destination)


@pytest.mark.parametrize(
    "change",
    ["running", "summary", "report", "receipt", "missing_audit", "acquisition", "pin"],
)
def test_incomplete_or_tampered_evidence_writes_no_paper_files(
    complete_export, monkeypatch, change
):
    base = complete_export
    if change == "running":
        write(base / "queue.json", {"status": "running"})
    elif change == "summary":
        summary = read(base / "summary.json")
        summary["inference"]["criterion_met"] = True
        write(base / "summary.json", summary)
    elif change == "report":
        # Updating the receipt does not make arbitrary report text acceptable.
        (base / "results.rst").write_bytes(b"Altered report\n")
        receipt = read(base / "export.json")
        receipt["report_sha256"] = sha(base / "results.rst")
        write(base / "export.json", receipt)
    elif change == "receipt":
        receipt = read(base / "export.json")
        receipt["criterion_met"] = True
        write(base / "export.json", receipt)
    elif change == "missing_audit":
        (base / "chunk_0000/audit.json").rename(base / "chunk_0000/saved_audit.json")
    elif change == "acquisition":
        monkeypatch.setattr(paper, "RETAINED_SOURCES", 1)
    else:
        monkeypatch.setattr(paper, "PROTOCOL_SHA", "unexpected")
    destination = base / "paper"
    with pytest.raises((ValueError, FileNotFoundError)):
        paper.export(base, destination)
    assert not destination.exists()


def displayed_fraction(text):
    match = re.fullmatch(r"\$(-?[\d.]+)\\times 10\^\{(-?\d+)\}\$", text)
    if match:
        return Fraction(match[1]) * Fraction(10) ** int(match[2])
    return Fraction(text)


@pytest.mark.parametrize(
    "value",
    ["1/100000000000000000000", "1/700000000", "9999999/1000000000", "1/3", "1", "0"],
)
def test_lower_display_never_rounds_up_or_erases_a_positive_bound(value):
    exact = Fraction(value) * 100
    shown = displayed_fraction(paper.pp(value, lower=True))
    assert 0 <= shown <= exact
    if exact:
        assert shown > 0
        assert (exact - shown) / exact < Fraction(1, 100_000)


def test_signed_realized_difference_and_rounding_carry_are_preserved():
    assert paper.pp("-1/3") == "-33.3333"
    assert paper.pp("99999999/1000000000") == "10.0000"
    assert paper.pp("99999999/1000000000", lower=True) == "9.99999"
    assert displayed_fraction(paper.pp("-1/700000000")) < 0
