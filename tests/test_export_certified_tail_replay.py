from copy import deepcopy
from fractions import Fraction

import pytest

from koikoi.layout import POINTS
from scripts.export_certified_tail_replay import aggregate, compare_coverage, export
from scripts.prospective_incumbent import cases


def rows():
    result = []
    for index, case in enumerate(cases()):
        final = [30] * 240
        final[POINTS + case["seat"]] = 40
        result.append(
            {
                "index": index,
                "status": "missing_root",
                "source": {**case, "final": final},
            }
        )
    return result


def test_conditional_and_actual_means_include_absent_roots_and_draws():
    data = rows()
    data[0].update(
        status="evaluated",
        arms={
            "controller": {
                "selection": {"gain": "1/4", "action": 1, "parent_q": "1/2"}
            },
            "root_only": {
                "selection": {"gain": "0", "action": None, "parent_q": "1/2"}
            },
        },
        assessment={
            "outcomes": {"controller": {"margin": 0}, "root_only": {"margin": 10}}
        },
    )
    result = aggregate(data)
    assert result[0, "controller"] == {
        "gain": Fraction(1, 1024),
        "actual": Fraction(-1, 512),
        "adoptions": 1,
    }
    assert result[1, "controller"]["gain"] == 0
    assert result[0, "root_only"]["actual"] == 0
    with pytest.raises(AssertionError):
        aggregate(data[:-1])


def test_coverage_distinguishes_parent_recovery_and_new_adoption():
    old = rows()
    old[0].update(
        status="evaluated",
        arms={
            arm: {"selection": {"gain": "0", "action": None, "parent_q": None}}
            for arm in ("controller", "root_only")
        },
    )
    new = deepcopy(old)
    new[0]["arms"]["controller"]["selection"].update(
        gain="1/8", action=4, parent_q="1/2"
    )
    changes = compare_coverage(old, new)
    assert changes["controller"]["new_parent_values"] == 1
    assert changes["controller"]["new_adoptions"] == 1
    assert changes["controller"]["greater_gain"] == 1
    assert changes["root_only"]["new_parent_values"] == 0
    new[0]["source"]["seat"] = 1
    with pytest.raises(AssertionError):
        compare_coverage(old, new)


def test_export_requires_a_completed_queue_before_writing(tmp_path, monkeypatch):
    from scripts import export_certified_tail_replay as module
    from scripts.run_arbitration_study import write

    monkeypatch.setattr(module, "check", lambda _: ({}, {}))
    write(tmp_path / "queue.json", {"status": "running"})
    with pytest.raises(AssertionError):
        export(tmp_path, tmp_path / "report.rst", tmp_path / "receipt.json")
    assert not (tmp_path / "report.rst").exists()


def test_complete_report_keeps_exact_means_and_rejects_changed_records(
    tmp_path, monkeypatch
):
    from scripts import export_certified_tail_replay as module
    from scripts.run_arbitration_study import read, sha, write

    source, candidate = tmp_path / "source", tmp_path / "candidate"
    protocol = {"source": str(source), "chunks": list(range(32))}
    monkeypatch.setattr(module, "check", lambda _: (protocol, {}))
    write(candidate / "protocol.json", protocol)
    write(candidate / "queue.json", {"status": "complete"})
    write(source / "summary.json", {"development_strength_gate": False})
    old = rows()
    old[0].update(
        status="evaluated",
        arms={
            arm: {"selection": {"gain": "0", "action": None, "parent_q": None}}
            for arm in ("controller", "root_only")
        },
        assessment={
            "outcomes": {"controller": {"margin": 10}, "root_only": {"margin": 10}}
        },
    )
    new = deepcopy(old)
    new[0]["arms"]["controller"]["selection"].update(
        gain="1/4", action=1, parent_q="1/2"
    )
    new[0]["assessment"]["outcomes"]["controller"]["margin"] = 0
    bindings = {}
    for chunk in range(32):
        for directory, data in ((source, old), (candidate, new)):
            folder = directory / f"chunk_{chunk:03}"
            write(folder / "records.json", data[16 * chunk : 16 * (chunk + 1)])
            write(
                folder / "complete.json",
                {"records_sha256": sha(folder / "records.json"), "artifacts": {}},
            )
            if directory == candidate:
                bindings[str(folder / "complete.json")] = sha(folder / "complete.json")
    write(
        candidate / "summary.json",
        {
            "status": "passed",
            "protocol_sha256": sha(candidate / "protocol.json"),
            "new_source_matches": 0,
            "reused_sources": 512,
            "chunk_bindings": bindings,
            "original_actual_mean": [0] * 4,
            "original_conditional_mean": [0] * 4,
            "certified_actual_mean": [-1 / 512, -1 / 512, 0, 0],
            "certified_conditional_mean": [1 / 1024, 1 / 1024, 0, 0],
            "costs": {
                v: {a: {} for a in ("controller", "root_only")}
                for v in ("original", "certified")
            },
            "work": {},
        },
    )
    report, receipt = tmp_path / "report.rst", tmp_path / "receipt.json"
    export(candidate, report, receipt)
    assert "1/1024" in report.read_text(encoding="utf-8")
    assert "-1/512" in report.read_text(encoding="utf-8")
    assert read(receipt)["status"] == "passed"
    assert read(receipt)["new_simulator_work"] == 0
    assert read(receipt)["coverage"]["controller"]["new_adoptions"] == 1
    write(candidate / "chunk_031/records.json", [])
    with pytest.raises(AssertionError):
        export(candidate, tmp_path / "altered.rst", tmp_path / "altered.json")
