from copy import deepcopy

import pytest

from scripts.export_symbolic_tail_replay import export
from tests.test_export_certified_tail_replay import rows


def test_export_requires_a_completed_queue_before_writing(tmp_path, monkeypatch):
    from scripts import export_symbolic_tail_replay as module
    from scripts.run_arbitration_study import write

    monkeypatch.setattr(module, "check", lambda _: ({}, {}))
    write(tmp_path / "queue.json", {"status": "running"})
    with pytest.raises(AssertionError):
        export(tmp_path, tmp_path / "report.rst", tmp_path / "receipt.json")
    assert not (tmp_path / "report.rst").exists()


def test_complete_report_keeps_exact_means_and_rejects_changed_records(
    tmp_path, monkeypatch
):
    from scripts import export_symbolic_tail_replay as module
    from scripts.run_arbitration_study import read, sha, write

    source, candidate = tmp_path / "source", tmp_path / "candidate"
    protocol = {"source": str(source), "chunks": list(range(32))}
    monkeypatch.setattr(module, "check", lambda _: (protocol, {}))
    write(candidate / "protocol.json", protocol)
    write(candidate / "queue.json", {"status": "complete"})
    original_source = tmp_path / "original"
    write(source / "protocol.json", {"source": str(original_source)})
    write(original_source / "summary.json", {"development_strength_gate": False})
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
            write(folder / "process.json", {"status": "complete"})
            write(
                folder / "complete.json",
                {
                    "records_sha256": sha(folder / "records.json"),
                    "artifacts": {},
                    "status": "complete",
                    "protocol_sha256": sha(candidate / "protocol.json"),
                },
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
            "opponent_only_actual_mean": [0] * 4,
            "opponent_only_conditional_mean": [0] * 4,
            "symbolic_actual_mean": [-1 / 512, -1 / 512, 0, 0],
            "symbolic_conditional_mean": [1 / 1024, 1 / 1024, 0, 0],
            "costs": {
                v: {a: {} for a in ("controller", "root_only")}
                for v in ("opponent_only", "symbolic")
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
    assert read(receipt)["exact_contrasts"]["symbolic"]["actual"] == [
        "-1/512",
        "-1/512",
        "0",
        "0",
    ]
    assert "original prospective strength gate failed" in report.read_text(
        encoding="utf-8"
    )
    with pytest.raises(ValueError):
        export(candidate, report, receipt)
    summary = read(candidate / "summary.json")
    altered = deepcopy(summary)
    altered["symbolic_actual_mean"][0] = 0
    write(candidate / "summary.json", altered)
    with pytest.raises(AssertionError):
        export(candidate, tmp_path / "bad_mean.rst", tmp_path / "bad_mean.json")
    assert not (tmp_path / "bad_mean.rst").exists()
    write(candidate / "summary.json", summary)
    write(candidate / "chunk_031/records.json", [])
    with pytest.raises(AssertionError):
        export(candidate, tmp_path / "altered.rst", tmp_path / "altered.json")
