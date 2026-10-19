"""Synthetic presentation tests: these fixtures are not strength evidence."""

import copy
import json

import pytest

from scripts import export_published_duration_results as exporter


def analysis():
    rows = {
        label: {
            "mean": 0.02 if "-" in label else 0.5,
            "marginal95": [-0.01, 0.05] if "-" in label else [0.48, 0.52],
        }
        for label in exporter.LABELS
    }
    return {
        "mode": "main",
        "compound_blocks": 4096,
        "distinct_initial_deals": 8192,
        "full_matches": 98304,
        "bootstrap_seed": 99410001,
        "bootstrap_replicates": 10000,
        "rows": rows,
        "qualified": False,
        "primary": {
            "deals": 4096,
            "replicates": 10000,
            "seed": 99410001,
            "family_size": 4,
            "critical": 2,
            "nondegenerate": 4,
            "rows": {
                label: {
                    "mean": 0.02,
                    "standard_error": 0.02,
                    "simultaneous95": [-0.02, 0.06],
                    "sign": 0,
                }
                for label in exporter.PRIMARY
            },
        },
    }


def write(path, value):
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode("utf-8"))


def completed(tmp_path, monkeypatch):
    root, queue = tmp_path / "study", tmp_path / "queue"
    root.mkdir()
    queue.mkdir()
    (root / "evaluation").mkdir()
    protocol = root / "evaluation_protocol.json"
    write(protocol, {"synthetic_fixture": True})
    digest = exporter.sha(protocol)
    monkeypatch.setattr(exporter, "PROTOCOL_SHA", digest)
    training = tmp_path / "training.json"
    write(
        training,
        {
            "status": "complete",
            "cases": {
                "fixture": {
                    "counters": {
                        "updates": 8040,
                        "decisions": 131727360,
                        "rounds": 8804257,
                        "matches": 1133134,
                    }
                }
            },
        },
    )
    monkeypatch.setattr(exporter, "TRAINING", training)
    monkeypatch.setattr(exporter, "TRAINING_SHA", exporter.sha(training))
    summary = analysis()
    summary["protocol_sha256"] = digest
    write(root / "summary.json", summary)
    dependencies = [protocol, root / "summary.json", training]
    for cell in exporter.CELLS:
        for suffix in (".json", ".jsonl"):
            path = root / "evaluation" / (cell + suffix)
            write(path, {"synthetic_fixture": cell})
            dependencies.append(path)
    write(
        root / "audit.json",
        {
            "status": "passed",
            "protocol_sha256": digest,
            "qualified": False,
            "work": {"full_matches": 98304},
            "inputs": {str(path): exporter.sha(path) for path in dependencies},
        },
    )
    audit_sha = exporter.sha(root / "audit.json")
    write(
        queue / "process.json",
        {
            "status": "complete",
            "completed": ["prepare", "evaluate", "audit"],
            "audit_sha256": audit_sha,
            "qualified": False,
            "deployment_changed": False,
        },
    )
    write(
        root / "evaluation_process.json",
        {"status": "complete_awaiting_audit", "completed": sorted(exporter.CELLS)},
    )
    return root, queue, audit_sha


def test_complete_export_and_dependency_tamper(tmp_path, monkeypatch):
    root, queue, digest = completed(tmp_path, monkeypatch)
    output = tmp_path / "export"
    evidence = exporter.export(root, queue, output, digest)
    assert evidence["summary"] == exporter.read(root / "summary.json")
    assert len(evidence["summary"]["rows"]) == 72
    rendered = (output / "results.rst").read_text(encoding="utf-8")
    assert "criterion fails" in rendered and "simultaneous 95%" in rendered
    assert "131727360" in rendered and "98,304 matches" in rendered
    for label in exporter.PRIMARY:
        assert label in rendered
    with pytest.raises(ValueError, match="Preserve existing"):
        exporter.export(root, queue, output, digest)
    (root / "evaluation" / "parent1_long_wp.jsonl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Missing audit binding"):
        exporter.export(root, queue, tmp_path / "tampered", digest)
    assert not (tmp_path / "tampered").exists()


def test_omission_nonfinite_and_capability_rejected():
    original = analysis()
    exporter.validate_analysis(original)
    for field in ("rows", "primary"):
        value = copy.deepcopy(original)
        rows = value[field] if field == "rows" else value[field]["rows"]
        rows.pop(next(iter(rows)))
        with pytest.raises(ValueError, match="Require all"):
            exporter.validate_analysis(value)
    value = copy.deepcopy(original)
    value["rows"][next(iter(exporter.LABELS))]["mean"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        exporter.validate_analysis(value)
    value = copy.deepcopy(original)
    value["mode"] = "capability"
    with pytest.raises(ValueError, match="Capability"):
        exporter.validate_analysis(value)


def test_gate_uses_all_four_unrounded_bands_and_zero_se():
    value = analysis()
    for label, row in value["primary"]["rows"].items():
        row.update(
            mean=0.02,
            standard_error=0.009999999999,
            simultaneous95=[0.02 - 2 * 0.009999999999, 0.02 + 2 * 0.009999999999],
            sign=1,
        )
    value["qualified"] = True
    exporter.validate_analysis(value)
    assert "criterion passes" in exporter.render(value, {}, {})
    label = next(label for label in exporter.PRIMARY if label.startswith("parent2"))
    row = value["primary"]["rows"][label]
    row.update(
        standard_error=0.010000000001,
        simultaneous95=[0.02 - 2 * 0.010000000001, 0.02 + 2 * 0.010000000001],
        sign=0,
    )
    with pytest.raises(ValueError, match="Qualification"):
        exporter.validate_analysis(value)
    value["qualified"] = False
    exporter.validate_analysis(value)
    row.update(standard_error=0, simultaneous95=[None, None], sign=0)
    value["primary"]["nondegenerate"] = 3
    exporter.validate_analysis(value)
    assert "[undefined]" in exporter.render(value, {}, {})
    row["sign"] = 1
    with pytest.raises(ValueError, match="Degenerate"):
        exporter.validate_analysis(value)


def test_incomplete_producer_or_queue_never_creates_export(tmp_path, monkeypatch):
    root, queue, digest = completed(tmp_path, monkeypatch)
    process = exporter.read(root / "evaluation_process.json")
    process["completed"].pop()
    write(root / "evaluation_process.json", process)
    with pytest.raises(ValueError, match="twelve"):
        exporter.export(root, queue, tmp_path / "output", digest)
    write(queue / "process.json", {"status": "evaluate"})
    with pytest.raises(ValueError, match="completion queue"):
        exporter.export(root, queue, tmp_path / "output", digest)
    assert not (tmp_path / "output").exists()
