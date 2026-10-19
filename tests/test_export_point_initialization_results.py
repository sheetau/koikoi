"""Synthetic presentation fixtures only; none of these values are study results."""

import copy
import json

import pytest

from scripts import export_point_initialization_results as exporter


def write(path, value):
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode("utf-8"))


def analysis():
    return {
        "mode": "main",
        "compound_blocks": 4096,
        "distinct_initial_deals": 8192,
        "full_matches": 131072,
        "bootstrap_seed": 104110001,
        "bootstrap_replicates": 10000,
        "rows": {
            label: {
                "mean": 0.02 if "-" in label else 0.5,
                "marginal95": [-0.01, 0.05] if "-" in label else [0.48, 0.52],
            }
            for label in exporter.LABELS
        },
        "qualified": False,
        "primary": {
            "deals": 4096,
            "replicates": 10000,
            "seed": 104110001,
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


def completed(tmp_path, monkeypatch):
    root, queue, training = [tmp_path / name for name in ("study", "queue", "training")]
    for folder in (root, queue, training):
        folder.mkdir()
    (root / "evaluation").mkdir()
    monkeypatch.setattr(exporter, "TRAINING", training)
    write(training / "protocol.json", {"synthetic_fixture": True})
    train_sha = exporter.sha(training / "protocol.json")
    monkeypatch.setattr(exporter, "TRAINING_SHA", train_sha)
    dependencies = [training / "protocol.json"]
    arms, audits, models = {}, {}, {}
    for arm in exporter.ARMS:
        model = training / f"{arm}.pt"
        model.write_bytes(arm.encode())
        models[arm] = model
        dependencies.append(model)
        counts = {
            "updates": 520,
            "decisions": 8519680,
            "rounds": 570000,
            "matches": 73000,
            "warmup_updates": 20,
            "policy_updates": 500,
            "warmup_decisions": 327680,
        }
        arms[arm] = {
            "status": "complete",
            "protocol_sha256": train_sha,
            "checkpoint_sha256": exporter.sha(model),
            "counters": counts,
        }
        for stage in (20, 520):
            audits[f"{arm}/{stage}"] = {
                "status": "passed",
                "arm": arm,
                "update": stage,
                "protocol_sha256": train_sha,
                "checkpoint": {"sha256": exporter.sha(model), "counters": counts},
            }
    training_work = {
        k: sum(a["counters"][k] for a in arms.values())
        for k in ("decisions", "rounds", "matches", "updates")
    }
    write(
        training / "complete.json",
        {
            "status": "complete",
            "protocol_sha256": train_sha,
            "arms": arms,
            "total_training_work": training_work,
        },
    )
    dependencies.append(training / "complete.json")
    for parent in (1, 2):
        p = training / f"parent{parent}.pt"
        p.write_bytes(b"fixture parent")
        models[f"parent{parent}"] = p
        dependencies.append(p)
    cells = {}
    for parent in (1, 2):
        for policy in exporter.POLICIES:
            model = models[
                f"parent{parent}" if policy == "parent" else f"seed{parent}_{policy}"
            ]
            for opponent in ("wp", "point"):
                cells[f"parent{parent}_{policy}_{opponent}"] = {
                    "parent": parent,
                    "policy": policy,
                    "opponent": opponent,
                    "checkpoint": str(model),
                    "first_seed": 103110001 if opponent == "wp" else 103210001,
                }
    protocol_path = root / "evaluation_protocol.json"
    write(
        protocol_path,
        {
            "mode": "main",
            "pairs": 4096,
            "cells": cells,
            "inputs": {str(p): exporter.sha(p) for p in dependencies},
            "training_audits": audits,
            "bootstrap": {"seed": 104110001, "replicates": 10000},
            "main_training_protocol_sha256": train_sha,
        },
    )
    protocol_sha = exporter.sha(protocol_path)
    dependencies.append(protocol_path)
    summary = analysis() | {"protocol_sha256": protocol_sha}
    write(root / "summary.json", summary)
    dependencies.append(root / "summary.json")
    cell_work = {"decisions": 900000, "round_endings": 62000, "full_matches": 8192}
    for key, cell in cells.items():
        rows = root / "evaluation" / f"{key}.jsonl"
        write(rows, {"synthetic_fixture": key})
        report = root / "evaluation" / f"{key}.json"
        write(
            report,
            {
                "status": "complete",
                "cell": key,
                **cell,
                "protocol_sha256": protocol_sha,
                "rows_sha256": exporter.sha(rows),
                "checkpoint_sha256": exporter.sha(cell["checkpoint"]),
                "work": cell_work,
            },
        )
        dependencies.extend((report, rows))
    write(
        root / "audit.json",
        {
            "status": "passed",
            "qualified": False,
            "protocol_sha256": protocol_sha,
            "inputs": {str(p): exporter.sha(p) for p in dependencies},
            "work": {k: 16 * v for k, v in cell_work.items()},
        },
    )
    digest = exporter.sha(root / "audit.json")
    write(
        queue / "protocol.json",
        {"inputs": {str(training / "protocol.json"): train_sha}},
    )
    queue_sha = exporter.sha(queue / "protocol.json")
    monkeypatch.setattr(exporter, "QUEUE_SHA", queue_sha)
    write(
        queue / "complete.json",
        {
            "status": "complete",
            "completed": ["prepare", "evaluate", "audit"],
            "protocol_sha256": queue_sha,
            "audit_sha256": digest,
            "qualified": False,
            "deployment_changed": False,
        },
    )
    write(
        root / "evaluation_complete.json",
        {
            "status": "complete_awaiting_audit",
            "completed": sorted(cells),
            "protocol_sha256": protocol_sha,
        },
    )
    return root, queue, digest


def test_complete_export_units_all_rows_and_non_overwrite(tmp_path, monkeypatch):
    root, queue, digest = completed(tmp_path, monkeypatch)
    output = tmp_path / "export"
    evidence = exporter.export(root, queue, output, digest)
    assert evidence["summary"] == exporter.read(root / "summary.json")
    assert len(evidence["summary"]["rows"]) == 120
    text = (output / "results.rst").read_text(encoding="utf-8")
    assert "criterion fails" in text and "131,072 matches" in text
    assert "+2.0000 [-2.0000, +6.0000]" in text
    assert "51118080" in text and "not exploration alone" in text
    assert all(label in text for label in exporter.PRIMARY)
    assert evidence["new_matches"] == evidence["new_neural_calls"] == 0
    with pytest.raises(ValueError, match="Preserve existing"):
        exporter.export(root, queue, output, digest)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_finish",
        "failure",
        "missing_cell",
        "changed_rows",
        "changed_training",
        "wrong_audit",
        "wrong_queue",
    ],
)
def test_incomplete_or_changed_evidence_refused_before_output(
    tmp_path, monkeypatch, fault
):
    root, queue, digest = completed(tmp_path, monkeypatch)
    if fault == "missing_finish":
        (queue / "complete.json").unlink()
    elif fault == "failure":
        write(root / "evaluation_failed.json", {"error": "failure"})
    elif fault == "missing_cell":
        p = root / "evaluation_complete.json"
        value = exporter.read(p)
        value["completed"].pop()
        write(p, value)
    elif fault == "changed_rows":
        next((root / "evaluation").glob("*.jsonl")).write_bytes(b"changed")
    elif fault == "changed_training":
        (exporter.TRAINING / "complete.json").write_bytes(b"changed")
    elif fault == "wrong_audit":
        digest = "0" * 64
    else:
        write(queue / "protocol.json", {"inputs": {}, "changed": True})
    output = tmp_path / "refused"
    with pytest.raises(ValueError):
        exporter.export(root, queue, output, digest)
    assert not output.exists()


def test_all_primary_bands_and_zero_variance_rules():
    summary = analysis()
    for row in summary["primary"]["rows"].values():
        row.update(standard_error=0.001, simultaneous95=[0.018, 0.022], sign=1)
    summary["qualified"] = True
    exporter.validate_analysis(summary)
    first = next(iter(summary["primary"]["rows"].values()))
    first.update(standard_error=0, simultaneous95=[None, None], sign=0)
    summary["primary"]["nondegenerate"] = 3
    with pytest.raises(ValueError, match="Qualification"):
        exporter.validate_analysis(summary)
    summary["qualified"] = False
    exporter.validate_analysis(summary)
    assert "undefined" in exporter.render(summary, {}, {})


def test_omissions_nonfinite_and_capability_are_rejected():
    for case in ("secondary", "primary", "nonfinite", "capability"):
        summary = copy.deepcopy(analysis())
        if case == "secondary":
            summary["rows"].pop(next(iter(summary["rows"])))
        elif case == "primary":
            summary["primary"]["rows"].pop(next(iter(summary["primary"]["rows"])))
        elif case == "nonfinite":
            next(iter(summary["rows"].values()))["mean"] = float("nan")
        else:
            summary["mode"] = "capability"
        with pytest.raises(ValueError):
            exporter.validate_analysis(summary)
