"""Synthetic export fixtures,not auxiliary study outcomes."""

import copy
import json

import pytest

from scripts import export_score_auxiliary_results as exporter


def write(path, data):
    path.write_bytes((json.dumps(data, indent=2) + "\n").encode("utf-8"))


def analysis():
    return {
        "mode": "main",
        "compound_blocks": 4096,
        "distinct_initial_deals": 8192,
        "full_matches": 131072,
        "bootstrap_seed": 109310001,
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
            "seed": 109310001,
            "family_size": 6,
            "critical": 2,
            "nondegenerate": 6,
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
    for path in (root, queue, training, root / "evaluation"):
        path.mkdir()
    monkeypatch.setattr(exporter, "TRAINING", training)
    write(training / "protocol.json", {"synthetic_fixture": True})
    training_sha = exporter.sha(training / "protocol.json")
    monkeypatch.setattr(exporter, "TRAINING_SHA", training_sha)
    dependencies = [training / "protocol.json"]
    arms, models, audited = {}, {}, {}
    for arm in exporter.ARMS:
        folder = training / arm / "update_00000520"
        folder.mkdir(parents=True)
        model = folder / "ppo.pt"
        model.write_bytes(arm.encode())
        write(folder / "manifest.json", {"files": {"ppo.pt": exporter.sha(model)}})
        dependencies.extend((model, folder / "manifest.json"))
        models[arm] = model
        ppo = {
            "updates": 520,
            "decisions": 8519680,
            "rounds": 550000,
            "matches": 71000,
            "warmup_updates": 20,
            "policy_updates": 500,
            "warmup_decisions": 327680,
        }
        aux = {
            "cycles": 100,
            "decisions": 1500000,
            "rounds": 99000,
            "matches": 12800,
            "retained_rows": 96000,
            "example_presentations": 0 if arm.endswith("none") else 576000,
        }
        arms[arm] = {
            "status": "complete",
            "protocol_sha256": training_sha,
            "endpoint": str(folder),
            "manifest_sha256": exporter.sha(folder / "manifest.json"),
            "ppo_work": ppo,
            "auxiliary_work": aux,
        }
        audited[arm] = {"endpoint_manifest_sha256": arms[arm]["manifest_sha256"]}
    totals = {
        group: {
            key: sum(arm[group][key] for arm in arms.values())
            for key in ("decisions", "rounds", "matches")
        }
        for group in ("ppo_work", "auxiliary_work")
    }
    write(
        training / "complete.json",
        {
            "status": "complete",
            "protocol_sha256": training_sha,
            "arms": arms,
            "totals": totals,
        },
    )
    dependencies.append(training / "complete.json")
    write(
        training / "training_audit.json",
        {
            "format": "score_auxiliary_training_audit_v1",
            "status": "passed",
            "protocol_sha256": training_sha,
            "complete_sha256": exporter.sha(training / "complete.json"),
            "arms": audited,
            "totals": totals,
            "inputs": {str(p): exporter.sha(p) for p in dependencies},
        },
    )
    dependencies.append(training / "training_audit.json")
    for seed in (1, 2):
        model = training / f"parent{seed}.pt"
        model.write_bytes(b"fixture parent")
        models[f"parent{seed}"] = model
        dependencies.append(model)
    cells = {}
    for seed in (1, 2):
        for policy in exporter.POLICIES:
            model = models[
                f"parent{seed}" if policy == "parent" else f"seed{seed}_{policy}"
            ]
            for opponent in ("wp", "point"):
                cells[f"parent{seed}_{policy}_{opponent}"] = {
                    "parent": seed,
                    "policy": policy,
                    "opponent": opponent,
                    "checkpoint": str(model),
                    "first_seed": 109110001 if opponent == "wp" else 109210001,
                }
    write(
        root / "evaluation_protocol.json",
        {
            "mode": "main",
            "pairs": 4096,
            "cells": cells,
            "inputs": {str(p): exporter.sha(p) for p in dependencies},
            "bootstrap": {"seed": 109310001, "replicates": 10000},
            "main_training_protocol_sha256": training_sha,
        },
    )
    protocol_sha = exporter.sha(root / "evaluation_protocol.json")
    dependencies.append(root / "evaluation_protocol.json")
    write(root / "summary.json", analysis() | {"protocol_sha256": protocol_sha})
    dependencies.append(root / "summary.json")
    work = {"decisions": 900000, "round_endings": 62000, "full_matches": 8192}
    for key, cell in cells.items():
        rows = root / "evaluation" / f"{key}.jsonl"
        report = root / "evaluation" / f"{key}.json"
        write(rows, {"fixture": key})
        write(
            report,
            {
                "status": "complete",
                "cell": key,
                **cell,
                "protocol_sha256": protocol_sha,
                "rows_sha256": exporter.sha(rows),
                "checkpoint_sha256": exporter.sha(cell["checkpoint"]),
                "work": work,
            },
        )
        dependencies.extend((rows, report))
    write(
        root / "audit.json",
        {
            "status": "passed",
            "qualified": False,
            "protocol_sha256": protocol_sha,
            "inputs": {str(p): exporter.sha(p) for p in dependencies},
            "work": {k: 16 * v for k, v in work.items()},
        },
    )
    digest = exporter.sha(root / "audit.json")
    write(
        queue / "protocol.json",
        {"inputs": {str(training / "protocol.json"): training_sha}},
    )
    queue_sha = exporter.sha(queue / "protocol.json")
    monkeypatch.setattr(exporter, "QUEUE_SHA", queue_sha)
    write(
        queue / "complete.json",
        {
            "status": "complete",
            "protocol_sha256": queue_sha,
            "completed": ["training_audit", "prepare", "evaluate", "evaluation_audit"],
            "training_audit_sha256": exporter.sha(training / "training_audit.json"),
            "evaluation_audit_sha256": digest,
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


def test_full_export_preserves_six_primary_all120_rows_and_actual_streams(
    tmp_path, monkeypatch
):
    root, queue, digest = completed(tmp_path, monkeypatch)
    output = tmp_path / "export"
    evidence = exporter.export(root, queue, output, digest)
    assert evidence["summary"] == exporter.read(root / "summary.json")
    assert len(evidence["summary"]["rows"]) == 120
    text = (output / "results.rst").read_text(encoding="utf-8")
    assert "criterion fails" in text and "+2.0000 [-2.0000, +6.0000]" in text
    assert all(label in text for label in exporter.PRIMARY)
    assert "51118080" in text and "76800" in text
    assert "auxiliary_work" in text and "ppo_work" in text
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
        "changed_training_audit",
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
        write(root / "evaluation_failed.json", {"error": "fixture"})
    elif fault == "missing_cell":
        value = exporter.read(root / "evaluation_complete.json")
        value["completed"].pop()
        write(root / "evaluation_complete.json", value)
    elif fault == "changed_rows":
        next((root / "evaluation").glob("*.jsonl")).write_bytes(b"changed")
    elif fault == "changed_training_audit":
        (exporter.TRAINING / "training_audit.json").write_bytes(b"changed")
    elif fault == "wrong_audit":
        digest = "0" * 64
    else:
        write(queue / "protocol.json", {"changed": True})
    output = tmp_path / "refused"
    with pytest.raises(ValueError):
        exporter.export(root, queue, output, digest)
    assert not output.exists()


def test_missing_outcome_only_gain_and_degeneracy_cannot_qualify():
    value = analysis()
    for row in value["primary"]["rows"].values():
        row.update(standard_error=0.001, simultaneous95=[0.018, 0.022], sign=1)
    value["qualified"] = True
    exporter.validate_analysis(value)
    row = value["primary"]["rows"]["seed2/mixture/score-win/win"]
    row.update(standard_error=0.02, simultaneous95=[-0.02, 0.06], sign=0)
    with pytest.raises(ValueError, match="Qualification"):
        exporter.validate_analysis(value)
    value["qualified"] = False
    exporter.validate_analysis(value)
    row.update(standard_error=0, simultaneous95=[None, None], sign=0)
    value["primary"]["nondegenerate"] = 5
    assert "undefined" in exporter.render(value, {}, {"arms": {}, "totals": {}})


def test_missing_coordinates_nonfinite_and_capability_rejected():
    for fault in ("secondary", "primary", "nonfinite", "capability"):
        value = copy.deepcopy(analysis())
        if fault == "secondary":
            value["rows"].pop(next(iter(value["rows"])))
        elif fault == "primary":
            value["primary"]["rows"].pop(next(iter(value["primary"]["rows"])))
        elif fault == "nonfinite":
            next(iter(value["rows"].values()))["mean"] = float("nan")
        else:
            value["mode"] = "capability"
        with pytest.raises(ValueError):
            exporter.validate_analysis(value)
