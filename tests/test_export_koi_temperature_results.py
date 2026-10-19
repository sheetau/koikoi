import copy
import csv
import io
import json

import pytest

from scripts import export_koi_temperature_results as exporter


def analysis_fixture():
    # Synthetic presentation fixture: these are not actual match outcomes.
    absolute, contrasts = [], []
    for seed in (1, 2):
        for scope in ("mixture", "wp", "point"):
            for policy in ("parent", "fixed", "calibrated"):
                absolute.append(
                    {
                        "parent": seed,
                        "policy": policy,
                        "scope": scope,
                        "win": 0.51,
                        "points": -2.5,
                        "win_interval": [0.49, 0.53],
                        "point_interval": [-3.0, -2.0],
                    }
                )
            for left, right in (
                ("calibrated", "fixed"),
                ("calibrated", "parent"),
                ("fixed", "parent"),
            ):
                contrasts.append(
                    {
                        "parent": seed,
                        "left": left,
                        "right": right,
                        "scope": scope,
                        "win_change": 0.0025,
                        "point_change": -0.5,
                        "win_interval": [-1e-10, 0.01],
                        "point_interval": [-1.0, 0.1],
                    }
                )
    return {
        "absolute": absolute,
        "contrasts": contrasts,
        "replicated_strength_gain": False,
        "bootstrap_seed": 97310001,
        "bootstrap_replicates": 10000,
    }


def write(path, value):
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode("utf-8"))


def completed_fixture(tmp_path, monkeypatch):
    root = tmp_path / "fixture_study"
    root.mkdir()
    write(root / "protocol.json", {"fixture": True})
    monkeypatch.setattr(exporter, "PROTOCOL_SHA", exporter.sha(root / "protocol.json"))
    write(root / "process.json", {"status": "complete_awaiting_audit"})
    write(root / "finish_process.json", {"status": "complete"})
    dependency = root / "fixture_dependency.txt"
    dependency.write_bytes(b"Synthetic dependency, not a neural model.\n")
    training = {
        f"seed{seed}_{mode}": {
            "updates": 520,
            "decisions": 8519680,
            "rounds": 100,
            "matches": 10,
        }
        for seed in (1, 2)
        for mode in ("fixed", "calibrated")
    }
    work = {"full_matches": 49152, "decisions": 491520, "round_endings": 98304}
    audit = {
        "status": "passed",
        "analysis": analysis_fixture(),
        "training_counters": training,
        "evaluation_work": work,
        "evaluation_replay_work": work,
        "shared_evaluation_deals": 4096,
        "inputs": {str(dependency): exporter.sha(dependency)},
    }
    write(root / "independent_audit.json", audit)
    digest = exporter.sha(root / "independent_audit.json")
    summary = copy.deepcopy(audit)
    summary.update(
        {
            "status": "complete",
            "deployment_changed": False,
            "training_work": {
                "updates": 2080,
                "decisions": 34078720,
                "rounds": 400,
                "matches": 40,
            },
        }
    )
    summary["inputs"][str(root / "independent_audit.json")] = digest
    write(root / "summary.json", summary)
    return root, digest, dependency


def test_all_rows_and_unrounded_gate_are_required():
    analysis = analysis_fixture()
    assert len(exporter.validate_analysis(analysis)) == 4
    for group in ("absolute", "contrasts"):
        changed = copy.deepcopy(analysis)
        changed[group][-1] = copy.deepcopy(changed[group][0])
        with pytest.raises(ValueError, match="missing, duplicated"):
            exporter.validate_analysis(changed)
    changed = copy.deepcopy(analysis)
    changed["replicated_strength_gain"] = True
    with pytest.raises(ValueError, match="unrounded intervals"):
        exporter.validate_analysis(changed)
    for row in analysis["contrasts"]:
        if row["scope"] == "mixture" and row["left"] == "calibrated":
            row["win_interval"][0] = 1e-10
    analysis["replicated_strength_gain"] = True
    assert len(exporter.validate_analysis(analysis)) == 4
    analysis["absolute"][0]["points"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        exporter.validate_analysis(analysis)


def test_completed_export_preserves_all_values_units_and_outputs(tmp_path, monkeypatch):
    root, digest, _ = completed_fixture(tmp_path, monkeypatch)
    output = tmp_path / "export"
    receipt = exporter.export(root, output, digest)
    assert receipt["absolute_rows"] == receipt["contrast_rows"] == 18
    assert receipt["primary_rows"] == 4
    for group in ("absolute", "contrasts"):
        source = analysis_fixture()[group]
        rows = list(
            csv.DictReader(
                io.StringIO((output / (group + ".csv")).read_text(encoding="utf-8"))
            )
        )
        assert len(rows) == 18
        for exported, original in zip(rows, source):
            for key in (
                ("win", "points")
                if group == "absolute"
                else ("win_change", "point_change")
            ):
                assert float(exported[key]) == original[key]
            for metric in ("win", "point"):
                assert [
                    float(exported[metric + "_lower"]),
                    float(exported[metric + "_upper"]),
                ] == original[metric + "_interval"]
    report = (output / "results.rst").read_text(encoding="utf-8")
    report_lines = report.splitlines()
    assert len(report_lines[0]) == len(report_lines[1])
    assert "criterion fails" in report and "+0.2500 [-0.0000, +1.0000]" in report
    assert "51.0000 [49.0000, 53.0000]" in report
    assert "divide by two" in report and "not simultaneous" in report
    assert all(
        exporter.sha(path) == value for path, value in receipt["outputs"].items()
    )
    for path in output.iterdir():
        assert not path.read_bytes().startswith(b"\xef\xbb\xbf")
        assert b"\r\n" not in path.read_bytes()
    with pytest.raises(ValueError, match="Preserve existing"):
        exporter.export(root, output, digest)


def test_incomplete_audit_or_changed_dependencies_never_create_export(
    tmp_path, monkeypatch
):
    root, digest, dependency = completed_fixture(tmp_path, monkeypatch)
    output = tmp_path / "must_not_exist"
    write(root / "finish_process.json", {"status": "auditing"})
    with pytest.raises(ValueError, match="finalizer"):
        exporter.export(root, output, digest)
    assert not output.exists()
    write(root / "finish_process.json", {"status": "complete"})
    with pytest.raises(ValueError, match="identity mismatch"):
        exporter.export(root, output, "0" * 64)
    dependency.write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="Changed dependency"):
        exporter.export(root, output, digest)
    assert not output.exists()


def test_summary_disagreement_is_rejected_before_output(tmp_path, monkeypatch):
    root, digest, _ = completed_fixture(tmp_path, monkeypatch)
    summary = exporter.read(root / "summary.json")
    summary["analysis"]["absolute"][0]["win"] = 0.99
    write(root / "summary.json", summary)
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="disagrees with audit: analysis"):
        exporter.export(root, output, digest)
    assert not output.exists()
