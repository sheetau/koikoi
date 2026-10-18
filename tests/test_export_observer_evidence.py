"""Paper export must reject incomplete/smoke evidence and preserve metric units."""

import hashlib
import json

import pytest

from scripts import export_observer_paper_evidence as exporter


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8", newline="\n")


def test_no_export_from_running_study(tmp_path, monkeypatch):
    base = tmp_path / "running"
    base.mkdir()
    write(base / "process.json", {"status": "running"})
    monkeypatch.setattr(exporter, "BASE", base)
    destination = tmp_path / "generated"
    with pytest.raises(ValueError, match="not complete"):
        exporter.export("0" * 64, destination)
    assert not destination.exists()
    with pytest.raises(ValueError, match="SHA256"):
        exporter.export("unpinned", destination)


def test_reduced_pipeline_cannot_be_published_as_full_study(tmp_path, monkeypatch):
    base = tmp_path / "smoke"
    base.mkdir()
    write(base / "process.json", {"status": "complete"})
    write(base / "independent_audit.json", {"status": "passed"})
    write(
        base / "analysis.json", {"status": "passed", "kind": "observer_selfplay_smoke"}
    )
    digest = hashlib.sha256((base / "independent_audit.json").read_bytes()).hexdigest()
    monkeypatch.setattr(exporter, "BASE", base)
    destination = tmp_path / "generated"
    with pytest.raises(AssertionError):
        exporter.export(digest, destination)
    assert not destination.exists()


def test_win_percentage_points_and_raw_point_margin_are_distinct():
    # A hand-chosen mapping tests display units; these are not study outcomes.
    contrasts = {
        f"parent{i}_observer-parent{i}{suffix}/{opponent}": {
            "win_change": 0.01234,
            "win_ci95": [-0.005, 0.025],
            "point_change": -0.375,
            "point_ci95": [-0.875, 0.125],
        }
        for i in (1, 2)
        for suffix in ("", "_signed")
        for opponent, _ in exporter.OPPONENTS
    }
    comparisons = (("observer", ""), ("observer", "_signed"))
    headers = ("Observer minus parent", "Observer minus signed")
    wins = exporter.comparison_table(contrasts, comparisons, headers)
    points = exporter.comparison_table(contrasts, comparisons, headers, metric="point")
    assert wins.count("$+1.234$ [-0.500, 2.500]") == 8
    assert points.count("$-0.375$ [-0.875, 0.125]") == 8
    assert "-37.500" not in points
    assert "Parent & Opponent" in wins and "2 & Point" in wins
    with pytest.raises(ValueError):
        exporter.comparison_table(contrasts, comparisons, headers, metric="score")
