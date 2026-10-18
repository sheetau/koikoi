"""Prevent incomplete or capability-only results from entering manuscript evidence."""

import json
from pathlib import Path

import pytest

from scripts.export_seat_observer_paper_evidence import export


def test_requires_explicit_hashes(tmp_path):
    with pytest.raises(ValueError, match="lowercase SHA256"):
        export(tmp_path, "pending", "0" * 64, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_live_study_cannot_export(tmp_path):
    base = Path("runs/observer_selfplay_study")
    state = json.loads((base / "process.json").read_text(encoding="utf-8"))
    # Use the actual live state when available; keep the refusal check meaningful
    # after the research run finishes without changing its artifacts.
    if state["status"] != "complete":
        candidate = base
    else:
        candidate = tmp_path / "unfinished"
        candidate.mkdir()
        (candidate / "process.json").write_text(
            '{"status": "running"}\n', encoding="utf-8", newline="\n"
        )
    with pytest.raises(ValueError, match="not complete"):
        export(candidate, "0" * 64, "0" * 64, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_completed_smoke_is_not_strength_evidence(tmp_path):
    (tmp_path / "process.json").write_text(
        '{"status": "complete"}\n', encoding="utf-8", newline="\n"
    )
    (tmp_path / "protocol.json").write_text(
        '{"kind": "seat_observer_smoke"}\n', encoding="utf-8", newline="\n"
    )
    with pytest.raises(ValueError, match="not strength evidence"):
        export(tmp_path, "0" * 64, "0" * 64, tmp_path / "output")
    assert not (tmp_path / "output").exists()
