"""Result reuse never turns stale inputs or failures into passing evidence."""

import json
import sys

import pytest

from scripts import check_game as checks


@pytest.fixture
def runner(tmp_path, monkeypatch):
    source = tmp_path / "input.txt"
    source.write_text("original", encoding="utf-8")
    counter = tmp_path / "executions.txt"
    monkeypatch.setattr(checks, "ROOT", tmp_path)
    monkeypatch.setattr(checks, "OUTPUT", tmp_path / "results")
    monkeypatch.setattr(checks, "inputs", lambda _: [source])
    monkeypatch.setattr(checks, "SUITES", {"quick": ["sample"], "full": ["match"]})
    code = "from pathlib import Path; p=Path('executions.txt'); p.write_text(p.read_text()+'x' if p.exists() else 'x')"
    monkeypatch.setattr(checks, "command", lambda _: [sys.executable, "-c", code])
    return source, counter


def test_identical_inputs_reuse_and_changed_inputs_execute(runner, monkeypatch, capsys):
    source, counter = runner
    monkeypatch.setattr(sys, "argv", ["check_game", "--reuse"])
    assert checks.main() == 0
    assert checks.main() == 0
    assert counter.read_text() == "x"
    assert "REUSED" in capsys.readouterr().out
    source.write_text("changed", encoding="utf-8")
    assert checks.main() == 0
    assert counter.read_text() == "xx"


def test_full_match_is_never_reused(runner, monkeypatch):
    _, counter = runner
    monkeypatch.setattr(sys, "argv", ["check_game", "--suite", "full", "--reuse"])
    assert checks.main() == checks.main() == 0
    assert counter.read_text() == "xx"


def test_script_errors_do_not_enter_pass_cache(runner, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["check_game"])
    monkeypatch.setattr(
        checks,
        "command",
        lambda _: [sys.executable, "-c", "print('SCRIPT ERROR: failure')"],
    )
    assert checks.main() == 1
    assert not json.loads((checks.OUTPUT / "passes.json").read_text())
