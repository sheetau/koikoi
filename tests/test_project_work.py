"""Management-only regressions; no game, model, or research verification."""

import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scripts import project_work as work

STATE = {
    "objective": "Build the strongest Koi-Koi AI",
    "scope": "Infrastructure only; model work deferred",
    "next_work": ["Complete log migration"],
    "limits": ["No training or research evaluation"],
    "anchors": ["docs/WORKFLOW.md"],
}


def event(number, payload, previous=""):
    value = {
        "version": 1,
        "id": number,
        "recorded_at": "2026-10-24T00:00:00+00:00",
        "previous": previous,
        "payload": payload,
    }
    value["sha256"] = work.sha(work.encode(value).encode("utf-8"))
    return value


def log(title="Migration", body="Structural change only"):
    return {
        "kind": "milestone",
        "title": title,
        "body": body,
        "commands": [],
        "outcome": "",
        "refs": [],
    }


@pytest.fixture
def project(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    first = event(1, {"kind": "state", "value": STATE})
    (tmp_path / work.JOURNAL).write_text(
        work.encode(first) + "\n", encoding="utf-8", newline="\n"
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    sources = []
    for tag in ("status", "experiments"):
        raw = f"# Old {tag}\r\n\r\n日本語の履歴 alpha beta\r\n\r\nClosed job PID 123.\r\n".encode()
        path = docs / f"{tag}.md"
        path.write_bytes(raw)
        sources.append(
            {
                "tag": tag,
                "path": path.relative_to(tmp_path).as_posix(),
                "sha256": work.sha(raw),
                "bytes": len(raw),
                "git_sha256": work.sha(raw.replace(b"\r\n", b"\n")),
                "git_bytes": len(raw.replace(b"\r\n", b"\n")),
            }
        )
    (tmp_path / work.SOURCES).write_text(
        json.dumps({"version": 1, "sources": sources}), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def store(project):
    instance = work.Store(project)
    try:
        yield instance
    finally:
        instance.close()


def test_brief_is_bounded_and_does_not_scan_historical_files(store, project):
    (project / work.SOURCES).unlink()
    result = store.brief()
    assert result == {"revision": 1, **STATE, "recent": []}
    assert len(work.encode(result)) < 1000
    assert store.db.execute("SELECT COUNT(*) FROM passages").fetchone()[0] == 0


def test_unchanged_journal_uses_sqlite_without_reparsing(store, monkeypatch):
    expected = store.brief()

    def unexpected_parse(*args):
        raise AssertionError("Unchanged journal should use existing SQL rows")

    monkeypatch.setattr(work, "load_journal", unexpected_parse)
    assert store.brief() == expected


def test_competing_writers_preserve_the_winner_and_reject_the_stale_write(project):
    def update(title):
        connection = work.Store(project)
        try:
            try:
                return connection.append(log(title), 1)["saved"]
            except ValueError as error:
                assert "Stale revision" in str(error)
                return False
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(update, ["First writer", "Second writer"]))
    assert sorted(results) == [False, True]
    connection = work.Store(project)
    try:
        assert connection.doctor()["events"] == 2
    finally:
        connection.close()


def test_append_conflicts_do_not_lose_or_duplicate_records(store, project):
    saved = store.append(log(), expected_revision=1)
    assert saved == {"id": "e:2", "revision": 2, "saved": True}
    original = (project / work.JOURNAL).read_bytes()
    with pytest.raises(ValueError, match="Stale revision"):
        store.append(log("Stale edit"), expected_revision=1)
    assert (project / work.JOURNAL).read_bytes() == original
    assert store.brief()["revision"] == 2
    assert store.show("e:2")["recorded_at"]


def test_current_state_replacement_preserves_prior_state_and_logs(store):
    store.append(log(), 1)
    new_state = dict(STATE, scope="Waiting for the next explicit instruction")
    store.append({"kind": "state", "value": new_state}, 2)
    assert store.brief()["scope"] == new_state["scope"]
    assert STATE["scope"] in store.show("e:1")["text"]
    assert store.brief()["recent"][0]["id"] == "e:2"


def test_interrupted_sql_commit_replays_the_durable_journal(store, project):
    # Emulate a process dying after fsync(journal), before SQLite commit.
    _, records = work.load_journal(project / work.JOURNAL)
    second = event(2, log("Durable before commit"), records[-1]["sha256"])
    with (project / work.JOURNAL).open("ab") as stream:
        stream.write((work.encode(second) + "\n").encode("utf-8"))
    assert store.brief()["revision"] == 2
    assert store.doctor()["events"] == 2


def test_deleting_database_rebuilds_from_recovery_journal(project):
    original = work.Store(project)
    original.append(log("Recoverable"), 1)
    expected = original.brief()
    original.close()
    (project / work.DATABASE).unlink()
    restored = work.Store(project)
    try:
        assert restored.brief() == expected
        assert restored.doctor()["status"] == "ok"
    finally:
        restored.close()


@pytest.mark.parametrize("mutation", ["partial", "digest", "sequence"])
def test_damaged_journal_is_rejected_without_silently_resetting(
    store, project, mutation
):
    path = project / work.JOURNAL
    raw = path.read_bytes()
    if mutation == "partial":
        raw = raw[:-1]
    else:
        record = json.loads(raw)
        record["payload"]["value"]["scope"] = "tampered"
        if mutation == "sequence":
            record["id"] = 3
        raw = (work.encode(record) + "\n").encode("utf-8")
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        store.brief()
    assert path.read_bytes() == raw


def test_missing_journal_is_not_treated_as_an_empty_backlog(tmp_path):
    with pytest.raises(ValueError, match="restore"):
        work.Store(tmp_path)
    assert not (tmp_path / ".koikoi-work").exists()


def test_literal_search_handles_japanese_and_sql_characters(store, project):
    originals = {p: p.read_bytes() for p in (project / "docs").iterdir()}
    result = store.history("日本語 alpha", limit=1, chars=100)
    row = result["matches"][0]
    assert row["historical"] and row["lines"] == [3, 3]
    assert "日本語" in row["text"]
    assert store.show(row["id"])["text"] == "日本語の履歴 alpha beta"
    assert not store.history("' OR 1=1 --")["matches"]
    assert {p: p.read_bytes() for p in originals} == originals


def test_frozen_archive_change_fails_even_when_already_indexed(store, project):
    store.history("alpha")
    path = project / "docs/status.md"
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b"alpha", b"omega"))  # same byte length
    with pytest.raises(ValueError, match="Frozen archive changed"):
        store.history("omega", source="status")
    # A missing/stale archive never prevents the short current handoff.
    assert store.brief()["revision"] == 1


def test_git_lf_checkout_remains_searchable_without_rewriting_workspace_history(
    store, project
):
    expected = store.history("alpha")["matches"]
    for path in (project / "docs").iterdir():
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    assert store.history("alpha")["matches"] == expected
    assert store.doctor()["frozen_sources"] == 2


def test_log_search_and_detail_pagination_preserve_all_text(store):
    body = "context " * 200 + "unique-marker" + " tail" * 500
    store.append(log(body=body), 1)
    match = store.history("unique-marker", source="log", chars=100)["matches"][0]
    assert match["truncated"] and "unique-marker" in match["text"]
    offset, parts = 0, []
    while offset is not None:
        page = store.show("e:2", offset=offset, chars=100)
        parts.append(page["text"])
        offset = page["next_offset"]
    assert json.loads("".join(parts))["body"] == body


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "execute", "command": "never run"},
        dict(log(), extra="unknown"),
        dict(log(), title="x" * 161),
        {"kind": "state", "value": dict(STATE, next_work=["x" * 701])},
    ],
)
def test_invalid_updates_leave_both_stores_unchanged(store, project, payload):
    raw = (project / work.JOURNAL).read_bytes()
    with pytest.raises(ValueError):
        store.append(payload, 1)
    assert (project / work.JOURNAL).read_bytes() == raw
    assert store.brief()["revision"] == 1


def test_backup_can_be_read_independently_and_never_overwrites(store, project):
    store.append(log("Backup source"), 1)
    destination = project / "backups/test.sqlite3"
    result = store.backup(destination)
    assert result["sha256"] == work.sha(destination.read_bytes())
    with sqlite3.connect(destination) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        store.backup(destination)
    assert destination.read_bytes() == before


def test_doctor_detects_database_rows_that_disagree_with_the_journal(store):
    store.brief()
    store.db.execute("UPDATE events SET payload='{}' WHERE id=1")
    store.db.commit()
    with pytest.raises(ValueError, match="differ from recovery"):
        store.doctor()


def test_cli_logs_commands_as_data_without_executing_them(project, capsys):
    marker = project / "must-not-exist"
    request = log("Command is a record", body="Not an execution request")
    request["commands"] = [f"create {marker}"]
    path = project / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    assert (
        work.main(
            ["log", "--file", str(path), "--expected-revision", "1"], root=project
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["revision"] == 2
    assert not marker.exists()


def test_import_does_not_load_research_or_third_party_modules():
    code = (
        "import sys; from scripts import project_work; "
        "assert not any(n.split('.')[0] in {'torch','numpy','koikoi'} for n in sys.modules)"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(work.__file__).resolve().parents[1],
        check=True,
    )
