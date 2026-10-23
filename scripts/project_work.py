"""Bounded project handoff and SQLite log search. Standard library only.

SQLite serves queries; the append-only UTF-8 journal is the portable recovery
record. Archived research files are indexed on demand and never rewritten.
No training, inference, experiment, or research verifier is imported or run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOURNAL = "data/work-log.jsonl"
SOURCES = "data/work-history.json"
DATABASE = ".koikoi-work/work.sqlite3"
VERSION = 1
KINDS = ("milestone", "decision", "note", "question", "run")
STATE_KEYS = {"objective", "scope", "next_work", "limits", "anchors"}


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bounded_text(value, name, maximum, *, empty=False):
    require(isinstance(value, str), f"{name} must be text")
    require(
        (empty or bool(value.strip())) and len(value) <= maximum,
        f"{name} must contain {'0' if empty else '1'}..{maximum} characters",
    )


def validate_payload(payload):
    require(isinstance(payload, dict), "Event payload must be an object")
    kind = payload.get("kind")
    if kind == "state":
        require(set(payload) == {"kind", "value"}, "Invalid state event fields")
        value = payload["value"]
        require(
            isinstance(value, dict) and set(value) == STATE_KEYS,
            "State requires objective, scope, next_work, limits, anchors",
        )
        for key in ("objective", "scope"):
            bounded_text(value[key], key, 700)
        for key in ("next_work", "limits", "anchors"):
            require(
                isinstance(value[key], list) and len(value[key]) <= 12,
                f"{key} must be a list of at most 12 strings",
            )
            for item in value[key]:
                bounded_text(item, key, 700)
        require(
            len(encode(value)) <= 4000,
            "Handoff exceeds 4000 characters; move detail to a log",
        )
        return
    require(kind in KINDS, f"Unknown event kind: {kind}")
    require(
        set(payload) == {"kind", "title", "body", "commands", "outcome", "refs"},
        "Log requires kind, title, body, commands, outcome, refs",
    )
    for key, maximum in (("title", 160), ("body", 20000), ("outcome", 2000)):
        bounded_text(payload[key], key, maximum, empty=key != "title")
    for key, count, maximum in (("commands", 10, 2000), ("refs", 20, 700)):
        require(
            isinstance(payload[key], list) and len(payload[key]) <= count,
            f"{key} must be a list of at most {count} strings",
        )
        for item in payload[key]:
            bounded_text(item, key, maximum)
    require(len(encode(payload)) <= 32000, "Log is too large; reference a detail file")


def load_journal(path, raw=None):
    require(
        path.is_file(),
        f"Missing recovery journal: {path}. Restore from Git; do not initialize an empty backlog.",
    )
    if raw is None:
        raw = path.read_bytes()
    require(
        raw.endswith(b"\n"),
        "Incomplete journal tail; preserve it and recover from backup/Git",
    )
    result, previous = [], ""
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        event = json.loads(line)
        require(
            isinstance(event, dict)
            and set(event)
            == {"version", "id", "recorded_at", "previous", "payload", "sha256"},
            f"Invalid journal event {number}",
        )
        require(
            event["version"] == VERSION and event["id"] == number,
            f"Invalid event sequence at {number}",
        )
        require(event["previous"] == previous, f"Broken journal chain at {number}")
        bounded_text(event["recorded_at"], "recorded_at", 40)
        validate_payload(event["payload"])
        unsigned = {k: v for k, v in event.items() if k != "sha256"}
        require(
            event["sha256"] == sha(encode(unsigned).encode("utf-8")),
            f"Journal digest differs at {number}",
        )
        previous = event["sha256"]
        result.append(event)
    require(
        result and result[0]["payload"]["kind"] == "state",
        "Journal must start with an explicit handoff",
    )
    return raw, result


class Store:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.journal = self.root / JOURNAL
        # Fail before creating a cache if the recoverable records are absent.
        require(
            self.journal.is_file(),
            f"Missing {JOURNAL}; restore the repository records first",
        )
        database = self.root / DATABASE
        database.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(database, timeout=15)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        require(version in (0, VERSION), "Unsupported local database version")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, recorded_at TEXT NOT NULL,
                kind TEXT NOT NULL, payload TEXT NOT NULL, digest TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS archive_sources (
                tag TEXT PRIMARY KEY, path TEXT NOT NULL, digest TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS passages (
                id TEXT PRIMARY KEY, source TEXT NOT NULL, path TEXT NOT NULL,
                title TEXT NOT NULL, first_line INTEGER NOT NULL,
                last_line INTEGER NOT NULL, body TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS passages_source ON passages(source);
        """)
        self.db.execute(f"PRAGMA user_version={VERSION}")

    def close(self):
        self.db.close()

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            raw = self.journal.read_bytes()
            digest = sha(raw)
            cached = self.db.execute(
                "SELECT value FROM meta WHERE key='journal'"
            ).fetchone()
            if cached is None or cached[0] != digest:
                _, events = load_journal(self.journal, raw)
                # Reconstruct after a clone, branch change, or interrupted write.
                self.db.execute("DELETE FROM events")
                self.db.executemany(
                    "INSERT INTO events VALUES (?,?,?,?,?)",
                    [
                        (
                            e["id"],
                            e["recorded_at"],
                            e["payload"]["kind"],
                            encode(e["payload"]),
                            e["sha256"],
                        )
                        for e in events
                    ],
                )
                self.db.execute(
                    "INSERT OR REPLACE INTO meta VALUES ('journal',?)", (digest,)
                )
            yield raw
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def append(self, payload, expected_revision):
        validate_payload(payload)
        with self.transaction() as raw:
            last = self.db.execute(
                "SELECT id,digest FROM events ORDER BY id DESC LIMIT 1"
            ).fetchone()
            revision = last["id"]
            require(
                expected_revision == revision,
                f"Stale revision: expected {expected_revision}, current {revision}; read brief again",
            )
            event = {
                "version": VERSION,
                "id": revision + 1,
                "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "previous": last["digest"],
                "payload": payload,
            }
            event["sha256"] = sha(encode(event).encode("utf-8"))
            addition = (encode(event) + "\n").encode("utf-8")
            # Durable recovery record first. If SQL commit fails, next open replays
            # this event; never truncate it or silently repeat the command.
            with self.journal.open("ab") as stream:
                stream.write(addition)
                stream.flush()
                os.fsync(stream.fileno())
            self.db.execute(
                "INSERT INTO events VALUES (?,?,?,?,?)",
                (
                    event["id"],
                    event["recorded_at"],
                    payload["kind"],
                    encode(payload),
                    event["sha256"],
                ),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO meta VALUES ('journal',?)",
                (sha(raw + addition),),
            )
        return {"id": f"e:{event['id']}", "revision": event["id"], "saved": True}

    def brief(self):
        with self.transaction():
            state = self.db.execute(
                "SELECT payload FROM events WHERE kind='state' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            latest = self.db.execute(
                "SELECT id,kind,payload FROM events WHERE kind!='state' ORDER BY id DESC LIMIT 3"
            ).fetchall()
            return {
                "revision": self.db.execute("SELECT MAX(id) FROM events").fetchone()[0],
                **json.loads(state[0])["value"],
                "recent": [
                    {
                        "id": f"e:{row['id']}",
                        "kind": row["kind"],
                        "title": json.loads(row["payload"])["title"],
                    }
                    for row in latest
                ],
            }

    def index_archives(self, source="all"):
        manifest = json.loads((self.root / SOURCES).read_text(encoding="utf-8"))
        require(manifest.get("version") == VERSION, "Unsupported history manifest")
        count = 0
        for item in manifest["sources"]:
            if source not in ("all", item["tag"]):
                continue
            path = (self.root / item["path"]).resolve()
            require(path.is_relative_to(self.root), "Archive path escapes workspace")
            raw = path.read_bytes()
            digest = sha(raw)
            identities = {(item["bytes"], item["sha256"])}
            if "git_sha256" in item:
                identities.add((item["git_bytes"], item["git_sha256"]))
            require(
                (len(raw), digest) in identities,
                f"Frozen archive changed: {item['path']}; restore its original bytes",
            )
            cached = self.db.execute(
                "SELECT digest FROM archive_sources WHERE tag=?", (item["tag"],)
            ).fetchone()
            if cached is None or cached[0] != digest:
                self.db.execute("DELETE FROM passages WHERE source=?", (item["tag"],))
                self.db.executemany(
                    "INSERT INTO passages VALUES (?,?,?,?,?,?,?)", passages(item, raw)
                )
                self.db.execute(
                    "INSERT OR REPLACE INTO archive_sources VALUES (?,?,?)",
                    (item["tag"], item["path"], digest),
                )
            count += 1
        require(count or source == "log", f"Unknown archive source: {source}")
        return count

    def history(self, query, *, limit=5, chars=700, source="all"):
        bounded_text(query, "query", 200)
        require(
            1 <= limit <= 20 and 100 <= chars <= 2000,
            "Search limits: 1..20 rows, 100..2000 chars each",
        )
        terms = query.split()
        matches = []
        with self.transaction():
            if source in ("all", "log"):
                for row in self.db.execute(
                    "SELECT * FROM events WHERE kind!='state' ORDER BY id DESC"
                ):
                    payload = json.loads(row["payload"])
                    body = encode(payload)
                    if all(term.casefold() in body.casefold() for term in terms):
                        matches.append(
                            {
                                "id": f"e:{row['id']}",
                                "source": "log",
                                "title": payload["title"],
                                "recorded_at": row["recorded_at"],
                                **excerpt(body, terms, chars),
                            }
                        )
                        if len(matches) == limit:
                            break
            if source != "log" and len(matches) < limit:
                self.index_archives(source)
                conditions = [
                    "instr(lower(title || char(10) || body), lower(?)) > 0"
                    for _ in terms
                ]
                params = list(terms)
                if source != "all":
                    conditions.append("source=?")
                    params.append(source)
                params.append(limit - len(matches))
                rows = self.db.execute(
                    "SELECT * FROM passages WHERE "
                    + " AND ".join(conditions)
                    + " ORDER BY source,first_line LIMIT ?",
                    params,
                )
                for row in rows:
                    matches.append(
                        {
                            "id": row["id"],
                            "source": row["source"],
                            "historical": True,
                            "path": row["path"],
                            "lines": [row["first_line"], row["last_line"]],
                            "title": row["title"],
                            **excerpt(row["body"], terms, chars),
                        }
                    )
        return {
            "matches": matches,
            "limit": limit,
            "hint": "Use show ID for details. Archived plans/process IDs are historical.",
        }

    def show(self, identifier, *, offset=0, chars=8000):
        require(offset >= 0 and 100 <= chars <= 12000, "Invalid detail window")
        with self.transaction():
            if identifier.startswith("e:"):
                row = self.db.execute(
                    "SELECT * FROM events WHERE id=?", (int(identifier[2:]),)
                ).fetchone()
                require(row is not None, "Unknown event ID")
                body = json.dumps(
                    json.loads(row["payload"]), ensure_ascii=False, indent=2
                )
                metadata = {
                    "id": identifier,
                    "recorded_at": row["recorded_at"],
                    "sha256": row["digest"],
                }
            else:
                parts = identifier.split(":")
                require(
                    len(parts) == 3 and parts[0] == "h",
                    "ID must be e:NUMBER or h:SOURCE:LINE",
                )
                self.index_archives(parts[1])
                row = self.db.execute(
                    "SELECT * FROM passages WHERE id=?", (identifier,)
                ).fetchone()
                require(row is not None, "Unknown archive ID")
                body = row["body"]
                metadata = {
                    "id": identifier,
                    "historical": True,
                    "path": row["path"],
                    "title": row["title"],
                    "lines": [row["first_line"], row["last_line"]],
                }
            return {
                **metadata,
                "text": body[offset : offset + chars],
                "offset": offset,
                "next_offset": offset + chars if len(body) > offset + chars else None,
            }

    def doctor(self):
        with self.transaction() as raw:
            _, events = load_journal(self.journal, raw)
            require(
                self.db.execute("PRAGMA integrity_check").fetchone()[0] == "ok",
                "SQLite integrity check failed",
            )
            # Compare all rows to the recovery record, not merely a cached digest.
            rows = self.db.execute("SELECT * FROM events ORDER BY id").fetchall()
            require(
                [
                    (r["id"], r["recorded_at"], r["kind"], r["payload"], r["digest"])
                    for r in rows
                ]
                == [
                    (
                        e["id"],
                        e["recorded_at"],
                        e["payload"]["kind"],
                        encode(e["payload"]),
                        e["sha256"],
                    )
                    for e in events
                ],
                "SQLite events differ from recovery journal; rebuild the disposable database",
            )
            sources = self.index_archives()
            return {
                "status": "ok",
                "revision": events[-1]["id"],
                "events": len(rows),
                "frozen_sources": sources,
                "indexed_passages": self.db.execute(
                    "SELECT COUNT(*) FROM passages"
                ).fetchone()[0],
                "research_checks_executed": False,
            }

    def backup(self, output):
        output = Path(output).resolve()
        require(
            output != self.journal and output != (self.root / DATABASE).resolve(),
            "Choose a new backup path",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive reservation prevents accidental replacement of prior evidence.
        with output.open("xb"):
            pass
        try:
            with self.transaction():
                pass
            destination = sqlite3.connect(output)
            try:
                self.db.backup(destination)
            finally:
                destination.close()
        except BaseException:
            output.unlink()
            raise
        return {"backup": str(output), "sha256": sha(output.read_bytes())}


def passages(source, raw):
    title, lines, start = "", [], 1
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if line.startswith("#"):
            title = line.lstrip("# ")
        if lines and (not line.strip() or sum(len(part) for part in lines) > 4000):
            yield (
                f"h:{source['tag']}:{start}",
                source["tag"],
                source["path"],
                title,
                start,
                number - 1,
                "\n".join(lines),
            )
            lines = []
        if line.strip():
            if not lines:
                start = number
            lines.append(line)
    if lines:
        yield (
            f"h:{source['tag']}:{start}",
            source["tag"],
            source["path"],
            title,
            start,
            start + len(lines) - 1,
            "\n".join(lines),
        )


def excerpt(body, terms, chars):
    positions = [body.casefold().find(term.casefold()) for term in terms]
    position = min((p for p in positions if p >= 0), default=0)
    start = max(0, position - chars // 4)
    return {
        "text": body[start : start + chars],
        "offset": start,
        "truncated": start > 0 or len(body) > chars,
    }


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "brief", help="Small current handoff; does not scan archives or load Torch"
    )
    history = commands.add_parser(
        "history", help="Bounded literal AND search; IDs permit reading exact details"
    )
    history.add_argument("query")
    history.add_argument("--limit", type=int, default=5)
    history.add_argument("--chars", type=int, default=700)
    history.add_argument(
        "--source", choices=("all", "status", "experiments", "log"), default="all"
    )
    show = commands.add_parser("show")
    show.add_argument("id")
    show.add_argument("--offset", type=int, default=0)
    show.add_argument("--chars", type=int, default=8000)
    log = commands.add_parser(
        "log", help="Record only; supplied commands are never executed"
    )
    log.add_argument(
        "--file",
        type=Path,
        required=True,
        help="UTF-8 JSON: kind,title,body,commands,outcome,refs",
    )
    log.add_argument("--expected-revision", type=int, required=True)
    state = commands.add_parser(
        "state", help="Replace the short handoff without editing old log entries"
    )
    state.add_argument(
        "--file",
        type=Path,
        required=True,
        help="UTF-8 JSON: objective,scope,next_work,limits,anchors",
    )
    state.add_argument("--expected-revision", type=int, required=True)
    commands.add_parser(
        "doctor", help="Check management records and frozen archive bytes only"
    )
    backup = commands.add_parser("backup")
    backup.add_argument("--output", type=Path, required=True)
    return cli


def main(argv=None, root=ROOT):
    args = parser().parse_args(argv)
    store = None
    try:
        store = Store(root)
        if args.command == "brief":
            result = store.brief()
        elif args.command == "history":
            result = store.history(
                args.query, limit=args.limit, chars=args.chars, source=args.source
            )
        elif args.command == "show":
            result = store.show(args.id, offset=args.offset, chars=args.chars)
        elif args.command in ("state", "log"):
            payload = json.loads(args.file.read_text(encoding="utf-8"))
            if args.command == "state":
                payload = {"kind": "state", "value": payload}
            else:
                require(isinstance(payload, dict), "Log request must be an object")
                require(payload.get("kind") in KINDS, "Use state to update the handoff")
            result = store.append(payload, args.expected_revision)
        elif args.command == "backup":
            result = store.backup(args.output)
        else:
            result = store.doctor()
        print(encode(result))
        return 0
    except (ValueError, OSError, sqlite3.Error) as error:
        print(
            encode(
                {
                    "error": str(error),
                    "hint": "No research command was executed. Preserve the journal; see docs/WORKFLOW.md.",
                }
            ),
            file=sys.stderr,
        )
        return 1
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
