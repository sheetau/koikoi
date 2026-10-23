# Project work and research records

Run commands from the repository root. Start with the current handoff, not the
historical Markdown:

```powershell
./.venv/Scripts/python.exe -X utf8 -m scripts.project_work brief
```

This standard-library command does not load PyTorch, inspect checkpoints, scan
`runs/`, query the GPU, or execute research checks. The handoff contains the
current scope, unresolved next work, constraints, and evidence entry points.
Its state is explicit; filenames, old process IDs, and old plans do not select
the next experiment. Imported research outcomes remain historical statements,
not results independently reproduced during the migration.

## SQLite and recovery

`.koikoi-work/work.sqlite3` is the local query store for work events, current
state, and historical passages. It uses WAL, a write lock, and durable writes.
`data/work-log.jsonl` is its tracked, append-only recovery journal; the CLI
updates both, so there is only one manual writing interface. Each event records
its recording time and previous-event hash. Recording time is not the date an
imported experiment occurred. Do not edit, truncate, reorder, or auto-merge old
events. Resolve concurrent Git branches deliberately, preserving both intents
as new events rather than inventing a historical order.

The journal is written and flushed before SQL commit. After interruption, a
clone, or a branch change, the next command reconstructs SQL events from the
journal when necessary. If a write reports an error, inspect `brief` and the
last event before retrying: the journal may already have committed the intent.
A missing journal is an error, never an empty backlog. The hash chain detects
accidental changes; it is not a signature or a replacement for Git/backups.
Unchanged journals are content-hashed, then read from SQL without reparsing all
events. File timestamps alone never decide whether cached records are current.

`docs/STATUS.md` and `docs/EXPERIMENTS.md` are frozen historical source files.
**Stop appending to them.** They remain at their original paths and retain
their original bytes so existing citations and evidence bindings keep working.
`data/work-history.json` pins their byte counts and SHA256 identities, including
the existing CRLF working copies and Git's already stored LF versions. This
permits a fresh LF checkout without rewriting the original local files or
accepting any other content change. They
are indexed lazily; ordinary `brief` does not read them. Old instructions in
these files to write STATUS or restart a pending job are historical. This
workflow and the current handoff govern new work.

Binary databases, sidecars, and backups are ignored. The journal and history
manifest are tracked and can contain project facts only: no credentials,
personal chat, hidden game state, raw trajectories, or copied model tensors.
Detailed research evidence stays in its existing report/protocol/run artifact,
referenced once by path and, where useful, digest.

## Retrieve only what is needed

```powershell
./.venv/Scripts/python.exe -X utf8 -m scripts.project_work history "frontier parent" --limit 3
./.venv/Scripts/python.exe -X utf8 -m scripts.project_work history "score auxiliary" --source experiments --chars 900
./.venv/Scripts/python.exe -X utf8 -m scripts.project_work history "migration" --source log
./.venv/Scripts/python.exe -X utf8 -m scripts.project_work show e:2
./.venv/Scripts/python.exe -X utf8 -m scripts.project_work show h:status:17 --chars 1500
```

Use IDs actually returned by the query; the archive ID above is illustrative.
Search uses literal case-insensitive AND terms (SQLite `lower` folds ASCII;
Japanese substring search works without tokenization). Defaults: 5 matches,
700 characters each. Maximum: 20 matches, 2000 characters each. Results include
source, title, line range, and whether they are historical. `show` defaults to
8000 characters and returns `next_offset` for another bounded window:

```powershell
./.venv/Scripts/python.exe -X utf8 -m scripts.project_work show e:2 --offset 8000
```

This avoids repeatedly loading the 1.2 MB combined history into conversation.
Use `rg -n` on specific code/report directories for implementation questions;
avoid unrestricted recursive searches of `runs/`, `external/`, or `.venv/`.
Read `docs/RULES.md` for rules work, `docs/DESIGN.md` for architecture changes,
and the relevant result/protocol before proposing an already tested experiment.

## Write at meaningful boundaries

Prepare a UTF-8 JSON request file outside tracked evidence (for example under
`.koikoi-work/`). A log request has exactly these fields:

```json
{
  "kind": "milestone",
  "title": "Short description of the structural change",
  "body": "What changed and why; reference detailed evidence instead of copying it.",
  "commands": ["the exact command actually used"],
  "outcome": "Measured outcome and limits of the check",
  "refs": ["docs/WORKFLOW.md", "tests/test_project_work.py"]
}
```

Kinds: `milestone`, `decision`, `note`, `question`, `run`. A recorded command is
data; the tool never executes it. Record commands/results once at a coherent
milestone, not after each edit. Use `run` for an already executed operation,
keeping source matches, rounds, policy decisions, planning transitions,
represented mass, and verification replay distinct in the referenced report.

```powershell
./.venv/Scripts/python.exe -X utf8 -m scripts.project_work brief
./.venv/Scripts/python.exe -X utf8 -m scripts.project_work log --file .koikoi-work/milestone.json --expected-revision 7
```

Replace `7` with the current revision. A stale revision refuses the write.
To update the handoff, supply a JSON object with exactly `objective` and `scope`
(strings), plus `next_work`, `limits`, and `anchors` (lists of strings):

```powershell
./.venv/Scripts/python.exe -X utf8 -m scripts.project_work state --file .koikoi-work/next-state.json --expected-revision 8
```

The handoff is limited to 4000 characters. Keep unresolved decisions and the
next concrete work there; log finished work separately. An imported task or
future proposal does not grant permission to implement it. The initial scope
is infrastructure only; do not begin model changes, learning, strength
evaluation, or research replay until the user asks to move to that phase.

## Proportionate checks and recovery

For management-tool changes, run only its regressions and linter:

```powershell
./.venv/Scripts/python.exe -X utf8 -m pytest tests/test_project_work.py -q
./.venv/Scripts/python.exe -X utf8 -m ruff check scripts/project_work.py tests/test_project_work.py
./.venv/Scripts/python.exe -X utf8 -m scripts.project_work doctor
```

`doctor` checks management records, SQLite integrity, and frozen history bytes.
It does not validate research conclusions, replay matches, or refresh evidence
hashes. Never treat cached test outcomes or an indexed claim as new strength
evidence. Do not rerun unchanged passing checks without a relevant change or
unresolved concern. Existing product/bootstrap commands retain their original
behavior and should be used only when their broader scope is needed.

```powershell
./.venv/Scripts/python.exe -X utf8 -m scripts.project_work backup --output .koikoi-work/backups/manual.sqlite3
```

Backup refuses an existing destination and uses SQLite's online backup API.
For complete recovery, retain Git history with the journal/manifest and the
original frozen Markdown. Backups contain logs too; keep them private.
An unreadable SQLite file is not automatically deleted: stop other work-tool
processes, move only `.koikoi-work/work.sqlite3` and its matching sidecars to a
new backup location, then run `brief` and `doctor` to rebuild. Never delete the
journal or research artifacts as a repair. Do not remove a live WAL file.

## Agent entry point and Git

The local root `AGENTS.md` points to `brief` and this guide. It remains ignored
under the existing repository convention; this tracked guide carries the
portable workflow. Preserve unrelated edits and processes. Never infer a live
job from an archived PID or restart a closed study to obtain an easier result.
Record a new measurement only when it was actually executed in its authorized
scope. Commit coherent milestones. Substantial verified project milestones
may be pushed under the root project rule, after checking that only intended
source, documentation, and recovery records are staged; no DB, backup, model,
run artifact, or local AGENTS file belongs in the commit.
