"""Prepare, then run the fixed two-model/two-comparator untouched final test.

Preparation requires the COMPLETE independently audited cold/warm study.
Neither import nor --prepare runs a game. --run consumes the fixed test set.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from koikoi.train import provenance
from koikoi.upstream import UPSTREAM_PATH, load_source
from scripts import audit_cold_ppo_screen as cold
from scripts.run_pretrained_ppo_screen import read, sha, write

BASE = Path("runs/final_published_test")
FIRST, PAIRS, WORKERS = 100_000_000, 10_000, 4
SEEDS = (6710001, 6720001)
AUDITOR = Path("scripts/audit_final_published_test.py")
EVALUATION = {
    "first_seed": FIRST,
    "pairs": PAIRS,
    "split": "test",
    "profile": "paper8",
    "device": "cpu",
    "cpu_threads": 2,
    "deterministic": True,
    "endgame": True,
    "search": "none",
}


def selected_arm(audit, summary):
    if audit.get("status") != "passed":
        raise ValueError("A completed independent development audit is required")
    decision = cold.independent_decision(audit["endpoints"], audit["paired"])
    if decision != audit["decision"] or decision != summary["provisional_decision"]:
        raise ValueError("Development decisions disagree")
    arm = decision["selected_arm"]
    if arm not in ("warm", "cold") or not decision["public_qualified"][arm]:
        raise ValueError("No qualified development method")
    if decision["canonical_case"] != f"seed{SEEDS[0]}_{arm}":
        raise ValueError("Canonical model must use the prespecified first seed")
    return arm


def development_inputs():
    # Read the terminal audit FIRST: a running study cannot create a final protocol.
    audit_path = cold.BASE / "independent_audit.json"
    audit = read(audit_path)
    source = cold.protocol()
    summary_path, manifest_path = [
        cold.BASE / name for name in ("summary.json", "process.json")
    ]
    summary, manifest = read(summary_path), read(manifest_path)
    assert manifest["status"] == "complete_awaiting_audit"
    cold.validate_schedule(manifest["jobs"], source["cases"], source["parallelism"])
    assert len(manifest["jobs"]) == 9
    assert all(
        r["status"] == "complete" and r["returncode"] == 0
        for r in manifest["jobs"].values()
    )
    assert audit["protocol_sha256"] == cold.PROTOCOL_SHA
    assert audit["summary_sha256"] == sha(summary_path)
    helpers = {
        "auditor_sha256": Path(cold.__file__),
        "accounting_helper_sha256": Path("scripts/audit_pretrained_ppo_screen.py"),
        "statistics_helper_sha256": Path("scripts/audit_round_continuation_screen.py"),
    }
    for key, path in helpers.items():
        assert audit[key] == sha(path)
    arm = selected_arm(audit, summary)
    paths = [audit_path, summary_path, manifest_path, cold.BASE / "protocol.json"]
    paths += list(helpers.values()) + [Path("scripts/run_pretrained_ppo_screen.py")]
    for case, endpoint in audit["endpoints"].items():
        path = cold.BASE / case / "evaluation.json"
        assert sha(path) == endpoint["report_sha256"]
        paths.append(path)
    models = {}
    for seed in SEEDS:
        case = f"seed{seed}_{arm}"
        folder = cold.BASE / case
        path = folder / "update_00000520.pt"
        rows = cold.metrics(folder / "metrics.jsonl", final=True)
        assert len(rows) == 520
        payload, check = cold.checkpoint(path, source["cases"][case], 520, rows, source)
        assert check == audit["checkpoints"][case + ":520"]
        models[case] = {
            "checkpoint": str(path),
            **check,
            "config": payload["config"],
            "training_provenance": payload["provenance"],
        }
        paths += [path, folder / "metrics.jsonl"]
    return (
        {
            "selected_arm": arm,
            "canonical_case": f"seed{SEEDS[0]}_{arm}",
            "audit_sha256": sha(audit_path),
            "summary_sha256": sha(summary_path),
        },
        models,
        {str(path): sha(path) for path in paths},
    )


def recorded_deals(value):
    ranges, seeds, pending = [], [], [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            start = item.get("seed_start", item.get("first_seed"))
            count = item.get("pairs", item.get("matches"))
            if type(start) is int and type(count) is int and count > 0:
                ranges.append((start, start + count))
            rows = item.get("records", [])
            if isinstance(rows, list):
                seeds.extend(
                    row["seed"]
                    for row in rows
                    if isinstance(row, dict) and type(row.get("seed")) is int
                )
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return ranges, seeds


def seed_scan(root=Path("runs")):
    """Inspect stored JSON report metadata AND actual recorded deal seeds."""
    reports, overlaps = {}, []
    for path in sorted(root.rglob("*.json")):
        if path.resolve().is_relative_to(BASE.resolve()):
            continue
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError(f"Cannot verify uninspected large JSON: {path}")
        # Historical PowerShell reports may carry a UTF-16 BOM. JSON bytes
        # support its documented UTF-8/16/32 encodings without rewriting files.
        try:
            r = json.loads(path.read_bytes())
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot inspect report JSON: {path}") from exc
        ranges, seeds = recorded_deals(r)
        if not ranges and not seeds:
            continue
        reports[str(path)] = sha(path)
        if any(start < FIRST + PAIRS and FIRST < end for start, end in ranges) or any(
            FIRST <= seed < FIRST + PAIRS for seed in seeds
        ):
            overlaps.append(str(path))
    return {
        "range": [FIRST, FIRST + PAIRS - 1],
        "scope": "recursive stored JSON ranges and raw recorded seeds; excludes this final run; no claim about unrecorded work",
        "reports": reports,
        "overlaps": overlaps,
    }


def job_plan(models):
    return {
        f"{case}__{variant}": {
            "case": case,
            "opponent": f"rl-{variant}-assisted",
            "variant": variant,
        }
        for variant in ("wp", "point")
        for case in models
    }


def upstream_inputs():
    # Recheck even after load_source's per-process cache has been populated.
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM_PATH, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=UPSTREAM_PATH,
        text=True,
    ).strip()
    if revision != cold.UPSTREAM or dirty:
        raise ValueError("Comparator checkout must remain clean and pinned")
    load_source()
    bindings = {}
    names = subprocess.check_output(
        ["git", "ls-files", "--", "*.py"], cwd=UPSTREAM_PATH, text=True
    ).splitlines()
    for name in names:
        path = UPSTREAM_PATH / name
        bindings[str(path)] = sha(path)
    comparators = {}
    for variant in ("wp", "point"):
        comparators[variant] = {}
        for name in ("discard", "pick", "koikoi"):
            path = UPSTREAM_PATH / f"model_agent/{name}_rl_{variant}.pt"
            comparators[variant][name] = sha(path)
            bindings[str(path)] = sha(path)
    return comparators, bindings


def prepare():
    if BASE.exists():
        raise FileExistsError(f"Refusing to replace {BASE}")
    development, models, bindings = development_inputs()
    current = provenance()
    assert current["source_sha256"] == cold.SOURCE
    comparators, upstream_bindings = upstream_inputs()
    bindings.update(upstream_bindings)
    scan = seed_scan()
    if scan["overlaps"]:
        raise ValueError("Final deal range already occurs in stored reports")
    p = {
        "prepared_at": time.time(),
        "provenance": current,
        "runner_sha256": sha(__file__),
        "auditor_sha256": sha(AUDITOR),
        "development": development,
        "models": models,
        "bindings": bindings,
        "comparator_hashes": comparators,
        "upstream_revision": cold.UPSTREAM,
        "seed_scan": scan,
        "evaluation": EVALUATION,
        "parallel_workers": WORKERS,
        "jobs": job_plan(models),
        "criteria": {
            "primary": "BOTH frozen model seeds: assisted RL-WP win-score95% CI lower bound >0.5.",
            "secondary": "Separately report assisted RL-Point on both models; replicated secondary criterion requires BOTH lower bounds >0.5. Points remain secondary and never replace win-score criteria.",
            "interpretation": "Pair bootstrap conditional on each fixed model; not exact coverage or a claim about all possible training seeds. The four evaluations share10000 deal seeds, not40000 independent deals. Canonical model remains the prespecified first training seed. No reranking or pooling.",
        },
        "stop_rule": "Exactlyfour evaluations x10000 duplicate pairs (80000 played matches). No training, adaptation, extension, early strength stopping or model selection on final outcomes. Failure aborts owned workers and retains artifacts. Audit all raw results before a conclusion; preserve failures. Concurrent decision timings are not isolated latency benchmarks.",
    }
    BASE.mkdir(exist_ok=False)
    write(BASE / "protocol.json", p)
    print({"prepared": True, "protocol_sha256": sha(BASE / "protocol.json")})


def validate_frozen():
    p = read(BASE / "protocol.json")
    if (BASE / "process.json").exists():
        assert read(BASE / "process.json")["protocol_sha256"] == sha(
            BASE / "protocol.json"
        )
    assert p["runner_sha256"] == sha(__file__) and p["auditor_sha256"] == sha(AUDITOR)
    assert p["evaluation"] == EVALUATION and p["parallel_workers"] == WORKERS
    assert provenance()["source_sha256"] == cold.SOURCE
    assert p["provenance"]["source_sha256"] == cold.SOURCE
    assert p["upstream_revision"] == cold.UPSTREAM
    for path, digest in p["bindings"].items():
        assert sha(path) == digest
    development, models, development_bindings = development_inputs()
    comparators, upstream_bindings = upstream_inputs()
    assert p["development"] == development and p["models"] == models
    assert p["bindings"] == development_bindings | upstream_bindings
    assert p["comparator_hashes"] == comparators
    assert p["jobs"] == job_plan(models) and len(p["jobs"]) == 4
    assert p["seed_scan"]["range"] == [FIRST, FIRST + PAIRS - 1]
    assert not p["seed_scan"]["overlaps"]
    for path, digest in p["seed_scan"]["reports"].items():
        assert sha(path) == digest
    return p


def evaluation_argv(p, name):
    job = p["jobs"][name]
    return [
        "koikoi.evaluate",
        "--agent",
        p["models"][job["case"]]["checkpoint"],
        "--opponent",
        job["opponent"],
        "--profile",
        "paper8",
        "--pairs",
        str(PAIRS),
        "--seed",
        str(FIRST),
        "--split",
        "test",
        "--device",
        "cpu",
        "--endgame",
        "--deterministic",
        "--search",
        "none",
        "--output",
        str(BASE / name / "evaluation.json"),
    ]


def worker(name):
    torch.set_num_threads(2)
    p = validate_frozen()
    assert name in p["jobs"]
    output = BASE / name / "evaluation.json"
    if output.exists():
        raise FileExistsError(output)
    from koikoi.evaluate import main

    sys.argv = evaluation_argv(p, name)
    main()
    validate_frozen()
    r = read(output)
    assert r["pairs"] == PAIRS and r["seed_start"] == FIRST and r["split"] == "test"
    assert len(r["records"]) == PAIRS


def conclusions(endpoints, jobs):
    result = {}
    for label, variant in (("primary_wp", "wp"), ("secondary_point", "point")):
        names = [name for name, job in jobs.items() if job["variant"] == variant]
        if len(names) != 2:
            raise ValueError("Both model seeds are required for each comparator")
        result[label + "_replicated"] = all(
            endpoints[name]["win_ci95"][0] > 0.5 for name in names
        )
    return result


def summarize(p):
    endpoints = {}
    for name in p["jobs"]:
        path = BASE / name / "evaluation.json"
        r = read(path)
        endpoints[name] = {
            "report_sha256": sha(path),
            "win_rate": r["draw_adjusted_win_rate"],
            "win_ci95": r["win_rate_ci95"],
            "point_difference": r["mean_point_difference"],
            "point_ci95": r["point_difference_ci95"],
        }
    write(
        BASE / "summary.json",
        {
            "status": "awaiting_independent_audit",
            "protocol_sha256": sha(BASE / "protocol.json"),
            "endpoints": endpoints,
            "provisional_conclusions": conclusions(endpoints, p["jobs"]),
        },
    )


def supervise(p):
    records, live = {}, {}
    manifest = {
        "runner_pid": os.getpid(),
        "started": time.time(),
        "status": "running",
        "jobs": records,
        "protocol_sha256": sha(BASE / "protocol.json"),
    }
    try:
        assert len(p["jobs"]) == p["parallel_workers"] == WORKERS
        write(BASE / "process.json", manifest)
        for name in p["jobs"]:
            folder = BASE / name
            folder.mkdir(exist_ok=False)
            stream = (folder / "evaluate.log").open("wb")
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "scripts.run_final_published_test",
                        "--worker",
                        name,
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            except BaseException:
                stream.close()
                raise
            records[name] = {
                "pid": process.pid,
                "started": time.time(),
                "status": "running",
                "resource": "cpu",
                "cpu_threads": 2,
            }
            live[name] = (process, stream)
            write(BASE / "process.json", manifest)
            print({"job": name, **records[name]}, flush=True)
        while live:
            for name, (process, stream) in list(live.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                del live[name]
                records[name].update(
                    returncode=code,
                    finished=time.time(),
                    status="complete" if code == 0 else "failed",
                )
                write(BASE / "process.json", manifest)
                if code:
                    raise RuntimeError(f"{name} failed; retain all artifacts")
            if live:
                time.sleep(0.5)
        summarize(p)
        manifest.update(status="complete_awaiting_audit", finished=time.time())
    except BaseException:
        for name, (process, stream) in live.items():
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        capture_output=True,
                        check=False,
                    )
                else:
                    process.terminate()
            code = process.wait()
            stream.close()
            records[name].update(
                returncode=code, finished=time.time(), status="aborted"
            )
        manifest.update(status="failed", finished=time.time())
        write(BASE / "process.json", manifest)
        raise
    write(BASE / "process.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.worker:
        worker(args.worker)
    else:
        p = validate_frozen()
        if (BASE / "process.json").exists() or any(
            (BASE / name).exists() for name in p["jobs"]
        ):
            raise FileExistsError(
                "Final test already started; inspect its original process, never silently rerun"
            )
        if seed_scan()["overlaps"]:
            raise ValueError("Final deal range appeared after preparation")
        supervise(p)


if __name__ == "__main__":
    main()
