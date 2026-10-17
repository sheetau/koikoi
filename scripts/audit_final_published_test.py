"""Independently audit frozen final-test identities, raw pairs and conclusions.

No production evaluator/statistics/runner decision functions are imported.
Reuses the earlier independent checkpoint and scalar-bootstrap auditors.
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

from scripts import audit_cold_ppo_screen as cold
from scripts.audit_round_continuation_screen import close, statistics
from scripts.run_pretrained_ppo_screen import write

BASE = Path("runs/final_published_test")
FIRST, PAIRS = 100_000_000, 10_000
SEEDS = (6710001, 6720001)
read, sha = cold.read, cold.sha


def assert_no_reserved_deals(value):
    if isinstance(value, dict):
        start = value.get("seed_start", value.get("first_seed"))
        count = value.get("pairs", value.get("matches"))
        if type(start) is int and type(count) is int and count > 0:
            assert start + count <= FIRST or start >= FIRST + PAIRS
        rows = value.get("records", [])
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and type(row.get("seed")) is int:
                    assert row["seed"] < FIRST or row["seed"] >= FIRST + PAIRS
        for child in value.values():
            assert_no_reserved_deals(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_reserved_deals(child)


def protocol():
    p = read(BASE / "protocol.json")
    assert p["runner_sha256"] == sha("scripts/run_final_published_test.py")
    assert p["auditor_sha256"] == sha(__file__)
    assert p["evaluation"] == {
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
    assert p["parallel_workers"] == 4 and p["upstream_revision"] == cold.UPSTREAM
    assert p["provenance"]["source_sha256"] == cold.SOURCE
    old = cold.protocol()
    previous = read(cold.BASE / "independent_audit.json")
    previous_manifest = read(cold.BASE / "process.json")
    assert previous_manifest["status"] == "complete_awaiting_audit"
    assert len(previous_manifest["jobs"]) == 9
    assert all(
        r["status"] == "complete" and r["returncode"] == 0
        for r in previous_manifest["jobs"].values()
    )
    cold.validate_schedule(previous_manifest["jobs"], old["cases"], old["parallelism"])
    assert previous["status"] == "passed"
    assert previous["protocol_sha256"] == cold.PROTOCOL_SHA
    assert previous["auditor_sha256"] == sha("scripts/audit_cold_ppo_screen.py")
    assert previous["summary_sha256"] == sha(cold.BASE / "summary.json")
    decision = cold.independent_decision(previous["endpoints"], previous["paired"])
    assert decision == previous["decision"]
    assert decision == read(cold.BASE / "summary.json")["provisional_decision"]
    arm = decision["selected_arm"]
    assert arm in ("warm", "cold") and decision["public_qualified"][arm]
    assert p["development"] == {
        "selected_arm": arm,
        "canonical_case": f"seed{SEEDS[0]}_{arm}",
        "audit_sha256": sha(cold.BASE / "independent_audit.json"),
        "summary_sha256": sha(cold.BASE / "summary.json"),
    }
    cases = [f"seed{seed}_{arm}" for seed in SEEDS]
    assert set(p["models"]) == set(cases)
    assert p["jobs"] == {
        f"{case}__{variant}": {
            "case": case,
            "opponent": f"rl-{variant}-assisted",
            "variant": variant,
        }
        for case in cases
        for variant in ("wp", "point")
    }
    for path, digest in p["bindings"].items():
        assert sha(path) == digest
    assert p["seed_scan"]["range"] == [FIRST, FIRST + PAIRS - 1]
    assert p["seed_scan"]["overlaps"] == []
    for path, digest in p["seed_scan"]["reports"].items():
        assert sha(path) == digest
        r = json.loads(Path(path).read_bytes())
        assert_no_reserved_deals(r)
    checkpoints = {}
    for case in cases:
        path = cold.BASE / case / "update_00000520.pt"
        rows = cold.metrics(cold.BASE / case / "metrics.jsonl", final=True)
        assert len(rows) == 520
        payload, check = cold.checkpoint(path, old["cases"][case], 520, rows, old)
        assert check == previous["checkpoints"][case + ":520"]
        assert p["models"][case] == {
            "checkpoint": str(path),
            **check,
            "config": payload["config"],
            "training_provenance": payload["provenance"],
        }
        checkpoints[case] = (payload, check)
    upstream = Path("external/KoiKoi-AI")
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=upstream, text=True
        ).strip()
        == cold.UPSTREAM
    )
    assert not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=upstream,
        text=True,
    ).strip()
    assert set(p["comparator_hashes"]) == {"wp", "point"}
    for variant in ("wp", "point"):
        assert set(p["comparator_hashes"][variant]) == {"discard", "pick", "koikoi"}
        for name, digest in p["comparator_hashes"][variant].items():
            assert sha(upstream / f"model_agent/{name}_rl_{variant}.pt") == digest
    assert p["comparator_hashes"]["wp"] == old["comparator_hashes"]
    return p, checkpoints


def report(path, job, payload, check, model_path):
    r = read(path)
    expected = {
        "protocol_version": 2,
        "profile": "paper8",
        "split": "test",
        "opponent": job["opponent"],
        "seed_start": FIRST,
        "pairs": PAIRS,
        "matches": 2 * PAIRS,
        "device": "cpu",
        "deterministic": True,
        "endgame": True,
        "opponent_endgame": True,
        "search": "none",
        "koi_agent": None,
        "upstream_revision": cold.UPSTREAM,
    }
    for key, value in expected.items():
        assert r[key] == value
    assert Path(r["agent"]) == Path(model_path)
    assert r["provenance"]["source_sha256"] == cold.SOURCE
    assert set(r["checkpoints"]) == {r["agent"]}
    saved = r["checkpoints"][r["agent"]]
    assert saved["sha256"] == check["sha256"]
    for key in ("config", "counters", "provenance"):
        assert saved["training_" + key] == payload[key]
    for key in (
        "search_simulated_decisions",
        "search_executed_decisions",
        "search_preparation_decisions",
        "searched_decisions",
        "changed_decisions",
    ):
        assert r[key] == 0
    decisions = 0
    for side in ("agent_decisions", "opponent_decisions"):
        counts = r[side]
        assert all(isinstance(value, int) and value >= 0 for value in counts.values())
        assert counts["continued"] + counts["stopped"] == counts["koi"]
        decisions += sum(
            counts[key] for key in ("discard", "pick_hand", "pick_draw", "koi")
        )
    assert decisions == r["decisions"]
    scores, points = cold.raw_pairs(r, first=FIRST, count=PAIRS)
    stats = statistics(scores, points)
    for key, field in (
        ("score_mean", "draw_adjusted_win_rate"),
        ("score_ci95", "win_rate_ci95"),
        ("point_mean", "mean_point_difference"),
        ("point_ci95", "point_difference_ci95"),
    ):
        close(stats[key], r[field])
    return stats | {"report_sha256": sha(path), "decisions": decisions}


def schedule(manifest, jobs, prepared_at):
    assert manifest["status"] == "complete_awaiting_audit"
    assert set(manifest["jobs"]) == set(jobs) and len(jobs) == 4
    start, end = manifest["started"], manifest["finished"]
    assert math.isfinite(start) and math.isfinite(end) and prepared_at <= start <= end
    events = []
    for r in manifest["jobs"].values():
        assert r["status"] == "complete" and r["returncode"] == 0
        assert r["resource"] == "cpu" and r["cpu_threads"] == 2
        assert start <= r["started"] < r["finished"] <= end
        assert isinstance(r["pid"], int) and r["pid"] > 0
        events += [(r["started"], 1), (r["finished"], -1)]
    active = 0
    for _, delta in sorted(events):
        active += delta
        assert 0 <= active <= 4
    assert active == 0


def conclusions(endpoints, jobs):
    bounds = {
        variant: [
            endpoints[name]["score_ci95"][0]
            for name, job in jobs.items()
            if job["variant"] == variant
        ]
        for variant in ("wp", "point")
    }
    assert len(bounds["wp"]) == len(bounds["point"]) == 2
    return {
        "primary_wp_replicated": min(bounds["wp"]) > 0.5,
        "secondary_point_replicated": min(bounds["point"]) > 0.5,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-only", action="store_true")
    args = parser.parse_args()
    p, checkpoints = protocol()
    if args.protocol_only:
        print(
            {
                "status": "passed",
                "scope": "frozen protocol, dependencies and selected models",
            }
        )
        return
    manifest = read(BASE / "process.json")
    assert manifest["protocol_sha256"] == sha(BASE / "protocol.json")
    schedule(manifest, p["jobs"], p["prepared_at"])
    summary = read(BASE / "summary.json")
    assert summary["protocol_sha256"] == sha(BASE / "protocol.json")
    assert set(summary["endpoints"]) == set(p["jobs"])
    endpoints = {}
    for name, job in p["jobs"].items():
        payload, check = checkpoints[job["case"]]
        stats = report(
            BASE / name / "evaluation.json",
            job,
            payload,
            check,
            p["models"][job["case"]]["checkpoint"],
        )
        saved = summary["endpoints"][name]
        assert saved["report_sha256"] == stats["report_sha256"]
        for key, field in (
            ("score_mean", "win_rate"),
            ("score_ci95", "win_ci95"),
            ("point_mean", "point_difference"),
            ("point_ci95", "point_ci95"),
        ):
            close(stats[key], saved[field])
        endpoints[name] = stats
    result = conclusions(endpoints, p["jobs"])
    assert result == summary["provisional_conclusions"]
    output = {
        "status": "passed",
        "scope": "frozen identities, accounting, recorded work, raw pairs and independently recomputed statistics; no trajectory/inference replay",
        "protocol_sha256": sha(BASE / "protocol.json"),
        "summary_sha256": sha(BASE / "summary.json"),
        "auditor_sha256": sha(__file__),
        "development_auditor_sha256": sha("scripts/audit_cold_ppo_screen.py"),
        "statistics_helper_sha256": sha("scripts/audit_round_continuation_screen.py"),
        "endpoints": endpoints,
        "conclusions": result,
        "canonical_case": p["development"]["canonical_case"],
        "played_matches": 80000,
        "unique_deal_seeds": PAIRS,
    }
    write(BASE / "independent_audit.json", output)
    print({"status": "passed", "conclusions": result})


if __name__ == "__main__":
    main()
