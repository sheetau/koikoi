"""Independent checkpoint/raw-record audit of the fixed cold/warm comparison.

Uses earlier independent accounting/statistics primitives, never production
training/evaluation/selection helpers. Does not replay trajectories or inference.
"""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.audit_pretrained_ppo_screen import tensor_digest, validate_accounting
from scripts.audit_round_continuation_screen import close, statistics

BASE = Path("runs/cold_ppo_screen")
PROTOCOL_SHA = "ae81104920a1b31f84c1ee45045c0e984f91540c0b279420c456ab4bcf279950"
RUNNER_SHA = "ab36a65f52b768f01ff068b357a1206c6f5cb1a61e68f870a635b555bfcbd202"
SOURCE = "4b4360ed783c1d12cb5258e255f4b85f2e30dfa2e111fd8a078cf3439ce612cb"
INITIAL = "7d684d9063b8c4bbe406c7b50f9b43ac90bb774f5917b13522d28be1a81925f4"
UPSTREAM = "89d789565c7c36c84b74662e43001d960bf80901"
SEEDS = (6710001, 6720001)
FIRST, PAIRS = 7200000, 5000


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol():
    assert sha(BASE / "protocol.json") == PROTOCOL_SHA
    assert sha("scripts/run_cold_ppo_screen.py") == RUNNER_SHA
    p = read(BASE / "protocol.json")
    assert p["script_sha256"] == RUNNER_SHA
    assert p["helper_sha256"] == sha("scripts/run_pretrained_ppo_screen.py")
    digest = hashlib.sha256()
    for path in sorted(Path("koikoi").rglob("*")):
        if path.suffix in (".py", ".h", ".cpp", ".cu"):
            digest.update(path.as_posix().encode())
            digest.update(path.read_bytes())
    assert digest.hexdigest() == p["provenance"]["source_sha256"] == SOURCE
    assert (
        p["upstream_revision"] == UPSTREAM and p["initial_parameters_sha256"] == INITIAL
    )
    for name, value in p["comparator_hashes"].items():
        assert sha(f"external/KoiKoi-AI/model_agent/{name}_rl_wp.pt") == value
    for path, value in p["dependency_hashes"].items():
        assert sha(path) == value
    assert sha(p["initial_checkpoint"]) == p["initial_checkpoint_sha256"]
    assert sha("runs/cold_ppo_seed_range_check.json") == p["seed_range_check_sha256"]
    assert p["parallelism"] == {"gpu": 1, "cpu": 2}
    assert (
        p["updates"] == 520 and p["warmup_updates"] == 20 and p["policy_updates"] == 500
    )
    assert p["evaluation"]["first_seed"] == FIRST and p["evaluation"]["pairs"] == PAIRS
    assert (
        p["decisions_per_case"] == 8519680 and p["warmup_decisions_per_case"] == 327680
    )
    assert set(p["cases"]) == {
        f"seed{seed}_{arm}" for seed in SEEDS for arm in ("warm", "cold")
    }
    for seed in SEEDS:
        warm, cold = [p["cases"][f"seed{seed}_{arm}"] for arm in ("warm", "cold")]
        assert warm["seed"] == cold["seed"] == seed
        assert warm["score_temperature"] == 0.05 and warm["lr"] == 3e-6
        assert cold["score_temperature"] == 0.01 and cold["lr"] == 6e-7
        assert {
            k: v for k, v in warm.items() if k not in ("lr", "score_temperature")
        } == {k: v for k, v in cold.items() if k not in ("lr", "score_temperature")}
    return p


def checkpoint(path, config, updates, rows, p):
    content = Path(path).read_bytes()
    payload = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
    assert payload["format"] == "published_ppo_v1"
    if config is not None:
        assert payload["config"] == config
    config = payload["config"]
    assert payload["provenance"]["source_sha256"] == SOURCE
    initial, current = payload["actor"]["initialization"], payload["actor"]["current"]
    assert payload["actor"]["format"] == "published_actor_v1"
    assert initial == payload["provenance"]["published_initialization"]
    assert initial["parameters_sha256"] == INITIAL
    for key, value in (
        ("variant", "wp"),
        ("upstream_revision", UPSTREAM),
        ("model_hashes", p["comparator_hashes"]),
    ):
        assert initial[key] == current[key] == value
    assert current["endgame"] is False
    digest = tensor_digest(current["weights"])
    assert digest == current["parameters_sha256"]
    assert (digest == INITIAL) == (updates <= 20)
    assert all(torch.isfinite(t).all() for t in payload["value"].values())
    for key, value in {
        "torch": p["provenance"]["torch"],
        "device": "cuda:0",
        "cpu_threads": 2,
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }.items():
        assert payload["runtime"][key] == value
    validate_accounting(payload, rows, updates)
    groups, state = payload["optimizer"]["param_groups"], payload["optimizer"]["state"]
    assert (
        len(groups) == 2
        and len(groups[0]["params"]) == 90
        and len(groups[1]["params"]) == 6
    )
    assert groups[0]["lr"] == config["lr"] and groups[1]["lr"] == config["value_lr"]
    assert groups[0]["eps"] == groups[1]["eps"] == 1e-5
    ids = groups[0]["params"] + groups[1]["params"]
    assert len(set(ids)) == 96 and set(state) <= set(ids)
    tensors = [
        tensor for phase in (0, 1, 3) for tensor in current["weights"][phase].values()
    ] + list(payload["value"].values())
    assert len(tensors) == 96
    expected = []
    for name in ("discard", "pick", "koi", "value"):
        expected += [payload["counters"]["optimizer_steps_" + name]] * (
            6 if name == "value" else 30
        )
    for index, count, tensor in zip(ids, expected, tensors):
        if not count:
            assert index not in state
        else:
            assert int(state[index]["step"]) == count
            for moment in ("exp_avg", "exp_avg_sq"):
                assert (
                    state[index][moment].shape == tensor.shape
                    and torch.isfinite(state[index][moment]).all()
                )
    return payload, {
        "sha256": hashlib.sha256(content).hexdigest(),
        "parameters_sha256": digest,
        "counters": payload["counters"],
    }


def raw_pairs(r, first=FIRST, count=PAIRS):
    rows = sorted(r["records"], key=lambda row: row["seed"])
    assert len(rows) == r["pairs"] == count
    assert [row["seed"] for row in rows] == list(range(first, first + count))
    scores, points = [], []
    for row in rows:
        for seat in (0, 1):
            score, point = row[f"seat{seat}_score"], row[f"seat{seat}_point_difference"]
            assert math.isfinite(point) and point == int(point) and int(point) % 2 == 0
            assert score == (1 if point > 0 else 0 if point < 0 else 0.5)
        scores.append((row["seat0_score"] + row["seat1_score"]) / 2)
        points.append(
            (row["seat0_point_difference"] + row["seat1_point_difference"]) / 2
        )
    return np.array(scores), np.array(points)


def report(path, checkpoint_path, payload, checkpoint_sha):
    r = read(path)
    for key, value in {
        "protocol_version": 2,
        "profile": "paper8",
        "split": "development",
        "opponent": "rl-wp-assisted",
        "seed_start": FIRST,
        "pairs": PAIRS,
        "matches": 2 * PAIRS,
        "device": "cpu",
        "deterministic": True,
        "endgame": True,
        "opponent_endgame": True,
        "search": "none",
        "koi_agent": None,
        "upstream_revision": UPSTREAM,
    }.items():
        assert r[key] == value
    for key in (
        "search_simulated_decisions",
        "search_executed_decisions",
        "search_preparation_decisions",
    ):
        assert r[key] == 0
    assert r["provenance"]["source_sha256"] == SOURCE
    assert Path(r["agent"]) == checkpoint_path
    saved = r["checkpoints"][r["agent"]]
    assert saved["sha256"] == checkpoint_sha
    for key in ("config", "counters", "provenance"):
        assert saved["training_" + key] == payload[key]
    decisions = 0
    for side in ("agent_decisions", "opponent_decisions"):
        phase = r[side]
        assert phase["continued"] + phase["stopped"] == phase["koi"]
        decisions += sum(
            phase[key] for key in ("discard", "pick_hand", "pick_draw", "koi")
        )
    assert decisions == r["decisions"]
    scores, points = raw_pairs(r)
    values = statistics(scores, points)
    for key, saved_key in (
        ("score_mean", "draw_adjusted_win_rate"),
        ("score_ci95", "win_rate_ci95"),
        ("point_mean", "mean_point_difference"),
        ("point_ci95", "point_difference_ci95"),
    ):
        close(values[key], r[saved_key])
    return scores, points, values | {"report_sha256": sha(path)}


def compare_saved(computed, saved):
    assert saved["shared_pairs"] == PAIRS and saved["seeds"] == list(
        range(FIRST, FIRST + PAIRS)
    )
    for key, field in (
        ("score_mean", "score_difference_first_minus_second"),
        ("score_ci95", "score_difference_ci95"),
        ("point_mean", "point_difference_first_minus_second"),
        ("point_ci95", "point_difference_ci95"),
    ):
        close(computed[key], saved[field])


def independent_decision(endpoints, paired):
    qualified = {
        arm: min(endpoints[f"seed{seed}_{arm}"]["score_ci95"][0] for seed in SEEDS)
        > 0.5
        for arm in ("warm", "cold")
    }
    benefit = min(paired[str(seed)]["score_ci95"][0] for seed in SEEDS) > 0
    selected = None
    if qualified["warm"]:
        selected = "cold" if qualified["cold"] and benefit else "warm"
    elif qualified["cold"]:
        selected = "cold"
    return {
        "public_qualified": qualified,
        "cold_paired_benefit": benefit,
        "selected_arm": selected,
        "canonical_case": None if selected is None else "seed6710001_" + selected,
    }


def metrics(path, final=False):
    if not path.exists():
        return []
    content = path.read_bytes()
    if final:
        assert content.endswith(b"\n")
    return [json.loads(row) for row in content.split(b"\n")[:-1]]


def validate_schedule(jobs, cases, limits):
    expected = {"evaluate:initial"} | {
        f"{stage}:{case}" for stage in ("train", "evaluate") for case in cases
    }
    assert set(jobs) == expected
    events = []
    for name, row in jobs.items():
        resource = "gpu" if name.startswith("train:") else "cpu"
        required = (
            []
            if resource == "gpu" or name == "evaluate:initial"
            else ["evaluate:initial", "train:" + name.split(":")[1]]
        )
        assert (
            row["id"] == name
            and row["resource"] == resource
            and row["requires"] == required
        )
        assert row["status"] == "complete" and row["returncode"] == 0
        assert math.isfinite(row["started"]) and math.isfinite(row["finished"])
        assert row["finished"] >= row["started"]
        assert all(row["started"] >= jobs[dep]["finished"] for dep in required)
        events.extend([(row["started"], 1, resource), (row["finished"], -1, resource)])
    counts = {resource: 0 for resource in limits}
    for _, change, resource in sorted(events):
        counts[resource] += change
        assert 0 <= counts[resource] <= limits[resource]
    assert all(count == 0 for count in counts.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--checkpoints-only", action="store_true")
    args = parser.parse_args()
    p = protocol()
    if args.protocol_only:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "scope": "fixed protocol/source/runner/dependencies",
                }
            )
        )
        return
    manifest = read(BASE / "process.json")
    initial_path = Path(p["initial_checkpoint"])
    initial, initial_check = checkpoint(initial_path, None, 0, [], p)
    checks, endpoints, vectors = {"initial": initial_check}, {}, {}
    initial_done = (
        manifest["jobs"].get("evaluate:initial", {}).get("status") == "complete"
    )
    if initial_done:
        scores, points, stats = report(
            BASE / "initial/evaluation.json",
            initial_path,
            initial,
            initial_check["sha256"],
        )
        assert (scores == 0.5).all() and (points == 0).all()
        endpoints["initial"], vectors["initial"] = stats, (scores, points)
    if not args.checkpoints_only:
        assert initial_done and len(manifest["jobs"]) == 9
        validate_schedule(manifest["jobs"], p["cases"], p["parallelism"])
        assert all(
            job["status"] == "complete" and job["returncode"] == 0
            for job in manifest["jobs"].values()
        )
        summary = read(BASE / "summary.json")
        assert summary["protocol_sha256"] == PROTOCOL_SHA
        assert summary["initial_report_sha256"] == endpoints["initial"]["report_sha256"]
    for case, config in p["cases"].items():
        folder = BASE / case
        rows = metrics(folder / "metrics.jsonl", final=not args.checkpoints_only)
        for updates in (20, 520):
            path = folder / f"update_{updates:08d}.pt"
            if args.checkpoints_only and not path.exists():
                continue
            payload, check = checkpoint(path, config, updates, rows[:updates], p)
            checks[f"{case}:{updates}"] = check
            if args.checkpoints_only or updates != 520:
                continue
            assert len(rows) == 520
            scores, points, stats = report(
                folder / "evaluation.json", path, payload, check["sha256"]
            )
            vectors[case] = (scores, points)
            delta = statistics(
                scores - vectors["initial"][0], points - vectors["initial"][1]
            )
            endpoints[case] = stats | {"versus_initial": delta}
            saved = summary["endpoints"][case]
            assert saved["report_sha256"] == stats["report_sha256"]
            close(saved["win_rate"], stats["score_mean"])
            close(saved["win_ci95"], stats["score_ci95"])
            compare_saved(delta, saved["versus_initial"])
    output = {
        "status": "passed",
        "scope": "stored identities/counts/Adam/tensors and available raw outcomes; no trajectory/inference replay",
        "protocol_sha256": PROTOCOL_SHA,
        "auditor_sha256": sha(__file__),
        "accounting_helper_sha256": sha("scripts/audit_pretrained_ppo_screen.py"),
        "statistics_helper_sha256": sha("scripts/audit_round_continuation_screen.py"),
        "checkpoints": checks,
        "endpoints": endpoints,
    }
    if not args.checkpoints_only:
        paired = {}
        for seed in SEEDS:
            cold, warm = vectors[f"seed{seed}_cold"], vectors[f"seed{seed}_warm"]
            paired[str(seed)] = statistics(cold[0] - warm[0], cold[1] - warm[1])
            compare_saved(paired[str(seed)], summary["paired"][str(seed)])
        selected = independent_decision(endpoints, paired)
        assert selected == summary["provisional_decision"]
        output.update(
            paired=paired, decision=selected, summary_sha256=sha(BASE / "summary.json")
        )
    name = (
        f"checkpoint_audit_{len(checks)}_{int(initial_done)}.json"
        if args.checkpoints_only
        else "independent_audit.json"
    )
    (BASE / name).write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "checkpoints": list(checks),
                "initial_evaluation_audited": initial_done,
                "decision": output.get("decision"),
                "output": str(BASE / name),
            }
        )
    )


if __name__ == "__main__":
    main()
