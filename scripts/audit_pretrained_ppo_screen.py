"""Independent stored-artifact audit; never imports the learner or screen runner.

Statistical primitives are reused from the independently written round-screen
auditor, not from production evaluation/comparison. No trajectory/timing replay.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from scripts.audit_round_continuation_screen import close, raw_pairs, statistics

BASE = Path("runs/pretrained_ppo_screen")
PROTOCOL_SHA = "5823b638c7c0a17246e06b7a1ba6c6d46854147c479739a3e389b90c7c39a718"
RUNNER_SHA = "827ecd17d5728b4ac2e6f1b8a60f1634277a7e133ed4ed930c693014c9e23f7c"
SOURCE = "4b4360ed783c1d12cb5258e255f4b85f2e30dfa2e111fd8a078cf3439ce612cb"
INITIAL = "7d684d9063b8c4bbe406c7b50f9b43ac90bb774f5917b13522d28be1a81925f4"
UPSTREAM = "89d789565c7c36c84b74662e43001d960bf80901"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def protocol():
    assert sha(BASE / "protocol.json") == PROTOCOL_SHA
    assert sha("scripts/run_pretrained_ppo_screen.py") == RUNNER_SHA
    p = read(BASE / "protocol.json")
    digest = hashlib.sha256()
    for path in sorted(Path("koikoi").rglob("*")):
        if path.suffix in (".py", ".h", ".cpp", ".cu"):
            digest.update(path.as_posix().encode())
            digest.update(path.read_bytes())
    assert digest.hexdigest() == SOURCE == p["provenance"]["source_sha256"]
    assert p["script_sha256"] == RUNNER_SHA and p["upstream_revision"] == UPSTREAM
    assert p["initial_parameters_sha256"] == INITIAL
    for name, value in p["comparator_hashes"].items():
        assert sha(f"external/KoiKoi-AI/model_agent/{name}_rl_wp.pt") == value
    assert (
        p["updates"] == 520 and p["policy_updates"] == 500 and p["warmup_updates"] == 20
    )
    assert (
        p["decisions_per_case"] == 8519680 and p["warmup_decisions_per_case"] == 327680
    )
    for seed, config in zip((6681001, 6691001), p["cases"].values()):
        assert config["seed"] == seed and config["critic_warmup_updates"] == 20
        assert config["envs"] == config["horizon"] == 128
        assert config["epochs"] == 3 and config["minibatch"] == 512
    return p


def tensor_digest(weights):
    h = hashlib.sha256()
    for phase in sorted(weights):
        for name in sorted(weights[phase]):
            tensor = weights[phase][name].contiguous()
            assert torch.isfinite(tensor).all()
            h.update(f"{phase}:{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def validate_accounting(payload, rows, updates):
    """Check recorded work against arithmetic and actual simulator episode counters."""
    config, counts, env = payload["config"], payload["counters"], payload["env"]
    n, h, warmup = config["envs"], config["horizon"], config["critic_warmup_updates"]
    steps = math.ceil(n * h / config["minibatch"]) * config["epochs"]
    assert len(rows) == updates == counts["updates"]
    previous = {key: 0 for key in counts}
    warmup_counts = None
    for i, row in enumerate(rows, 1):
        assert all(math.isfinite(value) for value in row.values())
        assert row["updates"] == i and row["decisions"] == i * n * h
        assert row["warmup_updates"] == min(i, warmup)
        assert row["policy_updates"] == max(0, i - warmup)
        assert row["warmup_decisions"] == min(i, warmup) * n * h
        assert row["critic_only"] == (i <= warmup)
        assert row["optimizer_steps_value"] == i * steps
        for key in counts:
            assert type(row[key]) is int and row[key] >= previous[key]
        assert (
            0
            <= row["matches"] - previous["matches"]
            <= row["rounds"] - previous["rounds"]
            <= n * h
        )
        for name in ("discard", "pick", "koi"):
            key = "optimizer_steps_" + name
            assert 0 <= row[key] - previous[key] <= (0 if i <= warmup else steps)
        if i <= warmup:
            warmup_counts = row
        for kind in ("rounds", "matches"):
            assert row["warmup_" + kind] == (
                0 if warmup_counts is None else warmup_counts[kind]
            )
        previous = row
    assert all(counts[k] == previous[k] for k in counts)
    assert env.shape == (n, 208) and env.dtype == torch.int32
    assert env[:, 110].tolist() == list(range(config["seed"], config["seed"] + n))
    assert (env[:, 111] >= 0).all() and int(env[:, 111].sum()) == counts["matches"]
    assert ((env[:, 99] >= 0) & (env[:, 99] <= 3)).all()
    assert ((env[:, 103] >= 0) & (env[:, 103] < 8)).all()
    partial_rounds = int(env[:, 103].sum())
    assert (
        counts["matches"] + partial_rounds
        <= counts["rounds"]
        <= 8 * counts["matches"] + partial_rounds
    )


def checkpoint(path, config, updates, rows, p):
    content = Path(path).read_bytes()
    import io

    payload = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
    assert payload["format"] == "published_ppo_v1" and payload["config"] == config
    assert payload["provenance"]["source_sha256"] == SOURCE
    current, initial = payload["actor"]["current"], payload["actor"]["initialization"]
    assert payload["actor"]["format"] == "published_actor_v1"
    assert initial == payload["provenance"]["published_initialization"]
    assert initial["parameters_sha256"] == INITIAL
    for key, value in (
        ("variant", "wp"),
        ("upstream_revision", UPSTREAM),
        ("model_hashes", p["comparator_hashes"]),
    ):
        assert current[key] == initial[key] == value
    assert current["endgame"] is False
    digest = tensor_digest(current["weights"])
    assert digest == current["parameters_sha256"]
    assert (digest == INITIAL) == (updates <= 20)
    assert all(torch.isfinite(t).all() for t in payload["value"].values())
    runtime = payload["runtime"]
    for key, value in {
        "torch": p["provenance"]["torch"],
        "device": "cuda:0",
        "cpu_threads": 2,
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }.items():
        assert runtime[key] == value
    validate_accounting(payload, rows, updates)
    groups, state = payload["optimizer"]["param_groups"], payload["optimizer"]["state"]
    assert (
        len(groups) == 2
        and len(groups[0]["params"]) == 90
        and len(groups[1]["params"]) == 6
    )
    assert groups[0]["lr"] == 3e-6 and groups[1]["lr"] == 1e-4
    assert groups[0]["eps"] == groups[1]["eps"] == 1e-5
    parameter_ids = groups[0]["params"] + groups[1]["params"]
    assert len(set(parameter_ids)) == 96 and set(state) <= set(parameter_ids)
    expected = []
    for name in ("discard", "pick", "koi", "value"):
        expected += [payload["counters"]["optimizer_steps_" + name]] * (
            6 if name == "value" else 30
        )
    for index, count in zip(parameter_ids, expected):
        if not count:
            assert index not in state
        else:
            assert int(state[index]["step"]) == count
            for key in ("exp_avg", "exp_avg_sq"):
                assert torch.isfinite(state[index][key]).all()
    return payload, {
        "sha256": hashlib.sha256(content).hexdigest(),
        "parameters_sha256": digest,
        "counters": payload["counters"],
    }


def report(path, ckpt_path, payload, ckpt_hash):
    r = read(path)
    expected = {
        "protocol_version": 2,
        "profile": "paper8",
        "split": "development",
        "opponent": "rl-wp-assisted",
        "seed_start": 1000000,
        "pairs": 1000,
        "matches": 2000,
        "device": "cpu",
        "deterministic": True,
        "endgame": True,
        "opponent_endgame": True,
        "search": "none",
        "koi_agent": None,
        "upstream_revision": UPSTREAM,
    }
    for key, value in expected.items():
        assert r[key] == value
    for key in (
        "search_simulated_decisions",
        "search_executed_decisions",
        "search_preparation_decisions",
    ):
        assert r[key] == 0
    assert r["provenance"]["source_sha256"] == SOURCE
    assert Path(r["agent"]) == ckpt_path
    metadata = r["checkpoints"][r["agent"]]
    assert metadata["sha256"] == ckpt_hash
    for short, long in (
        ("config", "training_config"),
        ("provenance", "training_provenance"),
        ("counters", "training_counters"),
    ):
        assert metadata[long] == payload[short]
    decisions = 0
    for side in ("agent_decisions", "opponent_decisions"):
        phases = r[side]
        assert phases["continued"] + phases["stopped"] == phases["koi"]
        decisions += sum(
            phases[k] for k in ("discard", "pick_hand", "pick_draw", "koi")
        )
    assert decisions == r["decisions"]
    scores, points = raw_pairs(r)
    stats = statistics(scores, points)
    for computed, saved in (
        ("score_mean", "draw_adjusted_win_rate"),
        ("score_ci95", "win_rate_ci95"),
        ("point_mean", "mean_point_difference"),
        ("point_ci95", "point_difference_ci95"),
    ):
        close(stats[computed], r[saved])
    return scores, points, stats | {"report_sha256": sha(path)}


def advance_criterion(endpoints, cases):
    return all(
        endpoints[case]["score_ci95"][0] > 0.5
        and endpoints[case]["versus_initial"]["score_ci95"][0] > 0
        for case in cases
    )


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
                    "scope": "fixed protocol/source/runner/teacher identities",
                }
            )
        )
        return
    initial_path = BASE / "initial/initial.pt"
    initial, initial_check = checkpoint(
        initial_path, next(iter(p["cases"].values())), 0, [], p
    )
    checks, endpoints = {"initial": initial_check}, {}
    initial_scores, initial_points, stats = report(
        BASE / "initial/evaluation.json", initial_path, initial, initial_check["sha256"]
    )
    assert (initial_scores == 0.5).all() and (initial_points == 0).all()
    endpoints["initial"] = stats
    summary = None if args.checkpoints_only else read(BASE / "summary.json")
    if summary is not None:
        assert summary["protocol_sha256"] == PROTOCOL_SHA
        assert summary["initial_report_sha256"] == stats["report_sha256"]
    for case, config in p["cases"].items():
        folder = BASE / case
        rows = (
            [
                json.loads(line)
                for line in (folder / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            if (folder / "metrics.jsonl").exists()
            else []
        )
        for updates in (20, 520):
            path = folder / f"update_{updates:08d}.pt"
            if args.checkpoints_only and not path.exists():
                continue
            payload, check = checkpoint(path, config, updates, rows[:updates], p)
            checks[f"{case}:{updates}"] = check
            if updates != 520 or args.checkpoints_only:
                continue
            assert len(rows) == 520
            scores, points, stats = report(
                folder / "evaluation.json", path, payload, check["sha256"]
            )
            delta = statistics(scores - initial_scores, points - initial_points)
            saved = summary["results"][case]
            assert saved["report_sha256"] == stats["report_sha256"]
            close(saved["win_rate"], stats["score_mean"])
            close(saved["win_ci95"], stats["score_ci95"])
            paired = saved["versus_initial"]
            assert paired["shared_pairs"] == 1000 and paired["seeds"] == list(
                range(1000000, 1001000)
            )
            for key, field in (
                ("score_mean", "score_difference_first_minus_second"),
                ("score_ci95", "score_difference_ci95"),
                ("point_mean", "point_difference_first_minus_second"),
                ("point_ci95", "point_difference_ci95"),
            ):
                close(delta[key], paired[field])
            endpoints[case] = stats | {"versus_initial": delta}
    output = {
        "status": "passed",
        "scope": "stored identities/counts/Adam/tensor digests and raw paired statistics; no trajectory replay",
        "protocol_sha256": PROTOCOL_SHA,
        "auditor_sha256": sha(__file__),
        "statistics_helper_sha256": sha("scripts/audit_round_continuation_screen.py"),
        "checkpoints": checks,
        "endpoints": endpoints,
    }
    if not args.checkpoints_only:
        output["replicated_advance_criterion_passed"] = advance_criterion(
            endpoints, p["cases"]
        )
        output["summary_sha256"] = sha(BASE / "summary.json")
    name = (
        f"checkpoint_audit_{len(checks)}.json"
        if args.checkpoints_only
        else "independent_audit.json"
    )
    write(BASE / name, output)
    print(
        json.dumps(
            {
                "status": "passed",
                "checkpoints": list(checks),
                "output": str(BASE / name),
                "criterion": output.get("replicated_advance_criterion_passed"),
            }
        )
    )


if __name__ == "__main__":
    main()
