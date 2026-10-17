"""Independent raw-record/checkpoint audit of the fixed WP50 screen.

Does not import runner, evaluation, comparison, training or batch-teacher helpers.
No inference/strength evaluation is performed. --checkpoints-only validates only
already completed update500 artifacts, without selecting intermediate models.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

BASE = Path("runs/published_wp50_screen")
PROTOCOL_SHA = "9409200f1b6990b1c0d31087ee1967f40c586c11a806566fd2773d381f59ce5f"
SOURCE_SHA = "1c36895f2b5164c4c0a86862e2270364a74c2ff0ee4d950f4b0c4cd970575914"
RUNNER_SHA = "c5b7c63a5881222ce3b896b1c3dd7e585d3d9970cb9deb311c882ad9f13422d2"
UPSTREAM_REV = "89d789565c7c36c84b74662e43001d960bf80901"
TEACHER_PARAMETERS = "7d684d9063b8c4bbe406c7b50f9b43ac90bb774f5917b13522d28be1a81925f4"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def interval(values):
    """Independent scalar-replicate bootstrap and explicit linear quantiles."""
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and len(values) >= 2 and np.isfinite(values).all()
    rng = np.random.default_rng(99)
    draws = sorted(
        float(values[rng.integers(0, len(values), len(values))].sum() / len(values))
        for _ in range(5000)
    )
    result = []
    for p in (0.025, 0.975):
        at = 4999 * p
        index = math.floor(at)
        result.append(draws[index] + (at - index) * (draws[index + 1] - draws[index]))
    return result


def raw_pairs(report, count=1000):
    rows = report["records"]
    assert len(rows) == report["pairs"] == count
    ordered = sorted(rows, key=lambda row: row["seed"])
    assert [r["seed"] for r in ordered] == list(range(1000000, 1000000 + count))
    scores, points = [], []
    for row in ordered:
        for seat in (0, 1):
            score = row[f"seat{seat}_score"]
            point = row[f"seat{seat}_point_difference"]
            assert score in (0, 0.5, 1) and math.isfinite(point)
            assert point == int(point) and int(point) % 2 == 0
            assert score == (1 if point > 0 else 0 if point < 0 else 0.5)
        scores.append((row["seat0_score"] + row["seat1_score"]) / 2)
        points.append(
            (row["seat0_point_difference"] + row["seat1_point_difference"]) / 2
        )
    return np.array(scores), np.array(points)


def statistics(scores, points):
    return {
        "score_mean": float(scores.sum() / len(scores)),
        "score_ci95": interval(scores),
        "point_mean": float(points.sum() / len(points)),
        "point_ci95": interval(points),
    }


def verify_report(path, checkpoint_path, expected_hash, payload=None):
    report = read(path)
    for key, value in {
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
        "search": "none",
        "search_simulated_decisions": 0,
        "upstream_revision": UPSTREAM_REV,
    }.items():
        assert report[key] == value, (path, key)
    # Old pinned no-search parent reports predate executed/preparation counters.
    for key in ("search_executed_decisions", "search_preparation_decisions"):
        assert (report[key] if payload is not None else report.get(key, 0)) == 0
    decisions = 0
    for side in ("agent_decisions", "opponent_decisions"):
        phase = report[side]
        decisions += sum(
            phase[key] for key in ("discard", "pick_hand", "pick_draw", "koi")
        )
        assert phase["continued"] + phase["stopped"] == phase["koi"]
    assert decisions == report["decisions"]
    assert report.get("opponent_endgame", True)
    assert report.get("koi_agent") is None
    assert Path(report["agent"]) == Path(checkpoint_path)
    entry = report["checkpoints"][report["agent"]]
    assert entry["sha256"] == expected_hash
    if payload is not None:
        assert entry["training_counters"] == payload["counters"]
        assert entry["training_config"] == payload["config"]
        assert entry["training_provenance"] == payload["provenance"]
        assert report["provenance"]["source_sha256"] == SOURCE_SHA
    scores, points = raw_pairs(report)
    result = statistics(scores, points)
    close(result["score_mean"], report["draw_adjusted_win_rate"])
    close(result["score_ci95"], report["win_rate_ci95"])
    close(result["point_mean"], report["mean_point_difference"])
    close(result["point_ci95"], report["point_difference_ci95"])
    return (scores, points), {
        **result,
        "report_sha256": sha(path),
        "checkpoint_sha256": expected_hash,
    }


def verify_protocol():
    assert sha(BASE / "protocol.json") == PROTOCOL_SHA
    protocol = read(BASE / "protocol.json")
    assert (
        sha("scripts/run_published_screen.py")
        == protocol["script_sha256"]
        == RUNNER_SHA
    )
    digest = hashlib.sha256()
    for path in sorted(Path("koikoi").rglob("*")):
        if path.suffix in (".py", ".h", ".cpp", ".cu"):
            digest.update(path.as_posix().encode())
            digest.update(path.read_bytes())
    assert digest.hexdigest() == protocol["provenance"]["source_sha256"] == SOURCE_SHA
    assert protocol["updates"] == 500
    assert protocol["simulator_decisions_per_case"] == 131072000
    assert sha("configs/ppo8_published_wp50.json") == protocol["config_sha256"]
    expected = [
        f"{seed}_{arm}"
        for seed in ("seed1", "seed10001")
        for arm in ("wp50", "control")
    ]
    assert list(protocol["cases"]) == expected
    for case, record in protocol["cases"].items():
        assert sha(record["parent"]) == record["parent_sha256"]
        cfg = record["config"]
        fixed = {
            "algorithm": "ppo",
            "device": "cuda",
            "profile": "paper8",
            "envs": 2048,
            "horizon": 128,
            "kind": "mlp",
            "width": 128,
            "layers": 2,
            "lr": 0.0001,
            "epochs": 3,
            "minibatch": 8192,
            "clip": 0.2,
            "entropy": 0.02,
            "value_weight": 0.5,
            "belief_weight": 0.0,
            "gae_lambda": 0.95,
            "reward": "match",
            "cpu_threads": 2,
            "precision": "float32",
            "returns_backend": "native",
            "endgame": True,
            "cuda_graph": True,
            "cuda_rollout": False,
            "ppo_value": "shared",
            "policy_scope": "all",
            "koi_width": 0,
            "deterministic_cards": False,
            "seed": 6300001 if case.startswith("seed1_") else 6310001,
            "opponent_fraction": 0.5 if case.endswith("wp50") else 0.0,
            "published_opponent": "wp" if case.endswith("wp50") else "",
        }
        for field, value in fixed.items():
            assert cfg[field] == value, (case, field)
    return protocol


def parameter_hash(weights):
    digest = hashlib.sha256()
    for phase in sorted(weights):
        for name in sorted(weights[phase]):
            tensor = weights[phase][name].detach().cpu().contiguous()
            label = f"{phase}:{name}:{tensor.dtype}:{tuple(tensor.shape)}"
            digest.update(label.encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def teacher_weights(protocol):
    # Share only the legacy-pickle compatibility loader, never inference helpers.
    from koikoi.upstream import UPSTREAM_PATH, LegacyPickle, load_source

    load_source()
    result = {}
    for phase, name in ((0, "discard"), (1, "pick"), (3, "koikoi")):
        path = UPSTREAM_PATH / f"model_agent/{name}_rl_wp.pt"
        assert sha(path) == protocol["teacher_hashes"][name]
        result[phase] = torch.load(
            path, map_location="cpu", weights_only=False, pickle_module=LegacyPickle
        ).state_dict()
    assert parameter_hash(result) == TEACHER_PARAMETERS
    return result


def verify_checkpoint(case, record, teacher, protocol):
    folder = BASE / case
    path = folder / "update_00000500.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["format"] == 1
    cfg, counts, prov = payload["config"], payload["counters"], payload["provenance"]
    assert cfg == record["config"] == read(folder / "config.json")
    assert prov["source_sha256"] == SOURCE_SHA
    parent = torch.load(record["parent"], map_location="cpu", weights_only=False)
    assert prov["initialized_from"] == {
        "sha256": record["parent_sha256"],
        "profile": "paper8",
        "counters": parent["counters"],
        "provenance": parent["provenance"],
    }
    assert counts["updates"] == 500 and counts["decisions"] == 131072000
    assert all(type(n) is int and n >= 0 for n in counts.values())
    state = payload["env"]
    assert state.device.type == "cpu" and state.dtype == torch.int32
    assert state.shape == (2048, 208)
    # Native ABI: round103, rounds104, bankruptcy105, seed110, episode111.
    assert (state[:, 104] == 8).all() and (state[:, 105] == 1).all()
    assert ((state[:, 103] >= 0) & (state[:, 103] < 8)).all()
    assert ((state[:, 99] >= 0) & (state[:, 99] < 4)).all()
    assert torch.equal(
        state[:, 110], torch.arange(cfg["seed"], cfg["seed"] + 2048, dtype=torch.int32)
    )
    assert int(state[:, 111].sum()) == counts["matches"]
    completed_rounds = counts["rounds"] - int(state[:, 103].sum())
    assert counts["matches"] <= completed_rounds <= 8 * counts["matches"]
    expected_keys = {"updates", "decisions", "rounds", "matches"}
    parameters = list(payload["model"].values())
    assert payload["model"].keys() == parent["model"].keys()
    assert any(
        not torch.equal(v, parent["model"][k]) for k, v in payload["model"].items()
    )
    if case.endswith("wp50"):
        expected_keys |= {
            "learner_decisions",
            "published_decisions",
            "published_rounds",
            "published_matches",
        }
        assert (
            counts["learner_decisions"] + counts["published_decisions"]
            == counts["decisions"]
        )
        assert 0 < counts["published_decisions"] < counts["decisions"] // 2
        assert int(state[:1024, 111].sum()) == counts["published_matches"]
        completed_teacher_rounds = counts["published_rounds"] - int(
            state[:1024, 103].sum()
        )
        assert (
            counts["published_matches"]
            <= completed_teacher_rounds
            <= 8 * counts["published_matches"]
        )
        assert counts["published_matches"] <= counts["matches"]
        assert counts["published_rounds"] <= counts["rounds"]
        snapshot = payload["published_opponent"]
        agent = snapshot["agent"]
        assert agent["variant"] == "wp" and agent["endgame"]
        assert agent["upstream_revision"] == UPSTREAM_REV
        assert agent["model_hashes"] == protocol["teacher_hashes"]
        assert (
            parameter_hash(agent["weights"])
            == agent["parameters_sha256"]
            == TEACHER_PARAMETERS
        )
        assert agent["weights"].keys() == teacher.keys()
        for phase in teacher:
            assert agent["weights"][phase].keys() == teacher[phase].keys()
            for name, value in teacher[phase].items():
                assert torch.equal(value, agent["weights"][phase][name])
        identity = prov["frozen_opponent"]
        for key in agent:
            if key != "weights":
                assert agent[key] == identity[key]
        assert identity["kind"] == "published-batch-v1"
        assert identity["selection"] == "deterministic_argmax"
        assert identity["grouping"] == "all_current_opponent_rows_by_phase"
        assert identity["frozen_envs"] == 1024
        runtime = snapshot["runtime"]
        assert runtime == identity["runtime"]
        assert runtime["device"] == "cuda:0" and runtime["cpu_threads"] == 2
        assert (
            not runtime["matmul_tf32"]
            and not runtime["cudnn_tf32"]
            and not runtime["cudnn_benchmark"]
        )
        assert runtime["torch"] == prov["torch"] and runtime["gpu"] == prov["gpu"]
        parameters += list(payload["opponent_value"].values())
    else:
        assert (
            not {"published_opponent", "frozen_model", "opponent_value"}
            & payload.keys()
        )
        assert "frozen_opponent" not in prov
    assert set(counts) == expected_keys
    assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in parameters)
    optimizer = payload["optimizer"]
    assert len(optimizer["param_groups"]) == 1
    group = optimizer["param_groups"][0]
    assert group["lr"] == 0.0001 and group["eps"] == 1e-5 and group["capturable"]
    assert len(group["params"]) == len(set(group["params"])) == len(parameters)
    assert set(group["params"]) == set(optimizer["state"])
    for key, parameter in zip(group["params"], parameters):
        opt = optimizer["state"][key]
        assert float(opt["step"]) == 500 * 3 * (2048 * 128 // 8192)
        for name in ("exp_avg", "exp_avg_sq"):
            assert (
                opt[name].shape == parameter.shape and torch.isfinite(opt[name]).all()
            )
        assert (opt["exp_avg_sq"] >= 0).all()
    logs = [
        json.loads(line)
        for line in (folder / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(logs) == 500
    prior = {key: 0 for key in counts}
    for update, row in enumerate(logs, 1):
        assert row["updates"] == update and row["decisions"] == update * 2048 * 128
        assert all(math.isfinite(value) for value in row.values())
        for key in counts:
            assert type(row[key]) is int and row[key] >= prior[key]
        if case.endswith("wp50"):
            assert (
                row["learner_decisions"] + row["published_decisions"]
                == row["decisions"]
            )
        prior = row
    assert {key: logs[-1][key] for key in counts} == counts
    return payload, {
        "checkpoint_sha256": sha(path),
        "metrics_sha256": sha(folder / "metrics.jsonl"),
        "counters": counts,
        "model_changed_from_parent": True,
        "adam_steps": 48000,
    }


def check_summary_difference(actual, stored):
    assert stored["shared_pairs"] == 1000
    assert stored["seeds"] == list(range(1000000, 1001000))
    close(actual["score_mean"], stored["score_difference_first_minus_second"])
    close(actual["score_ci95"], stored["score_difference_ci95"])
    close(actual["point_mean"], stored["point_difference_first_minus_second"])
    close(actual["point_ci95"], stored["point_difference_ci95"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    protocol = verify_protocol()
    teacher = teacher_weights(protocol)
    checkpoints, raw, results = {}, {}, {}
    for case, record in protocol["cases"].items():
        path = BASE / case / "update_00000500.pt"
        if args.checkpoints_only and not path.exists():
            continue
        payload, info = verify_checkpoint(case, record, teacher, protocol)
        checkpoints[case] = info
        if not args.checkpoints_only:
            raw[case], results[case] = verify_report(
                BASE / case / "evaluation.json",
                path,
                info["checkpoint_sha256"],
                payload,
            )
    audit = {
        "auditor_sha256": sha(__file__),
        "protocol_sha256": PROTOCOL_SHA,
        "production_source_sha256": SOURCE_SHA,
        "checkpoints": checkpoints,
        "scope": "Independent stored-state/budget/Adam/teacher-weight checks; raw paired-outcome arithmetic only in full mode. Does not replay all training or evaluation trajectories. Legacy upstream pickle loading is shared solely to read original teacher tensors.",
        "paired_outcomes_checked": not args.checkpoints_only,
    }
    if args.checkpoints_only:
        audit.update(
            status="checkpoint_checks_only",
            complete_cases=list(checkpoints),
            pending_cases=sorted(set(protocol["cases"]) - set(checkpoints)),
        )
        out = BASE / f"checkpoint_audit_{len(checkpoints)}.json"
    else:
        manifest = read(BASE / "process.json")
        assert manifest["status"] == "complete_awaiting_audit"
        assert [job["job"] for job in manifest["jobs"]] == [
            f"{stage}:{case}"
            for stage in ("train", "evaluate")
            for case in protocol["cases"]
        ]
        assert all(
            job["status"] == "complete" and job["returncode"] == 0
            for job in manifest["jobs"]
        )
        summary = read(BASE / "summary.json")
        assert summary["protocol_sha256"] == PROTOCOL_SHA
        for label in ("seed1", "seed10001"):
            parent_path = protocol["cases"][label + "_wp50"]["parent"]
            baseline_path = Path(parent_path).parent / "eval_5000_wp.json"
            assert sha(baseline_path) == protocol["parent_baseline_hashes"][label]
            parent_raw, parent_result = verify_report(
                baseline_path,
                parent_path,
                protocol["cases"][label + "_wp50"]["parent_sha256"],
            )
            results[label + "_parent"] = parent_result
            treatment, control = raw[label + "_wp50"], raw[label + "_control"]
            differences = {}
            for name, a, b in (
                ("wp50_vs_control", treatment, control),
                ("wp50_vs_parent", treatment, parent_raw),
                ("control_vs_parent", control, parent_raw),
            ):
                differences[name] = statistics(a[0] - b[0], a[1] - b[1])
                check_summary_difference(
                    differences[name], summary["results"][label][name]
                )
            stored = summary["results"][label]
            close(stored["wp50_win_rate"], results[label + "_wp50"]["score_mean"])
            close(stored["wp50_win_ci95"], results[label + "_wp50"]["score_ci95"])
            assert (
                stored["treatment_report_sha256"]
                == results[label + "_wp50"]["report_sha256"]
            )
            assert (
                stored["control_report_sha256"]
                == results[label + "_control"]["report_sha256"]
            )
            results[label + "_differences"] = differences
        eligibility = all(
            results[label + "_differences"][comparison]["score_ci95"][0] > 0
            for label in ("seed1", "seed10001")
            for comparison in ("wp50_vs_control", "wp50_vs_parent")
        )
        absolute = all(
            results[label + "_wp50"]["score_ci95"][0] > 0.5
            for label in ("seed1", "seed10001")
        )
        audit.update(
            status="passed",
            results=results,
            paired_eligibility=eligibility,
            replicated_absolute_development_signal=absolute,
            summary_sha256=sha(BASE / "summary.json"),
            limitations="Shared repeatedly explored development deals; two parents are not2000 independent pairs. No independent confirmation, final-test result, deployment adoption or universal strength conclusion follows.",
        )
        out = BASE / "independent_audit.json"
    out.write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "complete_cases": list(checkpoints),
                "output": str(out),
            }
        )
    )


if __name__ == "__main__":
    main()
