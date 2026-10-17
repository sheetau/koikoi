"""Independent raw-record/checkpoint audit of the fixed complete-match MC screen.

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

BASE = Path("runs/complete_mc_screen")
PROTOCOL_SHA = "b397ad2258f189b39265e83f88957d5a4ab66a44cddcf48d0adc78dd883b664b"
SOURCE_SHA = "005484abeab7451d383d77ad13f26077d9650564fe5e69e9d0c6188b532f1e91"
RUNNER_SHA = "6085505ed2b323445009f7e20e8860a8874e180fff9a2d6e678da37bd834d36e"
UPSTREAM_REV = "89d789565c7c36c84b74662e43001d960bf80901"


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
        sha("scripts/run_complete_screen.py") == protocol["script_sha256"] == RUNNER_SHA
    )
    digest = hashlib.sha256()
    for path in sorted(Path("koikoi").rglob("*")):
        if path.suffix in (".py", ".h", ".cpp", ".cu"):
            digest.update(path.as_posix().encode())
            digest.update(path.read_bytes())
    assert digest.hexdigest() == protocol["provenance"]["source_sha256"] == SOURCE_SHA
    for name, digest in protocol["comparator_hashes"].items():
        assert (
            sha(Path("external/KoiKoi-AI/model_agent") / f"{name}_rl_wp.pt") == digest
        )
    assert protocol["updates"] == 500
    assert protocol["complete_matches_per_case"] == 1024000
    assert sha("configs/ppo8_complete_mc.json") == protocol["config_sha256"]
    expected = [
        f"{seed}_{arm}" for seed in ("seed1", "seed10001") for arm in ("mc", "control")
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
            "horizon": 512,
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
            "gae_lambda": 1.0 if case.endswith("mc") else 0.95,
            "reward": "match",
            "cpu_threads": 2,
            "precision": "float32",
            "returns_backend": "native",
            "endgame": True,
            "cuda_graph": False,
            "cuda_rollout": False,
            "ppo_value": "shared",
            "policy_scope": "all",
            "koi_width": 0,
            "deterministic_cards": False,
            "seed": 6440001 if case.startswith("seed1_") else 6450001,
            "opponent_fraction": 0.0,
            "published_opponent": "",
            "complete_matches": True,
        }
        for field, value in fixed.items():
            assert cfg[field] == value, (case, field)
    return protocol


def verify_metrics(logs, cfg, counts, expected_updates=500):
    assert len(logs) == expected_updates
    assert set(counts) == {
        "updates",
        "decisions",
        "rounds",
        "matches",
        "collection_rows",
    }
    assert all(type(n) is int and n >= 0 for n in counts.values())
    prior = {key: 0 for key in counts}
    adam_steps = 0
    for update, row in enumerate(logs, 1):
        assert row["updates"] == update and row["matches"] == update * cfg["envs"]
        assert all(math.isfinite(value) for value in row.values())
        for key in counts:
            assert type(row[key]) is int and row[key] >= prior[key]
        delta = {key: row[key] - prior[key] for key in counts}
        assert cfg["envs"] <= delta["rounds"] <= 8 * cfg["envs"]
        assert cfg["envs"] <= delta["decisions"] <= delta["collection_rows"]
        assert delta["collection_rows"] <= cfg["envs"] * cfg["horizon"]
        assert delta["collection_rows"] % cfg["envs"] == 0
        adam_steps += cfg["epochs"] * (
            (delta["decisions"] + cfg["minibatch"] - 1) // cfg["minibatch"]
        )
        prior = row
    assert {key: logs[-1][key] for key in counts} == counts
    return adam_steps


def verify_checkpoint(case, record, protocol):
    from koikoi.reference import ReferenceEnv

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
    assert (
        counts["updates"] == 500
        and counts["matches"] == protocol["complete_matches_per_case"]
    )
    logs = [
        json.loads(line)
        for line in (folder / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    adam_steps = verify_metrics(logs, cfg, counts)
    state = payload["env"]
    assert (
        state.device.type == "cpu"
        and state.dtype == torch.int32
        and state.shape == (2048, 208)
    )
    assert (state[:, 111] == 500).all() and int(state[:, 111].sum()) == counts[
        "matches"
    ]
    assert (
        (state[:, 103] == 0).all()
        and (state[:, 99] == 0).all()
        and (state[:, 98] == 0).all()
    )
    # Reconstruct every newly reset match independently in the Python rules engine.
    reference = np.stack(
        [
            ReferenceEnv(seed, "paper8", 500).state
            for seed in range(cfg["seed"], cfg["seed"] + 2048)
        ]
    )
    assert np.array_equal(state.numpy(), reference)
    assert counts["matches"] <= counts["rounds"] <= 8 * counts["matches"]
    assert payload["model"].keys() == parent["model"].keys()
    assert all(v.shape == parent["model"][k].shape for k, v in payload["model"].items())
    assert any(
        not torch.equal(v, parent["model"][k]) for k, v in payload["model"].items()
    )
    assert (
        not {
            "published_opponent",
            "frozen_model",
            "opponent_value",
            "ppo_critic",
            "critic",
        }
        & payload.keys()
    )
    assert "frozen_opponent" not in prov
    parameters = list(payload["model"].values())
    assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in parameters)
    optimizer = payload["optimizer"]
    assert len(optimizer["param_groups"]) == 1
    group = optimizer["param_groups"][0]
    assert group["lr"] == 0.0001 and group["eps"] == 1e-5 and not group["capturable"]
    assert len(group["params"]) == len(set(group["params"])) == len(parameters)
    assert set(group["params"]) == set(optimizer["state"])
    for key, parameter in zip(group["params"], parameters):
        opt = optimizer["state"][key]
        assert float(opt["step"]) == adam_steps
        for name in ("exp_avg", "exp_avg_sq"):
            assert (
                opt[name].shape == parameter.shape and torch.isfinite(opt[name]).all()
            )
        assert (opt["exp_avg_sq"] >= 0).all()
    return payload, {
        "checkpoint_sha256": sha(path),
        "metrics_sha256": sha(folder / "metrics.jsonl"),
        "counters": counts,
        "model_changed_from_parent": True,
        "adam_steps": adam_steps,
        "all_reset_states_match_python_reference": True,
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
    checkpoints, raw, results = {}, {}, {}
    for case, record in protocol["cases"].items():
        path = BASE / case / "update_00000500.pt"
        if args.checkpoints_only and not path.exists():
            continue
        payload, info = verify_checkpoint(case, record, protocol)
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
        "scope": "Independent stored-state/budget/variable-batch Adam checks and Python reconstruction of all final reset states; raw paired-outcome arithmetic in full mode. Does not replay all training or evaluation trajectories.",
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
            parent_path = protocol["cases"][label + "_mc"]["parent"]
            baseline_path = Path(parent_path).parent / "eval_5000_wp.json"
            assert sha(baseline_path) == protocol["parent_baseline_hashes"][label]
            parent_raw, parent_result = verify_report(
                baseline_path,
                parent_path,
                protocol["cases"][label + "_mc"]["parent_sha256"],
            )
            results[label + "_parent"] = parent_result
            treatment, control = raw[label + "_mc"], raw[label + "_control"]
            differences = {}
            for name, a, b in (
                ("mc_vs_control", treatment, control),
                ("mc_vs_parent", treatment, parent_raw),
                ("control_vs_parent", control, parent_raw),
            ):
                differences[name] = statistics(a[0] - b[0], a[1] - b[1])
                check_summary_difference(
                    differences[name], summary["results"][label][name]
                )
            stored = summary["results"][label]
            close(stored["mc_win_rate"], results[label + "_mc"]["score_mean"])
            close(stored["mc_win_ci95"], results[label + "_mc"]["score_ci95"])
            assert (
                stored["treatment_report_sha256"]
                == results[label + "_mc"]["report_sha256"]
            )
            assert (
                stored["control_report_sha256"]
                == results[label + "_control"]["report_sha256"]
            )
            results[label + "_differences"] = differences
        eligibility = all(
            results[label + "_differences"][comparison]["score_ci95"][0] > 0
            for label in ("seed1", "seed10001")
            for comparison in ("mc_vs_control", "mc_vs_parent")
        )
        absolute = all(
            results[label + "_mc"]["score_ci95"][0] > 0.5
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
