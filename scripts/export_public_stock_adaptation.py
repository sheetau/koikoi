"""Audit V5A lineage and independently reduce all reserved qualification matches."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts.export_public_stock_results import read, reduce_endpoint, sha, write
from scripts.public_stock_adaptation import (
    EVALUATE,
    PROFILES,
    V3,
    compare,
    training_config,
    validate_evaluation,
    verify_protocol,
)

TRAIN_COUNTS = ("decisions", "rounds", "matches", "updates")
EVAL_COUNTS = ("matches", "rounds", "decisions", "python_replay_decisions", "rollouts", "transitions", "roots")


def same_tree(left, right):
    import torch

    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same_tree(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def audit_training(base, study, label, bindings):
    import torch

    from koikoi.public_stock_residual import load_residual

    torch.set_num_threads(1)
    config = training_config(base, study, label)
    folder = Path(config["out"])
    row = study["systems"][label]
    checkpoint = folder / f"policy_add_{config['updates']:05d}.pt"
    model, payload = load_residual(checkpoint, "cpu")
    old = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
    receipt = read(folder / "config.json")
    if payload["config"] != config or receipt["config"] != config:
        raise ValueError("Wrong final training settings")
    if payload.get("training_method") != "warm_start_fresh_adam_v1":
        raise ValueError("Wrong training method")
    lineage = payload["lineage"]
    if (
        lineage != receipt["lineage"]
        or Path(lineage["checkpoint"]).resolve() != Path(row["checkpoint"]).resolve()
        or lineage["checkpoint_sha256"] != row["checkpoint_sha256"]
        or sha(row["checkpoint"]) != row["checkpoint_sha256"]
        or lineage["prior_counters"] != old["counters"]
        or lineage["receipt"] != V3 or lineage["receipt_sha256"] != sha(V3)
        or payload["parent_sha256"] != old["parent_sha256"]
        or not same_tree(payload["parent"], old["parent"])
    ):
        raise ValueError("Changed initial lineage or frozen teacher")
    if payload["runtime"] != receipt["runtime"] or payload["runtime"]["device"] != config["device"]:
        raise ValueError("Training runtime receipt differs")
    counters = payload["counters"]
    if set(counters) != set(TRAIN_COUNTS) or any(type(v) is not int or v < 0 for v in counters.values()):
        raise ValueError("Invalid training counters")
    if counters["updates"] != config["updates"] or counters["decisions"] != config["updates"] * config["envs"] * config["horizon"]:
        raise ValueError("Wrong additional training budget")
    if payload["cumulative_counters"] != {k: old["counters"][k] + v for k, v in counters.items()}:
        raise ValueError("Additional and cumulative training were conflated")
    correction = model.correction_state()
    if not all(torch.isfinite(value).all() for value in correction.values()):
        raise ValueError("Nonfinite trained parameters")
    if torch.equal(correction["actor.weight"], old["model"]["actor.weight"]):
        raise ValueError("Actor weights did not change")
    if payload["bindings"] != receipt["bindings"]:
        raise ValueError("Training source receipts differ")
    sources = {Path(p).resolve(): h for p, h in study["sources"].items()}
    for path, digest in payload["bindings"].items():
        if sources.get(Path(path).resolve()) != digest or sha(path) != digest:
            raise ValueError(f"Training source changed or unbound: {path}")
        bindings[path] = digest
    rows = [json.loads(line) for line in (folder / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    expected_updates = list(range(config["save_every"], config["updates"] + 1, config["save_every"]))
    if not expected_updates or expected_updates[-1] != config["updates"]:
        expected_updates.append(config["updates"])
    if [r["updates"] for r in rows] != expected_updates:
        raise ValueError("Training checkpoint schedule differs")
    for r in rows:
        if (
            r["decisions"] != r["updates"] * config["envs"] * config["horizon"]
            or r["profile"] != row["profile"] or r["seed"] != config["seed"]
            or not math.isfinite(r["loss"]) or not math.isfinite(r["seconds"]) or r["seconds"] < 0
            or not 0 <= r["matches"] <= r["rounds"] <= r["decisions"]
        ):
            raise ValueError("Invalid training progress record")
        path = folder / f"policy_add_{r['updates']:05d}.pt"
        bindings[str(path)] = sha(path)
    if {k: rows[-1][k] for k in TRAIN_COUNTS} != counters:
        raise ValueError("Final training counters differ from progress")
    for name in ("config.json", "progress.jsonl"):
        bindings[str(folder / name)] = sha(folder / name)
    return {
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "additional": counters, "cumulative": payload["cumulative_counters"],
        "seconds": payload["seconds"], "runtime": payload["runtime"],
        "initial_checkpoint_sha256": row["checkpoint_sha256"],
        "status": "passed", "teacher_unchanged": True, "finite_parameters": True,
    }


def checked_endpoint(folder, study, checkpoint, opponent, profile, bindings):
    scores, result = reduce_endpoint(folder, bindings)
    receipt = read(folder / "protocol.json")
    if any(result.get(k) != v for k, v in receipt.items()):
        raise ValueError("Qualification pre-run receipt differs")
    validate_evaluation(result, study, checkpoint, opponent, profile, folder)
    bindings[str(opponent)] = sha(opponent)
    return scores, result


def reduce_study(base):
    study = read(base / "protocol.json")
    verify_protocol(study)
    bindings = {**study["inputs"], **study["sources"], str(base / "protocol.json"): sha(base / "protocol.json")}
    training = {label: audit_training(base, study, label, bindings) for label in study["systems"]}
    if read(base / "training_audit.json") != {"protocol_sha256": sha(base / "protocol.json"), "systems": training}:
        raise ValueError("Frozen pre-qualification training audit differs")
    bindings[str(base / "training_audit.json")] = sha(base / "training_audit.json")
    data, counts, curves = {}, {}, {}
    environments = set()
    for label, row in study["systems"].items():
        other = study["systems"][f"{row['profile']}_parent{1 - row['parent']}"]["checkpoint"]
        new = training[label]["checkpoint"]
        endpoints = (("new_own", new, row["checkpoint"]), ("new_other", new, other), ("old_other", row["checkpoint"], other))
        scores = []
        for kind, checkpoint, opponent in endpoints:
            folder = base / f"qualification_{label}_{kind}"
            score, result = checked_endpoint(folder, study, checkpoint, opponent, row["profile"], bindings)
            scores.append(score)
            counts[folder.name] = {k: result[k] for k in (*EVAL_COUNTS, "seconds")}
            environments.add((result["torch"], result["gpu"]))
        data[label] = np.asarray(scores)
        curves[label] = dict(zip((x[0] for x in endpoints), data[label].mean(1).tolist(), strict=True))
    if len(environments) != 1:
        raise ValueError("Qualification runtime/device identities differ")
    # Both old-parent orientations must reproduce exactly complementary paired scores.
    for profile in PROFILES:
        if not np.array_equal(data[f"{profile}_parent0"][2] + data[f"{profile}_parent1"][2], np.ones(EVALUATE["pairs"])):
            raise ValueError("Old-parent duplicate-seat baselines are not complementary")
    indices = np.random.default_rng(study["bootstrap_seed"]).integers(
        0, EVALUATE["pairs"], (study["bootstrap_replicates"], EVALUATE["pairs"]),
    )
    contrasts, passed, eligible = compare(data, indices)
    return {
        "method": study["method"], "qualification_duplicate_streams": EVALUATE["pairs"],
        "training": training, "training_totals": {k: sum(r["additional"][k] for r in training.values()) for k in TRAIN_COUNTS},
        "qualification_endpoints": counts, "qualification_totals": {k: sum(r[k] for r in counts.values()) for k in EVAL_COUNTS},
        "scores": curves, "comparisons": contrasts, "qualified_rules": passed,
        "eligible_search_finalists": eligible,
        "comparison_family": {"count": 12, "one_sided_lower_quantile": study["lower_quantile"], "replicates": study["bootstrap_replicates"], "seed": study["bootstrap_seed"]},
        "v4_work_targets_passed": study["v4_work_targets_passed"],
        "scope": "Improvement against both retained opponents only; no new-policy search gain measured. Training counters are recorded, not independently replayed. All scored qualification matches are Python-replayed and outcomes/counts independently reduced.",
        "bindings": bindings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    result = reduce_study(Path(args.study))
    write(args.out, result)
    print(json.dumps({"qualified_rules": result["qualified_rules"], "eligible_search_finalists": result["eligible_search_finalists"]}), flush=True)


if __name__ == "__main__":
    main()
