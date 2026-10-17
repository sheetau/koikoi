"""Matched weight-decay contrast on the fixed larger-data validation cohort."""

import argparse
import json
from pathlib import Path

import torch

from koikoi.arbitration import read_checkpoint
from scripts.compare_score_training_budgets import evaluate, paired_change
from scripts.paired_rollout_targets import match_split
from scripts.run_arbitration_study import sha


def matched_settings(regularized, unregularized):
    for key in ("format", "network_config", "dataset", "counters", "feature_cache"):
        if regularized.get(key) != unregularized.get(key):
            raise ValueError(f"Endpoint settings differ: {key}")
    left, right = (
        dict(regularized["training_config"]),
        dict(unregularized["training_config"]),
    )
    if left.pop("weight_decay") != 1e-4 or right.pop("weight_decay") != 0:
        raise ValueError("Expected weight decay 1e-4 versus zero")
    if left != right:
        raise ValueError("Other training settings differ")


def compare(regularized, unregularized, data_base, output):
    if output.exists():
        raise ValueError("Use a fresh output")
    torch.set_num_threads(2)
    inputs, endpoints, contrasts = {}, {}, {}
    for path in (
        Path(__file__),
        Path("scripts/compare_score_training_budgets.py"),
        Path("scripts/diagnose_score_targets.py"),
        Path("scripts/diagnose_paired_targets.py"),
        Path("scripts/train_score_distribution.py"),
        Path("scripts/feature_score_selector.py"),
    ):
        inputs[str(path)] = sha(path)
    for base in (regularized, unregularized):
        process = base / "process.json"
        if json.loads(process.read_text(encoding="utf-8"))["status"] != "complete":
            raise ValueError("Both conditions must be complete")
        inputs[str(process)] = sha(process)
    audit_path = data_base / "features" / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["status"] != "passed":
        raise ValueError("Feature audit must pass")
    inputs[str(audit_path)] = sha(audit_path)
    for parent in (1, 2):
        dataset = data_base / "data" / f"parent{parent}" / "targets.pt"
        cache_path = data_base / "features" / f"parent{parent}.pt"
        data, digest = read_checkpoint(dataset)
        cache, cache_digest = read_checkpoint(cache_path)
        if (
            audit["parents"][str(parent)]["dataset_sha256"] != digest
            or audit["parents"][str(parent)]["cache_sha256"] != cache_digest
        ):
            raise ValueError("Feature audit identities differ")
        inputs.update({str(dataset): digest, str(cache_path): cache_digest})
        groups = data["match_groups"][match_split(data["match_groups"])].numpy()
        for representation in ("raw", "features"):
            for kind in ("paired_win", "preference"):
                name = f"parent{parent}_{representation}_{kind}"
                paths = [
                    base / name / "final.pt" for base in (regularized, unregularized)
                ]
                fitted = [read_checkpoint(path)[0] for path in paths]
                matched_settings(*fitted)
                samples = []
                for condition, path, payload in zip(
                    ("regularized", "zero_decay"), paths, fitted, strict=True
                ):
                    report, wins, points = evaluate(
                        path, data, digest, cache, cache_digest
                    )
                    if (
                        report["training_dataset_roots"] != 262144
                        or report["epochs"] != 20
                    ):
                        raise ValueError("Unexpected data/update budget")
                    report["weight_decay"] = payload["training_config"]["weight_decay"]
                    report["first_layer_max_abs"] = float(
                        payload["model"]["body.0.weight"].double().abs().max()
                    )
                    endpoints[name + "/" + condition] = report
                    inputs[str(path)] = report["checkpoint_sha256"]
                    samples.append((wins, points))
                contrasts[name] = paired_change(samples[1], samples[0], groups)
        del data, cache
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError(f"Comparison input changed: {path}")
    result = {
        "scope": "Zero-decay minus default-decay on identical development roots. One changed action with frozen continuation; not repeated-policy strength. All eight contrasts retained.",
        "inputs": inputs,
        "endpoints": endpoints,
        "zero_minus_regularized": contrasts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regularized", type=Path, default=Path("runs/score_scaled_fit_screen")
    )
    parser.add_argument(
        "--unregularized", type=Path, default=Path("runs/score_zero_decay_control")
    )
    parser.add_argument(
        "--data-base", type=Path, default=Path("runs/scaled_paired_targets_recovered")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.regularized, args.unregularized, args.data_base, args.output)
    print({"endpoints": len(result["endpoints"]), "sha256": sha(args.output)})


if __name__ == "__main__":
    main()
