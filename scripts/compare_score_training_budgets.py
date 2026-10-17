"""Compare fixed training budgets on identical held-out source-match roots."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.arbitration import read_checkpoint
from scripts.analyze_arbitration_study import tensor_equal
from scripts.diagnose_arbitration import proposal_mask
from scripts.diagnose_paired_targets import cluster_interval, selected_point
from scripts.diagnose_score_targets import decision_summary
from scripts.feature_score_selector import FeatureScoreSelector, pack_features
from scripts.paired_rollout_targets import match_split
from scripts.run_arbitration_study import sha
from scripts.train_score_distribution import ScoreSelector, validate_roots


def disjoint(training_groups, evaluation_groups):
    left = set(map(tuple, training_groups.tolist()))
    right = set(map(tuple, evaluation_groups.tolist()))
    if left & right:
        raise ValueError("Evaluation source matches overlap training")


@torch.no_grad()
def evaluate(checkpoint, data, dataset_digest, cache=None, cache_digest=None):
    fitted, checkpoint_digest = read_checkpoint(checkpoint)
    if fitted["format"] not in (
        "paired_score_selector_v1",
        "paired_feature_score_selector_v1",
    ):
        raise ValueError("Unexpected selector checkpoint")
    training, training_digest = read_checkpoint(fitted["dataset"]["path"])
    if training_digest != fitted["dataset"]["sha256"]:
        raise ValueError("Training dataset changed")
    if (
        not tensor_equal(fitted["experts"], data["experts"])
        or fitted["parent"]["sha256"] != data["parent"]["sha256"]
    ):
        raise ValueError("Evaluation expert identities differ")
    if fitted["provenance"]["source_sha256"] != data["provenance"]["source_sha256"]:
        raise ValueError("Evaluation production source differs")
    valid = match_split(data["match_groups"])
    train = ~match_split(training["match_groups"])
    if not valid.any() or not train.any():
        raise ValueError("Need nonempty training and evaluation groups")
    disjoint(training["match_groups"][train], data["match_groups"][valid])
    public = data["public_observations"]
    validate_roots(public)
    config = fitted["network_config"]
    if fitted["format"] == "paired_feature_score_selector_v1":
        if (
            cache is None
            or cache_digest is None
            or cache["dataset_sha256"] != dataset_digest
            or cache["verified_roots"] != len(public)
        ):
            raise ValueError("Evaluation feature cache mismatch")
        public = pack_features(public, cache["embeddings"], cache["scores"])
        model = FeatureScoreSelector(config["kind"], config["width"])
    else:
        model = ScoreSelector(config["kind"], config["width"])
    model.load_state_dict(fitted["model"])
    model.eval()
    choices = []
    for rows in public[valid].split(1024):
        actions = model(rows, proposal_mask(rows[:, :562]))[0].argmax(-1)
        choices.append(selected_point(rows[:, :562], actions))
    chosen = torch.cat(choices).numpy()
    outcomes = data["outcomes"][valid].numpy()
    points = data["point_differences"][valid].numpy()
    groups = data["match_groups"][valid].numpy()
    selected_wins = np.where(chosen, outcomes[:, 1], outcomes[:, 0])
    selected_points = np.where(chosen, points[:, 1], points[:, 0])
    report = {
        "checkpoint_sha256": checkpoint_digest,
        "training_dataset_sha256": training_digest,
        "training_dataset_path": fitted["dataset"]["path"],
        "training_dataset_roots": len(training["public_observations"]),
        "evaluation_dataset_sha256": dataset_digest,
        "evaluation_feature_cache_sha256": cache_digest
        if "feature_cache" in fitted
        else None,
        "kind": config["kind"],
        "representation": "features" if "feature_cache" in fitted else "raw",
        "initial_shared_body_sha256": fitted["training_config"][
            "initial_shared_body_sha256"
        ],
        "minibatch_order_seed": fitted["training_config"]["minibatch_order_seed"],
        "training_roots": int(train.sum()),
        "epochs": fitted["training_config"]["epochs"],
        "optimizer_steps": fitted["counters"]["optimizer_steps"],
        "evaluation_roots": len(chosen),
        "evaluation_source_matches": len(np.unique(groups, axis=0)),
        "training_evaluation_groups_disjoint": True,
        "point_choice_fraction": float(chosen.mean()),
        "single_action_comparisons": decision_summary(chosen, outcomes, points, groups),
    }
    return report, selected_wins, selected_points


def paired_change(left, right, groups):
    wins = (left[0] - right[0]) / 2
    points = left[1] - right[1]
    return {
        "win_score_change": float(wins.mean()),
        "win_source_match_ci95": cluster_interval(wins, groups),
        "point_margin_change": float(points.mean()),
        "point_source_match_ci95": cluster_interval(points, groups),
    }


def compare(large_base, small80, large20, output):
    if output.exists():
        raise ValueError("Use a fresh comparison output")
    torch.set_num_threads(2)
    audit = json.loads(
        (large_base / "features" / "audit.json").read_text(encoding="utf-8")
    )
    if audit["status"] != "passed":
        raise ValueError("Evaluation cache audit must pass")
    reports, contrasts, inputs = {}, {}, {str(Path(__file__)): sha(__file__)}
    for path in (
        large_base / "features" / "audit.json",
        Path("scripts/diagnose_score_targets.py"),
        Path("scripts/diagnose_paired_targets.py"),
        Path("scripts/feature_score_selector.py"),
        Path("scripts/train_score_distribution.py"),
    ):
        inputs[str(path)] = sha(path)
    for parent in (1, 2):
        dataset = large_base / "data" / f"parent{parent}" / "targets.pt"
        cache_path = large_base / "features" / f"parent{parent}.pt"
        data, digest = read_checkpoint(dataset)
        cache, cache_digest = read_checkpoint(cache_path)
        if (
            audit["parents"][str(parent)]["dataset_sha256"] != digest
            or audit["parents"][str(parent)]["cache_sha256"] != cache_digest
        ):
            raise ValueError("Evaluation audit identity mismatch")
        inputs.update({str(dataset): digest, str(cache_path): cache_digest})
        groups = data["match_groups"][match_split(data["match_groups"])].numpy()
        for representation in ("raw", "features"):
            for kind in ("paired_win", "preference"):
                name = f"parent{parent}_{representation}_{kind}"
                small20 = Path(
                    "runs/paired_difference_fit_screen"
                    if kind == "paired_win"
                    else "runs/preference_fit_screen"
                )
                samples = {}
                initialization = None
                for condition, base in (
                    ("small20", small20),
                    ("small80", small80),
                    ("large20", large20),
                ):
                    if (
                        json.loads((base / "process.json").read_text(encoding="utf-8"))[
                            "status"
                        ]
                        != "complete"
                    ):
                        raise ValueError("All fitting conditions must be complete")
                    checkpoint = base / name / "final.pt"
                    key = name + "/" + condition
                    report, wins, points = evaluate(
                        checkpoint, data, digest, cache, cache_digest
                    )
                    expected_epochs = 80 if condition == "small80" else 20
                    expected_roots = 262144 if condition == "large20" else 65536
                    if (
                        report["epochs"] != expected_epochs
                        or report["kind"] != kind
                        or report["representation"] != representation
                        or report["training_dataset_roots"] != expected_roots
                    ):
                        raise ValueError("Endpoint condition mismatch")
                    current_initialization = (
                        report["initial_shared_body_sha256"],
                        report["minibatch_order_seed"],
                    )
                    if (
                        initialization is not None
                        and initialization != current_initialization
                    ):
                        raise ValueError("Initialization/shuffle differ across budgets")
                    initialization = current_initialization
                    reports[key], samples[condition] = report, (wins, points)
                    inputs[str(checkpoint)] = report["checkpoint_sha256"]
                    inputs[report["training_dataset_path"]] = report[
                        "training_dataset_sha256"
                    ]
                for left, right in (
                    ("small80", "small20"),
                    ("large20", "small20"),
                    ("large20", "small80"),
                ):
                    contrasts[name + "/" + left + "-" + right] = paired_change(
                        samples[left], samples[right], groups
                    )
        del data, cache
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError(f"Comparison input changed: {path}")
    result = {
        "scope": "All fixed endpoints on the same larger-data held-out source-match roots. One changed action followed by frozen parent continuation; development evidence, not repeated-policy match strength or untouched final confirmation.",
        "inputs": inputs,
        "endpoints": reports,
        "paired_condition_changes": contrasts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--large-base", type=Path, default=Path("runs/scaled_paired_targets_recovered")
    )
    parser.add_argument(
        "--small80", type=Path, default=Path("runs/score_update_budget_control")
    )
    parser.add_argument(
        "--large20", type=Path, default=Path("runs/score_scaled_fit_screen")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/score_data_scale_comparison.json")
    )
    args = parser.parse_args()
    result = compare(args.large_base, args.small80, args.large20, args.output)
    print(json.dumps(result["paired_condition_changes"], indent=2))


if __name__ == "__main__":
    main()
