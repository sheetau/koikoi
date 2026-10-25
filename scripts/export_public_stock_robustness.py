"""Independent V4 cohort reduction and fourteen fixed paired comparisons."""

import argparse
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from scripts.export_public_stock_results import (
    intervals,
    read,
    reduce_endpoint,
    sha,
    write,
)
from scripts.public_stock_robustness import (
    ALL_THREE,
    CONTROL,
    COUNTS,
    FIXED,
    PREVIEW_CHOICE,
    PROFILES,
    TARGETS,
    validate_result,
    verify_protocol,
    work_admissibility,
)


def reduce_checked(folder, study, label, stage, particles, bindings):
    scores, result = reduce_endpoint(folder, bindings)
    receipt = read(folder / "protocol.json")
    if any(result.get(k) != v for k, v in receipt.items()):
        raise ValueError("Pre-run endpoint receipt differs from final result")
    validate_result(result, study, label, stage, particles, folder)
    return scores, result


def reduce_calibration(base, study, bindings):
    """Reconstruct frozen particle counts from work, independently of the runner."""
    calibration = {
        "protocol_sha256": sha(base / "protocol.json"), "particles": {}, "costs": {},
    }
    work = {}
    for label in study["systems"]:
        folder = base / f"calibration_{label}"
        _, result = reduce_checked(folder, study, label, "calibration", 32, bindings)
        transitions, matches = result["transitions"], result["matches"]
        if transitions <= 0:
            raise ValueError("Calibration has no positive work")
        counts = [
            min(256, max(2, math.floor(Fraction(t * matches * 32, transitions) + Fraction(1, 2))))
            for t in TARGETS
        ]
        if not counts[0] < counts[1]:
            raise ValueError("Calibration budgets do not increase")
        calibration["particles"][label] = [0, *counts]
        calibration["costs"][label] = {
            "transitions": transitions, "matches": matches,
            "per_particle": transitions / matches / 32,
            "result_sha256": sha(folder / "result.json"),
        }
        work[label] = {k: result[k] for k in (*COUNTS, "seconds")}
    return calibration, work


def compare(data, indices):
    """Keep parents separate and preserve within-stream covariance."""
    arrays = {p: np.asarray(data[p], dtype=float) for p in PROFILES}
    expected_shape = arrays[CONTROL].shape
    if len(expected_shape) != 3 or expected_shape[:2] != (2, 3):
        raise ValueError("Expected parent/budget/duplicate-stream scores")
    if any(
        a.shape != expected_shape or not np.isfinite(a).all()
        or np.any((a < 0) | (a > 1)) for a in arrays.values()
    ):
        raise ValueError("Score arrays differ or contain invalid values")
    changes = {p: a[:, 2] - a[:, 0] for p, a in arrays.items()}
    vectors = {}
    for profile in (PREVIEW_CHOICE, ALL_THREE):
        for parent in range(2):
            label = f"{profile}_parent{parent}"
            vectors[f"{label}_gain"] = changes[profile][parent]
            vectors[f"{label}_excess"] = changes[profile][parent] - changes[CONTROL][parent]
            vectors[f"{label}_high_minus_low"] = arrays[profile][parent, 2] - arrays[profile][parent, 1]
    for parent in range(2):
        vectors[f"preview_minus_all_parent{parent}"] = changes[PREVIEW_CHOICE][parent] - changes[ALL_THREE][parent]
    assert len(vectors) == 14
    comparisons = {
        k: intervals(v, indices, FIXED["interval_quantile"]) for k, v in vectors.items()
    }
    passes = {
        profile: all(
            comparisons[f"{profile}_parent{p}_{name}"]["lower"] > 0
            for p in range(2) for name in ("gain", "excess", "high_minus_low")
        )
        for profile in (PREVIEW_CHOICE, ALL_THREE)
    }
    ranking = "unresolved"
    differences = [comparisons[f"preview_minus_all_parent{p}"] for p in range(2)]
    if all(row["lower"] > 0 for row in differences):
        ranking = PREVIEW_CHOICE
    elif all(row["upper"] < 0 for row in differences):
        ranking = ALL_THREE
    return comparisons, passes, ranking


def reduce_study(base):
    study = read(base / "protocol.json")
    verify_protocol(study)
    bindings = {
        **study["sources"], **study["inputs"],
        str(base / "protocol.json"): sha(base / "protocol.json"),
    }
    calibration, calibration_work = reduce_calibration(base, study, bindings)
    if read(base / "calibration.json") != calibration:
        raise ValueError("Frozen calibration does not reproduce from saved pilot work")
    bindings[str(base / "calibration.json")] = sha(base / "calibration.json")
    data, work, admissibility = {}, {}, {}
    environments = set()
    for profile in PROFILES:
        data[profile] = []
        for parent in range(2):
            label = f"{profile}_parent{parent}"
            rows = []
            for budget, particles in enumerate(calibration["particles"][label]):
                endpoint = f"evaluation_{label}_b{budget}"
                folder = base / endpoint
                scores, result = reduce_checked(folder, study, label, "evaluation", particles, bindings)
                if len(scores) != study["evaluation_pairs"]:
                    raise ValueError("Wrong independent cohort size")
                rows.append(scores)
                work[endpoint] = {k: result[k] for k in (*COUNTS, "seconds")}
                environments.add((result["torch"], result["gpu"]))
                if budget:
                    admissibility[f"{label}_b{budget}"] = work_admissibility(
                        result["transitions"], result["matches"], TARGETS[budget - 1]
                    )
            data[profile].append(rows)
        data[profile] = np.asarray(data[profile])
    if len(environments) != 1:
        raise ValueError("Evaluation runtime/device identities differ")
    indices = np.random.default_rng(study["bootstrap_seed"]).integers(
        0, study["evaluation_pairs"],
        (study["bootstrap_replicates"], study["evaluation_pairs"]),
    )
    comparisons, passed, ranking = compare(data, indices)
    cost_passed = all(row["passed"] for row in admissibility.values())
    return {
        "method": study["method"],
        "evaluation_duplicate_streams": study["evaluation_pairs"],
        "calibration": calibration,
        "curves": {p: a.mean(2).tolist() for p, a in data.items()},
        "comparisons": comparisons,
        "comparison_family": {
            "count": 14, "quantiles": [study["interval_quantile"], 1 - study["interval_quantile"]],
            "replicates": study["bootstrap_replicates"], "seed": study["bootstrap_seed"],
        },
        "score_tests_passed": passed,
        "work_admissibility": admissibility,
        "all_work_targets_passed": cost_passed,
        "matched_work_claim_passed": {p: bool(v and cost_passed) for p, v in passed.items()},
        "ranking_by_search_gain": ranking,
        "matched_work_ranking": ranking if cost_passed else "unresolved",
        "calibration_endpoints": calibration_work,
        "evaluation_endpoints": work,
        "calibration_totals": {k: sum(r[k] for r in calibration_work.values()) for k in COUNTS},
        "evaluation_totals": {k: sum(r[k] for r in work.values()) for k in COUNTS},
        "reused_table_count": 6,
        "new_training_decisions": 0,
        "new_preparation_rounds": 0,
        "audit_scope": "Scored matches fully Python-replayed; outcomes and evaluated-match counts independently reduced. Internal planning work is recorded, not independently replayed.",
        "bindings": bindings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    target = Path(args.out)
    if target.exists():
        raise FileExistsError(target)
    result = reduce_study(Path(args.study))
    write(target, result)
    print({k: result[k] for k in ("score_tests_passed", "all_work_targets_passed", "matched_work_ranking", "evaluation_totals")})


if __name__ == "__main__":
    main()
