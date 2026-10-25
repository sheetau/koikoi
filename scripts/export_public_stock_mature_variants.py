"""Independent V5C reduction with separate new and reused preparation counts."""

import argparse
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from scripts.export_public_stock_results import read, reduce_endpoint, sha, write
from scripts.public_stock_mature_variants import verify_protocol
from scripts.public_stock_robustness import COUNTS, TARGETS, work_admissibility
from scripts.public_stock_stronger import (
    compare,
    increasing,
    table_receipt,
    validate_endpoint,
)


def table_manifest(base, study):
    systems, counts, bindings = {}, {}, {}
    for label in study["systems"]:
        systems[label], counts[label] = table_receipt(study, label, bindings)
    return {
        "protocol_sha256": sha(base / "protocol.json"),
        "systems": systems,
        "preparation": counts,
        "bindings": bindings,
    }


def checked_tables(base, study):
    actual = table_manifest(base, study)
    if read(base / "tables.json") != actual:
        raise ValueError("Frozen preparation manifest changed")
    return actual


def endpoint_name(label, stage, budget):
    return f"initial_{label}" if stage == "initial" else f"{stage}_{label}_b{budget}"


def reduce_checked(base, study, tables, label, stage, budget, particles, bindings):
    folder = base / endpoint_name(label, stage, budget)
    scores, result = reduce_endpoint(folder, bindings)
    receipt = read(folder / "protocol.json")
    if any(result.get(k) != v for k, v in receipt.items()):
        raise ValueError("Pre-run endpoint receipt differs")
    validate_endpoint(result, study, tables, label, stage, particles, folder)
    bindings[study["systems"][label]["opponent"]] = study["systems"][label][
        "opponent_sha256"
    ]
    return scores, result


def reduce_calibration(base, study, tables, stage, bindings, initial=None):
    if stage not in ("initial", "refinement") or (
        stage == "refinement" and initial is None
    ):
        raise ValueError("Missing calibration stage or proposed counts")
    record = {
        "protocol_sha256": sha(base / "protocol.json"),
        "tables_sha256": sha(base / "tables.json"),
        "stage": stage,
        "particles": {},
        "costs": {},
    }
    work = {}
    for label in study["systems"]:
        corrected = []
        budgets = (
            [(0, study["initial_particles"])]
            if stage == "initial"
            else list(enumerate(initial["particles"][label][1:], 1))
        )
        for budget, particles in budgets:
            _, result = reduce_checked(
                base, study, tables, label, stage, budget, particles, bindings
            )
            t, m = result["transitions"], result["matches"]
            if t <= 0:
                raise ValueError("Calibration has no positive work")
            targets = TARGETS if stage == "initial" else [TARGETS[budget - 1]]
            # Independent exact rational rounding rather than the runner's formula.
            corrected.extend(
                min(
                    256,
                    max(
                        2,
                        math.floor(
                            Fraction(target * m * particles, t) + Fraction(1, 2)
                        ),
                    ),
                )
                for target in targets
            )
            name = endpoint_name(label, stage, budget)
            record["costs"][name] = {
                "transitions": t,
                "matches": m,
                "particles": particles,
                "result_sha256": sha(base / name / "result.json"),
            }
            work[name] = {k: result[k] for k in (*COUNTS, "seconds")}
        record["particles"][label] = increasing(corrected)
    return record, work


def reduce_study(base):
    study = read(base / "protocol.json")
    verify_protocol(study)
    if Path(study["out"]).resolve() != base.resolve():
        raise ValueError("Wrong V5C study directory")
    tables = checked_tables(base, study)
    bindings = {
        **study["inputs"],
        **study["sources"],
        **tables["bindings"],
        str(base / "protocol.json"): sha(base / "protocol.json"),
        str(base / "tables.json"): sha(base / "tables.json"),
    }
    initial, initial_work = reduce_calibration(base, study, tables, "initial", bindings)
    refined, refined_work = reduce_calibration(
        base, study, tables, "refinement", bindings, initial
    )
    for stage, record in (("initial", initial), ("refinement", refined)):
        path = base / f"{stage}.json"
        if read(path) != record:
            raise ValueError("Frozen work calibration does not reproduce")
        bindings[str(path)] = sha(path)
    data, work, admissibility, runtimes = {}, {}, {}, set()
    for profile in study["profiles"]:
        data[profile] = []
        for parent in range(2):
            label = f"{profile}_parent{parent}"
            rows = []
            for budget, particles in enumerate(refined["particles"][label]):
                scores, result = reduce_checked(
                    base,
                    study,
                    tables,
                    label,
                    "evaluation",
                    budget,
                    particles,
                    bindings,
                )
                if len(scores) != study["evaluation_pairs"]:
                    raise ValueError("Wrong independent cohort size")
                rows.append(scores)
                name = endpoint_name(label, "evaluation", budget)
                work[name] = {k: result[k] for k in (*COUNTS, "seconds")}
                runtimes.add((result["torch"], result["gpu"]))
                if budget:
                    admissibility[f"{label}_b{budget}"] = work_admissibility(
                        result["transitions"], result["matches"], TARGETS[budget - 1]
                    )
            data[profile].append(rows)
        data[profile] = np.asarray(data[profile])
    if len(runtimes) != 1:
        raise ValueError("Evaluation runtime/device identities differ")
    indices = np.random.default_rng(study["bootstrap_seed"]).integers(
        0,
        study["evaluation_pairs"],
        (study["bootstrap_replicates"], study["evaluation_pairs"]),
    )
    comparisons, passed, ranking, quantile = compare(data, indices)
    cost_passed = all(r["passed"] for r in admissibility.values())
    preparation = tables["preparation"]
    return {
        "method": study["method"],
        "profiles": study["profiles"],
        "evaluation_duplicate_streams": study["evaluation_pairs"],
        "initial_calibration": initial,
        "refined_calibration": refined,
        "curves": {p: a.mean(2).tolist() for p, a in data.items()},
        "comparisons": comparisons,
        "comparison_family": {
            "count": len(comparisons),
            "quantiles": [quantile, 1 - quantile],
            "replicates": study["bootstrap_replicates"],
            "seed": study["bootstrap_seed"],
        },
        "score_tests_passed": passed,
        "work_admissibility": admissibility,
        "all_work_targets_passed": cost_passed,
        "matched_work_claim_passed": {
            p: bool(v and cost_passed) for p, v in passed.items()
        },
        "ranking_by_search_gain": ranking,
        "matched_work_ranking": ranking if cost_passed else "unresolved",
        "initial_endpoints": initial_work,
        "refinement_endpoints": refined_work,
        "evaluation_endpoints": work,
        "initial_totals": {k: sum(r[k] for r in initial_work.values()) for k in COUNTS},
        "refinement_totals": {
            k: sum(r[k] for r in refined_work.values()) for k in COUNTS
        },
        "evaluation_totals": {k: sum(r[k] for r in work.values()) for k in COUNTS},
        "preparation": preparation,
        "new_preparation_totals": {
            k: sum(
                r[k]
                for label, r in preparation.items()
                if not study["systems"][label]["reuse_table"]
            )
            for k in next(iter(preparation.values()))
        },
        "reused_preparation_totals": {
            k: sum(
                r[k]
                for label, r in preparation.items()
                if study["systems"][label]["reuse_table"]
            )
            for k in next(iter(preparation.values()))
        },
        "policy_scope": "Qualified 512-update finalists; retained 128-update control. Within-rule budget comparisons share each fixed actor/opponent. Unequal adaptation budgets; not a matched-training rule-effect estimate.",
        "new_training_decisions": 0,
        "audit_scope": "All scored matches Python-replayed and outcomes/counts independently reduced. Recorded internal planning is not replayed. New finalist-table audits reduce all targets and replay 3776 fixed-stratum rounds per table. Retained control-table audits are verified without new preparation or replay. Rechecking audit receipts is not a new replay.",
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
    print(
        {
            k: result[k]
            for k in (
                "score_tests_passed",
                "all_work_targets_passed",
                "matched_work_ranking",
                "evaluation_totals",
            )
        }
    )


if __name__ == "__main__":
    main()
