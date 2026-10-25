"""Serial conditional V5B preparation, two-stage calibration and confirmation."""

import argparse
import json
from pathlib import Path

from scripts.export_public_stock_results import read, sha, write
from scripts.export_public_stock_stronger import (
    checked_tables,
    endpoint_name,
    reduce_calibration,
    reduce_checked,
    table_manifest,
)
from scripts.public_stock_robustness import TARGETS
from scripts.public_stock_stronger import (
    corrected_count,
    increasing,
    make_protocol,
    table_receipt,
    verify_protocol,
)
from scripts.run_public_stock_robustness import run


def evaluate(base, study, tables, label, stage, budget, particles):
    verify_protocol(study)
    checked_tables(base, study)
    row = study["systems"][label]
    name = endpoint_name(label, stage, budget)
    options = {k: row[k] for k in ("checkpoint", "opponent", "table")}
    options.update(
        {
            "out": str(base / name),
            "deal-seed": study[f"{stage}_seed"],
            "pairs": study[f"{stage}_pairs"],
            "particles": particles,
            "search-seed": study["search_seed"],
            **{k: study[k] for k in ("batch", "horizon", "confidence")},
        }
    )
    command = ["scripts.evaluate_public_stock_rounds", "evaluate"]
    for key, value in options.items():
        command.extend(["--" + key, str(value)])
    run(command, base / f"{name}.log")
    _, result = reduce_checked(base, study, tables, label, stage, budget, particles, {})
    print(json.dumps({"completed": name}), flush=True)
    return result


def calibrate(base, study, tables, stage, initial=None):
    record = {
        "protocol_sha256": sha(base / "protocol.json"),
        "tables_sha256": sha(base / "tables.json"),
        "stage": stage,
        "particles": {},
        "costs": {},
    }
    initial_sha = sha(base / "initial.json") if stage == "refinement" else None
    for label in study["systems"]:
        counts = []
        budgets = (
            [(0, study["initial_particles"])]
            if stage == "initial"
            else list(enumerate(initial["particles"][label][1:], 1))
        )
        for budget, particles in budgets:
            if initial_sha and sha(base / "initial.json") != initial_sha:
                raise ValueError("Initial calibration changed")
            result = evaluate(base, study, tables, label, stage, budget, particles)
            t, m = result["transitions"], result["matches"]
            targets = TARGETS if stage == "initial" else [TARGETS[budget - 1]]
            counts.extend(
                corrected_count(t, m, particles, target) for target in targets
            )
            name = endpoint_name(label, stage, budget)
            record["costs"][name] = {
                "transitions": t,
                "matches": m,
                "particles": particles,
                "result_sha256": sha(base / name / "result.json"),
            }
        record["particles"][label] = increasing(counts)
    independently_reduced, _ = reduce_calibration(
        base, study, tables, stage, {}, initial
    )
    if independently_reduced != record:
        raise ValueError("Independent calibration differs")
    write(base / f"{stage}.json", record)
    print(json.dumps({stage + "_particles": record["particles"]}), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    base = Path(args.out)
    if base.exists() or Path(args.evidence).exists():
        raise FileExistsError("V5B outputs exist; inspect before recovery")
    study = make_protocol(base)
    verify_protocol(study)
    base.mkdir(parents=True, exist_ok=False)
    write(base / "protocol.json", study)
    for label, row in study["systems"].items():
        verify_protocol(study)
        folder = Path(row["table"]).parent
        run(
            [
                "scripts.build_auditable_round_values",
                "--checkpoint",
                row["checkpoint"],
                "--out",
                str(folder),
                "--samples",
                str(study["table_samples"]),
                "--seed",
                str(row["table_seed"]),
            ],
            base / f"build_{label}.log",
        )
        run(
            ["scripts.audit_public_round_values", "--table-dir", str(folder)],
            base / f"audit_{label}.log",
        )
        table_receipt(study, label, {})
        print(json.dumps({"audited_table": label}), flush=True)
    tables = table_manifest(base, study)
    write(base / "tables.json", tables)
    tables_sha = sha(base / "tables.json")
    initial = calibrate(base, study, tables, "initial")
    refined = calibrate(base, study, tables, "refinement", initial)
    frozen = {p: sha(base / p) for p in ("initial.json", "refinement.json")}
    for label, counts in refined["particles"].items():
        for budget, particles in enumerate(counts):
            if sha(base / "tables.json") != tables_sha or any(
                sha(base / p) != h for p, h in frozen.items()
            ):
                raise ValueError("Preparation or calibration changed after freezing")
            evaluate(base, study, tables, label, "evaluation", budget, particles)
    run(
        [
            "scripts.export_public_stock_stronger",
            "--study",
            str(base),
            "--out",
            args.evidence,
        ],
        base / "export.log",
    )
    result = read(args.evidence)
    write(
        base / "finished.json",
        {"status": "complete", "evidence": args.evidence, "sha256": sha(args.evidence)},
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "score_tests_passed",
                    "all_work_targets_passed",
                    "matched_work_ranking",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
