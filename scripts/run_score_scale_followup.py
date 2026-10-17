"""Fixed larger-data fitting and common-cohort comparison after data recovery."""

import argparse
import json
import time
from pathlib import Path

from koikoi.train import provenance
from scripts.audit_paired_feature_cache import audit as audit_features
from scripts.audit_win_selector_screen import audit as audit_fits
from scripts.compare_score_training_budgets import compare
from scripts.run_arbitration_study import sha, write
from scripts.run_score_fit_screen import make_plan
from scripts.run_score_fit_screen import run as fit_screen


def checked(plan):
    if provenance()["source_sha256"] != plan["source"]["source_sha256"]:
        raise ValueError("Production changed")
    for path, digest in plan["inputs"].items():
        if sha(path) != digest:
            raise ValueError(f"Follow-up input changed: {path}")


def protocol(data_base):
    small = make_plan(("paired_win", "preference"))
    collection = json.loads((data_base / "protocol.json").read_text(encoding="utf-8"))
    if (
        collection["source"]["source_sha256"] != small["source"]["source_sha256"]
        or collection["collection"]["roots_per_parent"] != 262144
    ):
        raise ValueError("Expected fixed fourfold source data")
    inputs = {**small["inputs"], **collection["inputs"]}
    paths = [
        Path(__file__),
        Path("scripts/audit_paired_feature_cache.py"),
        Path("scripts/audit_win_selector_screen.py"),
        Path("scripts/compare_score_training_budgets.py"),
        data_base / "protocol.json",
        Path("runs/score_update_budget_control/fit_audit.json"),
    ]
    for case in small["cases"]:
        previous = (
            "paired_difference_fit_screen"
            if case["kind"] == "paired_win"
            else "preference_fit_screen"
        )
        for base in (previous, "score_update_budget_control"):
            paths.append(Path("runs") / base / case["name"] / "final.pt")
    inputs.update({str(path): sha(path) for path in paths})
    plan = {
        "source": provenance(),
        "inputs": inputs,
        "kinds": ["paired_win", "preference"],
        "epochs": 20,
        "data_base": str(data_base),
        "scope": "Fixed8 larger-data fits; all endpoints retained, audited and compared with small20/small80 on common held-out source matches. No selection, strength matchup, early stopping or final-confirmation data.",
    }
    checked(plan)
    return plan


def run(base, data_base, fit_base):
    if base.exists() or fit_base.exists():
        raise ValueError("Use fresh follow-up and fit directories")
    plan = protocol(data_base)
    base.mkdir(parents=True)
    write(base / "protocol.json", plan)
    try:
        write(base / "process.json", {"status": "waiting_for_data"})
        while True:
            try:
                status = json.loads(
                    (data_base / "process.json").read_text(encoding="utf-8")
                )
            except json.JSONDecodeError:
                time.sleep(1)
                continue
            if status["status"] == "failed":
                raise ValueError(f"Data generation failed: {status}")
            if status["status"] == "complete":
                break
            time.sleep(10)
        checked(plan)
        write(base / "process.json", {"status": "auditing_features"})
        audit_features(data_base / "data", data_base / "features")
        checked(plan)
        write(base / "process.json", {"status": "fitting"})
        fit_screen(
            fit_base,
            plan["kinds"],
            data_base / "data",
            data_base / "features",
            plan["epochs"],
        )
        checked(plan)
        write(base / "process.json", {"status": "auditing_fits"})
        audit_fits(fit_base, fit_base / "fit_audit.json")
        write(base / "process.json", {"status": "comparing"})
        compare(
            data_base,
            Path("runs/score_update_budget_control"),
            fit_base,
            base / "comparison.json",
        )
        checked(plan)
        write(
            base / "process.json",
            {"status": "complete", "cases": 8, "compared_endpoints": 24},
        )
        print(
            {"status": "complete", "comparison_sha256": sha(base / "comparison.json")},
            flush=True,
        )
    except Exception as error:
        write(base / "process.json", {"status": "failed", "error": repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("runs/score_scale_followup"))
    parser.add_argument(
        "--data-base", type=Path, default=Path("runs/scaled_paired_targets_recovered")
    )
    parser.add_argument(
        "--fit-base", type=Path, default=Path("runs/score_scaled_fit_screen")
    )
    args = parser.parse_args()
    run(args.base, args.data_base, args.fit_base)


if __name__ == "__main__":
    main()
