"""Fixed endpoint fitting screen; no match evaluation or automatic model selection."""

import argparse
import contextlib
import json
import time
from pathlib import Path

from koikoi.train import provenance
from scripts.diagnose_score_targets import diagnose
from scripts.run_arbitration_study import sha
from scripts.train_score_distribution import fit


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def make_plan(kinds=("categorical", "win", "mean")):
    if (
        not kinds
        or len(set(kinds)) != len(kinds)
        or not set(kinds).issubset(
            {"categorical", "win", "mean", "paired_win", "paired_mean", "preference"}
        )
    ):
        raise ValueError("Expected distinct supported supervision kinds")
    cache_audit = Path("runs/paired_public_features/audit.json")
    if json.loads(cache_audit.read_text(encoding="utf-8"))["status"] != "passed":
        raise ValueError("Feature cache audit must pass")
    paths = [
        Path(__file__),
        Path("scripts/train_score_distribution.py"),
        Path("scripts/feature_score_selector.py"),
        Path("scripts/diagnose_score_targets.py"),
        Path("scripts/diagnose_paired_targets.py"),
        Path("scripts/paired_rollout_targets.py"),
        Path("scripts/cache_paired_expert_features.py"),
        Path("scripts/train_paired_arbitration.py"),
        Path("scripts/diagnose_arbitration.py"),
        Path("scripts/run_arbitration_study.py"),
        cache_audit,
    ]
    cases = []
    for parent in (1, 2):
        dataset = Path(f"runs/paired_arbitration_study/parent{parent}/targets.pt")
        cache = Path(f"runs/paired_public_features/parent{parent}.pt")
        paths.extend((dataset, cache))
        for representation in ("raw", "features"):
            for kind in kinds:
                cases.append(
                    {
                        "name": f"parent{parent}_{representation}_{kind}",
                        "dataset": str(dataset),
                        "feature_cache": str(cache)
                        if representation == "features"
                        else None,
                        "kind": kind,
                        "seed": 19400001 + parent * 10000,
                    }
                )
    return {
        "scope": f"Exploratory supervised learning screen on existing paired targets. All{len(cases)} final endpoints are retained and diagnosed; no early stopping, no match evaluation and no automatic selection. Held-out source-match diagnostics are development evidence, not fresh confirmation of a chosen method.",
        "supervision_kinds": list(kinds),
        "source": provenance(),
        "inputs": {str(path): sha(path) for path in paths},
        "epochs": 20,
        "device": "cuda",
        "cases": cases,
        "accounting": "Zero new simulation during fitting. Original branch data and public feature replay are shared across heads, not multiplied into independent samples.",
        "sequence_revision": "Fit/diagnose now while existing paired/static CPU evaluations continue unchanged. Any new score-selector match study waits for those comparisons to finish.",
    }


def run(base, kinds=("categorical", "win", "mean")):
    if base.exists():
        raise ValueError("Use a fresh screen directory")
    plan = make_plan(kinds)
    base.mkdir(parents=True)
    write(base / "plan.json", plan)
    reports = {}
    try:
        for case in plan["cases"]:
            if provenance()["source_sha256"] != plan["source"]["source_sha256"]:
                raise ValueError("Production source changed")
            for path, digest in plan["inputs"].items():
                if sha(path) != digest:
                    raise ValueError(f"Screen input changed: {path}")
            folder = base / case["name"]
            folder.mkdir()
            write(base / "process.json", {"status": "fitting", "case": case["name"]})
            started = time.perf_counter()
            with (
                (folder / "fit.log").open("w", encoding="utf-8") as log,
                contextlib.redirect_stdout(log),
            ):
                fitted = fit(
                    case["dataset"],
                    folder / "final.pt",
                    kind=case["kind"],
                    seed=case["seed"],
                    device=plan["device"],
                    epochs=plan["epochs"],
                    feature_cache=case["feature_cache"],
                )
            receipt = {
                "checkpoint_sha256": sha(folder / "final.pt"),
                "elapsed_fit_seconds": time.perf_counter() - started,
                "counters": fitted["counters"],
                "training_config": fitted["training_config"],
            }
            write(folder / "fit_complete.json", receipt)
            report = diagnose(
                Path(case["dataset"]), folder / "final.pt", folder / "diagnostic.json"
            )
            reports[case["name"]] = report
            print(
                json.dumps(
                    {"completed": case["name"], "diagnostic": report["sections"]["all"]}
                ),
                flush=True,
            )
        write(base / "summary.json", reports)
        write(base / "process.json", {"status": "complete", "cases": len(reports)})
    except Exception as error:
        write(base / "process.json", {"status": "failed", "error": repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("runs/score_fit_screen"))
    parser.add_argument("--kinds", nargs="+", default=["categorical", "win", "mean"])
    args = parser.parse_args()
    run(args.base, args.kinds)


if __name__ == "__main__":
    main()
