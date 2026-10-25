"""Serial V4 calibration/evaluation; all prior cohorts and artifacts stay frozen."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.export_public_stock_results import read, reduce_endpoint, sha, write
from scripts.public_stock_robustness import (
    calibrate_particles,
    make_protocol,
    validate_result,
    verify_protocol,
)


def run(command, log):
    print(json.dumps({"command": command}), flush=True)
    with log.open("x", encoding="utf-8", newline="\n") as stream:
        subprocess.run(
            [sys.executable, "-X", "utf8", "-m", *command],
            stdout=stream, stderr=subprocess.STDOUT, check=True,
        )


def evaluate(base, study, label, stage, budget, particles):
    verify_protocol(study)
    system = study["systems"][label]
    name = f"calibration_{label}" if stage == "calibration" else f"evaluation_{label}_b{budget}"
    folder = base / name
    command = ["scripts.evaluate_public_stock_rounds", "evaluate"]
    for key in ("checkpoint", "opponent", "table"):
        command.extend([f"--{key}", system[key]])
    options = {
        "out": str(folder), "deal-seed": study[f"{stage}_seed"],
        "pairs": study[f"{stage}_pairs"], "particles": particles,
        "batch": study["batch"], "horizon": study["horizon"],
        "confidence": study["confidence"], "search-seed": study["search_seed"],
    }
    for key, value in options.items():
        command.extend([f"--{key}", str(value)])
    run(command, base / f"{name}.log")
    _, result = reduce_endpoint(folder, {})
    validate_result(result, study, label, stage, particles, folder)
    print(json.dumps({"completed": name}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained", default="runs/public_stock_residual_v2")
    parser.add_argument("--tables", default="runs/public_stock_rounds_v3")
    parser.add_argument("--out", required=True)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    base = Path(args.out)
    if base.exists() or Path(args.evidence).exists():
        raise FileExistsError("V4 outputs already exist; inspect before recovery")
    study = make_protocol(Path(args.trained), Path(args.tables))
    verify_protocol(study)
    base.mkdir(parents=True, exist_ok=False)
    write(base / "protocol.json", study)
    calibration = {"protocol_sha256": sha(base / "protocol.json"), "particles": {}, "costs": {}}
    for label in study["systems"]:
        result = evaluate(base, study, label, "calibration", 0, study["calibration_particles"])
        t, m = result["transitions"], result["matches"]
        calibration["particles"][label] = calibrate_particles(t, m)
        calibration["costs"][label] = {
            "transitions": t, "matches": m, "per_particle": t / m / 32,
            "result_sha256": sha(base / f"calibration_{label}" / "result.json"),
        }
    write(base / "calibration.json", calibration)
    # A separate implementation verifies the cost-derived counts before holdout use.
    from scripts.export_public_stock_robustness import reduce_calibration

    independently_reduced, _ = reduce_calibration(base, study, {})
    if independently_reduced != calibration:
        raise ValueError("Independent calibration reduction differs")
    frozen_calibration = sha(base / "calibration.json")
    print(json.dumps({"calibrated_particles": calibration["particles"]}), flush=True)
    for label, particles in calibration["particles"].items():
        for budget, count in enumerate(particles):
            if sha(base / "calibration.json") != frozen_calibration:
                raise ValueError("Calibration changed after freezing")
            evaluate(base, study, label, "evaluation", budget, count)
    run(
        ["scripts.export_public_stock_robustness", "--study", str(base), "--out", args.evidence],
        base / "export.log",
    )
    result = read(args.evidence)
    write(base / "finished.json", {"status": "complete", "evidence": args.evidence, "sha256": sha(args.evidence)})
    print(json.dumps({k: result[k] for k in ("score_tests_passed", "all_work_targets_passed", "matched_work_ranking")}), flush=True)


if __name__ == "__main__":
    main()
