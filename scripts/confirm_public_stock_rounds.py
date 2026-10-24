"""Untouched confirmation for the completed-round rule screen, only if gated."""

import argparse
import json
from pathlib import Path

from scripts.evaluate_public_stock_residual import sha, write
from scripts.run_public_stock_residual import CONTROL, run


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    base, out = Path(args.study), Path(args.out)
    study, screen = read(base / "protocol.json"), read(base / "screen_result.json")
    if not screen["confirmation_gate_passed"]:
        raise ValueError("Completed-round development gate failed")
    for path, digest in study["sources"].items():
        if sha(path) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    out.mkdir(parents=True, exist_ok=False)
    profiles = [CONTROL, screen["selected"]]
    write(
        out / "protocol.json",
        {
            "study": str(base),
            "study_protocol_sha256": sha(base / "protocol.json"),
            "screen_sha256": sha(base / "screen_result.json"),
            "profiles": profiles,
            "particles": study["confirmation_particles"],
            "seed": study["confirmation_seed"],
            "pairs": study["confirmation_pairs"],
            "bootstrap_seed": study["bootstrap_seed"],
            "bootstrap_replicates": 20000,
            "lower_quantile": 0.0125,
            "primary": "positive high-budget gain and selected-minus-control gain for both parents",
            "secondary": "high-budget minus eight-particle gain; exploratory intervals",
            "runner_sha256": sha(__file__),
        },
    )
    for profile in profiles:
        for parent in range(2):
            label = f"{profile}_parent{parent}"
            checkpoint = Path(study["trained"]) / label / "policy_00128.pt"
            table = base / f"table_{label}"
            audit = read(table / "audit.json")
            if audit["status"] != "passed" or audit["table_sha256"] != sha(
                table / "table.npz"
            ):
                raise ValueError("Unverified continuation table")
            for particles in study["confirmation_particles"]:
                dest = out / f"{label}_p{particles}"
                run(
                    [
                        "scripts.evaluate_public_stock_rounds",
                        "evaluate",
                        "--checkpoint",
                        str(checkpoint),
                        "--table",
                        str(table / "table.npz"),
                        "--out",
                        str(dest),
                        "--deal-seed",
                        str(study["confirmation_seed"]),
                        "--pairs",
                        str(study["confirmation_pairs"]),
                        "--batch",
                        "32",
                        "--particles",
                        str(particles),
                        "--horizon",
                        "80",
                        "--confidence",
                        "2.5",
                        "--search-seed",
                        "971",
                    ],
                    out / f"{dest.name}.log",
                )
                print(json.dumps({"completed": dest.name}), flush=True)
    write(
        out / "completed.json",
        {"status": "all twelve endpoints complete; independent reduction required"},
    )


if __name__ == "__main__":
    main()
