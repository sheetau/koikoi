"""Run only the predeclared, gated public-stock confirmation cohort."""

import argparse
import json
from pathlib import Path

from scripts.evaluate_public_stock_residual import sha, write
from scripts.run_public_stock_residual import CONTROL, run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    base, out = Path(args.study), Path(args.out)
    screen = json.loads((base / "screen_result.json").read_text(encoding="utf-8"))
    protocol = json.loads((base / "protocol.json").read_text(encoding="utf-8"))
    if not screen["confirmation_gate_passed"]:
        raise ValueError("The prespecified development gate failed")
    for path, digest in protocol["sources"].items():
        if sha(path) != digest:
            raise ValueError(f"Frozen source changed: {path}")
    out.mkdir(parents=True, exist_ok=False)
    write(
        out / "protocol.json",
        {
            "study": str(base),
            "study_protocol_sha256": sha(base / "protocol.json"),
            "screen_sha256": sha(base / "screen_result.json"),
            "profiles": [CONTROL, screen["selected"]],
            "particles": protocol["particles"],
            "seed": protocol["confirmation_reserved_seed"],
            "pairs": protocol["confirmation_pairs"],
            "bootstrap_seed": 951,
            "bootstrap_replicates": 20000,
            "lower_quantile": 0.0125,
            "primary": "positive high-budget gain and selected-minus-control gain for both parents",
            "runner_sha256": sha(__file__),
        },
    )
    for profile in [CONTROL, screen["selected"]]:
        for parent in range(2):
            for particles in protocol["particles"]:
                dest = out / f"{profile}_parent{parent}_p{particles}"
                checkpoint = base / f"{profile}_parent{parent}" / "policy_00128.pt"
                run(
                    [
                        "scripts.evaluate_public_stock_residual",
                        "evaluate",
                        "--checkpoint",
                        str(checkpoint),
                        "--out",
                        str(dest),
                        "--deal-seed",
                        str(protocol["confirmation_reserved_seed"]),
                        "--pairs",
                        str(protocol["confirmation_pairs"]),
                        "--batch",
                        "64",
                        "--particles",
                        str(particles),
                        "--horizon",
                        "32",
                        "--search-seed",
                        "941",
                    ],
                    out / f"{dest.name}.log",
                )
                print(json.dumps({"completed": dest.name}), flush=True)
    write(
        out / "completed.json",
        {"status": "all 16 endpoints complete; independent reduction required"},
    )


if __name__ == "__main__":
    main()
