"""Reuse the reviewed work/comparison layout for complete V5C evidence only."""

import argparse
from pathlib import Path

from scripts.export_public_stock_results import read, sha, write
from scripts.plot_public_stock_robustness import render
from scripts.public_stock_robustness import PROFILES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = read(args.evidence)
    if (
        data["method"] != "public_stock_mature_variants_v5c"
        or data["profiles"] != list(PROFILES)
        or len(data["comparisons"]) != 14
        or data["evaluation_totals"]["matches"] != 9216
    ):
        raise ValueError("Complete two-finalist V5C evidence is required")
    for path, digest in data["bindings"].items():
        if sha(path) != digest:
            raise ValueError(f"Evidence binding changed: {path}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    render({**data, "calibration": data["refined_calibration"]}, out)
    write(
        out / "figure_evidence.json",
        {
            "inputs": {
                p: sha(p)
                for p in (
                    args.evidence,
                    str(Path(__file__)),
                    "scripts/plot_public_stock_robustness.py",
                )
            },
            "outputs": {p.as_posix(): sha(p) for p in sorted(out.iterdir())},
            "scope": "All fourteen simultaneous comparisons and twelve observed-work checks. Qualified 512-update finalist policies, retained 128-update control; unequal adaptation budgets. Reuses the reviewed V4 layout without modifying V4 artifacts. No new strength evaluation.",
        },
    )


if __name__ == "__main__":
    main()
