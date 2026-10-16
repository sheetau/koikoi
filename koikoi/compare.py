"""Compare paired evaluation reports using only shared independently seeded deals."""

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import paired_interval


def compare_reports(first, second):
    for field in ("profile", "split", "protocol_version"):
        if first.get(field) != second.get(field):
            raise ValueError(f"Incompatible evaluation field: {field}")
    if first.get("protocol_version") != 2:
        raise ValueError("Pair-independent RNG protocol version 2 required")
    opponent = first["opponent"]
    guards = [
        report.get("opponent_endgame", opponent.endswith("-assisted"))
        for report in (first, second)
    ]
    if guards[0] != guards[1]:
        raise ValueError("Opponent endgame assistance differs")
    hashes = [
        report.get("checkpoints", {}).get(report["opponent"], {}).get("sha256")
        for report in (first, second)
    ]
    if hashes[0] != hashes[1]:
        raise ValueError("Opponent checkpoint bytes differ")
    if opponent != second["opponent"] and not hashes[0]:
        raise ValueError("Incompatible evaluation field: opponent")
    if hashes[0] and first.get("deterministic") != second.get("deterministic"):
        raise ValueError("Opponent action-selection settings differ")
    if opponent.startswith("rl-") and first.get("upstream_revision") != second.get(
        "upstream_revision"
    ):
        raise ValueError("Opponent upstream revisions differ")
    rows = []
    for report in (first, second):
        indexed = {row["seed"]: row for row in report["records"]}
        if len(indexed) != len(report["records"]):
            raise ValueError("Duplicate evaluation seeds")
        rows.append(indexed)
    seeds = sorted(rows[0].keys() & rows[1].keys())
    if len(seeds) < 2:
        raise ValueError("At least two shared duplicate deals are required")
    scores, points = [], []
    for seed in seeds:
        a, b = rows[0][seed], rows[1][seed]
        scores.append(
            (a["seat0_score"] + a["seat1_score"] - b["seat0_score"] - b["seat1_score"])
            / 2
        )
        points.append(
            (
                a["seat0_point_difference"]
                + a["seat1_point_difference"]
                - b["seat0_point_difference"]
                - b["seat1_point_difference"]
            )
            / 2
        )
    return {
        "shared_pairs": len(seeds),
        "seeds": seeds,
        "score_difference_first_minus_second": float(np.mean(scores)),
        "score_difference_ci95": paired_interval(scores),
        "point_difference_first_minus_second": float(np.mean(points)),
        "point_difference_ci95": paired_interval(points),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_reports(
        json.loads(args.first.read_text(encoding="utf-8")),
        json.loads(args.second.read_text(encoding="utf-8")),
    )
    result.update(first=str(args.first), second=str(args.second))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "seeds"}, indent=2))
