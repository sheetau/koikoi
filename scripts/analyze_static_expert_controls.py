"""Compare no-learning card/koi hybrids with learned selectors on shared deals."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_arbitration_study import interval, outcome, read, sha


def validate_report(report, spec, opponent, p):
    evaluation = p["evaluation"]
    assert report["agent"] == spec["cards"] and report["koi_agent"] == spec["koi"]
    assert report["opponent"] == opponent and report["profile"] == "paper8"
    assert report["endgame"] and report["opponent_endgame"] and report["deterministic"]
    assert report["device"] == "cpu" and report["search"] == "none"
    assert (
        report["pairs"] == evaluation["pairs"]
        and report["matches"] == 2 * evaluation["pairs"]
    )
    assert report["checkpoints"][spec["parent"]]["sha256"] == spec["parent_sha256"]
    assert report["provenance"]["source_sha256"] == p["source"]["source_sha256"]
    rows = report["records"]
    assert [row["seed"] for row in rows] == list(
        range(evaluation["first_seed"], evaluation["first_seed"] + evaluation["pairs"])
    )
    for row in rows:
        for seat in (0, 1):
            points = row[f"seat{seat}_point_difference"]
            assert type(points) is int
            assert row[f"seat{seat}_score"] == (
                1 if points > 0 else 0 if points < 0 else 0.5
            )
    decisions = sum(
        report[agent][phase]
        for agent in ("agent_decisions", "opponent_decisions")
        for phase in ("discard", "pick_hand", "pick_draw", "koi")
    )
    assert decisions == report["decisions"]
    wins = np.array([(row["seat0_score"] + row["seat1_score"]) / 2 for row in rows])
    points = np.array(
        [
            (row["seat0_point_difference"] + row["seat1_point_difference"]) / 2
            for row in rows
        ]
    )
    for key, value in (
        ("draw_adjusted_win_rate", wins.mean()),
        ("win_rate_ci95", interval(wins)),
        ("mean_point_difference", points.mean()),
        ("point_difference_ci95", interval(points)),
    ):
        np.testing.assert_allclose(report[key], value, rtol=0, atol=1e-12)
    return wins, points


def analyze(base, paired):
    p = read(base / "protocol.json")
    assert read(base / "process.json")["status"] == "complete"
    assert p["runner_sha256"] == sha("scripts/run_static_expert_controls.py")
    assert p["previous_protocol_sha256"] == sha(paired / "protocol.json")
    previous = read(paired / "protocol.json")
    assert previous["evaluation"] == p["evaluation"]
    comparisons, reports = {}, {}
    for case, spec in p["cases"].items():
        for opponent in p["evaluation"]["opponents"]:
            path = base / case / (opponent + ".json")
            report = read(path)
            wins, points = validate_report(report, spec, opponent, p)
            key = case + "/" + opponent
            reports[key] = {
                "sha256": sha(path),
                "win_score": float(wins.mean()),
                "win_ci95": interval(wins),
                "point_difference": float(points.mean()),
                "points_ci95": interval(points),
            }
            parent = case.split("_", 1)[0]
            for suffix in ("full", "no_match", "ppo_gate", "ppo_actor"):
                learned = parent + "_" + suffix
                _, other_wins, other_points = outcome(
                    paired, learned, opponent, previous
                )
                delta, point_delta = other_wins - wins, other_points - points
                comparisons[learned + "-" + case + "/" + opponent] = {
                    "win_difference": float(delta.mean()),
                    "win_ci95": interval(delta),
                    "point_difference": float(point_delta.mean()),
                    "points_ci95": interval(point_delta),
                }
    result = {
        "status": "passed",
        "protocol_sha256": sha(base / "protocol.json"),
        "analysis_script_sha256": sha(__file__),
        "outcomes": reports,
        "paired_comparisons": comparisons,
        "scope": "Development comparisons on shared deal pairs. Every predefined hybrid is retained; no additional independent deals or training runs are created by these comparisons.",
    }
    (base / "analysis.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("runs/static_expert_controls")
    )
    parser.add_argument(
        "--paired", type=Path, default=Path("runs/paired_arbitration_study")
    )
    args = parser.parse_args()
    analyze(args.base, args.paired)


if __name__ == "__main__":
    main()
