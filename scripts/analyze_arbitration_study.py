"""Recompute complete arbitration-study outcomes and paired arm differences."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(99)
    means = []
    for _ in range(50):
        indices = rng.integers(len(values), size=(100, len(values)))
        means.extend(values[indices].mean(axis=1))
    return np.quantile(means, [0.025, 0.975]).tolist()


def raw_outcomes(rows, first, count):
    if [r["seed"] for r in rows] != list(range(first, first + count)):
        raise ValueError("Missing, duplicate, reordered or wrong deal seeds")
    for row in rows:
        if type(row["decisions"]) is not int or row["decisions"] <= 0:
            raise ValueError("Invalid decision count")
        for seat in (0, 1):
            score, points = (
                row[f"seat{seat}_score"],
                row[f"seat{seat}_point_difference"],
            )
            if type(points) is not int:
                raise ValueError("Point difference must be integral")
            expected = 1.0 if points > 0 else 0.0 if points < 0 else 0.5
            if score != expected:
                raise ValueError("Win score and point difference disagree")
    scores = np.array([(r["seat0_score"] + r["seat1_score"]) / 2 for r in rows])
    points = np.array(
        [(r["seat0_point_difference"] + r["seat1_point_difference"]) / 2 for r in rows]
    )
    return scores, points


def outcome(base, case, opponent, p):
    target = base / case / (opponent + ".json")
    raw = target.with_suffix(".jsonl")
    report = read(target)
    if report["raw_sha256"] != sha(raw) or report["checkpoint_sha256"] != sha(
        report["candidate"]
    ):
        raise ValueError("Evaluation artifact changed")
    rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
    scores, points = raw_outcomes(
        rows, p["evaluation"]["first_seed"], p["evaluation"]["pairs"]
    )
    if (
        report["pairs"] != len(rows)
        or report["matches"] != 2 * len(rows)
        or report["decisions"] != sum(r["decisions"] for r in rows)
    ):
        raise ValueError("Evaluation accounting differs")
    for key, actual in (
        ("win_score", scores.mean()),
        ("point_difference", points.mean()),
        ("win_ci95", interval(scores)),
        ("points_ci95", interval(points)),
    ):
        np.testing.assert_allclose(report[key], actual, rtol=0, atol=1e-12)
    return report, scores, points


def tensor_equal(left, right):
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            tensor_equal(left[k], right[k]) for k in left
        )
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            tensor_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def training(base, case, spec, p):
    folder = base / case
    checkpoint = folder / "final.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    initial = torch.load(folder / "initial.pt", map_location="cpu", weights_only=False)
    parent = torch.load(spec["parent"]["path"], map_location="cpu", weights_only=False)
    receipt = read(folder / "training_complete.json")
    assert sha(checkpoint) == receipt["checkpoint_sha256"]
    assert sha(spec["parent"]["path"]) == spec["parent"]["sha256"]
    assert payload["config"] == spec["config"] == initial["config"]
    assert payload["provenance"]["source_sha256"] == p["source"]["source_sha256"]
    counts = payload["counters"]
    assert counts == receipt["counters"]
    assert counts["updates"] == p["updates"] == 520
    assert counts["policy_updates"] == 500 and counts["warmup_updates"] == 20
    assert counts["decisions"] == p["decisions_per_case"]
    logs = [
        json.loads(line)
        for line in (folder / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(logs) == 520
    for update, row in enumerate(logs, 1):
        assert row["updates"] == update and row["decisions"] == update * 16384
        assert row["critic_only"] == (update <= 20)
        assert all(
            np.isfinite(value)
            for value in row.values()
            if isinstance(value, (float, int))
        )
    assert all(logs[-1][key] == value for key, value in counts.items())
    if spec["arm"] != "ppo":
        assert tensor_equal(initial["experts"], payload["experts"])
        assert tensor_equal(payload["experts"][0], parent["actor"])
        assert counts["optimizer_steps"] == 520 * 96
        assert (
            0
            <= counts["point_choices"]
            <= counts["disagreement_decisions"]
            <= counts["decisions"]
        )
    else:
        assert tensor_equal(initial["actor"], parent["actor"])
    return receipt


def analyze(base):
    p = read(base / "protocol.json")
    assert read(base / "process.json")["status"] == "complete"
    assert p["runner_sha256"] == sha("scripts/run_arbitration_study.py")
    training_reports, reports, samples = {}, {}, {}
    for case, spec in p["cases"].items():
        training_reports[case] = training(base, case, spec, p)
    for case in ["parent1", "parent2", *p["cases"]]:
        opponents = (
            p["evaluation"]["opponents"]
            if case in p["cases"]
            else p["evaluation"]["opponents"][:2]
        )
        for opponent in opponents:
            key = case + "/" + opponent
            reports[key], wins, points = outcome(base, case, opponent, p)
            samples[key] = (wins, points)
    comparisons = {}
    for parent in ("parent1", "parent2"):
        for opponent in p["evaluation"]["opponents"][:2]:
            for arm in ("full", "phase", "ppo"):
                for control in (parent, parent + "_phase", parent + "_ppo"):
                    candidate = parent + "_" + arm
                    if candidate == control:
                        continue
                    left, right = (
                        samples[candidate + "/" + opponent],
                        samples[control + "/" + opponent],
                    )
                    difference, point_delta = left[0] - right[0], left[1] - right[1]
                    comparisons[candidate + "-" + control + "/" + opponent] = {
                        "win_difference": float(difference.mean()),
                        "win_ci95": interval(difference),
                        "point_difference": float(point_delta.mean()),
                        "points_ci95": interval(point_delta),
                    }
    qualification = {
        arm: all(
            comparisons[f"parent{i}_{arm}-parent{i}/{opponent}"]["win_ci95"][0] > 0
            for i in (1, 2)
            for opponent in p["evaluation"]["opponents"][:2]
        )
        for arm in ("full", "phase", "ppo")
    }
    result = {
        "status": "passed",
        "protocol_sha256": sha(base / "protocol.json"),
        "analysis_script_sha256": sha(__file__),
        "training": training_reports,
        "outcomes": reports,
        "paired_comparisons": comparisons,
        "replicated_gain_over_parent_both_opponents": qualification,
        "scope": "Development inference conditional on two parents. Pair bootstrap does not quantify training-seed uncertainty. Hash/accounting/statistical checks do not replay every trajectory.",
    }
    target = base / "analysis.json"
    target.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "qualification": qualification,
                "outcomes": reports,
                "comparisons": comparisons,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("runs/arbitration_study"))
    analyze(parser.parse_args().base)


if __name__ == "__main__":
    main()
