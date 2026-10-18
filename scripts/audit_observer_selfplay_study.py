"""Independently recompute the closed observer-self-play study from complete deal pairs.

Uses joint per-deal resampling rather than the production analysis helpers.
No inference replay, new simulation, outcome selection or training is performed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.train import provenance


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def same(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            same(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def audit(base, output, allow_smoke=False, optimizer_dir=None):
    if output.exists():
        raise ValueError("Use a fresh output path")
    torch.set_num_threads(2)
    state = read(base / "process.json")
    if state["status"] != "complete":
        raise ValueError("The observer-self-play study is not terminal and audited yet")
    p, analysis = read(base / "protocol.json"), read(base / "analysis.json")
    smoke = p["kind"] == "observer_selfplay_smoke"
    assert (allow_smoke and smoke) or p["kind"] == "observer_selfplay_development"
    assert analysis["status"] == "passed"
    assert analysis["kind"] == p["kind"]
    assert (
        state["protocol_sha256"]
        == analysis["protocol_sha256"]
        == sha(base / "protocol.json")
    )
    assert state["analysis_sha256"] == sha(base / "analysis.json")
    assert p["source"]["source_sha256"] == provenance()["source_sha256"]
    inputs = {
        str(path): sha(path)
        for path in (
            base / "protocol.json",
            base / "process.json",
            base / "analysis.json",
            Path(__file__),
        )
    }
    for path, digest in p["inputs"].items():
        assert sha(path) == digest, path
        inputs[path] = digest
    parents = ["parent1"] if smoke else ["parent1", "parent2"]
    count, first = (2, 26061000) if smoke else (5000, 27000000)
    cases = list(p["controls"]) + list(p["cases"])
    opponents = ["rl-wp-assisted", "rl-point-assisted"]
    assert set(p["cases"]) == {
        f"{parent}_{arm}" for parent in parents for arm in ("observer", "signed")
    }
    assert set(p["controls"]) == {
        parent + suffix for parent in parents for suffix in ("", "_selfplay")
    }
    assert p["resources"] == {"trainers": 1, "evaluators": 1 if smoke else 3}
    assert p["evaluation"]["opponents"] == opponents
    assert p["evaluation"]["pairs"] == count and p["evaluation"]["first_seed"] == first
    assert p["evaluation"]["threads"] == 2
    jobs = {(j["mode"], j["case"], j["opponent"]): j for j in state["jobs"]}
    expected = {("train", case, None) for case in p["cases"]}
    expected |= {
        ("evaluate", case, opponent) for case in cases for opponent in opponents
    }
    assert len(jobs) == len(state["jobs"]) and set(jobs) == expected
    events = []
    for (mode, case, _), job in jobs.items():
        assert job["returncode"] == 0 and job["started"] < job["ended"]
        events += [(job["started"], 1, mode), (job["ended"], -1, mode)]
        if mode == "evaluate" and case in p["cases"]:
            assert jobs["train", case, None]["ended"] <= job["started"]
    source = Path(p["source_study"])
    prior_state = read(source / "process.json")
    prior_audit = read(source / "independent_audit.json")
    assert prior_state["status"] == "complete" and prior_audit["status"] == "passed"
    assert prior_state["protocol_sha256"] == p["source_protocol_sha256"]
    assert prior_state["analysis_sha256"] == sha(source / "analysis.json")
    for path, digest in prior_audit["inputs"].items():
        assert sha(path) == digest, path
        inputs[path] = digest
    for name in ("process.json", "independent_audit.json"):
        path = source / name
        inputs[str(path)] = sha(path)
    for mode in ("train", "evaluate"):
        previous_end = max(j["ended"] for j in prior_state["jobs"] if j["mode"] == mode)
        next_start = min(j["started"] for j in jobs.values() if j["mode"] == mode)
        assert previous_end <= next_start
    for job in prior_state["jobs"]:
        assert job["returncode"] == 0 and job["started"] < job["ended"]
        events += [(job["started"], 1, job["mode"]), (job["ended"], -1, job["mode"])]
    active = {"train": 0, "evaluate": 0}
    peak = active.copy()
    for _, delta, mode in sorted(events):
        active[mode] += delta
        peak[mode] = max(peak[mode], active[mode])
        assert 0 <= active["train"] <= 1 and 0 <= active["evaluate"] <= 3
    assert all(value == 0 for value in active.values())
    training_totals = {
        key: 0 for key in ("decisions", "rounds", "matches", "value_rows")
    }
    optimizer_dir = (
        Path("runs/observer_selfplay_optimizer_audits")
        if optimizer_dir is None
        else Path(optimizer_dir)
    )
    for case in p["cases"]:
        report_path = base / case / "training_audit.json"
        optimizer_path = optimizer_dir / (case + ".json")
        training, optimizer = read(report_path), read(optimizer_path)
        assert training == analysis["training"][case] and training["status"] == "passed"
        assert optimizer["status"] == "passed" and optimizer["case"] == case
        for path, digest in optimizer["inputs"].items():
            assert sha(path) == digest, path
            inputs[path] = digest
        for path in (report_path, optimizer_path):
            inputs[str(path)] = sha(path)
        config = p["cases"][case]["config"]
        assert (
            training["counters"]["decisions"]
            == p["updates"] * config["envs"] * config["horizon"]
            == (128 if smoke else 8519680)
        )
        assert (
            training["counters"]["value_rows"] == 2 * training["counters"]["decisions"]
        )
        for key in training_totals:
            training_totals[key] += training["counters"][key]
    warmups = {}
    for parent in parents:
        pair = []
        for arm in ("observer", "signed"):
            path = base / f"{parent}_{arm}" / "warmup.pt"
            inputs[str(path)] = sha(path)
            pair.append(torch.load(path, map_location="cpu", weights_only=False))
        for key in (
            "actor",
            "value",
            "optimizer",
            "env",
            "counters",
            "torch_rng",
            "cuda_rng",
        ):
            assert same(pair[0][key], pair[1][key]), (parent, key)
        warmups[parent] = {
            "status": "passed",
            "observer_sha256": inputs[str(base / f"{parent}_observer" / "warmup.pt")],
            "signed_sha256": inputs[str(base / f"{parent}_signed" / "warmup.pt")],
        }
        assert warmups[parent] == analysis["matched_warmup"][parent]
    records, samples = {}, {}
    for case in cases:
        for opponent in opponents:
            key = case + "/" + opponent
            report_path = base / case / (opponent + ".json")
            raw_path = report_path.with_suffix(".jsonl")
            report = read(report_path)
            assert report == analysis["outcomes"][key]
            candidate = (
                base / case / "final.pt"
                if case in p["cases"]
                else Path(p["controls"][case]["path"])
            )
            assert Path(report["candidate"]).resolve() == candidate.resolve()
            assert report["checkpoint_sha256"] == sha(candidate)
            assert report["raw_sha256"] == sha(raw_path)
            assert report["case"] == case and report["opponent"] == opponent
            for path in (report_path, raw_path, candidate):
                inputs[str(path)] = sha(path)
            rows = [
                json.loads(line)
                for line in raw_path.read_text(encoding="utf-8").splitlines()
            ]
            assert [row["seed"] for row in rows] == list(range(first, first + count))
            values = []
            for row in rows:
                assert type(row["decisions"]) is int and row["decisions"] > 0
                for seat in (0, 1):
                    point = row[f"seat{seat}_point_difference"]
                    assert type(point) is int
                    assert row[f"seat{seat}_score"] == (
                        0.5 if point == 0 else float(point > 0)
                    )
                values.append(
                    [
                        (row["seat0_score"] + row["seat1_score"]) / 2,
                        (row["seat0_point_difference"] + row["seat1_point_difference"])
                        / 2,
                    ]
                )
            assert report["pairs"] == count and report["matches"] == 2 * count
            assert report["decisions"] == sum(row["decisions"] for row in rows)
            records[key], samples[key] = report, np.array(values)
    contrasts = {}
    for parent in parents:
        for opponent in opponents:
            for arm, control in (
                ("observer", ""),
                ("signed", ""),
                ("selfplay", ""),
                ("observer", "_selfplay"),
                ("observer", "_signed"),
                ("signed", "_selfplay"),
            ):
                left, right = parent + "_" + arm, parent + control
                key = left + "-" + right + "/" + opponent
                contrasts[key] = (
                    samples[left + "/" + opponent] - samples[right + "/" + opponent]
                )
    combined = list(samples.items()) + list(contrasts.items())
    values = np.column_stack([value for _, value in combined])
    rng = np.random.default_rng(99)
    bootstrap = np.empty((5000, values.shape[1]))
    for replicate in range(5000):
        indices = rng.integers(count, size=count)
        bootstrap[replicate] = values[indices].mean(axis=0)
    intervals = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    recomputed = {}
    for index, (key, value) in enumerate(combined):
        means, bounds = value.mean(axis=0), intervals[:, 2 * index : 2 * index + 2]
        report = records[key] if key in records else analysis["contrasts"][key]
        names = (
            ("win_score", "point_difference", "win_ci95", "points_ci95")
            if key in records
            else ("win_change", "point_change", "win_ci95", "point_ci95")
        )
        for name, actual in zip(
            names, (means[0], means[1], bounds[:, 0], bounds[:, 1]), strict=True
        ):
            np.testing.assert_allclose(report[name], actual, rtol=0, atol=1e-12)
        recomputed[key] = {
            "win": float(means[0]),
            "points": float(means[1]),
            "win_ci95": bounds[:, 0].tolist(),
            "points_ci95": bounds[:, 1].tolist(),
        }
    assert set(records) == set(analysis["outcomes"])
    assert set(contrasts) == set(analysis["contrasts"])
    primary = {
        f"{parent}_observer-{parent}{control}/{opponent}": recomputed[
            f"{parent}_observer-{parent}{control}/{opponent}"
        ]["win_ci95"][0]
        > 0
        for parent in parents
        for control in ("", "_signed")
        for opponent in opponents
    }
    qualified = None if smoke else all(primary.values())
    assert qualified == analysis["replicated_observer_gain"]
    assert all(sha(path) == digest for path, digest in inputs.items())
    result = {
        "status": "passed",
        "inputs": inputs,
        "resource_peaks": peak,
        "kind": p["kind"],
        "matched_warmup": warmups,
        "training_totals": training_totals,
        "evaluation_matches": len(cases) * len(opponents) * 2 * count,
        "evaluation_decisions": sum(row["decisions"] for row in records.values()),
        "shared_unique_deal_streams": count,
        "recomputed": recomputed,
        "primary_positive_lower_bounds": primary,
        "replicated_observer_gain": qualified,
        "scope": "Independent raw-pair outcomes and all contrasts with joint 5000-replicate seed99 bootstrap; combined original/new closed-job schedule and pinned training/optimizer receipts. No inference replay, new simulation, checkpoint selection or training-seed uncertainty estimate.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha(output),
                "replicated_observer_gain": result["replicated_observer_gain"],
                "evaluation_decisions": result["evaluation_decisions"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("runs/observer_selfplay_study")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--optimizer-dir", type=Path)
    args = parser.parse_args()
    audit(args.base, args.output, args.allow_smoke, args.optimizer_dir)
