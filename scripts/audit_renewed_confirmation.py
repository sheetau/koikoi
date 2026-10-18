"""Independent trajectory and scalar-statistics audit of final confirmation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_public_continuation_study import replay_match
from scripts.renewed_confirmation_gate import check
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.run_final_round_training_study import checked_inputs

POLICIES = ("parent", "fixed_public", "renewed_public")
BOOTSTRAP_DRAWS = 10000


def verify_identity(row, parent, opponent):
    assert type(row["parent"]) is int and row["parent"] == parent
    assert row["opponent"] == opponent
    index = POLICIES.index(row["policy"])
    rep, seat = row["replicate"], row["seat"]
    assert type(rep) is int and 0 <= rep < 4096
    assert type(seat) is int and seat in (0, 1)
    assert type(row["seed"]) is int
    assert row["seed"] == 101100001 + 10000 * (opponent == "point") + rep
    assert (row["memory"] is None) == (index == 0)
    scores = row["terminal_points"]
    assert len(scores) == 2 and all(type(p) is int for p in scores)
    assert sum(scores) == 60
    return index, rep, seat


def audit_cell(path, parent, opponent):
    points = np.empty((3, 4096, 2))
    seen, prefixes, memories = set(), {}, {}
    work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    checks = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = verify_identity(row, parent, opponent)
            assert key not in seen
            seen.add(key)
            index, rep, seat = key
            prefix, counts, nchecks = replay_match(row)
            checks += nchecks
            for name, count in counts.items():
                work[name] += count
            if index == 0:
                prefixes[rep, seat] = prefix
            else:
                assert prefix == prefixes[rep, seat][: len(prefix)]
                if row["memory"]["activation"] is None:
                    assert prefix == prefixes[rep, seat]
                evidence = {
                    name: row["memory"][name]
                    for name in (
                        "events",
                        "candidates",
                        "prediction_calls",
                        "first_identification",
                        "activation",
                    )
                }
                if index == 1:
                    memories[rep, seat] = evidence
                else:
                    assert evidence == memories[rep, seat]
            points[index, rep, seat] = (
                row["terminal_points"][seat] - row["terminal_points"][1 - seat]
            )
    assert len(seen) == work["completed_matches"] == 24576
    return points, work, checks


def interval(values):
    values = np.asarray(values, dtype=float)
    if np.all(np.ptp(values, axis=1) == 0):
        return [float(values.mean())] * 2
    rng = np.random.default_rng(101300001)
    samples = []
    for start in range(0, BOOTSTRAP_DRAWS, 128):
        count = min(128, BOOTSTRAP_DRAWS - start)
        indices = rng.integers(
            values.shape[1], size=(count, values.shape[0], values.shape[1])
        )
        averages = np.zeros((count, values.shape[0]))
        for group in range(values.shape[0]):
            averages[:, group] = values[group][indices[:, group]].mean(axis=1)
        samples.extend(averages.mean(axis=1))
    return np.percentile(samples, [2.5, 97.5]).tolist()


def independent_analysis(points):
    """No producer loader, bootstrap, outcome transform, or decision helper."""
    assert points.shape == (2, 3, 2, 4096, 2) and np.isfinite(points).all()
    contrasts, absolute = [], []
    for parent in (1, 2):
        for scope, groups in (("mixture", [0, 1]), ("wp", [0]), ("point", [1])):
            for candidate, baseline in ((2, 0), (2, 1), (1, 0)):
                row = {
                    "parent": parent,
                    "scope": scope,
                    "comparison": f"{POLICIES[candidate]}-minus-{POLICIES[baseline]}",
                }
                for metric in ("win", "point"):
                    outcomes = points[parent - 1].copy()
                    if metric == "win":
                        outcomes = np.where(
                            outcomes > 0, 1.0, np.where(outcomes < 0, 0.0, 0.5)
                        )
                    difference = (
                        outcomes[candidate, groups] - outcomes[baseline, groups]
                    )
                    paired = (difference[:, :, 0] + difference[:, :, 1]) / 2
                    row[metric] = {
                        "change": float(paired.mean()),
                        "interval95": interval(paired),
                    }
                contrasts.append(row)
            for candidate, name in enumerate(POLICIES):
                row = {"parent": parent, "scope": scope, "policy": name}
                for metric in ("win", "point"):
                    outcomes = points[parent - 1, candidate, groups].copy()
                    if metric == "win":
                        outcomes = np.where(
                            outcomes > 0, 1.0, np.where(outcomes < 0, 0.0, 0.5)
                        )
                    paired = (outcomes[:, :, 0] + outcomes[:, :, 1]) / 2
                    row[metric] = {
                        "mean": float(paired.mean()),
                        "interval95": interval(paired),
                    }
                absolute.append(row)
    primary = [
        row
        for row in contrasts
        if row["scope"] == "mixture"
        and row["comparison"]
        in (
            "renewed_public-minus-parent",
            "renewed_public-minus-fixed_public",
        )
    ]
    assert len(primary) == 4
    return {
        "contrasts": contrasts,
        "absolute": absolute,
        "confirmed_strength_gain": all(r["win"]["interval95"][0] > 0 for r in primary),
    }


def verify_plan(plan):
    assert plan["kind"] == "renewed_confirmation_v1"
    for key, expected in {
        "policies": list(POLICIES),
        "pairs_per_opponent": 4096,
        "deal_start": 101100001,
        "distinct_deals": 8192,
        "matches": 98304,
        "bootstrap_seed": 101300001,
        "bootstrap_replicates": 10000,
        "workers": 4,
        "torch_threads_per_worker": 2,
        "automatic_promotion": False,
    }.items():
        assert plan[key] == expected, key
    qualification = check(Path(plan["study"]))
    assert qualification["qualified"] and qualification == plan["qualification"]
    assert plan["parents"] == [
        {
            "index": index,
            "models": {
                "parent": path,
                "fixed_public": str(
                    Path("runs/public_continuation_study")
                    / f"parent{index}_public/final.pt"
                ),
                "renewed_public": str(
                    Path(plan["study"]) / f"parent{index}_public/final.pt"
                ),
            },
        }
        for index, (path, _) in enumerate(PARENTS, 1)
    ]
    bound = {Path(p).resolve(): digest for p, digest in plan["inputs"].items()}
    for parent in plan["parents"]:
        for path in parent["models"].values():
            assert bound[Path(path).resolve()] == sha(path)
    checked_inputs(plan)


def audit(root):
    output = root / "independent_audit.json"
    if output.exists():
        raise ValueError("Refusing to overwrite independent audit")
    assert read(root / "process.json")["status"] == "complete"
    plan, produced = read(root / "protocol.json"), read(root / "analysis.json")
    assert produced["status"] == "complete"
    verify_plan(plan)
    inputs = dict(plan["inputs"])
    for path, digest in produced["inputs"].items():
        assert path not in inputs or inputs[path] == digest
        inputs[path] = digest
    paths = [root / "protocol.json", root / "analysis.json"] + [
        root / f"eval_parent{p}_{opponent}.jsonl"
        for p in (1, 2)
        for opponent in ("wp", "point")
    ]
    assert all(
        str(path) in produced["inputs"]
        for path in paths
        if path.name != "analysis.json"
    )
    inputs[str(root / "analysis.json")] = sha(root / "analysis.json")
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    points = np.empty((2, 3, 2, 4096, 2))
    work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    checks = 0
    for parent in (1, 2):
        for group, opponent in enumerate(("wp", "point")):
            values, counts, checked = audit_cell(
                root / f"eval_parent{parent}_{opponent}.jsonl", parent, opponent
            )
            points[parent - 1, :, group] = values
            checks += checked
            for name, value in counts.items():
                work[name] += value
    result = independent_analysis(points)
    assert all(result[key] == produced[key] for key in result)
    assert work["completed_matches"] == 98304
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    write(
        output,
        {
            "status": "passed",
            **result,
            "inputs": inputs,
            "evaluation_replay_work": work,
            "strict_input_checks": checks,
            "distinct_evaluation_deals": 8192,
            "new_neural_inferences": 0,
            "new_independent_matches": 0,
            "development_mechanism_claims": plan["qualification"]["development_claims"],
            "scope": "Replays all recorded actions/public histories and independently recomputes statistics; does not rerun neural predictions or confirm mechanism effects.",
        },
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
