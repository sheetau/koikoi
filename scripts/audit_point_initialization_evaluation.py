"""Native CPU action replay and separate compound-block statistics verification."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.env import BatchEnv
from koikoi.layout import PHASE, POINTS, TERMINAL
from scripts.point_initialization_analysis import analyse
from scripts.run_arbitration_study import read, sha, write


def replay(row):
    env = BatchEnv(1, "cpu", "paper8", row["seed"])
    assert (
        hashlib.sha256(env.state.numpy().tobytes()).hexdigest() == row["initial_sha256"]
    )
    rounds = 0
    for action in row["actions"]:
        assert (
            type(action) is int and 0 <= action < 50 and env.state[0, PHASE] != TERMINAL
        )
        _, legal = env.observe()
        assert legal[0, action]
        result = env.step(torch.tensor([action]), freeze_done=True)[0]
        assert result[0] != 99
        rounds += int(result[1])
    assert env.state[0, PHASE] == TERMINAL
    assert env.state[0].tolist() == row["endpoint"]
    margin = int(env.state[0, POINTS] - env.state[0, POINTS + 1])
    assert type(row["points"]) is int and row["points"] == margin * (
        1 - 2 * row["seat"]
    )
    expected = {
        "decisions": len(row["actions"]),
        "round_endings": rounds,
        "full_matches": 1,
    }
    assert expected == row["work"]
    return expected


def check_statistics(points, summary, seed, replicates):
    """Explicit seat/opponent sums, separate quantiles and bootstrap chunking."""
    columns = {}
    for parent in (1, 2):
        for scope, opponents in (("mixture", (0, 1)), ("wp", (0,)), ("point", (1,))):
            for policy, name in enumerate(
                ("parent", "wp_fixed", "point_fixed", "point_matched")
            ):
                wins, margins = [], []
                for deal in range(points.shape[3]):
                    w, p = 0.0, 0.0
                    for opponent in opponents:
                        for seat in (0, 1):
                            value = int(
                                points[parent - 1, policy, opponent, deal, seat]
                            )
                            w += 1 if value > 0 else 0.5 if value == 0 else 0
                            p += value
                    wins.append(w / (2 * len(opponents)))
                    margins.append(p / (2 * len(opponents)))
                prefix = f"parent{parent}/{scope}/{name}/"
                columns[prefix + "win"] = np.array(wins)
                columns[prefix + "points"] = np.array(margins)
            for left, right in (
                ("point_matched", "wp_fixed"),
                ("point_matched", "point_fixed"),
                ("point_matched", "parent"),
                ("point_fixed", "wp_fixed"),
                ("point_fixed", "parent"),
                ("wp_fixed", "parent"),
            ):
                for metric in ("win", "points"):
                    prefix = f"parent{parent}/{scope}/"
                    columns[prefix + left + "-" + right + "/" + metric] = (
                        columns[prefix + left + "/" + metric]
                        - columns[prefix + right + "/" + metric]
                    )
    assert set(columns) == set(summary["rows"])
    labels = list(columns)
    values = np.stack(list(columns.values()), axis=1)
    n = len(values)
    primary_labels = [
        f"parent{p}/mixture/point_matched-{other}/win"
        for p in (1, 2)
        for other in ("wp_fixed", "parent")
    ]
    indices = [labels.index(label) for label in primary_labels]
    mean = values.sum(0) / n
    se = np.sqrt(np.square(values[:, indices] - mean[indices]).sum(0) / (n - 1) / n)
    rng = np.random.default_rng(seed)
    samples, maxdev = [], []
    active = se > 0
    for first in range(0, replicates, 37):
        count = rng.multinomial(n, [1 / n] * n, size=min(37, replicates - first))
        sample = np.einsum("bi,ij->bj", count, values) / n
        samples.extend(sample)
        if active.any():
            deviation = abs(
                (sample[:, indices][:, active] - mean[indices][active]) / se[active]
            )
            maxdev.extend(deviation.max(1))
        else:
            maxdev.extend([0.0] * len(sample))
    critical = sorted(maxdev)[int(np.ceil(0.95 * (replicates - 1)))]
    np.testing.assert_allclose(
        critical, summary["primary"]["critical"], atol=2e-12, rtol=0
    )
    ordered = np.sort(np.asarray(samples), axis=0)
    intervals = []
    for quantile in (0.025, 0.975):
        index = quantile * (replicates - 1)
        lo, hi = int(np.floor(index)), int(np.ceil(index))
        intervals.append(
            ordered[lo] * (hi - index) + ordered[hi] * (index - lo)
            if hi != lo
            else ordered[lo]
        )
    for col, label in enumerate(labels):
        row = summary["rows"][label]
        np.testing.assert_allclose(row["mean"], mean[col], atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            row["marginal95"], [x[col] for x in intervals], atol=1e-12, rtol=0
        )
    signs = []
    for col, label in enumerate(primary_labels):
        row = summary["primary"]["rows"][label]
        np.testing.assert_allclose(row["mean"], mean[indices[col]], atol=1e-12, rtol=0)
        np.testing.assert_allclose(row["standard_error"], se[col], atol=1e-12, rtol=0)
        expected = mean[indices[col]] + np.array([-1, 1]) * critical * se[col]
        if not active[col]:
            assert row["simultaneous95"] == [None, None]
            sign = 0
        else:
            np.testing.assert_allclose(
                row["simultaneous95"], expected, atol=2e-12, rtol=0
            )
            sign = 1 if expected[0] > 0 else -1 if expected[1] < 0 else 0
        assert row["sign"] == sign
        signs.append(sign)
    assert summary["qualified"] == all(sign == 1 for sign in signs)
    assert summary["full_matches"] == points.size
    assert summary["compound_blocks"] == points.shape[3]
    assert summary["distinct_initial_deals"] == 2 * points.shape[3]
    assert (
        summary["bootstrap_seed"] == seed
        and summary["bootstrap_replicates"] == replicates
    )
    assert summary["primary"]["family_size"] == 4
    assert summary["primary"]["nondegenerate"] == int(active.sum())


def audit(root):
    if (root / "audit.json").exists() or (root / "summary.json").exists():
        raise ValueError("Preserve completed outputs")
    path = root / "evaluation_protocol.json"
    protocol = read(path)
    assert (
        read(root / "evaluation_complete.json")["status"] == "complete_awaiting_audit"
    )
    assert not (root / "evaluation_failed.json").exists()
    for name, expected in protocol["inputs"].items():
        assert sha(name) == expected, name
    pairs = protocol["pairs"]
    points = np.empty((2, 4, 2, pairs, 2), dtype=np.int64)
    inputs = dict(protocol["inputs"])
    inputs[str(path)] = sha(path)
    total = {"decisions": 0, "round_endings": 0, "full_matches": 0}
    assert len(protocol["cells"]) == 16
    initial_hashes = {}
    for parent in (1, 2):
        for policy_index, policy in enumerate(
            ("parent", "wp_fixed", "point_fixed", "point_matched")
        ):
            for opponent_index, opponent in enumerate(("wp", "point")):
                key = f"parent{parent}_{policy}_{opponent}"
                cell = protocol["cells"][key]
                assert (cell["parent"], cell["policy"], cell["opponent"]) == (
                    parent,
                    policy,
                    opponent,
                )
                report_path = root / "evaluation" / f"{key}.json"
                rows_path = report_path.with_suffix(".jsonl")
                report = read(report_path)
                assert report["status"] == "complete" and report["cell"] == key
                assert all(report[k] == v for k, v in cell.items())
                assert report["checkpoint_sha256"] == sha(cell["checkpoint"])
                assert report["protocol_sha256"] == sha(path) and report[
                    "rows_sha256"
                ] == sha(rows_path)
                assert (
                    report["runtime"]["device"] == "cpu"
                    and report["runtime"]["cpu_threads"] == 2
                )
                counts = dict.fromkeys(total, 0)
                with rows_path.open(encoding="utf-8") as stream:
                    for deal in range(pairs):
                        for seat in (0, 1):
                            row = json.loads(next(stream))
                            assert all(
                                row[k] == v
                                for k, v in {
                                    "parent": parent,
                                    "opponent": opponent,
                                    "replicate": deal,
                                    "seat": seat,
                                    "seed": cell["first_seed"] + deal,
                                }.items()
                            )
                            for name, value in replay(row).items():
                                counts[name] += value
                            identity = (opponent, deal)
                            if identity not in initial_hashes:
                                initial_hashes[identity] = row["initial_sha256"]
                            assert initial_hashes[identity] == row["initial_sha256"]
                            points[
                                parent - 1, policy_index, opponent_index, deal, seat
                            ] = row["points"]
                    assert not stream.read().strip()
                assert counts == report["work"] and counts["full_matches"] == 2 * pairs
                for name, value in counts.items():
                    total[name] += value
                inputs.update({str(p): sha(p) for p in (rows_path, report_path)})
                print(
                    f"Verified {key}: {counts['full_matches']} complete recorded matches",
                    flush=True,
                )
    assert total["full_matches"] == 32 * pairs
    bootstrap = protocol["bootstrap"]
    summary = analyse(points, bootstrap["seed"], bootstrap["replicates"])
    check_statistics(points, summary, bootstrap["seed"], bootstrap["replicates"])
    if protocol["mode"] == "capability":
        # Both Point fixtures share the same original actor and deployment path.
        np.testing.assert_array_equal(points[:, 2], points[:, 3])
        assert (points[:, 1, 0].sum(-1) == 0).all()
    summary["mode"] = protocol["mode"]
    summary["protocol_sha256"] = sha(path)
    write(root / "summary.json", summary)
    inputs[str(root / "summary.json")] = sha(root / "summary.json")
    for name, expected in inputs.items():
        assert sha(name) == expected, name
    write(
        root / "audit.json",
        {
            "status": "passed",
            "protocol_sha256": sha(path),
            "inputs": inputs,
            "work": total,
            "auditor_sha256": sha(__file__),
            "qualified": summary["qualified"] if protocol["mode"] == "main" else False,
            "scope": "All recorded actions replayed through native CPU against reference producer endpoints; all initial deals, legal actions, work and points checked. Independent scalar compound-block statistics reproduce all120 marginal intervals and four simultaneous bands. No neural-inference replay. Capability is not strength evidence.",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    audit(args.input)


if __name__ == "__main__":
    main()
