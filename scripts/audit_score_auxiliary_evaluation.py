"""Native replay and separately computed six-contrast score-auxiliary evidence."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_point_initialization_evaluation import replay
from scripts.audit_score_auxiliary_statistics import check_statistics
from scripts.evaluate_point_initialization import write_new
from scripts.run_arbitration_study import read, sha
from scripts.score_auxiliary_analysis import analyse


def audit(root):
    if (root / "audit.json").exists() or (root / "summary.json").exists():
        raise ValueError("Preserve completed outputs")
    path = root / "evaluation_protocol.json"
    protocol = read(path)
    completion = read(root / "evaluation_complete.json")
    assert completion["status"] == "complete_awaiting_audit"
    assert completion["protocol_sha256"] == sha(path)
    assert len(completion["completed"]) == 16
    assert set(completion["completed"]) == set(protocol["cells"])
    assert not (root / "evaluation_failed.json").exists()
    for name, expected in protocol["inputs"].items():
        assert sha(name) == expected, name
    pairs = protocol["pairs"]
    assert protocol["mode"] in ("main", "capability")
    assert pairs == (4096 if protocol["mode"] == "main" else 2)
    assert (
        protocol["main_training_protocol_sha256"]
        == "9ba0bd7b1d187d59269af336537ceab890bf8884a969a9df073dcd626e686f46"
    )
    assert protocol["bootstrap"] == {"seed": 109310001, "replicates": 10000}
    points = np.empty((2, 4, 2, pairs, 2), dtype=np.int64)
    inputs = dict(protocol["inputs"])
    inputs[str(path)] = sha(path)
    inputs[str(root / "evaluation_complete.json")] = sha(
        root / "evaluation_complete.json"
    )
    total = {"decisions": 0, "round_endings": 0, "full_matches": 0}
    assert len(protocol["cells"]) == 16
    initial_hashes = {}
    for parent in (1, 2):
        for policy_index, policy in enumerate(("parent", "none", "win", "score")):
            for opponent_index, opponent in enumerate(("wp", "point")):
                key = f"parent{parent}_{policy}_{opponent}"
                cell = protocol["cells"][key]
                expected_first = (
                    (109110001, 109210001)
                    if protocol["mode"] == "main"
                    else (110010001, 110020001)
                )[opponent_index]
                assert cell["first_seed"] == expected_first
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
    summary["mode"] = protocol["mode"]
    summary["protocol_sha256"] = sha(path)
    write_new(root / "summary.json", summary)
    inputs[str(root / "summary.json")] = sha(root / "summary.json")
    for name, expected in inputs.items():
        assert sha(name) == expected, name
    write_new(
        root / "audit.json",
        {
            "status": "passed",
            "protocol_sha256": sha(path),
            "inputs": inputs,
            "work": total,
            "auditor_sha256": sha(__file__),
            "qualified": summary["qualified"] if protocol["mode"] == "main" else False,
            "scope": "All recorded actions replayed through native CPU against reference producer endpoints; all initial deals, legal actions, work and points checked. Independent scalar compound-block statistics reproduce all120 marginal intervals and six simultaneous bands. No neural-inference replay. Capability is not strength evidence.",
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
