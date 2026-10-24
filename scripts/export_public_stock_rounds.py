"""Reduce completed-round rule results without merging earlier research cohorts."""

import argparse
from pathlib import Path

import numpy as np

from koikoi.public_stock import PROFILES
from scripts.export_public_stock_results import (
    intervals,
    read,
    reduce_endpoint,
    sha,
    write,
)

CONTROL = "preview0_choice0_stakes0"
WORK = (
    "matches",
    "rounds",
    "decisions",
    "rollouts",
    "transitions",
    "roots",
    "python_replay_decisions",
)


def reduce(base, confirmation=None):
    study, screen = read(base / "protocol.json"), read(base / "screen_result.json")
    bindings = {
        str(base / p): sha(base / p) for p in ("protocol.json", "screen_result.json")
    }
    for path, digest in study["sources"].items():
        if sha(path) != digest:
            raise ValueError(f"Frozen study source changed: {path}")
        bindings[path] = digest
    curves, work, preparation = {}, {}, {}
    for profile in PROFILES:
        curves[profile] = []
        for parent in range(2):
            label = f"{profile}_parent{parent}"
            table = base / f"table_{label}"
            meta, audit = read(table / "result.json"), read(table / "audit.json")
            if (
                audit["status"] != "passed"
                or meta["table_sha256"] != sha(table / "table.npz")
                or audit["table_sha256"] != meta["table_sha256"]
            ):
                raise ValueError("Table audit does not bind this table")
            if (
                meta["checkpoint_sha256"] != study["checkpoints"][label]
                or meta["profile"] != profile
            ):
                raise ValueError("Wrong policy/rule continuation table")
            if (
                meta["config"]["samples"] != study["table_samples"]
                or meta["conditional_round_rollouts"]
                != 8 * 118 * study["table_samples"]
            ):
                raise ValueError("Table preparation budget differs")
            if (
                audit["all_target_values_reduced"] != meta["conditional_round_rollouts"]
                or audit["transitions"] != meta["transitions"]
            ):
                raise ValueError("Audit coverage differs")
            for path, digest in meta["sources"].items():
                if sha(path) != digest:
                    raise ValueError(f"Table source changed: {path}")
                bindings[path] = digest
            for path in [
                table / p
                for p in ("protocol.json", "result.json", "table.npz", "audit.json")
            ]:
                bindings[str(path)] = sha(path)
            for name, digest in meta["traces"].items():
                if sha(table / name) != digest:
                    raise ValueError("Conditional-round trace changed")
                bindings[str(table / name)] = digest
            preparation[label] = {
                k: meta[k]
                for k in (
                    "conditional_round_rollouts",
                    "terminal_round_rollouts",
                    "transitions",
                    "seconds",
                )
            }
            preparation[label].update(
                {
                    k: audit[k]
                    for k in ("python_round_replays", "python_replay_decisions")
                }
            )
            rows = []
            for particles in study["particles"]:
                endpoint = f"{label}_p{particles}"
                scores, result = reduce_endpoint(base / f"screen_{endpoint}", bindings)
                if (
                    result["config"]["deal_seed"] != study["development_seed"]
                    or result["config"]["pairs"] != study["development_pairs"]
                    or result["config"]["particles"] != particles
                    or result["profile"] != profile
                    or result["config"]["opponent"] != "self"
                    or any(
                        result["config"][key] != study[key]
                        for key in ("horizon", "confidence", "search_seed")
                    )
                ):
                    raise ValueError("Wrong development endpoint")
                if result["round_table_metadata_sha256"] != sha(table / "result.json"):
                    raise ValueError("Wrong table metadata")
                if (
                    result["round_table_sha256"] != meta["table_sha256"]
                    or result["checkpoint_sha256"] != study["checkpoints"][label]
                ):
                    raise ValueError("Wrong online table or actor")
                rows.append(scores)
                work[endpoint] = {k: result[k] for k in (*WORK, "seconds")}
            curves[profile].append(rows)
        curves[profile] = np.asarray(curves[profile])
        np.testing.assert_array_equal(
            curves[profile].mean(2), screen["curves"][profile]
        )
    gain = {p: rows[:, -1] - rows[:, 0] for p, rows in curves.items()}
    selected = min(
        (p for p in PROFILES if p != CONTROL), key=lambda p: (-gain[p].mean(1).min(), p)
    )
    gate = gain[selected].mean(1).min() > max(0.0, gain[CONTROL].mean(1).min())
    if (
        selected != screen["selected"]
        or bool(gate) != screen["confirmation_gate_passed"]
    ):
        raise ValueError("Selection or advancement gate does not reproduce")
    indices = np.random.default_rng(study["bootstrap_seed"]).integers(
        0, study["development_pairs"], (20000, study["development_pairs"])
    )
    summary = {
        "method": "completed_rounds",
        "particles": study["particles"],
        "development_pairs": study["development_pairs"],
        "selected": selected,
        "confirmation_gate_passed": bool(gate),
        "curves": {p: rows.mean(2).tolist() for p, rows in curves.items()},
        "development_gain_intervals_exploratory": {
            p: [intervals(row, indices) for row in rows] for p, rows in gain.items()
        },
        "development_curve_gain_intervals_exploratory": {
            p: [
                [intervals(budget - parent[0], indices) for budget in parent]
                for parent in rows
            ]
            for p, rows in curves.items()
        },
        "development_high_minus_low_exploratory": {
            p: [intervals(parent[-1] - parent[1], indices) for parent in rows]
            for p, rows in curves.items()
        },
        "screen_endpoints": work,
        "screen_totals": {k: sum(row[k] for row in work.values()) for k in WORK},
        "preparation": preparation,
        "preparation_totals": {
            k: sum(row[k] for row in preparation.values())
            for k in (
                "conditional_round_rollouts",
                "terminal_round_rollouts",
                "transitions",
                "python_round_replays",
                "python_replay_decisions",
            )
        },
        "new_policy_training_decisions": 0,
    }
    if confirmation is not None:
        cp = read(confirmation / "protocol.json")
        if (
            not gate
            or cp["profiles"] != [CONTROL, selected]
            or cp["screen_sha256"] != sha(base / "screen_result.json")
            or cp["study_protocol_sha256"] != sha(base / "protocol.json")
            or cp["seed"] != study["confirmation_seed"]
            or cp["pairs"] != study["confirmation_pairs"]
            or cp["particles"] != study["confirmation_particles"]
        ):
            raise ValueError("Unqualified confirmation")
        bindings[str(confirmation / "protocol.json")] = sha(
            confirmation / "protocol.json"
        )
        data, counters = {}, {}
        for profile in cp["profiles"]:
            data[profile] = []
            for parent in range(2):
                rows = []
                for particles in cp["particles"]:
                    label = f"{profile}_parent{parent}_p{particles}"
                    scores, result = reduce_endpoint(confirmation / label, bindings)
                    if (
                        result["config"]["deal_seed"] != cp["seed"]
                        or result["config"]["pairs"] != cp["pairs"]
                        or result["profile"] != profile
                        or result["config"]["particles"] != particles
                        or result["config"]["opponent"] != "self"
                        or any(
                            result["config"][key] != study[key]
                            for key in ("horizon", "confidence", "search_seed")
                        )
                        or result["checkpoint_sha256"]
                        != study["checkpoints"][f"{profile}_parent{parent}"]
                    ):
                        raise ValueError("Wrong confirmation stream")
                    rows.append(scores)
                    counters[label] = {k: result[k] for k in WORK}
                data[profile].append(rows)
            data[profile] = np.asarray(data[profile])
        indices = np.random.default_rng(cp["bootstrap_seed"]).integers(
            0, cp["pairs"], (cp["bootstrap_replicates"], cp["pairs"])
        )
        change = {p: rows[:, -1] - rows[:, 0] for p, rows in data.items()}
        checks = {
            "selected_gain": [
                intervals(row, indices, cp["lower_quantile"])
                for row in change[selected]
            ],
            "gain_minus_control": [
                intervals(row, indices, cp["lower_quantile"])
                for row in change[selected] - change[CONTROL]
            ],
        }
        summary["confirmation"] = {
            "checks": checks,
            "passed": all(row["lower"] > 0 for rows in checks.values() for row in rows),
            "curves": {p: rows.mean(2).tolist() for p, rows in data.items()},
            "high_minus_low_secondary": [
                intervals(row[-1] - row[1], indices) for row in data[selected]
            ],
            "endpoints": counters,
        }
    bindings[str(Path(__file__).resolve())] = sha(__file__)
    bindings["scripts/export_public_stock_results.py"] = sha(
        "scripts/export_public_stock_results.py"
    )
    summary["bindings"] = bindings
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", required=True)
    p.add_argument("--confirmation")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    result = reduce(
        Path(args.study), Path(args.confirmation) if args.confirmation else None
    )
    write(args.out, result)
    print(
        {
            k: result[k]
            for k in (
                "selected",
                "confirmation_gate_passed",
                "screen_totals",
                "preparation_totals",
            )
        }
    )


if __name__ == "__main__":
    main()
