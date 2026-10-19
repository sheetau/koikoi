"""Compare archived public WP/Point scores without new inference or outcomes."""

import argparse
import math
from pathlib import Path

import numpy as np

from koikoi.layout import PHASE
from scripts.koi_temperature_mechanism import CORPUS, CORPUS_SHA
from scripts.phase_exploration_calibration import choice_statistics
from scripts.run_arbitration_study import read, sha, write


def scalar_entropy(scores, mask, temperature):
    values = []
    for row, legal in zip(scores, mask, strict=True):
        q = [float(x) for x in row[legal]]
        largest = max(q)
        weights = [math.exp((x - largest) / temperature) for x in q]
        denominator = math.fsum(weights)
        probabilities = [w / denominator for w in weights]
        entropy = -math.fsum(p * math.log(p) for p in probabilities if p > 0)
        values.append(entropy / math.log(len(q)))
    return math.fsum(values) / len(values)


def run(output):
    if output.exists():
        raise ValueError("Preserve previous diagnostic")
    assert sha(CORPUS) == CORPUS_SHA
    paths = [
        CORPUS,
        Path(__file__),
        Path("koikoi/layout.py"),
        Path("scripts/phase_exploration_calibration.py"),
        Path("scripts/koi_temperature_mechanism.py"),
    ]
    inventory = []
    for path in sorted(Path("runs").rglob("config.json")):
        config = read(path)
        if config.get("kind") == "published":
            paths.append(path)
            inventory.append(
                {
                    "path": str(path),
                    "sha256": sha(path),
                    "initial_variant": config.get("initial_variant"),
                }
            )
    with np.load(CORPUS, allow_pickle=False) as data:
        selected = (data["sources"] >= 0) & (data["sources"] % 4 == 3)
        phases = data["states"][:, PHASE]
        mask = data["masks"] & (
            (data["guards"][:, None] < 0)
            | (np.arange(50)[None] == data["guards"][:, None])
        )
        groups = {}
        for phase in range(4):
            rows = selected & (phases == phase) & (mask.sum(1) > 1)
            legal = mask[rows]
            record = {"unforced_rows": int(rows.sum()), "models": {}}
            actions = {}
            for variant in ("wp", "point"):
                scores = data["cpu_" + variant][rows]
                assert np.isfinite(scores[legal]).all()
                record["models"][variant] = choice_statistics(scores, legal, 0.05)
                assert math.isclose(
                    scalar_entropy(scores, legal, 0.05),
                    record["models"][variant]["mean_normalized_entropy"],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                actions[variant] = np.where(legal, scores, -np.inf).argmax(1)
            record["different_greedy_actions"] = int(
                (actions["wp"] != actions["point"]).sum()
            )
            if phase == 3:
                assert (legal.sum(1) == 2).all() and legal[:, 48:].all()
                record["wp_stop_point_continue"] = int(
                    ((actions["wp"] == 48) & (actions["point"] == 49)).sum()
                )
                record["wp_continue_point_stop"] = int(
                    ((actions["wp"] == 49) & (actions["point"] == 48)).sum()
                )
            groups[str(phase)] = record
        source_count = len(np.unique(data["sources"][selected]))
        rows_count = int(selected.sum())
    write(
        output,
        {
            "status": "complete",
            "inputs": {str(p): sha(p) for p in paths},
            "inventory": inventory,
            "groups": groups,
            "corpus_selection": "nonnegative source ID modulo4 ==3, same old diagnostic split",
            "source_groups": source_count,
            "all_rows": rows_count,
            "temperature": 0.05,
            "scalar_entropy_check": "passed",
            "new_work": {
                "neural_calls": 0,
                "simulator_decisions": 0,
                "training_updates": 0,
                "matches": 0,
            },
            "scope": "Descriptive archived CPU score comparison, not inference replay, deployment-float argmax certification or strength evidence. Config inventory covers existing config.json files only, including capabilities and descendants; entries are not independent studies. No temperature or model selected.",
        },
    )
    print(
        {
            "all_rows": rows_count,
            "source_groups": source_count,
            "inventory_entries": len(inventory),
            "groups": groups,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
