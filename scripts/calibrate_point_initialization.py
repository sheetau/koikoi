"""Fix Point phase temperatures from archived public scores, without outcomes."""

import argparse
import math
from pathlib import Path

import numpy as np

from koikoi.layout import PHASE
from scripts.compare_published_initializations import scalar_entropy
from scripts.koi_temperature_mechanism import CORPUS, CORPUS_SHA
from scripts.phase_exploration_calibration import (
    calibrate_temperature,
    choice_statistics,
)
from scripts.run_arbitration_study import sha, write


def fit(data):
    sources = data["sources"]
    phases = data["states"][:, PHASE]
    legal = data["masks"] & (
        (data["guards"][:, None] < 0) | (np.arange(50)[None] == data["guards"][:, None])
    )
    eligible = (sources >= 0) & (legal.sum(1) > 1)
    calibration = eligible & (sources % 4 != 3)
    validation = eligible & (sources % 4 == 3)
    groups = {
        split: sorted(int(x) for x in np.unique(sources[selected]))
        for split, selected in (
            ("calibration", calibration),
            ("validation", validation),
        )
    }
    if not groups["calibration"] or not groups["validation"]:
        raise ValueError("Both source-group partitions must be nonempty")
    if set(groups["calibration"]) & set(groups["validation"]):
        raise ValueError("Source groups overlap")
    temperatures, records = [], {}
    for phase in range(4):
        rows = calibration & (phases == phase)
        target = choice_statistics(data["cpu_wp"][rows], legal[rows], 0.05)[
            "mean_normalized_entropy"
        ]
        temperature = calibrate_temperature(
            data["cpu_point"][rows], legal[rows], target
        )
        temperatures.append(temperature)
        record = {"target_normalized_entropy": target}
        for split, selected in (
            ("calibration", calibration),
            ("validation", validation),
        ):
            rows = selected & (phases == phase)
            values = {}
            for name, variant, divisor in (
                ("wp_fixed", "wp", 0.05),
                ("point_fixed", "point", 0.05),
                ("point_matched", "point", temperature),
            ):
                scores = data["cpu_" + variant][rows]
                stats = choice_statistics(scores, legal[rows], divisor)
                independent = scalar_entropy(scores, legal[rows], divisor)
                if not math.isclose(
                    independent,
                    stats["mean_normalized_entropy"],
                    abs_tol=1e-12,
                    rel_tol=1e-12,
                ):
                    raise ValueError("Scalar entropy verification failed")
                values[name] = stats
            record[split] = values
        actual = record["calibration"]["point_matched"]["mean_normalized_entropy"]
        if not math.isclose(actual, target, abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError("Calibration did not reach its fixed target")
        records[str(phase)] = record
    return {
        "phase_temperatures": temperatures,
        "source_groups": groups,
        "phases": records,
    }


def run(output):
    if output.exists():
        raise ValueError("Preserve completed calibration")
    if sha(CORPUS) != CORPUS_SHA:
        raise ValueError("Archived public-score corpus changed")
    with np.load(CORPUS, allow_pickle=False) as data:
        result = fit(data)
    paths = [
        CORPUS,
        Path(__file__),
        Path("koikoi/layout.py"),
        Path("scripts/phase_exploration_calibration.py"),
        Path("scripts/compare_published_initializations.py"),
        Path("scripts/koi_temperature_mechanism.py"),
        Path("scripts/run_arbitration_study.py"),
        Path("tests/test_calibrate_point_initialization.py"),
        Path("docs/POINT_INITIALIZATION_STUDY.rst"),
    ]
    result.update(
        {
            "status": "complete",
            "inputs": {str(p): sha(p) for p in paths},
            "selection": "Unforced nonnegative source IDs; modulo4 !=3 calibration, ==3 validation. Existing reused groups, not fresh held-out games.",
            "new_work": {
                "neural_calls": 0,
                "simulator_decisions": 0,
                "training_updates": 0,
                "matches": 0,
            },
            "scope": "Initial per-phase mean normalized entropy on archived CPU scores only. Does not match action distributions, gradients, on-policy visitation or later entropy. No outcomes or validation optimization. Not strength evidence or a novel algorithm.",
            "scalar_entropy_check": "passed",
        }
    )
    write(output, result)
    print(
        {
            "phase_temperatures": result["phase_temperatures"],
            "source_groups": {k: len(v) for k, v in result["source_groups"].items()},
        }
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
