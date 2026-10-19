"""Six fixed Point/WP learners; evaluation is separately gated and not run here."""

import argparse
import json
import math
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import torch

from scripts.audit_pretrained_ppo_screen import tensor_digest, validate_accounting
from scripts.phase_temperature_ppo import (
    PhaseTemperatureConfig,
    PhaseTemperatureTrainer,
)
from scripts.run_arbitration_study import PARENTS, read, sha

CAPABILITY = Path("runs/point_initialization_capability")
CALIBRATION = Path("runs/point_initialization_calibration.json")
CALIBRATION_SHA = "869ab1e45e7e8a5569ff20edaea55fdc5e2a2464d6d6a32a9c50df188bd701f3"
AUDIT_SHA = "f8d3036e9af6b55b21cf94275515fd0c19f61b737539931133f1e38b56538367"
SEED_SCAN = Path("runs/point_initialization_main_seed_scan.json")
SCAN_SHA = "dfa5688894853738b608b0c79c530027cc6a0e7dc0854d54b1b932244aa98e68"


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def checked(inputs):
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError(f"Changed frozen input: {path}")


def prepare(root):
    if root.exists():
        raise ValueError("Preserve existing study")
    suite = (
        ET.parse("runs/point_initialization_runner_tests_final.xml")
        .getroot()
        .find("testsuite")
    )
    assert int(suite.attrib["tests"]) >= 3
    assert all(int(suite.attrib[k]) == 0 for k in ("errors", "failures", "skipped"))
    assert sha(CALIBRATION) == CALIBRATION_SHA
    assert sha(CAPABILITY / "independent_audit.json") == AUDIT_SHA
    assert sha(SEED_SCAN) == SCAN_SHA
    scan = read(SEED_SCAN)
    assert not scan["overlaps"] and not scan["unreadable"]
    capability = read(CAPABILITY / "independent_audit.json")
    assert capability["status"] == "passed"
    inputs = dict(capability["inputs"])
    for path in (
        Path(__file__),
        CAPABILITY / "independent_audit.json",
        SEED_SCAN,
        Path("docs/POINT_INITIALIZATION_MAIN.rst"),
        Path("tests/test_train_point_initialization.py"),
        Path("runs/point_initialization_runner_tests_final.xml"),
    ):
        inputs[str(path)] = sha(path)
    for path, digest in PARENTS:
        assert sha(path) == digest
        inputs[str(path)] = digest
    checked(inputs)
    temperatures = read(CALIBRATION)["phase_temperatures"]
    configs = {}
    for index, seed in enumerate((102110001, 102210001), 1):
        for mode, variant, divisors in (
            ("wp_fixed", "wp", [0.05] * 4),
            ("point_fixed", "point", [0.05] * 4),
            ("point_matched", "point", temperatures),
        ):
            configs[f"seed{index}_{mode}"] = asdict(
                PhaseTemperatureConfig(
                    device="cuda",
                    seed=seed,
                    initial_variant=variant,
                    phase_temperatures=divisors,
                )
            )
    protocol = {
        "inputs": inputs,
        "configs": configs,
        "updates": 520,
        "warmup": 20,
        "decisions_per_arm": 8519680,
        "total_decisions": 51118080,
        "evaluation": {
            "pairs_per_opponent": 4096,
            "wp_first_seed": 103110001,
            "point_first_seed": 103210001,
            "bootstrap_seed": 104110001,
            "bootstrap_replicates": 10000,
            "full_matches": 131072,
            "primary": "Four simultaneous95% mixture-win lower bounds: point_matched minus wp_fixed and retained parent, both seeds; all strictly positive",
            "candidate": "seed1_point_matched",
            "secondary_coordinates": 120,
        },
        "scope": "Fixed endpoint initialization/control study. No outcome selection, promotion or evaluation in this runner.",
    }
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / "protocol.json", protocol)
    print(
        {
            "prepared": str(root),
            "inputs": len(inputs),
            "protocol_sha256": sha(root / "protocol.json"),
        },
        flush=True,
    )


def train_arm(root, name, config, protocol):
    checked(protocol["inputs"])
    folder = root / name
    folder.mkdir(exist_ok=False)
    trainer = PhaseTemperatureTrainer(PhaseTemperatureConfig(**config))
    trainer.save(folder / "initial.pt")
    initial = trainer.model.actor.checkpoint()
    assert (
        initial["current"]["parameters_sha256"]
        == initial["initialization"]["parameters_sha256"]
    )
    write_new(folder / "config.json", config)
    records = []
    start = time.perf_counter()
    with (folder / "metrics.jsonl").open("x", encoding="utf-8", newline="\n") as log:
        for update in range(1, protocol["updates"] + 1):
            metrics = trainer.update()
            if not all(
                math.isfinite(v)
                for v in metrics.values()
                if isinstance(v, (float, int))
            ):
                raise FloatingPointError("Nonfinite learning metrics")
            row = {
                "update": update,
                "metrics": metrics,
                "counters": trainer.counters.copy(),
                "seconds": time.perf_counter() - start,
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            records.append(metrics | trainer.counters)
            if update == 20 or update % 100 == 0 or update == protocol["updates"]:
                checkpoint = folder / f"update_{update:08d}.pt"
                trainer.save(checkpoint)
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                validate_accounting(payload, records, update)
                digest = tensor_digest(payload["actor"]["current"]["weights"])
                assert digest == payload["actor"]["current"]["parameters_sha256"]
                assert (digest == initial["initialization"]["parameters_sha256"]) == (
                    update <= 20
                )
                del payload
            if update % 20 == 0:
                print(json.dumps({"arm": name, "update": update}), flush=True)
    assert trainer.counters["decisions"] == protocol["decisions_per_arm"]
    checked(protocol["inputs"])
    result = {
        "status": "complete",
        "counters": trainer.counters.copy(),
        "checkpoint_sha256": sha(checkpoint),
        "metrics_sha256": sha(folder / "metrics.jsonl"),
        "protocol_sha256": sha(root / "protocol.json"),
    }
    write_new(folder / "complete.json", result)
    return result


def run(root):
    protocol = read(root / "protocol.json")
    checked(protocol["inputs"])
    if (root / "started.json").exists():
        raise ValueError("Study was already started; no automatic restart")
    write_new(
        root / "started.json",
        {"pid": os.getpid(), "protocol_sha256": sha(root / "protocol.json")},
    )
    try:
        results = {
            name: train_arm(root, name, config, protocol)
            for name, config in protocol["configs"].items()
        }
        checked(protocol["inputs"])
        totals = {
            key: sum(r["counters"][key] for r in results.values())
            for key in ("decisions", "rounds", "matches", "updates")
        }
        assert totals["decisions"] == protocol["total_decisions"]
        write_new(
            root / "complete.json",
            {
                "status": "complete",
                "arms": results,
                "total_training_work": totals,
                "evaluation": "not_started",
                "deployment_changed": False,
                "protocol_sha256": sha(root / "protocol.json"),
            },
        )
    except BaseException as error:
        write_new(root / "failed.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.command](args.output)
