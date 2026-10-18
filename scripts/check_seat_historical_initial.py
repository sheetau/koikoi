"""Check all planned CUDA initial states against archived observer controls.

Constructs and saves fresh initial models only: no collect, update or evaluation.
The full strength-study launch prerequisites still apply after this preflight.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from koikoi.arbitration import read_checkpoint
from scripts.analyze_arbitration_study import tensor_equal
from scripts.check_seat_observer_device import training_ready
from scripts.run_arbitration_study import read, sha, write
from scripts.run_observer_selfplay_study import checked
from scripts.run_seat_observer_study import compare_initial
from scripts.seat_observer_selfplay import SeatObserverConfig, SeatObserverTrainer


def run(output):
    if output.exists():
        raise ValueError("Use a fresh preflight directory")
    source = Path("runs/observer_selfplay_study")
    prior = checked(source)
    assert training_ready(read(source / "process.json"))
    capability_path = Path("runs/seat_observer_capability_cuda/audit.json")
    capability = read(capability_path)
    assert capability["status"] == "passed" and capability["kind"] == "cuda_capability"
    inputs = prior["inputs"] | capability["inputs"]
    for path in (
        Path(__file__),
        source / "protocol.json",
        capability_path,
        Path("scripts/run_seat_observer_study.py"),
    ):
        inputs[str(path)] = sha(path)
    for index in (1, 2):
        path = source / f"parent{index}_observer/initial.pt"
        inputs[str(path)] = sha(path)
    assert all(sha(path) == digest for path, digest in inputs.items())
    torch.set_num_threads(2)
    records = {}
    for index, parent in enumerate(prior["parents"], 1):
        original, _ = read_checkpoint(source / f"parent{index}_observer/initial.pt")
        pair = []
        for context in ("public", "masked"):
            config = dict(prior["cases"][f"parent{index}_observer"]["config"])
            config.update(seat_context=context, policy_returns="observer")
            config = SeatObserverConfig(**config)
            assert config.device == "cuda"
            trainer = SeatObserverTrainer.from_parent(config, parent["path"])
            path = output / f"parent{index}_{context}_initial.pt"
            trainer.save(path)
            initial, digest = read_checkpoint(path)
            comparison = compare_initial(initial, original)
            assert all(value == 0 for value in initial["counters"].values())
            assert not initial["optimizer"]["state"]
            assert initial["config"] == asdict(config)
            pair.append(initial)
            records[f"parent{index}_{context}"] = {
                "config": asdict(config),
                "initial_sha256": digest,
                "archived_comparison": comparison,
                "counters": initial["counters"],
            }
            del trainer
            torch.cuda.empty_cache()
        for key in (
            "actor",
            "value",
            "optimizer",
            "env",
            "counters",
            "runtime",
            "torch_rng",
            "cuda_rng",
        ):
            assert tensor_equal(pair[0][key], pair[1][key]), (index, key)
    assert all(sha(path) == digest for path, digest in inputs.items())
    report = {
        "status": "passed",
        "inputs": inputs,
        "conditions": records,
        "executed_work": {"decisions": 0, "rounds": 0, "matches": 0, "value_rows": 0},
        "scope": "Four planned CUDA initial models match archived actor/environment/RNG and zero-padded critic. Public/masked full initial states agree. No training, inference, simulation steps, strength estimate or full-study launch.",
    }
    write(output / "audit.json", report)
    print("initial_preflight_sha256=" + sha(output / "audit.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
