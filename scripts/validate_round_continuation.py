"""Archived complete-collection equivalence and fixed full-learning benchmark."""

import hashlib
import json
import subprocess
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path

import torch

import koikoi.complete_matches as current_collection
from koikoi.train import Config, Trainer, provenance
from scripts.run_round_continuation_screen import preflight

REVISION = "18e432258b77fc9a318b8e57bbc3c5ba53a421bc"


def archive_module(name, file):
    content = subprocess.check_output(["git", "show", f"{REVISION}:{file}"])
    module = types.ModuleType(name)
    module.__file__ = str(Path(file).resolve())
    sys.modules[name] = module
    # Exact local committed source; no user-provided executable input.
    exec(compile(content.decode("utf-8"), module.__file__, "exec"), module.__dict__)  # noqa: S102
    return module, hashlib.sha256(content).hexdigest()


def equal_tensors(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    old, trainer_sha = archive_module("koikoi._round_old_train", "koikoi/train.py")
    collector, collector_sha = archive_module(
        "koikoi._round_old_collect", "koikoi/complete_matches.py"
    )
    cases = []
    for device in ("cpu", "cuda"):
        for lam in (0.95, 1.0):
            config = Config(
                device=device,
                profile="paper8",
                envs=8,
                horizon=512,
                width=32,
                epochs=1,
                minibatch=128,
                cpu_threads=2,
                complete_matches=True,
                gae_lambda=lam,
                endgame=True,
            )
            older = asdict(config)
            del older["round_continuation"]
            first, second = old.Trainer(old.Config(**older)), Trainer(config)
            for _ in range(2):
                cpu_before = torch.get_rng_state()
                gpu_before = torch.cuda.get_rng_state() if device == "cuda" else None
                sys.modules["koikoi.complete_matches"] = collector
                try:
                    expected = first.update()
                finally:
                    sys.modules["koikoi.complete_matches"] = current_collection
                cpu_after = torch.get_rng_state()
                gpu_after = torch.cuda.get_rng_state() if device == "cuda" else None
                torch.set_rng_state(cpu_before)
                if gpu_before is not None:
                    torch.cuda.set_rng_state(gpu_before)
                assert second.update() == expected
                assert torch.equal(cpu_after, torch.get_rng_state())
                if gpu_after is not None:
                    assert torch.equal(gpu_after, torch.cuda.get_rng_state())
                assert first.counters == second.counters
                assert torch.equal(first.env.state, second.env.state)
                equal_tensors(first.model.state_dict(), second.model.state_dict())
                a, b = first.optimizer.state_dict(), second.optimizer.state_dict()
                assert a["param_groups"] == b["param_groups"]
                for key in a["state"]:
                    equal_tensors(a["state"][key], b["state"][key])
            cases.append(
                {
                    "device": device,
                    "lambda": lam,
                    "updates": 2,
                    "counters": second.counters,
                }
            )
    print("Four archived complete-collection equivalence cases passed", flush=True)
    protocol = preflight()
    record = protocol["cases"]["seed1_round"]
    config = Config(**(record["config"] | {"seed": 6620001}))
    trainer = Trainer(config)
    trainer.initialize_weights(record["parent"])
    for _ in range(5):
        trainer.update()
    torch.cuda.synchronize()
    initial = trainer.counters.copy()
    start = time.perf_counter()
    for _ in range(20):
        trainer.update()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    counts = {key: value - initial[key] for key, value in trainer.counters.items()}
    report = {
        "status": "passed",
        "provenance": provenance(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "archived_revision": REVISION,
        "archived_trainer_sha256": trainer_sha,
        "archived_collector_sha256": collector_sha,
        "equivalence_cases": cases,
        "equivalence_scope": "Actual archived Trainer and collector: metrics/model/Adam/environment/counters and CPU/CUDA RNG exact after two updates per case; no target override. Other dependencies unchanged.",
        "benchmark": {
            "config": asdict(config),
            "initialized_from": trainer.provenance["initialized_from"],
            "warmup_counters": initial,
            "timed_counters": counts,
            "seconds": elapsed,
            "active_decisions_per_second": counts["decisions"] / elapsed,
            "complete_matches_per_second": counts["matches"] / elapsed,
            "includes_learning": True,
            "strength_evidence": False,
        },
    }
    path = Path("runs/round_continuation_validation.json")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "seconds": elapsed,
                "timed_counters": counts,
                "output": str(path),
            }
        )
    )


if __name__ == "__main__":
    main()
