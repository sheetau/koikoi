"""Compare real archived Trainer source with unchanged default execution modes."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import types
from dataclasses import asdict
from pathlib import Path

import torch

from koikoi.train import Config, Trainer, provenance

REVISION = "3a395248493f4037ea1fe43afb0a56d6c48ce3cb"


def tensors(first, second):
    assert first.keys() == second.keys()
    for key in first:
        assert torch.equal(first[key], second[key]), key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/published_training_default_equivalence.json"),
    )
    args = parser.parse_args()
    source = subprocess.check_output(["git", "show", f"{REVISION}:koikoi/train.py"])
    name = "koikoi._archived_training_defaults"
    old = types.ModuleType(name)
    old.__file__ = str(Path("koikoi/train.py").resolve())
    sys.modules[name] = old
    # Execute only the exact repository revision above, never caller-supplied code.
    exec(compile(source.decode("utf-8"), old.__file__, "exec"), old.__dict__)  # noqa: S102
    results = []
    with tempfile.TemporaryDirectory() as folder:
        opponent = Path(folder) / "opponent.pt"
        Trainer(
            Config(device="cpu", profile="paper8", envs=4, width=32, cpu_threads=2)
        ).save(opponent)
        for device, capture, fraction in [
            ("cpu", False, 0.0),
            ("cpu", False, 0.5),
            ("cuda", False, 0.0),
            ("cuda", True, 0.0),
            ("cuda", True, 0.5),
        ]:
            config = Config(
                device=device,
                profile="paper8",
                seed=63009,
                envs=8,
                horizon=32,
                width=32,
                epochs=1,
                minibatch=128,
                cpu_threads=2,
                cuda_graph=capture,
                cuda_rollout=capture,
                opponent_fraction=fraction,
            )
            old_config = asdict(config)
            del old_config["published_opponent"]
            del old_config["complete_matches"]
            first = old.Trainer(old.Config(**old_config))
            second = Trainer(config)
            if fraction:
                first.set_frozen_opponent(opponent)
                second.set_frozen_opponent(opponent)
            for _ in range(2):
                before = torch.get_rng_state().clone()
                gpu_before = (
                    torch.cuda.get_rng_state().clone() if device == "cuda" else None
                )
                expected = first.update()
                after = torch.get_rng_state().clone()
                gpu_after = (
                    torch.cuda.get_rng_state().clone() if device == "cuda" else None
                )
                torch.set_rng_state(before)
                if gpu_before is not None:
                    torch.cuda.set_rng_state(gpu_before)
                assert second.update() == expected
                assert torch.equal(after, torch.get_rng_state())
                if gpu_after is not None:
                    assert torch.equal(gpu_after, torch.cuda.get_rng_state())
                assert first.counters == second.counters
                assert torch.equal(first.env.state, second.env.state)
                tensors(first.trainable.state_dict(), second.trainable.state_dict())
                for a, b in zip(
                    first.optimizer.state.values(), second.optimizer.state.values()
                ):
                    tensors(a, b)
            results.append(
                {
                    "device": device,
                    "capture": capture,
                    "fraction": fraction,
                    "updates": 2,
                    "counters": second.counters,
                }
            )
    report = {
        "passed": True,
        "provenance": provenance(),
        "archived_revision": REVISION,
        "archived_trainer_sha256": hashlib.sha256(source).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": results,
        "scope": "Actual archived Trainer versus current: metrics, model/critic/Adam, simulator, counters and CPU/CUDA RNG equal after two updates in each fixed mode; no strength evidence.",
    }
    path = args.output
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"passed": True, "cases": len(results), "output": str(path)}))


if __name__ == "__main__":
    main()
