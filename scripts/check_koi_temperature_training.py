"""Production-size warmup, optimizer and exact-resume capability for both arms."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from scripts.koi_temperature_ppo import KoiTemperatureConfig, KoiTemperatureTrainer
from scripts.run_arbitration_study import read, sha, write


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def run(output):
    if output.exists():
        raise ValueError("Preserve prior capability runs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    calibration = Path("runs/phase_exploration_calibration.json")
    assert (
        sha(calibration)
        == "d191ca3bafe0c3bb1f268f612a3dfb80f2223dc97292ed87d0571c6024487e0a"
    )
    inputs = dict(read(calibration)["inputs"])
    paths = [
        Path(__file__),
        Path("scripts/koi_temperature_ppo.py"),
        Path("tests/test_koi_temperature_ppo.py"),
        Path("runs/koi_temperature_ppo_tests_initial.xml"),
        calibration,
    ]
    paths += [
        p for p in Path("koikoi").rglob("*") if p.suffix in (".py", ".h", ".cpp", ".cu")
    ]
    inputs.update({str(p): sha(p) for p in paths})
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "protocol.json",
        {"inputs": inputs, "updates": 22, "warmup": 20, "seed": 96990001},
    )
    arms = {}
    try:
        for mode, temperature in (
            ("fixed", 0.05),
            ("calibrated", read(calibration)["phase_temperatures"][3]),
        ):
            folder = output / mode
            config = KoiTemperatureConfig(
                device="cuda", seed=96990001, koi_temperature=temperature
            )
            trainer = KoiTemperatureTrainer(config)
            trainer.save(folder / "initial.pt")
            write(folder / "config.json", asdict(config))
            metrics = []
            for index in range(1, 23):
                row = trainer.update()
                assert all(
                    torch.isfinite(torch.tensor(v))
                    for v in row.values()
                    if isinstance(v, (float, int))
                )
                metrics.append(
                    {
                        "update": index,
                        "metrics": row,
                        "counters": trainer.counters.copy(),
                    }
                )
                if index in (20, 21, 22):
                    trainer.save(folder / f"update_{index:08d}.pt")
                write(
                    output / "process.json",
                    {"status": "running", "arm": mode, "update": index},
                )
            write(folder / "metrics.json", metrics)
            initial, warm, final = (
                load(folder / name)
                for name in ("initial.pt", "update_00000020.pt", "update_00000022.pt")
            )
            equal(initial["actor"]["current"], warm["actor"]["current"])
            assert any(
                not torch.equal(value, final["actor"]["current"]["weights"][phase][key])
                for phase, state in initial["actor"]["current"]["weights"].items()
                for key, value in state.items()
            )
            restarted = KoiTemperatureTrainer.restore(folder / "update_00000021.pt")
            actual = restarted.update()
            assert actual == metrics[-1]["metrics"]
            restarted.save(folder / "resumed_22.pt")
            replay = load(folder / "resumed_22.pt")
            for key in (
                "actor",
                "value",
                "optimizer",
                "env",
                "counters",
                "torch_rng",
                "cuda_rng",
            ):
                equal(replay[key], final[key])
            replay_work = {
                key: metrics[-1]["counters"][key] - metrics[-2]["counters"][key]
                for key in ("decisions", "rounds", "matches")
            }
            arms[mode] = {
                "exact_resume": True,
                "counters": final["counters"],
                "replay_work": replay_work,
                "optimizer_states": len(final["optimizer"]["state"]),
            }
            print(
                json.dumps(
                    {"arm": mode, "status": "passed", "counters": final["counters"]}
                ),
                flush=True,
            )
            del trainer, restarted
        a, b = load(output / "fixed/initial.pt"), load(output / "calibrated/initial.pt")
        for key in ("actor", "value", "optimizer", "env", "torch_rng", "cuda_rng"):
            equal(a[key], b[key])
        for path in output.rglob("*"):
            if path.is_file() and path.name != "process.json":
                inputs[str(path)] = sha(path)
        for path, digest in inputs.items():
            assert sha(path) == digest, path
        write(
            output / "audit.json",
            {
                "status": "passed",
                "inputs": inputs,
                "arms": arms,
                "matched_initial_exact": True,
                "actor_unchanged_during_warmup": True,
                "scope": "Production-size execution and exact state checks, not a strength comparison.",
            },
        )
        write(output / "process.json", {"status": "complete"})
    except BaseException as error:
        write(output / "process.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
