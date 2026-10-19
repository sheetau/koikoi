"""Production-shape capability for three fixed initialization/exploration arms."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from koikoi.upstream import UPSTREAM_PATH, load_source
from koikoi.upstream_batch import parameter_digest
from scripts.phase_temperature_ppo import (
    PhaseTemperatureConfig,
    PhaseTemperatureTrainer,
)
from scripts.published_duration_study import assert_equal
from scripts.run_arbitration_study import read, sha

CALIBRATION = Path("runs/point_initialization_calibration.json")
CALIBRATION_SHA = "869ab1e45e7e8a5569ff20edaea55fdc5e2a2464d6d6a32a9c50df188bd701f3"
SEED = 102010001


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def checked(inputs):
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError(f"Changed input: {path}")


def scan_seed():
    """Scoped prior config/protocol inventory; not a proof about unrecorded work."""
    records, overlaps = {}, []

    def visit(value, trail):
        if isinstance(value, dict):
            for key, item in value.items():
                if "seed" in key.lower() and isinstance(item, int) and item == SEED:
                    overlaps.append(trail + "/" + key)
                visit(item, trail + "/" + key)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, trail + "/" + str(index))

    for path in sorted(Path("runs").rglob("*.json")):
        if path.name not in ("config.json", "protocol.json"):
            continue
        value = read(path)
        records[str(path)] = sha(path)
        visit(value, str(path))
    if overlaps:
        raise ValueError(f"Recorded capability seed already used: {overlaps}")
    return {
        "seed": SEED,
        "recorded_files": records,
        "overlaps": overlaps,
        "scope": "Exact seed-valued fields in existing config.json/protocol.json; does not enumerate all future schedules or unrecorded work.",
    }


def run(root):
    if root.exists():
        raise ValueError("Preserve existing capability")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if sha(CALIBRATION) != CALIBRATION_SHA:
        raise ValueError("Calibration changed")
    calibration = read(CALIBRATION)
    inputs = dict(calibration["inputs"])
    load_source()
    paths = [
        CALIBRATION,
        Path(__file__),
        Path("scripts/phase_temperature_ppo.py"),
        Path("scripts/published_duration_study.py"),
        Path("tests/test_phase_temperature_ppo.py"),
        Path("runs/phase_temperature_ppo_tests.xml"),
    ]
    paths += [
        p for p in Path("koikoi").rglob("*") if p.suffix in (".py", ".h", ".cpp", ".cu")
    ]
    paths += list(UPSTREAM_PATH.glob("*.py"))
    paths += [
        UPSTREAM_PATH / f"model_agent/{phase}_rl_{variant}.pt"
        for phase in ("discard", "pick", "koikoi")
        for variant in ("wp", "point")
    ]
    inputs.update({str(p): sha(p) for p in paths})
    checked(inputs)
    scan = scan_seed()
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / "seed_scan.json", scan)
    inputs[str(root / "seed_scan.json")] = sha(root / "seed_scan.json")
    arms = [
        ("wp_fixed", "wp", [0.05] * 4),
        ("point_fixed", "point", [0.05] * 4),
        ("point_matched", "point", calibration["phase_temperatures"]),
    ]
    write_new(
        root / "protocol.json",
        {
            "inputs": inputs,
            "seed": SEED,
            "warmup_updates": 20,
            "policy_updates": 2,
            "arms": arms,
            "pid": os.getpid(),
            "scope": "Capability, not strength",
        },
    )
    results = {}
    try:
        for name, variant, temperatures in arms:
            folder = root / name
            folder.mkdir()
            cfg = PhaseTemperatureConfig(
                device="cuda",
                seed=SEED,
                initial_variant=variant,
                phase_temperatures=temperatures,
            )
            trainer = PhaseTemperatureTrainer(cfg)
            trainer.save(folder / "initial.pt")
            write_new(folder / "config.json", asdict(cfg))
            initial = load(folder / "initial.pt")
            assert initial["actor"]["initialization"]["variant"] == variant
            digest = parameter_digest(initial["actor"]["current"]["weights"])
            assert digest == initial["actor"]["initialization"]["parameters_sha256"]
            assert (trainer.model.value[-1].weight == 0).all()
            assert (trainer.model.value[-1].bias == 0).all()
            metrics = []
            with (folder / "metrics.jsonl").open(
                "x", encoding="utf-8", newline="\n"
            ) as log:
                for index in range(1, 23):
                    row = trainer.update()
                    assert all(
                        torch.isfinite(torch.tensor(v))
                        for v in row.values()
                        if isinstance(v, (float, int))
                    )
                    record = {
                        "update": index,
                        "metrics": row,
                        "counters": trainer.counters.copy(),
                    }
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    metrics.append(record)
                    if index in (20, 21, 22):
                        trainer.save(folder / f"update_{index:08d}.pt")
                    if index % 5 == 0 or index == 22:
                        print(json.dumps({"arm": name, "update": index}), flush=True)
            warm = load(folder / "update_00000020.pt")
            final = load(folder / "update_00000022.pt")
            assert_equal(initial["actor"]["current"], warm["actor"]["current"])
            assert final["actor"]["current"]["parameters_sha256"] != digest
            assert final["counters"]["decisions"] == 22 * 128 * 128
            assert final["counters"]["warmup_updates"] == 20
            assert final["counters"]["policy_updates"] == 2
            del trainer
            restored = PhaseTemperatureTrainer.restore(folder / "update_00000021.pt")
            assert_equal(restored.update(), metrics[-1]["metrics"])
            restored.save(folder / "resumed_22.pt")
            actual = load(folder / "resumed_22.pt")
            for key in (
                "actor",
                "value",
                "optimizer",
                "env",
                "counters",
                "torch_rng",
                "cuda_rng",
                "python_rng",
                "numpy_rng",
            ):
                assert_equal(actual[key], final[key])
            results[name] = {
                "status": "passed",
                "initial_actor_sha256": digest,
                "counters": final["counters"],
                "duplicate_resume_work": {
                    k: metrics[-1]["counters"][k] - metrics[-2]["counters"][k]
                    for k in ("decisions", "rounds", "matches")
                },
            }
            del restored
            print(json.dumps({"arm": name, "status": "passed"}), flush=True)
        initial = {name: load(root / name / "initial.pt") for name, _, _ in arms}
        for name in ("point_fixed", "point_matched"):
            for key in (
                "value",
                "optimizer",
                "env",
                "torch_rng",
                "cuda_rng",
                "python_rng",
                "numpy_rng",
            ):
                assert_equal(initial["wp_fixed"][key], initial[name][key])
        assert_equal(initial["point_fixed"]["actor"], initial["point_matched"]["actor"])
        checked(inputs)
        outputs = {str(p): sha(p) for p in root.rglob("*") if p.is_file()}
        write_new(
            root / "complete.json",
            {
                "status": "passed",
                "inputs": inputs,
                "outputs": outputs,
                "arms": results,
                "common_initial_critic_env_optimizer_rng": True,
                "point_arms_initial_actor_equal": True,
                "scope": "Production128x128,20warmup+2policy updates per arm and exact repeated next update. Small capability, not playing-strength evidence.",
            },
        )
    except BaseException as error:
        write_new(root / "failed.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
