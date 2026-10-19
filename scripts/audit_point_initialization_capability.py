"""Check saved production capability states and work without rerunning learning."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from koikoi.upstream_batch import BatchPublishedAgent
from scripts.audit_pretrained_ppo_screen import tensor_digest, validate_accounting
from scripts.phase_temperature_ppo import PhaseTemperatureConfig
from scripts.published_duration_study import assert_equal
from scripts.run_arbitration_study import read, sha


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def audit(root):
    output = root / "independent_audit.json"
    if output.exists():
        raise ValueError("Preserve completed audit")
    producer = read(root / "complete.json")
    protocol = read(root / "protocol.json")
    assert producer["status"] == "passed" and not (root / "failed.json").exists()
    inputs = producer["inputs"] | producer["outputs"]
    inputs[str(root / "complete.json")] = sha(root / "complete.json")
    inputs[str(Path(__file__))] = sha(__file__)
    inputs["scripts/audit_pretrained_ppo_screen.py"] = sha(
        "scripts/audit_pretrained_ppo_screen.py"
    )
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    source = hashlib.sha256()
    for path in sorted(Path("koikoi").rglob("*")):
        if path.suffix in (".py", ".h", ".cpp", ".cu"):
            source.update(path.as_posix().encode())
            source.update(path.read_bytes())
    references = {
        v: BatchPublishedAgent(v, "cpu").checkpoint() for v in ("wp", "point")
    }
    total = dict.fromkeys(("decisions", "rounds", "matches"), 0)
    duplicate = total.copy()
    stages, initials = {}, {}
    for name, variant, temperatures in protocol["arms"]:
        folder = root / name
        cfg = asdict(
            PhaseTemperatureConfig(
                device="cuda",
                seed=protocol["seed"],
                initial_variant=variant,
                phase_temperatures=temperatures,
            )
        )
        assert json.dumps(read(folder / "config.json"), sort_keys=True) == json.dumps(
            cfg, sort_keys=True
        )
        initial = load(folder / "initial.pt")
        initials[name] = initial
        assert_equal(
            initial["actor"]["current"]["weights"], references[variant]["weights"]
        )
        assert all(v == 0 for v in initial["counters"].values())
        raw = [
            json.loads(line)
            for line in (folder / "metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [r["update"] for r in raw] == list(range(1, 23))
        rows = [r["metrics"] | r["counters"] for r in raw]
        stages[name] = {}
        for update in (20, 21, 22):
            payload = load(folder / f"update_{update:08d}.pt")
            assert payload["format"] == "phase_temperature_ppo_v1"
            assert payload["config"] == cfg
            assert payload["provenance"]["source_sha256"] == source.hexdigest()
            assert payload["provenance"]["phase_temperature_source"] == sha(
                "scripts/phase_temperature_ppo.py"
            )
            actor = payload["actor"]
            assert actor["initialization"] == initial["actor"]["initialization"]
            assert (
                actor["initialization"]
                == payload["provenance"]["published_initialization"]
            )
            assert actor["initialization"]["parameters_sha256"] == tensor_digest(
                references[variant]["weights"]
            )
            digest = tensor_digest(actor["current"]["weights"])
            assert digest == actor["current"]["parameters_sha256"]
            assert (digest == actor["initialization"]["parameters_sha256"]) == (
                update == 20
            )
            assert all(torch.isfinite(v).all() for v in payload["value"].values())
            validate_accounting(payload, rows[:update], update)
            assert payload["counters"]["decisions"] == update * 16384
            assert payload["runtime"] == initial["runtime"]
            assert payload["runtime"]["device"] == "cuda:0"
            groups = payload["optimizer"]["param_groups"]
            state = payload["optimizer"]["state"]
            assert len(groups) == 2
            assert [len(g["params"]) for g in groups] == [90, 6]
            assert [g["lr"] for g in groups] == [3e-6, 1e-4]
            assert all(g["eps"] == 1e-5 for g in groups)
            ids = groups[0]["params"] + groups[1]["params"]
            assert len(set(ids)) == 96 and set(state) <= set(ids)
            expected = (
                [payload["counters"]["optimizer_steps_" + k]] * size
                for k, size in (
                    ("discard", 30),
                    ("pick", 30),
                    ("koi", 30),
                    ("value", 6),
                )
            )
            for key, steps in zip(
                ids, [n for group in expected for n in group], strict=True
            ):
                if not steps:
                    assert key not in state
                else:
                    assert int(state[key]["step"]) == steps
                    assert torch.isfinite(state[key]["exp_avg"]).all()
                    assert torch.isfinite(state[key]["exp_avg_sq"]).all()
            stages[name][str(update)] = payload["counters"]
        repeated = load(folder / "resumed_22.pt")
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
            assert_equal(repeated[key], payload[key])
        assert producer["arms"][name]["counters"] == payload["counters"]
        for key in total:
            total[key] += payload["counters"][key]
            work = stages[name]["22"][key] - stages[name]["21"][key]
            assert producer["arms"][name]["duplicate_resume_work"][key] == work
            duplicate[key] += work
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
            assert_equal(initials[name][key], initials["wp_fixed"][key])
    assert_equal(initials["point_fixed"]["actor"], initials["point_matched"]["actor"])
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    result = {
        "status": "passed",
        "inputs": inputs,
        "stages": stages,
        "unique_training_work": total,
        "duplicate_resume_work": duplicate,
        "new_inference_calls": 0,
        "new_simulator_steps": 0,
        "scope": "Saved weights against pinned originals; metrics, simulator counters, per-parameter Adam states and resumed endpoint equality. No training or inference replay; capability is not strength evidence.",
    }
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print({"status": "passed", "unique_work": total, "duplicate_work": duplicate})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    torch.set_num_threads(2)
    audit(parser.parse_args().input)
