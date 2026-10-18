"""Production-shape continuation capability on existing audited training roots."""

import argparse
import json
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from scripts.audit_final_round_training_study import tensor_equal
from scripts.public_memory_continuation import ContinuationConfig, ContinuationTrainer
from scripts.run_arbitration_study import PARENTS, read, sha, write

STUDY_SHA = "3e1282183196b0f76bfbaa383160304f34377c12d482578bf4cec112a627c7c1"
BANK_SHA = "f2e49203737b4b51e54ab39cd5f73fc669e47d24c68c10081226e224f2556120"


def require_completed_study(root):
    if read(root / "process.json")["status"] != "complete":
        raise ValueError("Wait for the complete round8 comparison")
    path = root / "independent_audit.json"
    if not path.exists() or read(path)["status"] != "passed":
        raise ValueError("Wait for the independent round8 audit")
    audit = read(path)
    for name, digest in audit["inputs"].items():
        if sha(name) != digest:
            raise ValueError(f"Changed comparison input: {name}")
    return {**audit["inputs"], str(path): sha(path)}


def assert_same_endpoint(left, right):
    for key in (
        "actor",
        "value",
        "optimizer",
        "bank",
        "bank_rng",
        "memory",
        "counters",
        "torch_rng",
        "cuda_rng",
        "python_rng",
        "prefix_actor",
        "opponents",
        "runtime",
    ):
        tensor_equal(left[key], right[key])
    assert left["numpy_rng"][0] == right["numpy_rng"][0]
    np.testing.assert_array_equal(left["numpy_rng"][1], right["numpy_rng"][1])
    assert left["numpy_rng"][2:] == right["numpy_rng"][2:]


def run(output):
    if output.exists():
        raise ValueError("Use a fresh capability directory")
    study = Path("runs/public_memory_study")
    inputs = require_completed_study(study)
    if sha(study / "protocol.json") != STUDY_SHA:
        raise ValueError("Unexpected round8 comparison")
    bank_root = Path("runs/public_memory_continuation_bank")
    receipt = bank_root / "audit.json"
    if sha(receipt) != BANK_SHA:
        raise ValueError("Expected the completed continuation bank")
    bank_audit = read(receipt)
    inputs.update(bank_audit["inputs"])
    inputs.update(bank_audit["outputs"])
    inputs[str(receipt)] = sha(receipt)
    parent, digest = PARENTS[0]
    inputs[parent] = digest
    files = [
        Path(__file__),
        Path("scripts/public_memory_continuation.py"),
        Path("scripts/public_memory_continuation_match.py"),
        Path("scripts/audit_final_round_training_study.py"),
        Path("tests/test_public_memory_continuation.py"),
        Path("tests/test_public_memory_continuation_match.py"),
        Path("tests/test_public_memory_continuation_capability.py"),
        Path("runs/public_memory_continuation_cpu_tests.xml"),
        Path("runs/public_memory_continuation_cuda_tests.xml"),
        Path("runs/public_memory_continuation_match_tests.xml"),
        Path("runs/public_memory_continuation_capability_tests.xml"),
    ]
    inputs.update({str(path): sha(path) for path in files})
    cases = list(
        ET.parse("runs/public_memory_continuation_cuda_tests.xml").iter("testcase")
    )
    if (
        len(cases) != 1
        or "[cuda]" not in cases[0].attrib["name"]
        or any(cases[0].find(k) is not None for k in ("skipped", "failure", "error"))
    ):
        raise ValueError("Successful real CUDA verification required")
    for path, expected in inputs.items():
        if sha(path) != expected:
            raise ValueError(f"Changed capability input: {path}")
    if not torch.cuda.is_available():
        raise ValueError("CUDA required")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    bank = torch.load(bank_root / "parent1.pt", map_location="cpu", weights_only=False)
    assert bank["format"] == "public_memory_continuation_bank_v1"
    data = tuple(bank[k] for k in ("states", "seats", "opponents", "memory"))
    configs = {
        mode: ContinuationConfig(
            device="cuda",
            seed=9209001,
            critic_warmup_updates=1,
            public_memory=mode == "public",
        )
        for mode in ("public", "masked")
    }
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "protocol.json",
        {
            "inputs": inputs,
            "configs": {k: asdict(v) for k, v in configs.items()},
            "logical_updates_per_arm": 3,
            "replayed_updates_per_arm": 1,
            "scope": "Production capability only; no playing-strength evaluation.",
        },
    )
    results = {}
    try:
        for mode, config in configs.items():
            write(output / "process.json", {"status": "running", "mode": mode})
            folder = output / mode
            trainer = ContinuationTrainer.from_parent(config, parent, *data)
            trainer.save(folder / "initial.pt")
            metrics, start = [], time.perf_counter()
            for update in range(1, 4):
                metrics.append(trainer.update())
                trainer.save(folder / f"update_{update}.pt")
                print(
                    json.dumps(
                        {"mode": mode, "update": update, "counters": trainer.counters}
                    ),
                    flush=True,
                )
            before_replay = torch.load(
                folder / "update_2.pt", map_location="cpu", weights_only=False
            )
            final = torch.load(
                folder / "update_3.pt", map_location="cpu", weights_only=False
            )
            initial = torch.load(
                folder / "initial.pt", map_location="cpu", weights_only=False
            )
            tensor_equal(final["prefix_actor"], initial["prefix_actor"])
            tensor_equal(final["opponents"], initial["opponents"])
            steps = {}
            for name, parameter in trainer.model.named_parameters():
                state = trainer.optimizer.state[parameter]
                assert all(bool(torch.isfinite(v).all()) for v in state.values())
                phase = (
                    {"0": "discard", "1": "pick", "3": "koi"}[name.split(".")[2]]
                    if name.startswith("actor.")
                    else "value"
                )
                steps[name] = int(state["step"])
                assert steps[name] == trainer.counters["optimizer_steps_" + phase]
            column_l1 = sum(
                float(m.encoder_block.f1.weight[:, 2:4].detach().abs().sum())
                for m in trainer.model.actor.models.values()
            )
            assert (column_l1 > 0) if mode == "public" else column_l1 == 0
            c = trainer.counters
            assert (
                c["conditional_tail_completions"] == 384
                and c["conditional_round_endings"] > 384
            )
            assert c["matches"] == c["rounds"] == 0
            assert c["decisions"] == c["learner_decisions"] + c["opponent_decisions"]
            del trainer
            resumed = ContinuationTrainer.restore(folder / "update_2.pt")
            assert resumed.update() == metrics[-1]
            resumed.save(folder / "replayed_update_3.pt")
            replay = torch.load(
                folder / "replayed_update_3.pt", map_location="cpu", weights_only=False
            )
            assert_same_endpoint(final, replay)
            replay_work = {
                k: final["counters"][k] - before_replay["counters"][k]
                for k in (
                    "decisions",
                    "learner_decisions",
                    "opponent_decisions",
                    "conditional_tail_completions",
                    "conditional_round_endings",
                    "matches",
                    "rounds",
                )
            }
            results[mode] = {
                "counters": final["counters"],
                "replay_work": replay_work,
                "optimizer_steps": steps,
                "memory_column_l1": column_l1,
                "metrics": metrics,
                "seconds_including_replay": time.perf_counter() - start,
                "exact_resume": True,
            }
            del resumed
        for endpoint in ("initial.pt", "update_1.pt"):
            left, right = [
                torch.load(
                    output / mode / endpoint, map_location="cpu", weights_only=False
                )
                for mode in ("public", "masked")
            ]
            assert_same_endpoint(left, right)
        for path, expected in inputs.items():
            if sha(path) != expected:
                raise ValueError(f"Capability input changed during execution: {path}")
        inputs.update({str(p): sha(p) for p in output.rglob("*.pt")})
        inputs[str(output / "protocol.json")] = sha(output / "protocol.json")
        write(
            output / "audit.json",
            {
                "status": "passed",
                "inputs": inputs,
                "arms": results,
                "matched_initial_and_warmup_exact": True,
                "new_source_simulation": 0,
                "scope": "Two three-update CUDA continuations plus two exact-resume updates; no strength evidence.",
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
