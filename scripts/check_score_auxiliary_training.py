"""Fixed production-shape capability, not a strength comparison."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.published_train import PublishedConfig, PublishedTrainer
from scripts.score_auxiliary_trainer import AuxiliaryConfig, ScoreAuxiliaryTrainer, sha


def equal(a, b):
    if isinstance(a, torch.Tensor):
        if not torch.equal(a, b):
            raise AssertionError("Tensor mismatch")
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError("Key mismatch")
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError("Length mismatch")
        for left, right in zip(a, b, strict=True):
            equal(left, right)
    elif a != b:
        raise AssertionError("Scalar mismatch")


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def equal_ppo(a, b):
    # Restoring records a different provenance parent, not a different learner.
    for key in a:
        if key != "provenance":
            equal(a[key], b[key])
    equal(a["provenance"]["source_sha256"], b["provenance"]["source_sha256"])


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = PublishedConfig(
        device="cuda", initial_variant="point", seed=105510001, critic_warmup_updates=1
    )
    records = []
    started = time.perf_counter()
    for kind in ("none", "win", "score"):
        torch.cuda.reset_peak_memory_stats()
        auxiliary = AuxiliaryConfig(
            kind=kind,
            interval=1,
            deal_seed=105610001,
            action_seed=105710001,
            reservoir_seed=105810001,
            order_seed=105910001,
        )
        trainer = ScoreAuxiliaryTrainer(PublishedTrainer(config), auxiliary)
        root = output / kind
        trainer.save(root / "initial")
        metrics = [trainer.update()]
        trainer.save(root / "warmup")
        metrics.append(trainer.update())
        trainer.save(root / "cycle1")
        first_ppo, first_aux = trainer.ppo.counters.copy(), trainer.counters.copy()
        metrics.append(trainer.update())
        trainer.save(root / "cycle2")
        final_ppo, final_aux = trainer.ppo.counters.copy(), trainer.counters.copy()
        del trainer
        restored = ScoreAuxiliaryTrainer.restore(root / "cycle1")
        equal(restored.update(), metrics[-1])
        restored.save(root / "repeated_cycle2")
        del restored
        for filename in ("ppo.pt", "auxiliary.pt"):
            actual, expected = (
                load(root / "repeated_cycle2" / filename),
                load(root / "cycle2" / filename),
            )
            (equal_ppo if filename == "ppo.pt" else equal)(actual, expected)
        record = {
            "kind": kind,
            "ppo_config": asdict(config),
            "auxiliary_config": asdict(auxiliary),
            "metrics": metrics,
            "unique_ppo_work": {
                k: final_ppo[k] for k in ("decisions", "rounds", "matches")
            },
            "repeated_ppo_work": {
                k: final_ppo[k] - first_ppo[k]
                for k in ("decisions", "rounds", "matches")
            },
            "unique_auxiliary_work": final_aux,
            "repeated_auxiliary_work": {
                k: final_aux[k] - first_aux[k] for k in final_aux
            },
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "exact_resume": True,
        }
        if kind != "none":
            for stage in ("initial", "warmup"):
                equal_ppo(
                    load(root / stage / "ppo.pt"),
                    load(output / "none" / stage / "ppo.pt"),
                )
            equal(
                load(root / "cycle1/auxiliary.pt")["last_collection"],
                load(output / "none/cycle1/auxiliary.pt")["last_collection"],
            )
        else:
            base = PublishedTrainer.restore(root / "cycle1/ppo.pt")
            base.update()
            base.save(root / "original_ppo_control.pt")
            equal_ppo(
                load(root / "original_ppo_control.pt"), load(root / "cycle2/ppo.pt")
            )
            record["additional_original_ppo_control_work"] = record["repeated_ppo_work"]
            del base
        with (root / "capability.json").open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(json.dumps(record, indent=2) + "\n")
        records.append(record)
        print(
            json.dumps(
                {
                    "completed": kind,
                    "exact_resume": True,
                    "peak_cuda_allocated_bytes": record["peak_cuda_allocated_bytes"],
                }
            ),
            flush=True,
        )
    report = {
        "format": "score_auxiliary_training_capability_v1",
        "passed": True,
        "strength_evidence": False,
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
        "files": {str(p): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
    }
    with (output / "report.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
