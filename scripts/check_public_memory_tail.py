"""Production-shape matched-memory capability; no playing-strength evaluation."""

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from scripts.public_memory_tail import MemoryTailConfig, MemoryTailTrainer
from scripts.run_arbitration_study import PARENTS, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    bank_path = Path("runs/final_round_training_capability/initial.pt")
    old = json.loads(
        Path("runs/final_round_training_capability/audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert old["inputs"][str(bank_path)] == sha(bank_path)
    bank = torch.load(bank_path, weights_only=False, map_location="cpu")["bank"]
    memory = torch.tensor([0, 1, -1, 1])
    parent, digest = PARENTS[0]
    assert sha(parent) == digest
    results, warm = {}, None
    for mode in ("public", "masked"):
        config = MemoryTailConfig(
            device="cuda",
            seed=9009001,
            critic_warmup_updates=1,
            public_memory=mode == "public",
        )
        folder = args.output / mode
        trainer = MemoryTailTrainer.from_parent(config, parent, *bank, memory)
        trainer.save(folder / "initial.pt")
        prefix = copy.deepcopy(trainer.prefix_actor)
        metrics, start = [], time.perf_counter()
        for update in range(3):
            metrics.append(trainer.update())
            if update == 0:
                current = {
                    k: v.detach().cpu().clone()
                    for k, v in trainer.model.state_dict().items()
                }
                if warm is None:
                    warm = current
                else:
                    assert all(torch.equal(current[k], v) for k, v in warm.items())
            print(
                json.dumps(
                    {"mode": mode, "update": update + 1, "counters": trainer.counters}
                ),
                flush=True,
            )
        steps = {}
        for name, parameter in trainer.model.named_parameters():
            entry = trainer.optimizer.state[parameter]
            assert all(bool(torch.isfinite(v).all()) for v in entry.values())
            steps[name] = int(entry["step"])
        for phase, weights in prefix["current"]["weights"].items():
            for name, value in weights.items():
                assert torch.equal(
                    value, trainer.prefix_actor["current"]["weights"][phase][name]
                )
        for name, actor in trainer.opponents.items():
            assert (
                actor.checkpoint()["current"]["parameters_sha256"]
                == trainer.provenance["tail_opponents"][name]
            )
        column_l1 = sum(
            float(m.encoder_block.f1.weight[:, 2:4].detach().abs().sum())
            for m in trainer.model.actor.models.values()
        )
        assert (column_l1 > 0) if mode == "public" else (column_l1 == 0)
        assert trainer.counters["conditional_tail_completions"] == 384
        assert trainer.counters["matches"] == trainer.counters["rounds"] == 0
        trainer.save(folder / "final.pt")
        results[mode] = {
            "config": asdict(config),
            "counters": trainer.counters.copy(),
            "metrics": metrics,
            "optimizer_steps": steps,
            "memory_column_l1": column_l1,
            "seconds": time.perf_counter() - start,
        }
        del trainer
    files = (
        [Path(__file__), bank_path, Path(parent)]
        + [
            Path(p)
            for p in (
                "scripts/public_memory_tail.py",
                "scripts/public_memory_match.py",
                "tests/test_public_memory_tail.py",
                "tests/test_public_memory_match.py",
                "runs/public_memory_tail_tests.xml",
                "runs/public_memory_match_tests.xml",
                "scripts/final_round_training.py",
                "scripts/fixed_microbatch_continuation.py",
            )
        ]
        + list(args.output.rglob("*.pt"))
    )
    write(
        args.output / "audit.json",
        {
            "status": "passed",
            "inputs": {str(p): sha(p) for p in files},
            "arms": results,
            "matched_warmup_exact": True,
            "new_source_simulation": 0,
            "scope": "Two three-update CUDA capability arms on four reused toy roots; no strength evidence.",
        },
    )


if __name__ == "__main__":
    main()
