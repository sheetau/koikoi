"""Fixed production-size capability; compare full matched critic warmups."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from koikoi.train import provenance
from scripts.observer_selfplay import ObserverConfig, ObserverTrainer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    """Canonical nested state digest, independent of torch serialization IDs."""
    result = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            result.update(str((item.dtype, tuple(item.shape))).encode())
            result.update(item.numpy().tobytes())
        elif isinstance(item, dict):
            result.update(b"dict")
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            result.update(str((type(item).__name__, len(item))).encode())
            for entry in item:
                visit(entry)
        else:
            result.update(repr(item).encode())
        result.update(b"\0")

    visit(value)
    return result.hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def run(base, device, warmup, updates):
    if base.exists() or warmup < 1 or updates <= warmup:
        raise ValueError("Use a fresh directory and at least one warmup/policy update")
    parent = Path("runs/cold_ppo_screen/seed6710001_warm/update_00000520.pt")
    previous = Path("runs/learner_return_study/protocol.json")
    prior = json.loads(previous.read_text(encoding="utf-8"))
    assert all(sha(path) == expected for path, expected in prior["inputs"].items())
    production = provenance()["source_sha256"]
    assert production == prior["source"]["source_sha256"]
    paths = [Path(__file__), Path("docs/OBSERVER_SELFPLAY.rst"), parent, previous]
    paths += [
        Path("scripts") / name
        for name in ("observer_selfplay.py", "observer_selfplay_returns.py")
    ]
    paths += [
        Path("tests") / name
        for name in ("test_observer_selfplay.py", "test_observer_selfplay_returns.py")
    ]
    inputs = {str(path): sha(path) for path in paths}
    base.mkdir(parents=True)
    conditions = {}
    for mode in ("observer", "signed"):
        config = ObserverConfig(
            device=device,
            envs=128,
            horizon=128,
            minibatch=512,
            epochs=3,
            critic_width=256,
            critic_warmup_updates=warmup,
            seed=26040001,
            policy_returns=mode,
        )
        conditions[mode] = asdict(config)
    write(
        base / "protocol.json",
        {
            "inputs": inputs,
            "production_source_sha256": production,
            "conditions": conditions,
            "updates": updates,
            "purpose": "Capability and matched warmup, not strength",
        },
    )
    records, warmup_states = {}, {}
    for mode, config_dict in conditions.items():
        trainer = ObserverTrainer.from_parent(ObserverConfig(**config_dict), parent)
        assert not trainer.optimizer.state
        initial = trainer.model.actor.checkpoint()["current"]["parameters_sha256"]
        start = time.perf_counter()
        with (base / f"{mode}.jsonl").open("x", encoding="utf-8", newline="\n") as log:
            for index in range(updates):
                metrics = trainer.update()
                actor = trainer.model.actor.checkpoint()["current"]["parameters_sha256"]
                assert (actor == initial) == (index < warmup)
                row = (
                    metrics
                    | trainer.counters
                    | {"elapsed_seconds": time.perf_counter() - start}
                )
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps({"mode": mode} | row), flush=True)
                if index + 1 == warmup:
                    state = {
                        "model": trainer.model.state_dict(),
                        "optimizer": trainer.optimizer.state_dict(),
                        "env": trainer.env.state,
                        "counters": trainer.counters,
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(),
                    }
                    warmup_states[mode] = {
                        key: digest(value) for key, value in state.items()
                    }
                    trainer.save(base / f"{mode}_warmup.pt")
                    if mode == "signed":
                        assert warmup_states["signed"] == warmup_states["observer"]
        c = trainer.counters
        assert c["decisions"] == updates * 16384
        assert c["value_rows"] == 2 * c["decisions"]
        assert c["warmup_decisions"] == warmup * 16384
        assert c["warmup_value_rows"] == 2 * c["warmup_decisions"]
        assert c["policy_updates"] == updates - warmup
        assert c["optimizer_steps_value"] == updates * 96
        steps = {}
        for name, parameter in trainer.model.named_parameters():
            assert torch.isfinite(parameter).all()
            state = trainer.optimizer.state[parameter]
            phase = (
                "value"
                if name.startswith("value.")
                else {"0": "discard", "1": "pick", "3": "koi"}[name.split(".")[2]]
            )
            expected = c["optimizer_steps_" + phase]
            assert int(state["step"]) == expected > 0
            assert all(torch.isfinite(value).all() for value in state.values())
            steps[name] = expected
        trainer.save(base / f"{mode}_final.pt")
        records[mode] = {
            "config": config_dict,
            "counters": c.copy(),
            "optimizer_steps": steps,
            "initial_actor": initial,
            "final_actor": actor,
            "checkpoint_sha256": sha(base / f"{mode}_final.pt"),
            "warmup_checkpoint_sha256": sha(base / f"{mode}_warmup.pt"),
            "metrics_sha256": sha(base / f"{mode}.jsonl"),
            "elapsed_seconds": time.perf_counter() - start,
        }
        del trainer
        if device == "cuda":
            torch.cuda.empty_cache()
    assert all(sha(path) == expected for path, expected in inputs.items())
    assert all(sha(path) == expected for path, expected in prior["inputs"].items())
    assert provenance()["source_sha256"] == production
    report = {
        "status": "passed",
        "inputs": inputs,
        "production_source_sha256": production,
        "conditions": records,
        "matched_warmup_states": warmup_states,
        "total_work": {
            key: sum(row["counters"][key] for row in records.values())
            for key in ("decisions", "rounds", "matches", "value_rows")
        },
        "scope": "Production-size capability and exact matched warmup; no strength estimate, no replay, no promotion",
    }
    write(base / "audit.json", report)
    print(
        json.dumps(
            {
                "status": "passed",
                "audit_sha256": sha(base / "audit.json"),
                "total_work": report["total_work"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--warmup-updates", type=int, default=20)
    parser.add_argument("--updates", type=int, default=22)
    args = parser.parse_args()
    run(args.output, args.device, args.warmup_updates, args.updates)
