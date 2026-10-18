"""Fixed small CPU integration check from both retained published parents."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from koikoi.agents import NeuralAgent
from scripts.diagnose_observer_seat_information import sha
from scripts.seat_observer_selfplay import (
    SeatObserverConfig,
    SeatObserverTrainer,
    load_policy,
)


def run(base):
    if base.exists():
        raise ValueError("Use a fresh capability directory")
    protocol_path = Path("runs/observer_selfplay_study/protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    parents = [
        Path(f"runs/cold_ppo_screen/seed{s}_warm/update_00000520.pt")
        for s in (6710001, 6720001)
    ]
    inputs = {
        str(p): sha(p)
        for p in [
            Path(__file__),
            Path("scripts/seat_observer_selfplay.py"),
            protocol_path,
            *parents,
        ]
    }
    assert all(sha(p) == h for p, h in protocol["inputs"].items())
    config = SeatObserverConfig(
        device="cpu",
        cpu_threads=2,
        envs=8,
        horizon=64,
        minibatch=256,
        epochs=1,
        critic_width=256,
        critic_warmup_updates=2,
        seed=26092001,
    )
    base.mkdir(parents=True)
    plan = {
        "inputs": inputs,
        "config": asdict(config),
        "updates_per_case": 3,
        "parents": [str(p) for p in parents],
        "contexts": ["public", "masked"],
        "scope": "CPU integration and actual-gradient check,not playing strength or full-size CUDA capability",
    }
    (base / "protocol.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    cases = {}
    for index, parent in enumerate(parents):
        for context in plan["contexts"]:
            name = f"parent{index + 1}_{context}"
            config.seat_context = context
            trainer = SeatObserverTrainer.from_parent(config, parent)
            obs, mask = trainer.observe()
            baseline = NeuralAgent(parent, deterministic=True)
            with torch.no_grad():
                expected = baseline.model(obs, mask)
                actual = trainer.model(obs, mask)
                assert torch.equal(expected[0], actual[0])
                torch.testing.assert_close(expected[1], actual[1], rtol=1e-6, atol=1e-7)
            initial_actor = trainer.model.actor.checkpoint()["current"][
                "parameters_sha256"
            ]
            trainer.save(base / f"{name}_initial.pt")
            metrics = []
            for update in range(3):
                row = trainer.update() | trainer.counters.copy()
                assert row["critic_only"] == (update < 2)
                if update < 2:
                    assert (
                        trainer.model.actor.checkpoint()["current"]["parameters_sha256"]
                        == initial_actor
                    )
                metrics.append(row)
            final = base / f"{name}_final.pt"
            trainer.save(final)
            assert (
                trainer.model.actor.checkpoint()["current"]["parameters_sha256"]
                != initial_actor
            )
            assert trainer.counters["decisions"] == 1536
            assert trainer.counters["value_rows"] == 3072
            seat_l1 = float(trainer.model.value[0].weight[:, 561].detach().abs().sum())
            assert (seat_l1 > 0) if context == "public" else (seat_l1 == 0)
            optimizer = trainer.optimizer.state_dict()
            value_ids = optimizer["param_groups"][1]["params"]
            assert len(value_ids) == 6
            for key in value_ids:
                assert int(optimizer["state"][key]["step"]) == 6
            assert all(
                torch.isfinite(t).all()
                for s in optimizer["state"].values()
                for t in s.values()
            )
            obs, mask = trainer.observe()
            loaded = load_policy(final)
            with torch.no_grad():
                for a, b in zip(trainer.model(obs, mask), loaded(obs, mask)):
                    assert torch.equal(a, b)
            cases[name] = {
                "counters": trainer.counters.copy(),
                "seat_column_l1": seat_l1,
                "value_adam_steps": 6,
                "optimizer_parameter_states": len(optimizer["state"]),
                "initial_sha256": sha(base / f"{name}_initial.pt"),
                "final_sha256": sha(final),
                "metrics": metrics,
            }
            print(
                json.dumps(
                    {
                        "case": name,
                        "counters": trainer.counters,
                        "seat_column_l1": seat_l1,
                    }
                ),
                flush=True,
            )
    assert all(sha(p) == h for p, h in inputs.items())
    assert all(sha(p) == h for p, h in protocol["inputs"].items())
    result = {
        "status": "passed",
        "inputs": inputs,
        "protocol_sha256": sha(base / "protocol.json"),
        "cases": cases,
        "total_work": {
            k: sum(r["counters"][k] for r in cases.values())
            for k in ("decisions", "rounds", "matches", "value_rows")
        },
        "scope": plan["scope"],
    }
    (base / "audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "audit_sha256": sha(base / "audit.json"),
                "total_work": result["total_work"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
