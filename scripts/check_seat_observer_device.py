"""Matched seat-context capability with exact restart and guarded GPU scheduling."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.env import BatchEnv
from koikoi.layout import PLAYER
from koikoi.published_train import packed_observation
from koikoi.reference import ReferenceEnv
from scripts.check_observer_selfplay import digest, sha, write
from scripts.seat_observer_selfplay import (
    SeatObserverConfig,
    SeatObserverFeatures,
    SeatObserverTrainer,
)

STUDY = Path("runs/observer_selfplay_study")
CASES = {f"parent{i}_{mode}" for i in (1, 2) for mode in ("observer", "signed")}
WORK = ("decisions", "rounds", "matches", "value_rows")


def training_ready(state):
    if state["status"] not in ("running", "complete"):
        raise ValueError("Current study is not running or successfully complete")
    rows = [j for j in state["jobs"] if j["mode"] == "train"]
    cases = {j["case"] for j in rows}
    if not cases <= CASES or len(cases) != len(rows):
        raise ValueError("Unexpected or duplicate current training jobs")
    for j in rows:
        if j.get("returncode") not in (None, 0):
            raise ValueError("A current training worker failed")
        if j.get("returncode") == 0 and not j["ended"] > j["started"]:
            raise ValueError("Invalid completed-worker timing")
    return cases == CASES and all(j.get("returncode") == 0 for j in rows)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def signatures(trainer):
    return {
        key: digest(value)
        for key, value in {
            "actor": trainer.model.actor.checkpoint(),
            "value": trainer.model.value.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "env": trainer.env.state,
            "counters": trainer.counters,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        }.items()
    }


@torch.no_grad()
def check_features(trainer, states):
    env = BatchEnv(len(states), trainer.env.device, "paper8", seed=0)
    env.state.copy_(torch.from_numpy(states).to(env.device))
    before = env.state.clone()
    features = SeatObserverFeatures(env, trainer.config.seat_context)(env)
    expected = []
    for state in states:
        seats = []
        for seat in (0, 1):
            ref = ReferenceEnv.__new__(ReferenceEnv)
            ref.state = state.copy()
            ref.state[PLAYER] = seat
            seats.append(
                np.r_[
                    ref.observe(),
                    float(state[PLAYER] == seat),
                    float(seat) if trainer.config.seat_context == "public" else 0,
                ].astype("float32")
            )
        expected.append(seats)
    np.testing.assert_array_equal(features.cpu().numpy(), np.array(expected))
    obs, _ = packed_observation(env)
    assert torch.equal(
        trainer.model.active_features(obs),
        features[
            torch.arange(len(states), device=env.device), env.state[:, PLAYER].long()
        ],
    )
    assert torch.equal(before, env.state)


def run(base, smoke=False, wait=False):
    if base.exists():
        raise ValueError("Use a fresh capability directory")
    if smoke and wait:
        raise ValueError("CPU smoke mode cannot wait for GPU training")
    protocol_path = STUDY / "protocol.json"
    prior = read(protocol_path)
    paths = [
        Path(__file__),
        Path("scripts/seat_observer_selfplay.py"),
        Path("scripts/check_observer_selfplay.py"),
        Path("tests/test_seat_observer_device.py"),
        Path("docs/SEAT_OBSERVER_STUDY.rst"),
        protocol_path,
        Path("runs/cold_ppo_screen/seed6710001_warm/update_00000520.pt"),
        Path("runs/observer_seat_information_v2/parent1_states.npy"),
    ]
    inputs = prior["inputs"] | {str(p): sha(p) for p in paths}
    assert all(sha(p) == h for p, h in inputs.items())
    current = read(STUDY / "process.json")
    assert current["protocol_sha256"] == sha(protocol_path)
    ready = training_ready(current)
    if not smoke and not ready and not wait:
        raise ValueError("The current study still has unfinished GPU training")
    warmup, updates = (1, 3) if smoke else (20, 22)
    conditions = {
        context: asdict(
            SeatObserverConfig(
                device="cpu" if smoke else "cuda",
                cpu_threads=2,
                envs=4 if smoke else 128,
                horizon=64 if smoke else 128,
                minibatch=128 if smoke else 512,
                epochs=1 if smoke else 3,
                critic_width=256,
                critic_warmup_updates=warmup,
                seed=26095001,
                seat_context=context,
                policy_returns="observer",
            )
        )
        for context in ("public", "masked")
    }
    base.mkdir(parents=True)
    write(
        base / "protocol.json",
        {
            "inputs": inputs,
            "conditions": conditions,
            "updates": updates,
            "replay_updates_per_condition": 1,
            "kind": "cpu_smoke" if smoke else "cuda_capability",
            "scope": "Capability,not strength; no partial outcome selection",
        },
    )
    if not smoke:
        if not ready:
            print(
                json.dumps(
                    {"status": "waiting_for_existing_gpu_training", "output": str(base)}
                ),
                flush=True,
            )
        while not ready:
            time.sleep(20)
            assert all(sha(p) == h for p, h in inputs.items())
            current = read(STUDY / "process.json")
            assert current["protocol_sha256"] == inputs[str(protocol_path)]
            ready = training_ready(current)
        for case in CASES:
            audit = read(STUDY / case / "training_audit.json")
            assert audit["status"] == "passed" and audit["counters"]["updates"] == 520
            assert sha(STUDY / case / "final.pt") == audit["checkpoint_sha256"]
        print(json.dumps({"status": "gpu_training_prerequisites_passed"}), flush=True)
    torch.set_num_threads(2)
    states = np.load(paths[-1], allow_pickle=False)
    parent = paths[-2]
    records, initial, warm = {}, {}, {}
    for context, spec in conditions.items():
        trainer = SeatObserverTrainer.from_parent(SeatObserverConfig(**spec), parent)
        check_features(trainer, states)
        initial[context] = signatures(trainer)
        trainer.save(base / f"{context}_initial.pt")
        if context == "masked":
            assert initial["public"] == initial["masked"]
        initial_actor = initial[context]["actor"]
        resume_counters = None
        with (base / f"{context}.jsonl").open(
            "x", encoding="utf-8", newline="\n"
        ) as log:
            for index in range(updates):
                metrics = trainer.update()
                row = metrics | trainer.counters.copy()
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps({"context": context} | row), flush=True)
                if index < warmup:
                    assert digest(trainer.model.actor.checkpoint()) == initial_actor
                if index + 1 == warmup:
                    warm[context] = signatures(trainer)
                    trainer.save(base / f"{context}_warmup.pt")
                    if context == "masked":
                        for key in (
                            "actor",
                            "env",
                            "counters",
                            "torch_rng",
                            "cuda_rng",
                        ):
                            assert warm["public"][key] == warm["masked"][key], key
                if index + 1 == updates - 1:
                    trainer.save(base / f"{context}_resume.pt")
                    resume_counters = trainer.counters.copy()
        expected = signatures(trainer)
        assert expected["actor"] != initial_actor
        column_l1 = float(trainer.model.value[0].weight[:, 561].detach().abs().sum())
        assert (column_l1 > 0) if context == "public" else (column_l1 == 0)
        trainer.save(base / f"{context}_final.pt")
        counters = trainer.counters.copy()
        steps = {}
        for name, parameter in trainer.model.named_parameters():
            assert torch.isfinite(parameter).all()
            state = trainer.optimizer.state[parameter]
            phase = (
                "value"
                if name.startswith("value.")
                else {"0": "discard", "1": "pick", "3": "koi"}[name.split(".")[2]]
            )
            count = counters["optimizer_steps_" + phase]
            assert int(state["step"]) == count > 0
            assert all(torch.isfinite(t).all() for t in state.values())
            steps[name] = count
        del trainer
        restored = SeatObserverTrainer.restore(base / f"{context}_resume.pt")
        assert restored.update() == metrics
        assert signatures(restored) == expected
        replay = {k: counters[k] - resume_counters[k] for k in WORK}
        records[context] = {
            "counters": counters,
            "replay_work": replay,
            "physical_work": {k: counters[k] + replay[k] for k in WORK},
            "optimizer_steps": steps,
            "seat_column_l1": column_l1,
            "final_sha256": sha(base / f"{context}_final.pt"),
            "resume_sha256": sha(base / f"{context}_resume.pt"),
            "warmup_sha256": sha(base / f"{context}_warmup.pt"),
            "initial_sha256": sha(base / f"{context}_initial.pt"),
            "metrics_sha256": sha(base / f"{context}.jsonl"),
        }
        del restored
        if not smoke:
            torch.cuda.empty_cache()
    assert all(sha(p) == h for p, h in inputs.items())
    result = {
        "status": "passed",
        "kind": "cpu_smoke" if smoke else "cuda_capability",
        "inputs": inputs,
        "protocol_sha256": sha(base / "protocol.json"),
        "conditions": records,
        "initial_signatures": initial,
        "warmup_signatures": warm,
        "feature_probes_per_condition": len(states),
        "total_physical_work": {
            k: sum(r["physical_work"][k] for r in records.values()) for k in WORK
        },
        "scope": "Exact saved restart includes one additional executed update per condition; no strength or promotion",
    }
    write(base / "audit.json", result)
    print(
        json.dumps(
            {
                "status": "passed",
                "sha256": sha(base / "audit.json"),
                "total_physical_work": result["total_physical_work"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    run(args.output, args.smoke, args.wait)
