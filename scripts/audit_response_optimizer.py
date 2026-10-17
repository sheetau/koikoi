"""Independently derive endpoint Adam budgets and measure parameter movement.

This companion check does not call the runner's training auditor. Source checks
use its existing immutable protocol verifier. It measures implementation facts,
not policy quality, and never selects a checkpoint or changes a running study.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.run_response_study import checked


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(base, case, output):
    if output.exists():
        raise ValueError("Use a fresh audit output")
    torch.set_num_threads(2)
    protocol = checked(base)
    spec = protocol["cases"][case]
    folder = base / case
    paths = [
        folder / name
        for name in (
            "initial.pt",
            "final.pt",
            "training_complete.json",
            "training_audit.json",
        )
    ] + [base / "protocol.json", Path(__file__)]
    inputs = {str(path): sha(path) for path in paths}
    prior = torch.load(folder / "initial.pt", map_location="cpu", weights_only=False)
    final = torch.load(folder / "final.pt", map_location="cpu", weights_only=False)
    receipt = json.loads(
        (folder / "training_complete.json").read_text(encoding="utf-8")
    )
    assert receipt["checkpoint_sha256"] == inputs[str(folder / "final.pt")]
    config = spec["config"]
    assert prior["config"] == final["config"] == config
    updates, warmup = protocol["updates"], config["critic_warmup_updates"]
    batch = config["envs"] * config["horizon"]
    steps_per_update = (
        (batch + config["minibatch"] - 1) // config["minibatch"]
    ) * config["epochs"]
    actor_steps = (updates - warmup) * steps_per_update
    critic_steps = updates * steps_per_update
    blocks = []
    for phase, name in ((0, "discard"), (1, "pick"), (3, "koi")):
        blocks.append(
            (
                name,
                prior["actor"]["current"]["weights"][phase],
                final["actor"]["current"]["weights"][phase],
                actor_steps,
            )
        )
    blocks.append(("value", prior["value"], final["value"], critic_steps))
    if spec["arm"] == "response":
        blocks.append(
            (
                "opponent_value",
                prior["opponent_value"],
                final["opponent_value"],
                critic_steps,
            )
        )
    groups, states = final["optimizer"]["param_groups"], final["optimizer"]["state"]
    ids = [index for group in groups for index in group["params"]]
    assert len(set(ids)) == len(ids) == sum(len(b[2]) for b in blocks)
    assert set(ids) == set(states)
    assert len(groups[0]["params"]) == sum(len(b[2]) for b in blocks[:3])
    assert [len(g["params"]) for g in groups[1:]] == [len(b[2]) for b in blocks[3:]]
    offset, report = 0, {}
    for name, initial, current, expected in blocks:
        assert list(initial) == list(current)
        assert final["counters"]["optimizer_steps_" + name] == expected
        delta_squared = initial_squared = 0.0
        changed = 0
        for index, key in zip(
            ids[offset : offset + len(current)], current, strict=True
        ):
            old, new = initial[key], current[key]
            state = states[index]
            assert old.shape == new.shape
            assert torch.isfinite(new).all()
            assert int(state["step"]) == expected
            for moment in ("exp_avg", "exp_avg_sq"):
                assert (
                    state[moment].shape == new.shape
                    and torch.isfinite(state[moment]).all()
                )
            changed += not torch.equal(old, new)
            delta_squared += float((new.double() - old.double()).square().sum())
            initial_squared += float(old.double().square().sum())
        offset += len(current)
        report[name] = {
            "parameter_tensors": len(current),
            "changed_tensors": changed,
            "expected_and_actual_adam_steps": expected,
            "delta_l2": delta_squared**0.5,
            "initial_l2": initial_squared**0.5,
        }
    checked(base)
    assert all(sha(path) == digest for path, digest in inputs.items())
    result = {
        "status": "passed",
        "case": case,
        "inputs": inputs,
        "blocks": report,
        "scope": "Adam steps derived from frozen update/minibatch/warmup settings, checked against actual per-parameter state; initial-to-final parameter movement. No strength estimate, checkpoint selection, replay or new simulation.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps({"output": str(output), "sha256": sha(output), "blocks": report}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("runs/response_study"))
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.case, args.output)
