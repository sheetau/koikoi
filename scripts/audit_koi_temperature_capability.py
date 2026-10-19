"""Independently inspect saved production-size capability states and accounting."""

import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.run_arbitration_study import read, sha, write


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b


def audit(root):
    output = root / "independent_audit.json"
    if output.exists():
        raise ValueError("Preserve completed independent audit")
    original = read(root / "audit.json")
    assert original["status"] == "passed"
    inputs = dict(original["inputs"])
    inputs[str(root / "audit.json")] = sha(root / "audit.json")
    inputs[str(Path(__file__))] = sha(__file__)
    for name, digest in inputs.items():
        assert sha(name) == digest, name
    initial_states, results = [], {}
    for mode in ("fixed", "calibrated"):
        folder = root / mode
        initial, warm, prior, final, replayed = [
            torch.load(folder / name, map_location="cpu", weights_only=False)
            for name in (
                "initial.pt",
                "update_00000020.pt",
                "update_00000021.pt",
                "update_00000022.pt",
                "resumed_22.pt",
            )
        ]
        initial_states.append(initial)
        assert all(
            p["format"] == "koi_temperature_ppo_v1"
            for p in (initial, warm, prior, final, replayed)
        )
        assert initial["optimizer"]["state"] == {}
        same(initial["actor"], warm["actor"])
        assert (
            final["counters"]["updates"] == 22
            and final["counters"]["decisions"] == 22 * 128 * 128
        )
        assert (
            final["counters"]["warmup_updates"] == 20
            and final["counters"]["policy_updates"] == 2
        )
        groups = final["optimizer"]["param_groups"]
        assert (groups[0]["lr"], groups[1]["lr"]) == (3e-6, 1e-4)
        assert {
            int(final["optimizer"]["state"][i]["step"].item())
            for i in groups[0]["params"]
        } == {192}
        assert {
            int(final["optimizer"]["state"][i]["step"].item())
            for i in groups[1]["params"]
        } == {2112}
        assert not set(groups[0]["params"]) & set(warm["optimizer"]["state"])
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
            same(final[key], replayed[key])
        logs = read(folder / "metrics.json")
        assert [r["update"] for r in logs] == list(range(1, 23))
        for row in logs:
            assert row["counters"]["decisions"] == row["update"] * 16384
            assert bool(row["metrics"]["critic_only"]) == (row["update"] <= 20)
            assert all(
                np.isfinite(v)
                for v in row["metrics"].values()
                if isinstance(v, (float, int))
            )
        same(logs[-1]["counters"], final["counters"])
        replay_work = {
            k: final["counters"][k] - prior["counters"][k]
            for k in ("decisions", "rounds", "matches")
        }
        assert replay_work == original["arms"][mode]["replay_work"]
        results[mode] = {
            "counters": final["counters"],
            "replay_work": replay_work,
            "optimizer_states": len(final["optimizer"]["state"]),
        }
    a, b = initial_states
    assert {k for k in a["config"] if a["config"][k] != b["config"][k]} == {
        "koi_temperature"
    }
    for key in (
        "actor",
        "value",
        "optimizer",
        "env",
        "torch_rng",
        "cuda_rng",
        "python_rng",
        "numpy_rng",
    ):
        same(a[key], b[key])
    write(
        output,
        {
            "status": "passed",
            "inputs": inputs,
            "arms": results,
            "scope": "Independent checkpoint/RNG/optimizer/counter checks; zero new training or policy inferences.",
        },
    )
    print(
        {
            "status": "passed",
            "logical_decisions": sum(
                r["counters"]["decisions"] for r in results.values()
            ),
            "replayed_decisions": sum(
                r["replay_work"]["decisions"] for r in results.values()
            ),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
