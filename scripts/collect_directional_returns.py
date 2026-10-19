"""Collect complete frozen-parent trajectories; policies see only public inputs."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import NeuralAgent
from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER, ROUND, TERMINAL
from koikoi.native import build_cpu
from koikoi.published_opponent import runtime_identity
from koikoi.published_train import packed_observation
from scripts.directional_return_probe import (
    directional_scores,
    estimate_advantages,
    seat_contributions,
)
from scripts.directional_return_statistics import ESTIMATORS
from scripts.run_arbitration_study import PARENTS, read, sha, write


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def bindings():
    paths = [
        p for p in Path("koikoi").rglob("*") if p.suffix in (".py", ".h", ".cpp", ".cu")
    ]
    paths += [
        Path(p)
        for p in (
            __file__,
            "scripts/audit_directional_returns.py",
            "scripts/directional_return_probe.py",
            "scripts/directional_return_statistics.py",
            "scripts/run_arbitration_study.py",
            "docs/DIRECTIONAL_RETURN_PROBE.rst",
            "tests/test_directional_return_probe.py",
            "tests/test_directional_return_statistics.py",
        )
    ]
    paths += [Path(p) for p, _ in PARENTS] + [build_cpu()]
    return {p.resolve().relative_to(Path.cwd()).as_posix(): sha(p) for p in paths}


def checked(inputs):
    for path, expected in inputs.items():
        if sha(path) != expected:
            raise ValueError(f"Input changed: {path}")


def model_digest(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def collect(model, deal_seed, sampling_seed, offset_seed):
    env = BatchEnv(64, "cuda", "paper8", deal_seed)
    torch.manual_seed(sampling_seed)
    initial = env.state.cpu().numpy().copy()
    offsets = np.random.default_rng(offset_seed).integers(0, 128, size=64)
    records = {
        k: []
        for k in (
            "action",
            "player",
            "phase",
            "round",
            "valid",
            "value",
            "weights",
            "legal",
            "result",
        )
    }
    state_hashes, public_hashes = [], []
    for _ in range(1024):
        state = env.state
        valid = state[:, PHASE] != TERMINAL
        if not valid.any():
            break
        state_hashes.append(digest(state.cpu().numpy()))
        obs, mask = packed_observation(env)
        guard = env.solved_actions()
        mask &= (guard[:, None] < 0) | (
            torch.arange(50, device="cuda")[None] == guard[:, None]
        )
        public_hashes.append(digest(obs.cpu().numpy()))
        logits, value, _ = model(obs, mask)
        weights = logits.log_softmax(-1).exp()
        action = torch.multinomial(weights, 1).squeeze(-1)
        for key, tensor in {
            "action": action,
            "player": state[:, PLAYER],
            "phase": state[:, PHASE],
            "round": state[:, ROUND],
            "valid": valid,
            "value": value.masked_fill(~valid, 0),
            "weights": weights,
            "legal": mask,
        }.items():
            records[key].append(tensor.cpu().numpy().copy())
        result = env.step(action, freeze_done=True)
        if (result[:, 0] == 99).any():
            raise ValueError("Illegal recorded action")
        records["result"].append(result.cpu().numpy().copy())
    if not (env.state[:, PHASE] == TERMINAL).all():
        raise ValueError("Safety limit reached with incomplete matches")
    arrays = {k: np.stack(v) for k, v in records.items()}
    arrays.update(
        initial=initial,
        final=env.state.cpu().numpy().copy(),
        offsets=offsets,
        state_hashes=np.asarray(state_hashes),
        public_hashes=np.asarray(public_hashes),
    )
    scores = directional_scores(
        arrays["weights"],
        arrays["action"],
        arrays["phase"],
        arrays["legal"],
        arrays["valid"],
    )
    advantages = estimate_advantages(
        arrays["result"][..., 0],
        arrays["value"],
        arrays["player"],
        arrays["valid"],
        arrays["result"][..., 2],
        offsets,
    )
    arrays["scores"] = scores
    for key, value in advantages.items():
        arrays["adv_" + key] = value
    arrays["contributions"] = np.stack(
        [
            seat_contributions(
                scores, advantages[key], arrays["player"], arrays["valid"]
            )
            for key in ESTIMATORS
        ],
        axis=1,
    )
    counts = {
        "matches": int(arrays["result"][..., 2].sum()),
        "rounds": int(arrays["result"][..., 1].sum()),
        "active_decisions": int(arrays["valid"].sum()),
        "padded_rows": int((~arrays["valid"]).sum()),
        "model_forward_batches": len(arrays["valid"]),
        "updates": 0,
    }
    assert counts["matches"] == 64
    assert counts["rounds"] == int((arrays["final"][:, ROUND] + 1).sum())
    return arrays, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("capability", "main"), required=True)
    parser.add_argument("--capability", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    inputs = bindings()
    for path, expected in PARENTS:
        assert sha(path) == expected
    capability = None
    if args.mode == "main":
        if args.capability is None:
            raise ValueError("Main collection requires passed capability audit")
        audit = read(args.capability / "audit.json")
        prior = read(args.capability / "protocol.json")
        assert audit["status"] == "passed" and prior["mode"] == "capability"
        assert audit["protocol_sha256"] == sha(args.capability / "protocol.json")
        assert audit["report_sha256"] == sha(args.capability / "report.json")
        assert prior["inputs"] == inputs
        capability = {
            "path": str(args.capability),
            "audit_sha256": sha(args.capability / "audit.json"),
        }
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "mode": args.mode,
        "inputs": inputs,
        "parents": PARENTS,
        "batches": 1 if args.mode == "capability" else 64,
        "envs": 64,
        "first_seed": 98100001 if args.mode == "capability" else 98110001,
        "sampling_first": 98200001 if args.mode == "capability" else 98210001,
        "offset_first": 98300000 if args.mode == "capability" else 98300001,
        "capability": capability,
        "bootstrap": {
            "replicates": 10000,
            "seed": 98410001,
            "family": "70 means and paired differences; max fixed-SE deviation; null band for zero SE",
        },
    }
    write(args.output / "protocol.json", protocol)
    report = {
        "protocol_sha256": sha(args.output / "protocol.json"),
        "batches": [],
        "models": [],
    }
    for parent, (path, _) in enumerate(PARENTS):
        agent = NeuralAgent(path, "cuda")
        assert (
            agent.profile == "paper8"
            and agent.requires_endgame
            and agent.model.temperature == 0.05
        )
        before = model_digest(agent.model)
        runtime = runtime_identity("cuda")
        for batch in range(protocol["batches"]):
            start = time.monotonic()
            deal = protocol["first_seed"] + 64 * batch
            sampling = protocol["sampling_first"] + 10000 * parent + batch
            offset = protocol["offset_first"] + batch
            arrays, counts = collect(agent.model, deal, sampling, offset)
            name = f"parent{parent + 1}_{batch:03}.npz"
            np.savez_compressed(args.output / name, **arrays)
            row = {
                "parent": parent,
                "batch": batch,
                "path": name,
                "sha256": sha(args.output / name),
                "deal_seed": deal,
                "sampling_seed": sampling,
                "offset_seed": offset,
                "counts": counts,
                "seconds": time.monotonic() - start,
            }
            report["batches"].append(row)
            write(args.output / "progress.json", report)
            print(json.dumps(row), flush=True)
        after = model_digest(agent.model)
        assert before == after and runtime == runtime_identity("cuda")
        report["models"].append(
            {
                "parent": parent,
                "before_sha256": before,
                "after_sha256": after,
                "runtime": runtime,
            }
        )
        del agent
    checked(inputs)
    report["status"] = "complete"
    write(args.output / "report.json", report)


if __name__ == "__main__":
    main()
