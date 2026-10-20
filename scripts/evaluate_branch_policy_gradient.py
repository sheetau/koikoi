"""Fixed GPU development evaluation; separate from strict CPU paper results."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.published_actor import PublishedActor
from koikoi.train import provenance
from scripts.branch_policy_gradient import public_inputs
from scripts.train_branch_policy_gradient import cpu, setup, sha, write


@torch.no_grad()
def play_chunk(env, candidate, opponent, first_seed):
    count = env.count
    if count % 2:
        raise ValueError("Evaluation batch must contain complete seat pairs")
    seeds = torch.arange(
        first_seed, first_seed + count // 2, dtype=torch.int32, device=env.device
    ).repeat_interleave(2)
    env.backend.call("kk_init", count, env.state, seeds, count, 8, 1)
    initial = env.state.clone()
    seats = torch.arange(count, device=env.device).remainder(2)
    transcript = []
    work = dict.fromkeys(("decisions", "round_endings", "completions"), 0)
    for _ in range(512):
        active = env.state[:, PHASE] != TERMINAL
        if not active.any():
            break
        features, phases, legal = public_inputs(env)
        actions = torch.zeros(count, dtype=torch.long, device=env.device)
        ours = env.state[:, PLAYER] == seats
        for selected, actor in ((active & ours, candidate), (active & ~ours, opponent)):
            if selected.any():
                actions[selected] = actor(
                    features[selected], phases[selected], legal[selected]
                ).argmax(-1)
        transcript.append(actions.to(torch.int16).cpu())
        result = env.step(actions, freeze_done=True)
        work["decisions"] += int(active.sum())
        work["round_endings"] += int(result[:, 1].sum())
        work["completions"] += int(result[:, 2].sum())
    if (env.state[:, PHASE] != TERMINAL).any() or work["completions"] != count:
        raise RuntimeError("Evaluation failed to complete every original match")
    points = (
        (env.state[:, POINTS] - env.state[:, POINTS + 1]) * (1 - 2 * seats)
    ).reshape(-1, 2)
    return cpu(
        {
            "initial": initial,
            "final": env.state.clone(),
            "actions": torch.stack(transcript),
            "points": points,
            "work": work,
        }
    )


def contrasts(points, order):
    """Rows are independent compound deal blocks; columns are fixed contrasts."""
    names = []
    rows = []
    for parent in (1, 2):
        shared = (
            np.sign(points[order.index(f"parent{parent}_shared_actions")]) + 1
        ) / 2
        for comparator in ("parent", "independent_actions", "sampled_return"):
            name = (
                f"parent{parent}"
                if comparator == "parent"
                else f"parent{parent}_{comparator}"
            )
            other = (np.sign(points[order.index(name)]) + 1) / 2
            rows.append((shared - other).mean(axis=(0, 2)))
            names.append(f"parent{parent}_shared_minus_{comparator}")
    return names, np.stack(rows, axis=1)


def report(points, order, settings):
    names, rows = contrasts(points, order)
    means = rows.mean(0)
    rng = np.random.default_rng(settings["bootstrap_seed"])
    maxima = []
    for offset in range(0, settings["bootstrap_replicates"], 100):
        count = min(100, settings["bootstrap_replicates"] - offset)
        indices = rng.integers(len(rows), size=(count, len(rows)))
        maxima.extend(np.abs(rows[indices].mean(1) - means).max(-1))
    radius = float(np.quantile(maxima, 0.95))
    primary = [
        {
            "name": name,
            "win_change": float(mean),
            "simultaneous_95": [float(mean - radius), float(mean + radius)],
        }
        for name, mean in zip(names, means)
    ]
    return {
        "primary": primary,
        "progression_criterion": all(
            item["simultaneous_95"][0] > 0 for item in primary
        ),
        "secondary_means": {
            name: {
                opponent: {
                    "win": float(((np.sign(points[i, j]) + 1) / 2).mean()),
                    "point_margin": float(points[i, j].mean()),
                }
                for j, opponent in enumerate(("wp", "point"))
            }
            for i, name in enumerate(order)
        },
    }


def run(base):
    base = Path(base)
    plan = json.loads((base / "protocol.json").read_text(encoding="utf-8"))
    completed = json.loads((base / "complete.json").read_text(encoding="utf-8"))
    if completed["status"] != "complete" or completed["protocol_sha256"] != sha(
        base / "protocol.json"
    ):
        raise ValueError("Training is incomplete or its protocol changed")
    directory = base / "evaluation"
    if directory.exists():
        raise ValueError("Use the unconsumed fixed evaluation only once")
    settings = plan["evaluation"]
    setup(settings["bootstrap_seed"])
    models = {name: str(base / name / "update_0128.pt") for name in plan["cases"]}
    models.update({f"parent{i + 1}": p["path"] for i, p in enumerate(plan["parents"])})
    for name, item in completed["cases"].items():
        if sha(models[name]) != item["endpoint_sha256"]:
            raise ValueError("Endpoint changed")
    for i, parent in enumerate(plan["parents"]):
        if sha(models[f"parent{i + 1}"]) != parent["sha256"]:
            raise ValueError("Parent changed")
    if provenance()["source_sha256"] != plan["source"]["source_sha256"]:
        raise ValueError("Production source changed")
    for name, digest in plan["bindings"].items():
        if sha(name) != digest:
            raise ValueError("Training implementation changed")
    directory.mkdir()
    protocol = {
        "settings": settings,
        "training_protocol_sha256": sha(base / "protocol.json"),
        "training_complete_sha256": sha(base / "complete.json"),
        "models": {
            name: {"path": path, "sha256": sha(path)} for name, path in models.items()
        },
        "source": provenance(),
        "bindings": {
            name: sha(name)
            for name in (
                __file__,
                "scripts/train_branch_policy_gradient.py",
                "scripts/branch_policy_gradient.py",
            )
        },
    }
    write(directory / "protocol.json", protocol)
    order = list(models)
    points = np.empty(
        (len(order), 2, settings["pairs_per_opponent"], 2), dtype=np.int32
    )
    env = BatchEnv(settings["batch"], settings["device"], debug=True)
    opponents = {
        kind: PublishedActor.from_published(kind, settings["device"]).requires_grad_(
            False
        )
        for kind in ("wp", "point")
    }
    protocol["opponents"] = {
        kind: actor.checkpoint()["current"]["parameters_sha256"]
        for kind, actor in opponents.items()
    }
    write(directory / "protocol.json", protocol)
    work = dict.fromkeys(("decisions", "round_endings", "completions"), 0)
    for index, name in enumerate(order):
        payload = torch.load(models[name], map_location="cpu", weights_only=False)
        actor = PublishedActor(payload["actor"], settings["device"]).requires_grad_(
            False
        )
        for opponent_index, (kind, opponent) in enumerate(opponents.items()):
            target = directory / f"{name}_{kind}"
            target.mkdir()
            for offset in range(0, settings["pairs_per_opponent"], env.count // 2):
                data = play_chunk(
                    env, actor, opponent, settings[f"{kind}_seed"] + offset
                )
                points[index, opponent_index, offset : offset + env.count // 2] = data[
                    "points"
                ].numpy()
                torch.save(data, target / f"chunk_{offset:04}.pt")
                for key in work:
                    work[key] += data["work"][key]
            print(
                json.dumps({"model": name, "opponent": kind, "work": work}), flush=True
            )
        del actor
    np.save(directory / "points.npy", points)
    result = report(points, order, settings)
    result.update(
        {
            "status": "complete",
            "model_order": order,
            "work": work,
            "protocol_sha256": sha(directory / "protocol.json"),
            "artifacts": {
                str(p.relative_to(directory)): sha(p)
                for p in sorted(directory.rglob("*.pt"))
            },
            "points_sha256": sha(directory / "points.npy"),
        }
    )
    write(directory / "summary.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="runs/branch_policy_gradient_training")
    run(parser.parse_args().input)
