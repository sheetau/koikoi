"""Describe controller choices on a fixed cached corpus and input interventions.

These are policy diagnostics, not new matches or estimates of playing strength.
Expert proposals stay fixed during the controller-only context interventions.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.arbitration import ArbitrationAgent, read_checkpoint
from koikoi.env import BatchEnv
from koikoi.train import provenance
from scripts.benchmark_final_latency import CORPUS, CORPUS_SHA, spread
from scripts.run_arbitration_study import sha, write
from scripts.train_paired_arbitration import PairedRolloutAgent

ROUNDS = (0, 3, 6)
OWN_POINTS = (10, 20, 30, 40, 50)


def proposal_mask(public):
    proposals = public[:, 560:].long()
    return torch.zeros(
        (len(public), 50), dtype=torch.bool, device=public.device
    ).scatter_(1, proposals, True)


@torch.no_grad()
def context_grid(model, public):
    """Modify only explicit round/score features, holding proposals fixed."""
    mask = proposal_mask(public)
    choices = []
    for round_index in ROUNDS:
        row = []
        for own_points in OWN_POINTS:
            modified = public.clone()
            modified[:, 534] = round_index / 12
            modified[:, 535] = 8 / 12
            modified[:, 536] = own_points / 64
            modified[:, 537] = (60 - own_points) / 64
            row.append(model(modified, mask)[0].argmax(-1))
        choices.append(torch.stack(row))
    return torch.stack(choices)


def grouped(public, selected):
    proposals = public[:, 560:].long()
    different = proposals[:, 0] != proposals[:, 1]
    point = different & (selected == proposals[:, 1])
    phase = public[:, 528:532].argmax(-1)
    round_index = (public[:, 534] * 12).round().int()
    lead = public[:, 536] - public[:, 537]
    groups = {f"phase{p}": phase == p for p in range(4)}
    groups.update(
        {
            "early": round_index < 3,
            "middle": (round_index >= 3) & (round_index < 6),
            "late": round_index >= 6,
            "trailing": lead < 0,
            "tied": lead == 0,
            "leading": lead > 0,
            "all": torch.ones(len(public), dtype=torch.bool),
        }
    )
    result = {}
    for name, active in groups.items():
        n, d, chosen = (
            int(active.sum()),
            int((active & different).sum()),
            int((active & point).sum()),
        )
        result[name] = {
            "states": n,
            "disagreements": d,
            "point_choices": chosen,
            "point_fraction_at_disagreement": chosen / d if d else None,
        }
    return result


def diagnose(checkpoints, output, count=2048):
    if output.exists():
        raise ValueError("Use a fresh diagnostics directory")
    if sha(CORPUS) != CORPUS_SHA:
        raise ValueError("Cached corpus changed")
    torch.set_num_threads(2)
    with np.load(CORPUS, allow_pickle=False) as data:
        indices = spread(np.flatnonzero(data["sources"] >= 0), count)
        states = torch.from_numpy(data["states"][indices].copy())
    output.mkdir(parents=True)
    report = {
        "scope": "Descriptive cached random-play states, not playing strength or causal match effects. Context grid changes controller inputs only; expert proposals, legal mask and other features remain fixed. Public guard and experts are not recomputed on hypothetical contexts. States may be outside the controller's training distribution.",
        "corpus": str(CORPUS),
        "corpus_sha256": CORPUS_SHA,
        "script_sha256": sha(__file__),
        "source": provenance(),
        "states": count,
        "indices": indices.tolist(),
        "expert_batch": 64,
        "controller_batch": count,
        "round_indices": list(ROUNDS),
        "own_points": list(OWN_POINTS),
        "models": {},
    }
    for checkpoint in checkpoints:
        payload, digest = read_checkpoint(checkpoint)
        cls = (
            PairedRolloutAgent
            if payload["format"] == "paired_rollout_arbitration_v1"
            else ArbitrationAgent
        )
        agent = cls(checkpoint)
        parts = []
        for chunk in states.split(64):
            env = BatchEnv(len(chunk), "cpu", "paper8")
            env.state.copy_(chunk)
            parts.append(agent.bank.observe(env)[0])
        public = torch.cat(parts)
        mask = proposal_mask(public)
        with torch.no_grad():
            original = agent.model(public, mask)[0].argmax(-1)
            grid = context_grid(agent.model, public)
        proposals = public[:, 560:].long()
        different = proposals[:, 0] != proposals[:, 1]
        point_grid = grid[:, :, different] == proposals[different, 1][None, None, :]
        sensitivity = (grid != original[None, None, :]).any(dim=0).any(dim=0)
        if agent.config.context in ("phase", "no_match") and sensitivity.any():
            raise ValueError(
                "Context-ablated controller unexpectedly responds to context"
            )
        name = checkpoint.parent.name
        if name in report["models"]:
            raise ValueError("Checkpoint directory names must be unique")
        raw = output / (name + ".npz")
        np.savez_compressed(
            raw,
            public=public.numpy(),
            selected=original.numpy(),
            context_actions=grid.numpy(),
        )
        row = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest,
            "context": agent.config.context,
            "raw_sha256": sha(raw),
            "groups": grouped(public, original),
            "context_point_fraction": point_grid.float().mean(-1).tolist(),
            "context_sensitive_disagreements": int((different & sensitivity).sum()),
            "expert_digests": [
                s["current"]["parameters_sha256"] for s in payload["experts"]
            ],
        }
        report["models"][name] = row
        print(
            json.dumps(
                {
                    "model": name,
                    "groups": row["groups"],
                    "context_point_fraction": row["context_point_fraction"],
                }
            ),
            flush=True,
        )
    write(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=2048)
    args = parser.parse_args()
    diagnose(args.checkpoints, args.output, args.states)


if __name__ == "__main__":
    main()
