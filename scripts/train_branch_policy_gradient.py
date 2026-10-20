"""Small matched-source study of terminal-outcome actor updates.

This module adds an experimental checkpoint format. It does not overwrite or
promote the playable checkpoint, and does not label its updates as PPO.
"""

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from koikoi.env import BatchEnv
from koikoi.layout import PLAYER, POINTS
from koikoi.published_actor import PublishedActor
from koikoi.train import provenance
from scripts.branch_policy_gradient import (
    BranchWorkspace,
    collect_roots,
    expected_return_loss,
    probabilities,
)
from scripts.run_arbitration_study import PARENTS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    return value


def write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def setup(seed):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass(frozen=True)
class Config:
    mode: str
    seed: int
    matches: int = 64
    device: str = "cuda"
    temperature: float = 0.05
    lr: float = 1e-5
    clip_norm: float = 0.5

    def __post_init__(self):
        if self.mode not in ("shared_actions", "independent_actions", "sampled_return"):
            raise ValueError("Unknown gradient mode")
        if type(self.matches) is not int or self.matches < 1:
            raise ValueError("Invalid match batch")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31 - 4000000:
            raise ValueError("Invalid seed")
        if any(
            not math.isfinite(x) or x <= 0
            for x in (self.temperature, self.lr, self.clip_norm)
        ):
            raise ValueError("Invalid optimizer or temperature setting")


def sampled_return_loss(probs, actions, returns, counts):
    """Root-sampled REINFORCE, with no fitted or batch-dependent baseline."""
    selected = probs.gather(1, actions[:, None]).squeeze(1)
    if (selected <= 0).any() or not torch.isfinite(selected).all():
        raise ValueError("Recorded behavior action has zero/nonfinite probability")
    return -(selected.log() * returns.detach() * counts.detach()).mean()


class Trainer:
    def __init__(self, parent, config):
        self.config = config
        setup(config.seed)
        self.parent = {"path": str(parent), "sha256": sha(parent)}
        snapshot = torch.load(parent, map_location="cpu", weights_only=False)
        self.actor = PublishedActor(snapshot["actor"], config.device)
        self.env = BatchEnv(config.matches, config.device, seed=config.seed, debug=True)
        self.workspace = BranchWorkspace(config.matches * 8, config.device)
        self.optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.lr, eps=1e-5
        )
        self.generators = {
            name: torch.Generator(device=config.device).manual_seed(
                config.seed + offset
            )
            for name, offset in (
                ("actions", 1000000),
                ("reservoir", 2000000),
                ("branches", 3000000),
            )
        }
        self.counters = {
            "updates": 0,
            "source": dict.fromkeys(("decisions", "round_endings", "completions"), 0),
            "branches": dict.fromkeys(("decisions", "round_endings", "completions"), 0),
            "eligible_decisions": 0,
        }
        self.source = provenance()

    def update(self):
        c = self.config
        source = collect_roots(
            self.env,
            self.actor,
            self.generators["actions"],
            self.generators["reservoir"],
            c.temperature,
        )
        features, phases, legal = self.workspace.root_inputs(source["states"])
        branch = None
        if c.mode != "sampled_return":
            branch = self.workspace.run(
                source["states"],
                self.actor,
                self.generators["branches"],
                temperature=c.temperature,
                shared_actions=c.mode == "shared_actions",
            )
        probs = probabilities(self.actor, features, phases, legal, c.temperature)
        before = probs.detach().clone()
        counts = source["eligible_decisions"]
        if branch is None:
            # selected_step is CPU-indexed because transcripts are stored on CPU.
            steps = source["selected_step"].cpu().clamp_min(0)
            actions = (
                source["actions"][steps, torch.arange(c.matches)].long().to(c.device)
            )
            # A zero-weight placeholder may be an initial forced decision.
            actions = torch.where(counts > 0, actions, legal.long().argmax(-1))
            returns = (
                (
                    (source["final"][:, POINTS] - source["final"][:, POINTS + 1])
                    * (1 - 2 * source["states"][:, PLAYER])
                )
                .sign()
                .float()
            )
            loss = sampled_return_loss(probs, actions, returns, counts)
        else:
            loss = expected_return_loss(probs, branch["q"], counts)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), c.clip_norm, error_if_nonfinite=True
        )
        self.optimizer.step()
        with torch.no_grad():
            after = probabilities(self.actor, features, phases, legal, c.temperature)
            kl = (
                (
                    before
                    * (before.clamp_min(1e-38).log() - after.clamp_min(1e-38).log())
                )
                .sum(-1)
                .mean()
            )
        self.counters["updates"] += 1
        self.counters["eligible_decisions"] += int(counts.sum())
        for key in source["work"]:
            self.counters["source"][key] += source["work"][key]
            if branch is not None:
                self.counters["branches"][key] += branch["work"][key]
        metrics = {
            "update": self.counters["updates"],
            "gradient_norm_before_clip": float(norm),
            "root_kl_after_update": float(kl),
            "root_action_changes": int((before.argmax(-1) != after.argmax(-1)).sum()),
            "source_work": source["work"],
            "branch_work": None if branch is None else branch["work"],
        }
        return metrics, {"source": cpu(source), "branch": cpu(branch)}

    def save(self, path):
        payload = {
            "format": "branch_policy_gradient_v1",
            "config": asdict(self.config),
            "parent": self.parent,
            "actor": self.actor.checkpoint(),
            "optimizer": self.optimizer.state_dict(),
            "env": self.env.state,
            "generators": {
                key: gen.get_state() for key, gen in self.generators.items()
            },
            "counters": self.counters,
            "source": self.source,
            "implementation": {
                name: sha(name)
                for name in (
                    "scripts/branch_policy_gradient.py",
                    "scripts/train_branch_policy_gradient.py",
                )
            },
        }
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    @classmethod
    def restore(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "branch_policy_gradient_v1":
            raise ValueError("Unexpected branch-gradient checkpoint format")
        if sha(payload["parent"]["path"]) != payload["parent"]["sha256"]:
            raise ValueError("Parent identity changed")
        if provenance()["source_sha256"] != payload["source"]["source_sha256"]:
            raise ValueError("Production source identity changed")
        if any(
            sha(name) != digest for name, digest in payload["implementation"].items()
        ):
            raise ValueError("Training implementation changed")
        result = cls(payload["parent"]["path"], Config(**payload["config"]))
        actor = PublishedActor(payload["actor"], result.config.device)
        result.actor.load_state_dict(actor.state_dict())
        result.optimizer.load_state_dict(payload["optimizer"])
        result.env.state.copy_(payload["env"])
        for key, state in payload["generators"].items():
            result.generators[key].set_state(state)
        result.counters = payload["counters"]
        result.source = payload["source"]
        return result


def run(base):
    base = Path(base)
    if base.exists():
        raise ValueError("Use a fresh study directory")
    base.mkdir(parents=True)
    names = (
        "scripts/branch_policy_gradient.py",
        "scripts/train_branch_policy_gradient.py",
        "scripts/run_arbitration_study.py",
        "docs/BRANCH_POLICY_GRADIENT_TRAINING.rst",
    )
    plan = {
        "source": provenance(),
        "bindings": {name: sha(name) for name in names},
        "parents": [{"path": name, "sha256": digest} for name, digest in PARENTS],
        "updates_per_arm": 128,
        "save_interval": 16,
        "cases": {
            f"parent{index + 1}_{mode}": {
                "parent_index": index,
                "config": asdict(Config(mode, seed)),
            }
            for index, seed in enumerate((111110001, 111210001))
            for mode in ("shared_actions", "independent_actions", "sampled_return")
        },
        "evaluation": {
            "device": "cuda",
            "batch": 128,
            "pairs_per_opponent": 1024,
            "wp_seed": 113110001,
            "point_seed": 113210001,
            "bootstrap_seed": 113310001,
            "bootstrap_replicates": 10000,
            "greedy": True,
            "endgame": True,
            "primary": "For each parent shared-parent, shared-independent and shared-sampled; equal WP/Point mixture draw-adjusted win change. All six simultaneous 95% lower bounds must exceed zero. Development only; no canonical promotion.",
        },
    }
    for parent in plan["parents"]:
        if sha(parent["path"]) != parent["sha256"]:
            raise ValueError("Parent identity mismatch")
    write(base / "protocol.json", plan)
    started = time.perf_counter()
    complete = {}
    for name, case in plan["cases"].items():
        directory = base / name
        directory.mkdir()
        parent = plan["parents"][case["parent_index"]]
        trainer = Trainer(parent["path"], Config(**case["config"]))
        for step in range(1, plan["updates_per_arm"] + 1):
            metrics, evidence = trainer.update()
            torch.save(evidence, directory / f"evidence_{step:04}.pt")
            with (directory / "metrics.jsonl").open(
                "a", encoding="utf-8", newline="\n"
            ) as f:
                f.write(json.dumps(metrics, allow_nan=False) + "\n")
            if step % plan["save_interval"] == 0:
                trainer.save(directory / f"update_{step:04}.pt")
                print(json.dumps({"case": name, **metrics}), flush=True)
        complete[name] = {
            "counters": trainer.counters,
            "endpoint_sha256": sha(directory / "update_0128.pt"),
            "evidence": {
                p.name: sha(p) for p in sorted(directory.glob("evidence_*.pt"))
            },
            "metrics_sha256": sha(directory / "metrics.jsonl"),
        }
        write(base / "progress.json", complete)
        del trainer
    if any(sha(name) != value for name, value in plan["bindings"].items()):
        raise RuntimeError("Bound source changed during training")
    write(
        base / "complete.json",
        {
            "status": "complete",
            "protocol_sha256": sha(base / "protocol.json"),
            "cases": complete,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/branch_policy_gradient_training")
    run(parser.parse_args().output)
