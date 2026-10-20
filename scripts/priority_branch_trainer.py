"""Separate root-priority/step-size experiment over the frozen branch machinery."""

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from koikoi.published_actor import PublishedActor
from koikoi.train import provenance
from scripts.branch_policy_gradient import expected_return_loss, probabilities
from scripts.priority_branch_roots import collect_roots
from scripts.train_branch_policy_gradient import Config as BaseConfig
from scripts.train_branch_policy_gradient import Trainer as BaseTrainer
from scripts.train_branch_policy_gradient import cpu, sha


@dataclass(frozen=True)
class Config(BaseConfig):
    sampler: str = "sensitivity"

    def __post_init__(self):
        super().__post_init__()
        if self.mode != "shared_actions" or self.sampler not in (
            "uniform",
            "sensitivity",
        ):
            raise ValueError(
                "Priority study requires shared branches and a known sampler"
            )


class Trainer(BaseTrainer):
    def update(self):
        c = self.config
        source = collect_roots(
            self.env,
            self.actor,
            self.generators["actions"],
            self.generators["reservoir"],
            c.sampler,
            c.temperature,
        )
        features, phases, legal = self.workspace.root_inputs(source["states"])
        branch = self.workspace.run(
            source["states"],
            self.actor,
            self.generators["branches"],
            temperature=c.temperature,
            shared_actions=True,
        )
        p = probabilities(self.actor, features, phases, legal, c.temperature)
        before = p.detach().clone()
        loss = expected_return_loss(p, branch["q"], source["importance"].float())
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
        self.counters["eligible_decisions"] += int(source["eligible_decisions"].sum())
        for key in source["work"]:
            self.counters["source"][key] += source["work"][key]
            self.counters["branches"][key] += branch["work"][key]
        metrics = {
            "update": self.counters["updates"],
            "gradient_norm_before_clip": float(norm),
            "root_kl_after_update": float(kl),
            "root_action_changes": int((before.argmax(-1) != after.argmax(-1)).sum()),
            "root_confident_fraction": float(
                (before.max(-1).values >= 0.99).float().mean()
            ),
            "constant_outcome_fraction": float(
                (
                    branch["q"].masked_fill(~legal, -2).max(-1).values
                    == branch["q"].masked_fill(~legal, 2).min(-1).values
                )
                .float()
                .mean()
            ),
            "mean_importance": float(source["importance"].mean()),
            "source_work": source["work"],
            "branch_work": branch["work"],
        }
        return metrics, {"source": cpu(source), "branch": cpu(branch)}

    def save(self, path):
        # Reuse the established payload construction, then label this new format
        # and bind both the original and new numerical implementations.
        staging = Path(str(path) + ".staging")
        super().save(staging)
        payload = torch.load(staging, map_location="cpu", weights_only=False)
        payload["format"] = "priority_branch_gradient_v1"
        payload["implementation"].update(
            {
                name: sha(name)
                for name in (
                    "scripts/priority_branch_roots.py",
                    "scripts/priority_branch_trainer.py",
                )
            }
        )
        temporary = Path(str(path) + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        staging.unlink()

    @classmethod
    def restore(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "priority_branch_gradient_v1":
            raise ValueError("Unexpected priority checkpoint")
        if sha(payload["parent"]["path"]) != payload["parent"]["sha256"]:
            raise ValueError("Parent changed")
        if provenance()["source_sha256"] != payload["source"]["source_sha256"]:
            raise ValueError("Production sources changed")
        if any(
            sha(name) != digest for name, digest in payload["implementation"].items()
        ):
            raise ValueError("Numerical implementation changed")
        result = cls(payload["parent"]["path"], Config(**payload["config"]))
        actor = PublishedActor(payload["actor"], result.config.device)
        result.actor.load_state_dict(actor.state_dict())
        result.optimizer.load_state_dict(payload["optimizer"])
        result.env.state.copy_(payload["env"])
        for key, state in payload["generators"].items():
            result.generators[key].set_state(state)
        result.counters, result.source = payload["counters"], payload["source"]
        return result
