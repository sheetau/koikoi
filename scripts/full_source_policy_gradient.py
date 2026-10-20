"""Single-step complete-trajectory Monte Carlo control for the retained actors."""

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from koikoi.layout import PHASE, PLAYER, POINTS, STATE_SIZE, TERMINAL
from koikoi.published_actor import PublishedActor
from koikoi.train import provenance
from scripts.branch_policy_gradient import Work, probabilities, public_actions
from scripts.train_branch_policy_gradient import Config as BaseConfig
from scripts.train_branch_policy_gradient import Trainer as BaseTrainer
from scripts.train_branch_policy_gradient import cpu, sha


@dataclass(frozen=True)
class Config(BaseConfig):
    decision_scope: str = "all"
    gradient_batch: int = 256

    def __post_init__(self):
        super().__post_init__()
        if self.decision_scope not in ("all", "root"):
            raise ValueError("Unknown decision scope")
        if self.decision_scope == "all" and self.mode != "sampled_return":
            raise ValueError("Full-source scope requires terminal sampled returns")
        if type(self.gradient_batch) is not int or self.gradient_batch < 1:
            raise ValueError("Invalid gradient batch")
        if self.decision_scope == "all" and self.gradient_batch > self.matches * 8:
            raise ValueError("Gradient batch exceeds the reusable workspace")


@torch.no_grad()
def collect_full_source(
    env, actor, action_rng, reservoir_rng, temperature=0.05, max_steps=512
):
    """Keep every eligible decision, preserving the old source and reservoir RNG."""
    env.reset_done()
    initial = env.state.clone()
    roots = initial.clone()
    counts = torch.zeros(env.count, dtype=torch.long, device=env.device)
    selected_step = torch.full_like(counts, -1)
    work, transcript = Work(), []
    states, choices, lanes, steps = [], [], [], []
    for step in range(max_steps):
        active = env.state[:, PHASE] != TERMINAL
        if not active.any():
            break
        uniforms = torch.rand(env.count, device=env.device, generator=action_rng)
        actions, legal = public_actions(env, actor, uniforms, temperature)
        eligible = active & (legal.sum(-1) > 1)
        indices = torch.where(eligible)[0]
        if len(indices):
            states.append(env.state[indices].clone())
            choices.append(actions[indices].clone())
            lanes.append(indices)
            steps.append(torch.full_like(indices, step))
        counts += eligible
        choose = eligible & (
            torch.rand(env.count, device=env.device, generator=reservoir_rng)
            < 1 / counts.clamp_min(1).double()
        )
        roots[choose] = env.state[choose]
        selected_step[choose] = step
        transcript.append(actions.to(torch.int16).cpu())
        result = env.step(actions, freeze_done=True)
        work.add(active, result)
    if (env.state[:, PHASE] != TERMINAL).any():
        raise RuntimeError("Source matches exceeded completion bound")
    if work.completions != env.count:
        raise RuntimeError("Source accounting mismatch")
    empty = torch.empty(0, dtype=torch.long, device=env.device)
    decisions = (
        torch.cat(states)
        if states
        else torch.empty((0, STATE_SIZE), dtype=torch.int32, device=env.device)
    )
    decision_lanes = torch.cat(lanes) if lanes else empty
    margin = env.state[:, POINTS] - env.state[:, POINTS + 1]
    returns = (margin[decision_lanes] * (1 - 2 * decisions[:, PLAYER])).sign().float()
    return {
        "initial": initial,
        "final": env.state.clone(),
        "states": roots,
        "eligible_decisions": counts,
        "selected_step": selected_step,
        "actions": torch.stack(transcript),
        "work": work.dictionary(),
        "decision_states": decisions,
        "decision_actions": torch.cat(choices) if choices else empty,
        "decision_lanes": decision_lanes,
        "decision_steps": torch.cat(steps) if steps else empty,
        "decision_returns": returns,
    }


def full_return_loss(probs, actions, returns, matches):
    """A sum over decisions divided by matches, independent of chunk lengths."""
    if type(matches) is not int or matches < 1:
        raise ValueError("Expected a positive complete-match count")
    if (
        probs.ndim != 2
        or actions.shape != (len(probs),)
        or returns.shape != (len(probs),)
    ):
        raise ValueError("Incompatible decision arrays")
    selected = probs.gather(1, actions[:, None]).squeeze(1)
    if (
        (selected <= 0).any()
        or not torch.isfinite(selected).all()
        or not torch.isfinite(returns).all()
    ):
        raise ValueError("Invalid probability or terminal return")
    return -(selected.log() * returns.detach()).sum() / matches


class Trainer(BaseTrainer):
    def update(self):
        c = self.config
        if c.decision_scope == "root":
            metrics, data = super().update()
            metrics["gradient_rows"] = int(
                (data["source"]["eligible_decisions"] > 0).sum()
            )
            return metrics, data
        source = collect_full_source(
            self.env,
            self.actor,
            self.generators["actions"],
            self.generators["reservoir"],
            c.temperature,
        )
        root_inputs = self.workspace.root_inputs(source["states"])
        with torch.no_grad():
            before = probabilities(self.actor, *root_inputs, c.temperature)
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        rows = len(source["decision_states"])
        for start in range(0, rows, c.gradient_batch):
            part = slice(start, start + c.gradient_batch)
            inputs = self.workspace.root_inputs(source["decision_states"][part])
            p = probabilities(self.actor, *inputs, c.temperature)
            loss = full_return_loss(
                p,
                source["decision_actions"][part],
                source["decision_returns"][part],
                c.matches,
            )
            loss.backward()
            loss_sum += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), c.clip_norm, error_if_nonfinite=True
        )
        self.optimizer.step()
        with torch.no_grad():
            after = probabilities(self.actor, *root_inputs, c.temperature)
            kl = (
                (
                    before
                    * (before.clamp_min(1e-38).log() - after.clamp_min(1e-38).log())
                )
                .sum(-1)
                .mean()
            )
        self.counters["updates"] += 1
        self.counters["eligible_decisions"] += rows
        for key in source["work"]:
            self.counters["source"][key] += source["work"][key]
        metrics = {
            "update": self.counters["updates"],
            "gradient_rows": rows,
            "loss": loss_sum,
            "gradient_norm_before_clip": float(norm),
            "root_kl_after_update": float(kl),
            "root_action_changes": int((before.argmax(-1) != after.argmax(-1)).sum()),
            "source_work": source["work"],
            "branch_work": None,
        }
        return metrics, {"source": cpu(source), "branch": None}

    def save(self, path):
        staging = Path(str(path) + ".staging")
        super().save(staging)
        payload = torch.load(staging, map_location="cpu", weights_only=False)
        payload["format"] = "full_source_policy_gradient_v1"
        payload["implementation"][__file__] = sha(__file__)
        temporary = Path(str(path) + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        staging.unlink()

    @classmethod
    def restore(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "full_source_policy_gradient_v1":
            raise ValueError("Unexpected full-source checkpoint")
        if sha(payload["parent"]["path"]) != payload["parent"]["sha256"]:
            raise ValueError("Parent changed")
        if provenance()["source_sha256"] != payload["source"]["source_sha256"]:
            raise ValueError("Production source changed")
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
        result.counters, result.source = payload["counters"], payload["source"]
        return result
