"""Training-only paired action outcomes; raw hidden worlds never enter a model.

Branches start from the actual sampled self-play world and share future deal
streams. Both seats subsequently use the same frozen public-observation policy.
Only the public root observation, proposals and outcome targets are exported.
"""

from dataclasses import dataclass

import torch

from koikoi.env import BatchEnv
from koikoi.layout import EPISODE, PHASE, PLAYER, POINTS, SEED, TERMINAL


@dataclass
class RolloutWork:
    decisions: int = 0
    completed_branches: int = 0
    round_endings: int = 0


class PublicContinuation:
    """Adapter for a frozen PublishedActor; observations and legal masks only."""

    def __init__(self, actor):
        self.actor = actor.eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, env):
        features, legal = env.published_observe()
        solved = env.solved_actions()
        legal &= (solved[:, None] < 0) | (
            torch.arange(50, device=env.device)[None] == solved[:, None]
        )
        return self.actor(features, env.state[:, PHASE], legal).argmax(-1)


@torch.no_grad()
def paired_outcomes(states, proposals, continuation, max_steps=512):
    """Return root-player outcomes for both forced first actions.

    Hidden simulator states are used solely to instantiate training trajectories.
    The continuation receives its own ordinary observations through its adapter.
    Inputs are immutable. Branch endpoints are simulation work, not new deals.
    """
    if states.ndim != 2 or states.shape[1] != 208 or states.dtype != torch.int32:
        raise ValueError("Expected packed int32 simulator states")
    if proposals.shape != (len(states), 2):
        raise ValueError("Expected two proposals per root")
    if (states[:, PHASE] == TERMINAL).any():
        raise ValueError("Roots must be active")
    count = len(states)
    env = BatchEnv(2 * count, states.device, "paper8", debug=True)
    env.state.copy_(states.repeat_interleave(2, dim=0))
    player = states[:, PLAYER].repeat_interleave(2)
    actions = proposals.reshape(-1)
    work = RolloutWork()
    for step in range(max_steps):
        active = env.state[:, PHASE] != TERMINAL
        if not active.any():
            break
        if step:
            actions = continuation(env)
        result = env.step(actions, freeze_done=True)
        work.decisions += int(active.sum())
        work.round_endings += int(result[:, 1].sum())
        work.completed_branches += int(result[:, 2].sum())
    if (env.state[:, PHASE] != TERMINAL).any():
        raise RuntimeError("Branch rollout exceeded the fixed completion bound")
    if work.completed_branches != 2 * count:
        raise RuntimeError("Branch completion accounting mismatch")
    points0 = env.state[:, POINTS] - env.state[:, POINTS + 1]
    points = points0 * (1 - 2 * player)
    wins = points.sign().float()
    return wins.reshape(count, 2), points.reshape(count, 2), work


@torch.no_grad()
def collect_root_batch(env, bank, count, generator, max_steps=10000):
    """Sample disagreement roots before observing their counterfactual outcomes.

    Behavior chooses the parent90% and Point10% at disagreements. Match identifiers
    are grouping metadata for leakage-free train/validation splits, never features.
    """
    if count < 1:
        raise ValueError("Root count must be positive")
    states, observations, groups = [], [], []
    total = 0
    source_work = {"decisions": 0, "completed_matches": 0, "round_endings": 0}
    for _ in range(max_steps):
        obs, _ = bank.observe(env)
        proposals = obs[:, 560:].long()
        different = proposals[:, 0] != proposals[:, 1]
        indices = torch.where(different)[0]
        # Randomize within a simultaneous batch; never prefer early env indices.
        if len(indices):
            order = torch.randperm(len(indices), device=env.device, generator=generator)
            indices = indices[order[: count - total]]
            states.append(env.state[indices].clone())
            observations.append(obs[indices].clone())
            groups.append(env.state[indices][:, [SEED, EPISODE]].clone())
            total += len(indices)
        choose_point = (
            torch.rand(env.count, device=env.device, generator=generator) < 0.1
        )
        actions = proposals.gather(1, choose_point.long()[:, None]).squeeze(1)
        result = env.step(actions)
        source_work["decisions"] += env.count
        source_work["round_endings"] += int(result[:, 1].sum())
        source_work["completed_matches"] += int(result[:, 2].sum())
        env.reset_done()
        if total == count:
            return (
                torch.cat(states),
                torch.cat(observations),
                torch.cat(groups),
                source_work,
            )
    raise RuntimeError("Insufficient disagreement roots within collection bound")


def match_split(groups, validation_modulus=5):
    """Assign whole source matches to validation, including all their decisions."""
    groups = groups.long()
    mixed = (groups[:, 0] * 73856093) ^ (groups[:, 1] * 19349663)
    return mixed.remainder(validation_modulus) == 0
