"""Experimental fixed-opponent returns at the learner's decision times.

Only the learner's observation values are used. This removes the intermediate
opponent-observation bootstrap; it does not make finite-observation PPO unbiased.
No production trainer imports this module yet.
"""

import math

import torch


@torch.no_grad()
def learner_gae(rewards, values, dones, learner, next_value, next_learner, lam=0.95):
    """Return (advantages, targets, valid) in the primitive [time, lane] layout.

    Rewards must already be in the fixed learner's payoff convention for each
    episode, including rewards on opponent steps. Convert BEFORE resetting roles.
    Values and the tail value are meaningful only at learner observations; entries
    at opponent observations are ignored. ``dones`` marks match, not round, ends.
    Discount is one. A gap of k simulator actions decays eligibility by lam**k.

    An unfinished final learner transition is excluded if the rollout tail is an
    opponent observation. Its starting value still bootstraps the previous learner
    transition. All excluded rows have zero advantages AND targets; callers must
    mask them out of actor/value losses and advantage normalization using valid.
    Count all simulated actions, learner actions, and excluded learner rows apart.
    The function detaches targets and does not modify inputs or consume RNG.
    """
    if rewards.ndim != 2 or min(rewards.shape) < 1:
        raise ValueError("Expected nonempty [time, lane] rewards")
    if any(x.shape != rewards.shape for x in (values, dones, learner)):
        raise ValueError("Rewards, values, dones and learner shapes must match")
    if any(x.shape != rewards.shape[1:] for x in (next_value, next_learner)):
        raise ValueError("Tail arrays must have one entry per lane")
    arrays = (rewards, values, dones, learner, next_value, next_learner)
    if any(x.device != rewards.device for x in arrays):
        raise ValueError("All return inputs must share a device")
    if not rewards.is_floating_point() or any(
        x.dtype != rewards.dtype for x in (values, next_value)
    ):
        raise ValueError("Rewards and values must share a floating dtype")
    if any(x.dtype != torch.bool for x in (dones, learner, next_learner)):
        raise ValueError("Done and learner indicators must be boolean")
    if not math.isfinite(lam) or not 0 <= lam <= 1:
        raise ValueError("Lambda must be finite and in [0, 1]")

    advantages = torch.zeros_like(values)
    targets = torch.zeros_like(values)
    valid = torch.zeros_like(learner)
    future_value = torch.where(next_learner, next_value, 0.0)
    future_advantage = torch.zeros_like(next_value)
    ready = next_learner.clone()
    accumulated = torch.zeros_like(next_value)
    live = torch.ones_like(next_learner)
    trace = torch.ones_like(next_value)
    for t in range(rewards.shape[0] - 1, -1, -1):
        # A terminal reward belongs to this episode even when the next row has
        # reset into a new episode with the opposite learner seat.
        accumulated = rewards[t] + torch.where(dones[t], 0.0, accumulated)
        live = live & ~dones[t]
        trace = torch.where(dones[t], 0.0, trace * lam)
        ready = ready | dones[t]
        own = learner[t]
        valid[t] = own & ready
        advantage = (
            accumulated
            + torch.where(live, future_value, 0.0)
            - values[t]
            + trace * future_advantage
        )
        advantages[t] = torch.where(valid[t], advantage, 0.0)
        targets[t] = torch.where(valid[t], advantage + values[t], 0.0)
        # Register the next known learner observation even if its own action has
        # an unfinished outcome. Its unavailable advantage contributes zero.
        future_value = torch.where(own, values[t], future_value)
        future_advantage = torch.where(own, advantages[t], future_advantage)
        ready = ready | own
        accumulated = torch.where(own, 0.0, accumulated)
        live = live | own
        trace = torch.where(own, 1.0, trace)
    return advantages, targets, valid
