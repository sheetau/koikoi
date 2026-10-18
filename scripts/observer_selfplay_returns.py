"""Experimental fixed-seat observations and returns for two-player self-play.

Each value row sees one observer's hand and public information. Rows must never
be joined into a critic that sees both hands. No production trainer imports this
module. A finite observation is not asserted to be a sufficient history.
"""

import math

import torch

from koikoi.env import BatchEnv
from koikoi.layout import PLAYER
from koikoi.train import signed_gae


class ObserverFeatures:
    """Extract [lane, seat, 561] independently, without changing the live game.

    The first 560 columns are the ordinary observation relative to the viewer;
    the last says whether that viewer is the actual actor. Phase, before-turn
    yaku points and history still describe the actual public turn. Only the
    copied PLAYER selector is changed. Scratch legal masks/labels are discarded;
    the scratch states are never stepped or supplied to an actor.
    """

    def __init__(self, env):
        self.scratch = BatchEnv(2 * env.count, env.device, env.profile, seed=0)
        self.seats = torch.arange(2, device=env.device).repeat(env.count)

    @torch.no_grad()
    def __call__(self, env):
        if (
            2 * env.count != self.scratch.count
            or env.device != self.scratch.device
            or env.profile != self.scratch.profile
            or torch.get_default_dtype() != torch.float32
        ):
            raise ValueError(
                "Observer extractor requires its original layout/device/profile"
            )
        self.scratch.state.copy_(env.state.repeat_interleave(2, dim=0))
        actors = self.scratch.state[:, PLAYER].clone()
        self.scratch.state[:, PLAYER] = self.seats
        ordinary, _ = self.scratch.observe()
        role = (actors == self.seats).float().unsqueeze(-1)
        return torch.cat((ordinary, role), dim=-1).reshape(env.count, 2, 561)


@torch.no_grad()
def observer_gae(rewards, values, dones, next_values, lam=0.95):
    """GAE per fixed seat on every primitive step, with match-reset boundaries.

    rewards: [time, lane], in the engine's seat-zero payoff convention.
    values: [time, lane, 2], each conditioned on that seat's own observation.
    dones: [time, lane] match ends; round ends do not stop eligibility.
    next_values: [lane, 2] after reset; ignored for terminal final transitions.

    Gamma is one. Seat-one rewards are the negative of seat-zero rewards, but
    neither seat's value is defined as the negative of the OTHER observation's
    value. All primitive rows have a target; no unfinished-learner-tail omission.
    """
    if rewards.ndim != 2 or min(rewards.shape) < 1:
        raise ValueError("Expected nonempty [time, lane] rewards")
    if (
        values.shape != (*rewards.shape, 2)
        or dones.shape != rewards.shape
        or next_values.shape != (*rewards.shape[1:], 2)
    ):
        raise ValueError("Expected two value columns per lane and boolean match ends")
    if any(x.device != rewards.device for x in (values, dones, next_values)):
        raise ValueError("All inputs must share a device")
    if not rewards.is_floating_point() or any(
        x.dtype != rewards.dtype for x in (values, next_values)
    ):
        raise ValueError("Rewards and values must share a floating dtype")
    if dones.dtype != torch.bool:
        raise ValueError("Match ends must be boolean")
    if not math.isfinite(lam) or not 0 <= lam <= 1:
        raise ValueError("Lambda must be finite and in [0, 1]")
    advantages = torch.empty_like(values)
    carry = torch.zeros_like(next_values)
    future = next_values
    for t in range(len(rewards) - 1, -1, -1):
        reward = torch.stack((rewards[t], -rewards[t]), dim=-1)
        terminal = dones[t, :, None]
        delta = reward + torch.where(terminal, 0.0, future) - values[t]
        carry = delta + lam * torch.where(terminal, 0.0, carry)
        advantages[t] = carry
        future = values[t]
    return advantages, advantages + values


@torch.no_grad()
def policy_advantages(
    mode, rewards, values, dones, players, next_values, next_players, lam=0.95
):
    """Select fixed-observer or signed active-observer policy advantages.

    Both experimental arms must fit the SAME observer_gae value targets. Only
    policy advantages change here. The signed control is not the historical
    trainer: it shares the new role-aware critic and per-observer value fitting.
    """
    if mode not in ("observer", "signed"):
        raise ValueError("Unknown policy return mode")
    advantages, _ = observer_gae(rewards, values, dones, next_values, lam)
    for seats, shape in ((players, rewards.shape), (next_players, rewards.shape[1:])):
        if (
            seats.shape != shape
            or seats.device != rewards.device
            or seats.dtype not in (torch.int32, torch.int64)
            or bool(((seats < 0) | (seats > 1)).any())
        ):
            raise ValueError(
                "Players must be integer seat indices with matching layout"
            )
    selected = players.long().unsqueeze(-1)
    if mode == "observer":
        return advantages.gather(-1, selected).squeeze(-1)
    active_value = values.gather(-1, selected).squeeze(-1)
    tail = next_values.gather(-1, next_players.long().unsqueeze(-1)).squeeze(-1)
    return signed_gae(
        rewards * (1 - 2 * players),
        active_value,
        dones,
        players,
        tail,
        next_players,
        lam,
    )[0]
