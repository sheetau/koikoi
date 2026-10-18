"""One-round simulation and finite public-context policy evaluation.

Backups define an independent-fresh-deal round model. They do not prove that
finite PRNG seeds in the original multi-round simulator are independent, or
that estimated boundary values reduce action-estimation error in practice.
"""

from dataclasses import dataclass

import numpy as np
import torch

from koikoi.env import BatchEnv
from koikoi.layout import BANKRUPT, DEALER, PHASE, POINTS, ROUND, ROUNDS, TERMINAL
from koikoi.reference import ReferenceEnv


def fresh_round_states(contexts, seeds):
    """Construct artificial round starts from (round, points0, dealer), not a game.

    Both policies will receive freshly dealt own/public observations. The public
    context is selected before dealing; no actual match's future cards are used.
    """
    contexts, seeds = np.asarray(contexts), np.asarray(seeds)
    if contexts.ndim != 2 or contexts.shape[1] != 3 or seeds.shape != (len(contexts),):
        raise ValueError("Expected N public contexts and N seeds")
    if contexts.dtype.kind not in "iu" or seeds.dtype.kind not in "iu":
        raise ValueError("Contexts and seeds must be integers")
    if not len(contexts):
        raise ValueError("At least one round start is required")
    if not ((contexts[:, 0] >= 0) & (contexts[:, 0] < 8)).all():
        raise ValueError("Round indices must lie in 0..7")
    if not ((contexts[:, 1] >= 1) & (contexts[:, 1] <= 59)).all():
        raise ValueError("Live player-zero points must lie in 1..59")
    if not np.isin(contexts[:, 2], [0, 1]).all():
        raise ValueError("Dealer must be an absolute seat")
    if not ((seeds >= 0) & (seeds < 2**31 - 1)).all():
        raise ValueError("Seeds must fit the simulator range")
    rows = []
    for (round_index, points0, dealer), seed in zip(contexts, seeds):
        env = ReferenceEnv(int(seed), "paper8")
        env.state[ROUND] = round_index
        env.state[POINTS : POINTS + 2] = (points0, 60 - points0)
        env.state[DEALER] = dealer
        env.deal()
        rows.append(env.state)
    return np.stack(rows)


@dataclass
class RoundWork:
    decisions: int = 0
    round_endings: int = 0
    terminal_match_endings: int = 0


class RoundWorkspace:
    """Stop each lane at its FIRST round ending, retaining its real endpoint.

    A temporary terminal phase freezes completed scratch lanes between calls.
    Returned states retain the true phase, including a freshly dealt next round.
    The continuation never acts on that next round. Artificial context endpoints
    are not independent complete matches started from the ordinary initial state.
    """

    def __init__(self, count, device="cpu"):
        self.env = BatchEnv(count, device, "paper8", debug=True)

    @torch.no_grad()
    def run(self, states, continuation, first_actions=None, max_steps=64):
        env = self.env
        if states.shape != env.state.shape or states.dtype != torch.int32:
            raise ValueError("Expected one packed int32 state per lane")
        if states.device != env.state.device:
            raise ValueError("State and workspace devices must match")
        if (
            (states[:, ROUNDS] != 8)
            | (states[:, BANKRUPT] != 1)
            | (states[:, PHASE] == TERMINAL)
        ).any():
            raise ValueError("Expected live paper8 states")
        if first_actions is not None and first_actions.shape != (env.count,):
            raise ValueError("Expected one optional first action per lane")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be positive")
        env.state.copy_(states)
        done = torch.zeros(env.count, dtype=torch.bool, device=states.device)
        endpoints = torch.empty_like(states)
        work = RoundWork()
        for step in range(max_steps):
            if done.all():
                break
            actions = (
                first_actions
                if step == 0 and first_actions is not None
                else continuation(env)
            )
            result = env.step(actions, freeze_done=True)
            work.decisions += int((~done).sum())
            work.round_endings += int(result[:, 1].sum())
            work.terminal_match_endings += int(result[:, 2].sum())
            ended = (~done) & (result[:, 1] != 0)
            endpoints[ended] = env.state[ended]
            done |= ended
            env.state[done, PHASE] = TERMINAL
        if not done.all():
            raise RuntimeError("Round exceeded the fixed decision bound")
        assert work.round_endings == env.count
        # Do not leave a synthetic terminal flag in the actual reported endpoint.
        env.state.copy_(endpoints)
        return endpoints, work


def backup_round(points0, next_dealer, terminal, next_values, root_player):
    """Back up all 59x2 contexts from equally weighted one-round samples.

    next_values is signed match utility for the same fixed player/model pair.
    Exact terminal wins/draws take precedence over any next-round table lookup.
    """
    points0, next_dealer, terminal = map(np.asarray, (points0, next_dealer, terminal))
    values = np.asarray(next_values, dtype=np.float64)
    if points0.ndim != 3 or points0.shape[:2] != (59, 2) or points0.shape[2] < 1:
        raise ValueError("Expected 59x2xN round endpoints")
    if points0.shape != next_dealer.shape or points0.shape != terminal.shape:
        raise ValueError("Endpoint arrays must have equal shape")
    if (
        points0.dtype.kind not in "iu"
        or next_dealer.dtype.kind not in "iu"
        or terminal.dtype != bool
    ):
        raise ValueError("Expected integer points/dealers and boolean terminal flags")
    if (
        values.shape != (59, 2)
        or not np.isfinite(values).all()
        or (np.abs(values) > 1).any()
    ):
        raise ValueError("Expected bounded finite next-round signed values")
    if root_player not in (0, 1):
        raise ValueError("Root player must be an absolute seat")
    live = ~terminal
    if not ((points0[live] >= 1) & (points0[live] <= 59)).all():
        raise ValueError("Nonterminal endpoints must have live point totals")
    if not np.isin(next_dealer, [0, 1]).all():
        raise ValueError("Invalid next dealer")
    target = np.sign(points0.astype(np.int64) - 30).astype(np.float64) * (
        1 - 2 * root_player
    )
    target[live] = values[points0[live] - 1, next_dealer[live]]
    return target.mean(axis=2)


def backward_values(points0, next_dealer, terminal):
    """Evaluate sampled round kernels for one parent and both WP/Point models.

    Axes: round(8), parent seat(2), opponent model(2), points0(59), dealer(2),
    deal sample(N). Absolute seat/dealer axes are never folded by symmetry.
    """
    points0, next_dealer, terminal = map(np.asarray, (points0, next_dealer, terminal))
    if points0.ndim != 6 or points0.shape[:-1] != (8, 2, 2, 59, 2):
        raise ValueError("Expected every round/player/model/score/dealer context")
    if points0.shape != next_dealer.shape or points0.shape != terminal.shape:
        raise ValueError("Endpoint arrays must have equal shape")
    if not terminal[7].all():
        raise ValueError("The eighth round must end the match")
    values = np.zeros(points0.shape[:-1], dtype=np.float64)
    for round_index in range(7, -1, -1):
        for player in (0, 1):
            for model in (0, 1):
                key = (round_index, player, model)
                after = (
                    values[round_index + 1, player, model]
                    if round_index < 7
                    else np.zeros((59, 2))
                )
                values[key] = backup_round(
                    points0[key], next_dealer[key], terminal[key], after, player
                )
    return values
