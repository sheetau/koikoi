"""Terminal reference paths from artificial public predeal contexts.

Two chance modes share their first deal. Native keeps its seed across rounds;
independent redeals only at a nonterminal boundary from a fresh round/sample key.
"""

from dataclasses import dataclass

import numpy as np
import torch

from koikoi.env import BatchEnv
from koikoi.layout import (
    BANKRUPT,
    DEALER,
    DISCARD,
    PHASE,
    PLAYER,
    POINTS,
    ROUND,
    ROUNDS,
    TERMINAL,
    TURN,
)
from scripts.round_transition_values import fresh_round_states


class FreshDealBank:
    """Independent round/replicate deals; context fields are set before policy use."""

    def __init__(self, base_seed, samples, device):
        if type(samples) is not int or samples < 1:
            raise ValueError("Use a positive sample count")
        if type(base_seed) is not int or not 0 <= base_seed < 2**31 - 7000000 - samples:
            raise ValueError("Invalid round seed range")
        self.samples, self.base_seed = samples, base_seed
        self.device = torch.device(device)
        contexts = np.array(
            [[r, 30, 0] for r in range(8) for _ in range(samples)], dtype=np.int32
        )
        seeds = np.concatenate(
            [base_seed + 1000000 * r + np.arange(samples) for r in range(8)]
        )
        self.states = (
            torch.from_numpy(fresh_round_states(contexts, seeds))
            .to(device)
            .reshape(8, samples, -1)
        )

    def starts(self, contexts, indices):
        contexts = torch.as_tensor(contexts, device=self.device)
        indices = torch.as_tensor(indices, device=self.device)
        if (
            contexts.ndim != 2
            or contexts.shape[1] != 3
            or indices.shape != contexts.shape[:1]
        ):
            raise ValueError(
                "Expected public contexts and one replicate index per lane"
            )
        if contexts.dtype not in (torch.int32, torch.int64) or indices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("Contexts and indices must be integers")
        if not len(indices) or not ((indices >= 0) & (indices < self.samples)).all():
            raise ValueError("Invalid replicate indices")
        r, p, d = contexts.unbind(1)
        if not ((r >= 0) & (r < 8) & (p >= 1) & (p <= 59) & (d >= 0) & (d <= 1)).all():
            raise ValueError("Invalid public context")
        state = self.states[r.long(), indices.long()].clone()
        state[:, POINTS] = p
        state[:, POINTS + 1] = 60 - p
        state[:, PLAYER] = state[:, DEALER] = d
        return state

    def replace_boundaries(self, state, ended, indices):
        """Copy fresh deals only for genuine nonterminal round endpoints."""
        selected = torch.where(ended & (state[:, PHASE] != TERMINAL))[0]
        if len(selected):
            contexts = state[selected][:, [ROUND, POINTS, DEALER]]
            state[selected] = self.starts(contexts, indices[selected])
        return len(selected)


@dataclass
class ReferenceWork:
    decisions: int = 0
    round_endings: int = 0
    terminal_paths: int = 0
    reseeded_rounds: int = 0


class TerminalReference:
    def __init__(self, count, device):
        self.env = BatchEnv(count, device, "paper8", debug=True)

    @torch.no_grad()
    def run(self, states, continuation, indices, bank=None, max_steps=512):
        env = self.env
        if (
            states.shape != env.state.shape
            or states.dtype != torch.int32
            or states.device != env.state.device
        ):
            raise ValueError("Expected packed int32 states on the workspace device")
        if (
            (states[:, ROUNDS] != 8)
            | (states[:, BANKRUPT] != 1)
            | (states[:, PHASE] != DISCARD)
            | (states[:, TURN] != 0)
        ).any():
            raise ValueError("Reference paths must start at fresh paper8 rounds")
        indices = torch.as_tensor(indices, device=states.device)
        if indices.shape != (env.count,) or indices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("Expected integer replicate indices")
        if bank is not None and (
            bank.device != env.device
            or not ((indices >= 0) & (indices < bank.samples)).all()
        ):
            raise ValueError(
                "Deal bank and replicate indices do not match the workspace"
            )
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("Use a positive decision bound")
        env.state.copy_(states)
        used = ReferenceWork()
        for _ in range(max_steps):
            active = env.state[:, PHASE] != TERMINAL
            if not active.any():
                break
            actions = continuation(env)
            result = env.step(actions, freeze_done=True)
            used.decisions += int(active.sum())
            used.round_endings += int(result[:, 1].sum())
            used.terminal_paths += int(result[:, 2].sum())
            if bank is not None:
                used.reseeded_rounds += bank.replace_boundaries(
                    env.state, result[:, 1] != 0, indices
                )
        if (env.state[:, PHASE] != TERMINAL).any():
            raise RuntimeError("Reference paths exceeded the fixed completion bound")
        assert used.terminal_paths == env.count
        if bank is not None:
            assert used.reseeded_rounds == used.round_endings - used.terminal_paths
        return env.state.clone(), used
