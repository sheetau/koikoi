"""Terminal-aware CUDA blocks for deterministic policy evaluation."""

import torch

from .capture import CapturedRollout
from .layout import PLAYER


class CapturedSearchRollout:
    """Keep first-match outcomes; omit work only after every particle finishes.

    The caller replaces the entire scratch environment before every invocation.
    Post-terminal scratch state need not equal a full-horizon rollout. Model,
    horizon, block size and endgame setting must stay fixed after construction.
    """

    @torch.no_grad()
    def __init__(self, model, env, actions, sign, horizon, chunk_size, endgame):
        self.model, self.env = model, env
        self.actions, self.sign = actions, sign
        self.endgame = endgame
        self.alive = torch.ones(len(env.state), dtype=torch.bool, device=env.device)
        self.values = torch.zeros(len(env.state), device=env.device)
        self.blocks = [
            (start == 0, min(chunk_size, horizon - start))
            for start in range(0, horizon, chunk_size)
        ]
        self.graphs = {}
        self.preparation_decisions = 0
        self.last_steps = 0
        for key in dict.fromkeys(self.blocks):
            alive, values = self.alive.clone(), self.values.clone()
            self.graphs[key] = CapturedRollout(
                lambda key=key: self._block(*key), env.state
            )
            # The generic capture restores env and torch RNG. These persistent
            # accumulator tensors also need rollback after its three warmups.
            self.alive.copy_(alive)
            self.values.copy_(values)
            self.preparation_decisions += 3 * len(env.state) * key[1]

    def _block(self, first, length):
        if first:
            self.alive.fill_(True)
            self.values.zero_()
        for depth in range(length):
            if first and depth == 0:
                actions = self.actions
            else:
                obs, mask = self.env.observe()
                actions = self.model(obs, mask)[0].argmax(-1)
            if self.endgame:
                solved = self.env.solved_actions()
                actions = torch.where(solved >= 0, solved, actions)
            result = self.env.step(actions)
            ended = result[:, 2].bool()
            self.values.copy_(
                torch.where(
                    self.alive & ended,
                    result[:, 0].float() * self.sign,
                    self.values,
                )
            )
            self.alive.logical_and_(~ended)
            self.env.reset_done()
        return {"any_alive": self.alive.any()}, None

    @torch.no_grad()
    def __call__(self):
        self.last_steps = 0
        for key in self.blocks:
            pending = bool(self.graphs[key]()[0]["any_alive"])
            self.last_steps += key[1]
            if not pending:
                break
        if pending:
            # Preserve the original horizon bootstrap for unfinished particles.
            obs, mask = self.env.observe()
            estimate = (
                self.model(obs, mask)[1]
                * (1 - 2 * self.env.state[:, PLAYER])
                * self.sign
            )
            values = torch.where(self.alive, estimate, self.values)
        else:
            values = self.values
        self.batch = {"values": values}
        return self.batch.copy(), None
