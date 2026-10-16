"""Batched, device-resident simulator. Observations never include private labels."""

import torch

from .layout import OBS_SIZE, STATE_SIZE, PROFILES
from .native import CPU, CUDA


class BatchEnv:
    def __init__(self, count=1024, device="cpu", profile="paper8", seed=1, debug=False):
        if count < 1 or not 0 <= seed < 2**31-count:
            raise ValueError("Invalid environment count or seed")
        self.count, self.device, self.profile, self.debug = count, torch.device(device), profile, debug
        self.backend = CUDA(self.device) if self.device.type == "cuda" else CPU()
        self.state = torch.empty((count, STATE_SIZE), dtype=torch.int32, device=self.device)
        seeds = torch.arange(seed, seed+count, dtype=torch.int32, device=self.device)
        rounds, bankrupt = PROFILES[profile]
        self.backend.call("kk_init", count, self.state, seeds, count, rounds, bankrupt)

    def observe(self, labels=False):
        obs = torch.empty((self.count, OBS_SIZE), device=self.device)
        mask = torch.empty((self.count, 50), dtype=torch.bool, device=self.device)
        target = torch.empty((self.count, 48), device=self.device)
        self.backend.call("kk_observe", self.count, self.state, obs, mask, target, self.count)
        return (obs, mask, target) if labels else (obs, mask)

    def step(self, actions):
        if actions.shape != (self.count,):
            raise ValueError("Expected one action per environment")
        actions = actions.to(device=self.device, dtype=torch.int32).contiguous()
        result = torch.empty((self.count, 4), dtype=torch.int32, device=self.device)
        self.backend.call("kk_step", self.count, self.state, actions, result, self.count)
        if self.debug and (result[:, 0] == 99).any():
            raise ValueError("Illegal action; affected states were not changed")
        return result

    def reset_done(self):
        self.backend.call("kk_reset", self.count, self.state, self.count)
