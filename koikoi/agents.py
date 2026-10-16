"""Observation-only reference opponents and trained-policy adapters."""

import hashlib
import io
from pathlib import Path

import numpy as np
import torch

from .model import PolicyValue
from .reference import LIGHTS, RIBBONS, TANE


class RandomAgent:
    def __init__(self, seed=1):
        self.rng = np.random.default_rng(seed)

    def act(self, env):
        return int(self.rng.choice(np.flatnonzero(env.legal())))


class GreedyAgent:
    def act(self, env):
        # Only the observer's public/own-card encoding is used.
        obs, legal = env.observe(), env.legal()
        cards = obs[:528].reshape(48, 11)
        if legal[48]:
            return 48
        weights = np.ones(48)
        weights[list(RIBBONS)] = 2
        weights[list(TANE)] = 3
        weights[list(LIGHTS)] = 5
        if obs[528] > 0:
            value = np.zeros(48)
            for c in np.flatnonzero(legal[:48]):
                matches = [j for j in range(c // 4 * 4, c // 4 * 4 + 4) if cards[j, 1]]
                value[c] = (
                    weights[c] + sum(weights[matches]) if matches else -weights[c]
                )
        else:
            value = weights
        return int(np.where(legal[:48], value, -np.inf).argmax())


class NeuralAgent:
    def __init__(self, checkpoint, device="cpu", deterministic=False):
        content = Path(checkpoint).read_bytes()
        self.checkpoint_sha256 = hashlib.sha256(content).hexdigest()
        payload = torch.load(
            io.BytesIO(content), map_location="cpu", weights_only=False
        )
        self.training_counters = payload["counters"]
        self.training_provenance = payload["provenance"]
        config = payload["config"]
        self.model = (
            PolicyValue(config["kind"], config["width"], config["layers"])
            .to(device)
            .eval()
        )
        self.model.load_state_dict(payload["model"])
        self.device, self.deterministic = device, deterministic
        self.profile = config["profile"]
        self.belief_trained = config["belief_weight"] > 0

    @torch.no_grad()
    def act(self, env):
        obs = torch.from_numpy(env.observe()).to(self.device).unsqueeze(0)
        mask = torch.from_numpy(env.legal()).to(self.device).unsqueeze(0)
        logits, _, _ = self.model(obs, mask)
        return int(
            logits.argmax(-1)
            if self.deterministic
            else torch.multinomial(logits.softmax(-1), 1)
        )
