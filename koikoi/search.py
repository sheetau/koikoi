"""Belief-sampled, information-respecting policy rollouts (not equilibrium search)."""

import numpy as np
import torch

from .env import BatchEnv
from .layout import *
from .model import sample_action


def sample_hidden(reference, rng, logits=None):
    """Cardinality-constrained weighted proposal, conditioned on valid initial hand.

    Learned weights define a Plackett-Luce proposal, not an exact joint posterior.
    Public action likelihoods beyond the network's prediction are not reweighted.
    """
    s = reference.state.copy()
    p, dealer = int(s[PLAYER]), int(s[DEALER])
    obs = reference.observe()[:528].reshape(48, 11)
    unknown = np.flatnonzero(obs[:, 4])
    count = int(round(reference.observe()[528+13]*8))
    played = [int(s[HISTORY+t*5])-1 for t in range(16)
              if (dealer+t) % 2 != p and s[HISTORY+t*5] > 0]
    weights = None
    if logits is not None:
        weights = np.exp(np.clip(np.asarray(logits)[unknown], -12, 12))
        weights /= weights.sum()
    for _ in range(10000):
        hand = rng.choice(unknown, size=count, replace=False, p=weights)
        initial_hand = played + hand.tolist()
        if max(np.bincount(np.asarray(initial_hand, dtype=int)//4, minlength=12)) < 4:
            break
    else:
        raise RuntimeError("Hidden-hand sampler could not satisfy initial deal constraints")
    s[unknown] = -1
    s[hand] = 1-p
    stock = np.array([c for c in unknown if c not in set(hand.tolist())], dtype=np.int32)
    rng.shuffle(stock)
    s[72:49+s[STOCK]] = stock
    # Initial hand order is irrelevant, but initial hidden contents must also be replaced.
    start = 48 + 8*(1-p)
    s[start:start+8] = sorted(initial_hand)
    # Future round deals must not depend on the real world's hidden RNG seed.
    s[SEED] = rng.integers(0, 2**31-1)
    return s


class SearchAgent:
    def __init__(self, neural, particles=32, horizon=32, seed=123, use_belief=True, koi_only=True):
        if particles < 1 or horizon < 1:
            raise ValueError("Search budgets must be positive")
        self.neural, self.particles, self.horizon = neural, particles, horizon
        self.rng = np.random.default_rng(seed)
        self.use_belief, self.koi_only = use_belief, koi_only
        if use_belief and not neural.belief_trained:
            raise ValueError("Checkpoint has no trained belief head; select uniform search")
        self.simulated_decisions = 0
        self.cache = {}

    @torch.no_grad()
    def act(self, reference):
        legal = np.flatnonzero(reference.legal())
        if len(legal) == 1 or (self.koi_only and reference.state[PHASE] != KOI):
            return self.neural.act(reference)
        device, model = self.neural.device, self.neural.model
        belief = None
        if self.use_belief:
            obs = torch.from_numpy(reference.observe()).to(device)[None]
            mask = torch.from_numpy(reference.legal()).to(device)[None]
            belief = model(obs, mask)[2][0].cpu().numpy()
        worlds = np.stack([sample_hidden(reference, self.rng, belief) for _ in range(self.particles)])
        n = len(legal)*self.particles
        profile = "paper8" if reference.state[BANKRUPT] else "research12"
        key = (n, profile)
        if key not in self.cache:
            self.cache[key] = BatchEnv(n, device, profile)
        env = self.cache[key]
        env.state.copy_(torch.from_numpy(np.tile(worlds, (len(legal), 1))).to(device))
        actions = torch.from_numpy(np.repeat(legal, self.particles)).to(device)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        values = torch.zeros(n, device=device)
        root_sign = 1-2*int(reference.state[PLAYER])
        for depth in range(self.horizon):
            if depth:
                obs, mask = env.observe()
                actions, _ = sample_action(model(obs, mask)[0])
            result = env.step(actions)
            ended = result[:, 2].bool()
            values = torch.where(alive & ended, result[:, 0].float()*root_sign, values)
            alive &= ~ended
            env.reset_done()
        obs, mask = env.observe()
        estimate = model(obs, mask)[1] * (1-2*env.state[:, PLAYER]) * root_sign
        values = torch.where(alive, estimate, values)
        self.simulated_decisions += n*self.horizon
        mean = values.reshape(len(legal), self.particles).mean(1)
        return int(legal[int(mean.argmax())])
