"""Belief-sampled, information-respecting policy rollouts (not equilibrium search)."""

import numpy as np
import torch

from .agents import solved_koi_action
from .capture import CapturedRollout
from .env import BatchEnv
from .layout import BANKRUPT, DEALER, HISTORY, KOI, PHASE, PLAYER, SEED, STOCK
from .model import sample_action


def sample_hidden(reference, rng, logits=None):
    """Draw one hidden world using the same proposal as batched search."""
    return sample_hidden_batch(reference, rng, 1, logits)[0]


def sample_hidden_batch(reference, rng, particles, logits=None):
    """Share public root preprocessing, preserving sequential proposal RNG draws.

    Learned weights define a Plackett-Luce proposal, not an exact joint posterior.
    Public action likelihoods beyond the network's prediction are not reweighted.
    """
    if particles < 1:
        raise ValueError("At least one hidden-world particle is required")
    root = reference.state.copy()
    p, dealer = int(root[PLAYER]), int(root[DEALER])
    observation = reference.observe()
    unknown = np.flatnonzero(observation[:528].reshape(48, 11)[:, 4])
    count = round(observation[528 + 13] * 8)
    played = [
        int(root[HISTORY + t * 5]) - 1
        for t in range(16)
        if (dealer + t) % 2 != p and root[HISTORY + t * 5] > 0
    ]
    weights = None
    if logits is not None:
        weights = np.exp(np.clip(np.asarray(logits)[unknown], -12, 12))
        weights /= weights.sum()
    worlds = np.repeat(root[None], particles, axis=0)
    for s in worlds:
        for _ in range(10000):
            hand = rng.choice(unknown, size=count, replace=False, p=weights)
            initial_hand = played + hand.tolist()
            months = np.bincount(np.asarray(initial_hand, dtype=int) // 4, minlength=12)
            if max(months) < 4:
                break
        else:
            raise RuntimeError(
                "Hidden-hand sampler could not satisfy initial deal constraints"
            )
        s[unknown] = -1
        s[hand] = 1 - p
        chosen = set(hand.tolist())
        stock = np.array([c for c in unknown if c not in chosen], dtype=np.int32)
        rng.shuffle(stock)
        s[72 : 49 + s[STOCK]] = stock
        # Replace the opponent's initial hidden contents as well as its current hand.
        start = 48 + 8 * (1 - p)
        s[start : start + 8] = sorted(initial_hand)
        # Future deals must not depend on the real world's hidden RNG seed.
        s[SEED] = rng.integers(0, 2**31 - 1)
    return worlds


class SearchAgent:
    def __init__(
        self,
        neural,
        particles=32,
        horizon=32,
        seed=123,
        use_belief=True,
        koi_only=True,
        confidence=2.5,
        endgame=False,
        cuda_graph=False,
    ):
        if particles < 1 or horizon < 1:
            raise ValueError("Search budgets must be positive")
        self.neural, self.particles, self.horizon = neural, particles, horizon
        if cuda_graph and torch.device(neural.device).type != "cuda":
            raise ValueError("Captured search requires a CUDA inference device")
        self.cuda_graph = cuda_graph
        self.graph_cache = {}
        self.rng = np.random.default_rng(seed)
        self.use_belief, self.koi_only = use_belief, koi_only
        if confidence < 0 or (confidence > 0 and particles < 2):
            raise ValueError(
                "Confidence guard needs at least two particles and a nonnegative multiplier"
            )
        self.confidence = confidence
        if use_belief and not neural.belief_trained:
            raise ValueError(
                "Checkpoint has no trained belief head; select uniform search"
            )
        self.simulated_decisions = 0
        self.preparation_decisions = 0
        self.searched_decisions = 0
        self.changed_decisions = 0
        self.cache = {}
        self.requires_endgame = endgame or getattr(neural, "requires_endgame", False)
        self.solved_decisions = 0

    @torch.no_grad()
    def act(self, reference):
        if self.requires_endgame:
            action = solved_koi_action(reference)
            if action is not None:
                self.solved_decisions += 1
                return action
        legal = np.flatnonzero(reference.legal())
        if len(legal) == 1 or (self.koi_only and reference.state[PHASE] != KOI):
            return self.neural.act(reference)
        device, model = self.neural.device, self.neural.model
        obs = torch.from_numpy(reference.observe()).to(device)[None]
        mask = torch.from_numpy(reference.legal()).to(device)[None]
        root_logits, _, root_belief = model(obs, mask)
        deterministic = getattr(self.neural, "deterministic", True)
        base_action = (
            int(root_logits.argmax(-1))
            if deterministic
            else int(sample_action(root_logits)[0])
        )
        base_index = int(np.flatnonzero(legal == base_action)[0])
        belief = None
        if self.use_belief:
            belief = root_belief[0].cpu().numpy()
        worlds = sample_hidden_batch(reference, self.rng, self.particles, belief)
        n = len(legal) * self.particles
        profile = "paper8" if reference.state[BANKRUPT] else "research12"
        key = (n, profile)
        if key not in self.cache:
            self.cache[key] = BatchEnv(n, device, profile)
        env = self.cache[key]
        env.state.copy_(torch.from_numpy(np.tile(worlds, (len(legal), 1))).to(device))
        actions = torch.from_numpy(np.repeat(legal, self.particles)).to(device)
        root_sign = 1 - 2 * int(reference.state[PLAYER])
        if self.cuda_graph:
            if key not in self.graph_cache:
                static_actions = actions.clone()
                static_sign = torch.tensor(
                    root_sign, dtype=torch.float32, device=device
                )
                graph = CapturedRollout(
                    lambda: (
                        {"values": self.rollout(env, static_actions, static_sign)},
                        None,
                    ),
                    env.state,
                )
                # CapturedRollout executes three warmups before recording.
                # Their simulator/RNG effects are rolled back, but count the work.
                self.preparation_decisions += 3 * n * self.horizon
                self.graph_cache[key] = graph, static_actions, static_sign
            graph, static_actions, static_sign = self.graph_cache[key]
            static_actions.copy_(actions)
            static_sign.fill_(root_sign)
            values = graph()[0]["values"]
        else:
            values = self.rollout(env, actions, root_sign)
        self.simulated_decisions += n * self.horizon
        values = values.reshape(len(legal), self.particles)
        differences = values - values[base_index : base_index + 1]
        lower = differences.mean(1)
        if self.confidence:
            lower -= self.confidence * differences.std(dim=1) / self.particles**0.5
        index = int(lower.argmax())
        if float(lower[index]) <= 0:
            index = base_index
        self.searched_decisions += 1
        self.changed_decisions += index != base_index
        return int(legal[index])

    @torch.no_grad()
    def rollout(self, env, actions, root_sign):
        """Tensor-only policy evaluation; root actions/sign can be graph inputs."""
        device, model = self.neural.device, self.neural.model
        deterministic = getattr(self.neural, "deterministic", True)
        n = len(env.state)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        values = torch.zeros(n, device=device)
        for depth in range(self.horizon):
            if depth:
                obs, mask = env.observe()
                logits = model(obs, mask)[0]
                if deterministic:
                    actions = logits.argmax(-1)
                else:
                    # Common policy randomness reduces candidate-comparison noise.
                    uniforms = torch.rand(self.particles, device=device).repeat(
                        n // self.particles
                    )
                    cdf = logits.softmax(-1).cumsum(-1)
                    cdf = cdf / cdf[:, -1:]
                    actions = (cdf <= uniforms[:, None]).sum(-1)
            if self.requires_endgame:
                solved = env.solved_actions()
                actions = torch.where(solved >= 0, solved, actions)
            result = env.step(actions)
            ended = result[:, 2].bool()
            values = torch.where(
                alive & ended, result[:, 0].float() * root_sign, values
            )
            alive &= ~ended
            env.reset_done()
        obs, mask = env.observe()
        estimate = model(obs, mask)[1] * (1 - 2 * env.state[:, PLAYER]) * root_sign
        return torch.where(alive, estimate, values)
