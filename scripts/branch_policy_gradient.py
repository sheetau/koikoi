"""Training-only complete-match action values for a public stochastic actor.

Raw states instantiate simulator branches, never actor inputs. Each source match
contributes one uniformly reservoir-sampled unforced decision, with its number
of eligible decisions as weight. This estimates a sum over decisions rather
than giving short matches disproportionate weight. Both seats use the current
actor, held fixed during collection; returns belong to the root player.
"""

from dataclasses import asdict, dataclass

import torch

from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER, POINTS, STATE_SIZE, TERMINAL


@dataclass
class Work:
    decisions: int = 0
    round_endings: int = 0
    completions: int = 0

    def add(self, active, result):
        self.decisions += int(active.sum())
        self.round_endings += int(result[:, 1].sum())
        self.completions += int(result[:, 2].sum())

    def dictionary(self):
        return asdict(self)


def public_inputs(env):
    features, legal = env.published_observe()
    solved = env.solved_actions()
    legal &= (solved[:, None] < 0) | (
        torch.arange(50, device=env.device)[None] == solved[:, None]
    )
    return features, env.state[:, PHASE].clone(), legal


def probabilities(actor, features, phases, legal, temperature=0.05):
    if temperature <= 0 or not torch.isfinite(torch.tensor(temperature)):
        raise ValueError("Temperature must be finite and positive")
    if not legal.any(-1).all():
        raise ValueError("Policy probabilities require active rows")
    return (actor(features, phases, legal) / temperature).softmax(-1)


def inverse_cdf(probs, uniforms):
    """Stable inverse CDF, including zero-mass leading/trailing actions."""
    if probs.ndim != 2 or uniforms.shape != (len(probs),):
        raise ValueError("Expected one uniform per probability row")
    if (
        not torch.isfinite(probs).all()
        or (probs < 0).any()
        or (probs.sum(-1) <= 0).any()
        or not ((uniforms >= 0) & (uniforms < 1)).all()
    ):
        raise ValueError("Invalid categorical probabilities or uniforms")
    cdf = probs.double().cumsum(-1)
    cdf /= cdf[:, -1:].clone()
    return (uniforms.double()[:, None] >= cdf).sum(-1)


@torch.no_grad()
def public_actions(env, actor, uniforms, temperature=0.05):
    features, phases, legal = public_inputs(env)
    active = phases != TERMINAL
    actions = torch.zeros(env.count, dtype=torch.long, device=env.device)
    if active.any():
        probs = probabilities(
            actor, features[active], phases[active], legal[active], temperature
        )
        actions[active] = inverse_cdf(probs, uniforms[active])
    return actions, legal


@torch.no_grad()
def collect_roots(
    env, actor, action_rng, reservoir_rng, temperature=0.05, max_steps=512
):
    """Finish one match per lane; store an unbiased sample of its decision sum.

    The independent reservoir stream must not affect behavior sampling. Forced
    decisions have identically zero masked-policy gradient and are omitted.
    Zero-eligible matches retain zero weight and an initial-state placeholder.
    Full action transcripts permit independent transition/work verification.
    """
    env.reset_done()
    initial = env.state.clone()
    roots = initial.clone()
    counts = torch.zeros(env.count, dtype=torch.long, device=env.device)
    selected_step = torch.full_like(counts, -1)
    work, transcript = Work(), []
    for step in range(max_steps):
        active = env.state[:, PHASE] != TERMINAL
        if not active.any():
            break
        uniforms = torch.rand(env.count, device=env.device, generator=action_rng)
        actions, legal = public_actions(env, actor, uniforms, temperature)
        eligible = active & (legal.sum(-1) > 1)
        counts += eligible
        choose = eligible & (
            torch.rand(env.count, device=env.device, generator=reservoir_rng)
            < 1 / counts.clamp_min(1).double()
        )
        roots[choose] = env.state[choose]
        selected_step[choose] = step
        transcript.append(actions.to(torch.int16).cpu())
        result = env.step(actions, freeze_done=True)
        work.add(active, result)
    if (env.state[:, PHASE] != TERMINAL).any():
        raise RuntimeError("Source matches exceeded the completion bound")
    if work.completions != env.count:
        raise RuntimeError("Source completion accounting mismatch")
    return {
        "states": roots,
        "eligible_decisions": counts,
        "selected_step": selected_step,
        "initial": initial,
        "final": env.state.clone(),
        "actions": torch.stack(transcript),
        "work": work.dictionary(),
    }


class BranchWorkspace:
    """Reuse one native backend; execute only actual legal-action branches."""

    def __init__(self, capacity, device="cuda"):
        self.env = BatchEnv(capacity, device, "paper8", debug=True)
        self.storage = self.env.state
        self.capacity = capacity

    def _load(self, states):
        if (
            states.ndim != 2
            or states.shape[1] != STATE_SIZE
            or states.dtype != torch.int32
            or states.device != self.storage.device
            or not 0 < len(states) <= self.capacity
            or (states[:, PHASE] == TERMINAL).any()
        ):
            raise ValueError("Invalid active roots or workspace capacity")
        self.env.count = len(states)
        self.env.state = self.storage[: len(states)]
        self.env.state.copy_(states)

    @torch.no_grad()
    def root_inputs(self, states):
        self._load(states)
        return public_inputs(self.env)

    @torch.no_grad()
    def run(
        self,
        states,
        actor,
        generator,
        temperature=0.05,
        shared_actions=True,
        max_steps=512,
    ):
        """Enumerate all legal first actions, then sample public continuations.

        All branches share their actual root world and round-indexed deals.
        shared_actions additionally couples continuation inverse-CDF uniforms
        by root and relative simulator step; its control uses independent
        uniforms. This control does NOT independently resample hidden worlds.
        """
        _, _, legal = self.root_inputs(states)
        root, action = torch.where(legal)
        initial = states[root].clone()
        self._load(initial)
        env = self.env
        root_player = initial[:, PLAYER].clone()
        work, transcript = Work(), []
        for step in range(max_steps):
            active = env.state[:, PHASE] != TERMINAL
            if not active.any():
                break
            if step:
                count = len(states) if shared_actions else len(root)
                uniforms = torch.rand(count, device=env.device, generator=generator)
                if shared_actions:
                    uniforms = uniforms[root]
                action, _ = public_actions(env, actor, uniforms, temperature)
            transcript.append(action.to(torch.int16).cpu())
            result = env.step(action, freeze_done=True)
            work.add(active, result)
        if (env.state[:, PHASE] != TERMINAL).any():
            raise RuntimeError("Branches exceeded the completion bound")
        if work.completions != len(root):
            raise RuntimeError("Branch completion accounting mismatch")
        points = (env.state[:, POINTS] - env.state[:, POINTS + 1]) * (
            1 - 2 * root_player
        )
        q = torch.zeros(legal.shape, device=env.device)
        root, first_action = torch.where(legal)
        q[root, first_action] = points.sign().float()
        return {
            "q": q,
            "legal": legal,
            "root_indices": root,
            "first_actions": first_action,
            "points": points,
            "final": env.state.clone(),
            "actions": torch.stack(transcript),
            "work": work.dictionary(),
        }


def expected_return_loss(probs, q, counts):
    """Single on-policy expected-action gradient; no learned critic or clipping.

    q is sampled with a frozen continuation. Centering is a detached
    action-independent baseline. Match counts, not their sum, normalize loss.
    Reusing q after actor updates changes this into a fixed-root surrogate.
    """
    if probs.shape != q.shape or counts.shape != (len(probs),):
        raise ValueError("Incompatible gradient inputs")
    if not torch.isfinite(q).all() or (counts < 0).any():
        raise ValueError("Invalid outcome values or decision weights")
    target = q.detach()
    baseline = (probs.detach() * target).sum(-1, keepdim=True)
    return -((probs * (target - baseline)).sum(-1) * counts.detach()).mean()
