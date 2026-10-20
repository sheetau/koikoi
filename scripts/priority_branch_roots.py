"""Public-policy weighted reservoirs for complete-match branch gradients."""

import torch

from koikoi.layout import PHASE, TERMINAL
from scripts.branch_policy_gradient import (
    Work,
    inverse_cdf,
    probabilities,
    public_inputs,
)


def policy_priority(probs):
    """A fixed public proxy, not an outcome estimate or optimality claim."""
    return 0.05 + (1 - probs.double().square().sum(-1)).clamp_min(0).sqrt()


def reservoir_step(total, weight, uniforms):
    """One weighted reservoir insertion per lane, retaining zero-weight lanes."""
    if (
        total.shape != weight.shape
        or uniforms.shape != weight.shape
        or not torch.isfinite(weight).all()
        or not torch.isfinite(total).all()
        or (weight < 0).any()
        or (total < 0).any()
        or not ((uniforms >= 0) & (uniforms < 1)).all()
    ):
        raise ValueError("Invalid weighted reservoir inputs")
    updated = total + weight
    choose = (weight > 0) & (uniforms < weight / updated.clamp_min(1e-30))
    return updated, choose


@torch.no_grad()
def collect_roots(
    env,
    actor,
    action_rng,
    reservoir_rng,
    sampler="sensitivity",
    temperature=0.05,
    max_steps=512,
):
    if sampler not in ("uniform", "sensitivity"):
        raise ValueError("Unknown root sampler")
    env.reset_done()
    initial = env.state.clone()
    roots = initial.clone()
    counts = torch.zeros(env.count, dtype=torch.long, device=env.device)
    selected_step = torch.full_like(counts, -1)
    total = torch.zeros(env.count, dtype=torch.float64, device=env.device)
    chosen_weight = torch.zeros_like(total)
    work = Work()
    transcript, priorities, uniforms_kept = [], [], []
    for step in range(max_steps):
        active = env.state[:, PHASE] != TERMINAL
        if not active.any():
            break
        uniforms = torch.rand(env.count, device=env.device, generator=action_rng)
        features, phases, legal = public_inputs(env)
        p = probabilities(
            actor, features[active], phases[active], legal[active], temperature
        )
        actions = torch.zeros(env.count, dtype=torch.long, device=env.device)
        actions[active] = inverse_cdf(p, uniforms[active])
        eligible = active & (legal.sum(-1) > 1)
        counts += eligible
        weights = torch.zeros_like(total)
        weights[active] = policy_priority(p) if sampler == "sensitivity" else 1
        weights *= eligible
        selection_uniforms = torch.rand(
            env.count, device=env.device, generator=reservoir_rng
        )
        total, choose = reservoir_step(total, weights, selection_uniforms)
        roots[choose] = env.state[choose]
        chosen_weight[choose] = weights[choose]
        selected_step[choose] = step
        transcript.append(actions.to(torch.int16).cpu())
        priorities.append(weights.cpu())
        uniforms_kept.append(selection_uniforms.cpu())
        result = env.step(actions, freeze_done=True)
        work.add(active, result)
    if (env.state[:, PHASE] != TERMINAL).any() or work.completions != env.count:
        raise RuntimeError("Source matches exceeded the completion bound")
    importance = torch.where(
        counts > 0, total / chosen_weight.clamp_min(1e-30), torch.zeros_like(total)
    )
    return {
        "states": roots,
        "eligible_decisions": counts,
        "selected_step": selected_step,
        "initial": initial,
        "final": env.state.clone(),
        "actions": torch.stack(transcript),
        "work": work.dictionary(),
        "sampler": sampler,
        "priority_total": total,
        "selected_priority": chosen_weight,
        "importance": importance,
        "priorities": torch.stack(priorities),
        "selection_uniforms": torch.stack(uniforms_kept),
    }
