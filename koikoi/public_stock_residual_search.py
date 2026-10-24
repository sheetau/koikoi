"""Rollout search using the frozen-parent public feature encoder."""

import torch

from .layout import PLAYER
from .public_stock_search import sample_batch, view


@torch.no_grad()
def rollout_actions(batch, model, particles, horizon, generator, confidence=1.0):
    """Evaluate every legal root action on common worlds; later actors see only
    their own observations. Horizon is decisions, and leaves use a shared critic.
    No hidden-state critic, perfect-information optimizer, or original score bound.
    """
    if particles < 1 or horizon < 1 or confidence < 0 or (confidence and particles < 2):
        raise ValueError("Positive rollout budgets required")
    encode = getattr(model, "encode", lambda env: env.observe())
    obs, legal = encode(batch)
    logits, _ = model(obs, legal)
    fallback = logits.argmax(1)
    active = (legal.sum(1) > 1).nonzero().flatten()
    if not len(active):
        return fallback, {"rollouts": 0, "transitions": 0, "roots": 0}
    roots = view(batch.state[active], batch)
    _, root_mask = encode(roots)
    root_id, action_id = root_mask.nonzero(as_tuple=True)
    worlds = sample_batch(roots, particles, generator).reshape(
        len(active), particles, -1
    )
    states = worlds[root_id].reshape(-1, batch.state.shape[1])
    sim = view(states, batch)
    root_player = sim.state[:, PLAYER].clone()
    actions = action_id.repeat_interleave(particles)
    result = sim.step(actions)
    transitions = len(actions)
    completed = result[:, 2].bool()
    utility = result[:, 0].float() * (1 - 2 * root_player)
    for _ in range(horizon - 1):
        alive = (~completed).nonzero().flatten()
        if not len(alive):
            break
        current = view(sim.state[alive], sim)
        o, mask = encode(current)
        scores, _ = model(o, mask)
        result = current.step(scores.argmax(1))
        sim.state[alive] = current.state
        transitions += len(alive)
        utility[alive] = result[:, 0].float() * (1 - 2 * root_player[alive])
        completed[alive] = result[:, 2].bool()
    alive = (~completed).nonzero().flatten()
    if len(alive):
        current = view(sim.state[alive], sim)
        o, mask = encode(current)
        _, values = model(o, mask)
        utility[alive] = values.clamp(-1, 1) * torch.where(
            current.state[:, PLAYER] == root_player[alive], 1, -1
        )
    samples = utility.reshape(-1, particles)
    baseline_rows = action_id == fallback[active][root_id]
    baseline = samples[baseline_rows]
    differences = samples - baseline[root_id]
    means = differences.mean(1)
    if confidence:
        means -= confidence * differences.std(1) / particles**0.5
    scores = torch.full(root_mask.shape, -torch.inf, device=batch.device)
    scores[root_id, action_id] = means
    # Exact ties retain the actor's ordering. This avoids gratuitous card-index
    # bias when the critic assigns identical values to several root actions.
    maxima = scores.max(1).values
    tied = scores == maxima[:, None]
    root_logits, _ = model(obs[active], legal[active])
    fallback[active] = root_logits.masked_fill(~tied, -torch.inf).argmax(1)
    return fallback, {
        "rollouts": len(actions),
        "transitions": transitions,
        "roots": len(active),
    }
