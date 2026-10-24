"""Reusable sampled continuation values at public round boundaries.

The cache is specific to a rule and one fixed policy playing
both sides. Values are Monte Carlo estimates, never certified bounds.
"""

import torch

from .layout import DEALER, PHASE, PLAYER, POINTS, ROUND, TERMINAL
from .public_stock_search import sample_batch, view


def boundary_value(states, player, table):
    """Value after round settlement, before conditioning on the fresh deal."""
    terminal = states[:, PHASE] == TERMINAL
    result = torch.sign(states[:, POINTS] - states[:, POINTS + 1]).float() * (
        1 - 2 * player
    )
    live = (~terminal).nonzero().flatten()
    if len(live):
        dealer = states[live, DEALER].long()
        points = states[live, POINTS].long()
        rounds = states[live, ROUND].long()
        result[live] = table[rounds, dealer, points] * (1 - 2 * player[live])
    return result


@torch.no_grad()
def round_rollout_actions(batch, model, particles, horizon, generator, confidence=2.5):
    """Complete the current round and reuse the fixed-policy continuation table.

    ``horizon`` is a hard engineering limit, not a critic cutoff. Failure to
    finish a round raises rather than silently replacing it by a neural value.
    """
    if particles < 2 or horizon < 1 or confidence < 0:
        raise ValueError("At least two particles and a positive step limit required")
    obs, legal = model.encode(batch)
    logits, _ = model(obs, legal)
    baseline = logits.argmax(1)
    answer = baseline.clone()
    active = (legal.sum(1) > 1).nonzero().flatten()
    if not len(active):
        return answer, {"rollouts": 0, "transitions": 0, "roots": 0}
    roots = view(batch.state[active], batch)
    root_mask = legal[active]
    root_id, action_id = root_mask.nonzero(as_tuple=True)
    worlds = sample_batch(roots, particles, generator).reshape(
        len(active), particles, -1
    )
    sim = view(worlds[root_id].reshape(-1, batch.state.shape[1]), batch)
    player = sim.state[:, PLAYER].clone()
    result = sim.step(action_id.repeat_interleave(particles))
    finished = result[:, 1] > 0
    transitions = len(result)
    for _ in range(horizon - 1):
        alive = (~finished).nonzero().flatten()
        if not len(alive):
            break
        current = view(sim.state[alive], sim)
        inputs, mask = model.encode(current)
        scores, _ = model(inputs, mask)
        result = current.step(scores.argmax(1))
        sim.state[alive] = current.state
        finished[alive] = result[:, 1] > 0
        transitions += len(alive)
    if not finished.all():
        raise RuntimeError("Incomplete round; no neural leaf substitution is permitted")
    samples = boundary_value(sim.state, player, model.round_table).reshape(
        -1, particles
    )
    baseline_rows = action_id == baseline[active][root_id]
    delta = samples - samples[baseline_rows][root_id]
    guarded = delta.mean(1) - confidence * delta.std(1) / particles**0.5
    scores = torch.full(root_mask.shape, -torch.inf, device=batch.device)
    scores[root_id, action_id] = guarded
    tied = scores == scores.max(1).values[:, None]
    answer[active] = logits[active].masked_fill(~tied, -torch.inf).argmax(1)
    return answer, {
        "rollouts": len(samples) * particles,
        "transitions": transitions,
        "roots": len(active),
    }
