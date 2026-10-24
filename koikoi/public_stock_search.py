"""Information-respecting sampled policy rollout search for public stock rules."""

import torch

from .layout import DEALER, HISTORY, PLAYER, SEED, STOCK
from .public_stock import PUBLIC
from .public_stock_native import PublicStockBatch


def view(states, template):
    batch = object.__new__(PublicStockBatch)
    batch.state = states.contiguous()
    batch.count = len(states)
    batch.device, batch.backend, batch.variant = (
        template.device,
        template.backend,
        template.variant,
    )
    return batch


def sample_batch(batch, particles, generator):
    """Vectorized uniform proposal with initial-hand rejection conditioning."""
    if particles < 1:
        raise ValueError("Positive particle count required")
    obs, _ = batch.observe()
    states = batch.state.repeat_interleave(particles, dim=0)
    obs = obs.repeat_interleave(particles, dim=0)
    n, device = len(states), states.device
    rows = torch.arange(n, device=device)
    cards = torch.arange(48, device=device)
    unknown = obs[:, :528].reshape(n, 48, 11)[:, :, 4].bool()
    hand_n = (obs[:, 541] * 8).round().long()
    played = states[:, HISTORY::5].long() - 1
    turns = torch.arange(16, device=device)
    opponent_turn = (states[:, DEALER, None] + turns) % 2 != states[:, PLAYER, None]
    played_valid = opponent_turn & (played >= 0)
    played_months = torch.zeros((n, 12), dtype=torch.int64, device=device)
    played_months.scatter_add_(1, played.clamp_min(0) // 4, played_valid.long())
    order = torch.zeros((n, 48), dtype=torch.int64, device=device)
    pending = rows
    for _ in range(10000):
        random = torch.rand((len(pending), 48), device=device, generator=generator)
        proposal = random.masked_fill(~unknown[pending], 2).argsort(dim=1)
        order[pending] = proposal
        counts = played_months[pending].clone()
        counts.scatter_add_(1, proposal // 4, (cards < hand_n[pending, None]).long())
        pending = pending[(counts >= 4).any(1)]
        if not len(pending):
            break
    else:
        raise RuntimeError("No deal-consistent hidden hand")
    states[:, :48][unknown] = -1
    hand_valid = cards < hand_n[:, None]
    hand_rows = rows[:, None].expand(-1, 48)[hand_valid]
    states[hand_rows, order[hand_valid]] = 1 - states[hand_rows, PLAYER]
    stock_n = (
        states[:, STOCK] - 23 - (states[:, PUBLIC : PUBLIC + 2] != 0).sum(1)
    ).long()
    positions = torch.arange(24, device=device)
    stock_valid = positions < stock_n[:, None]
    stock_rows = rows[:, None].expand(-1, 24)[stock_valid]
    stock_cards = order.gather(1, (hand_n[:, None] + positions).clamp_max(47))
    states[stock_rows, (72 + positions).expand(n, -1)[stock_valid]] = stock_cards[
        stock_valid
    ].int()
    # Replace original hidden hand storage, too, rather than leaving real-world
    # private identities in unused deck slots of a simulation.
    initial = (
        torch.cat(
            (played.masked_fill(~played_valid, 99), order.masked_fill(~hand_valid, 99)),
            dim=1,
        )
        .sort(dim=1)
        .values[:, :8]
    )
    start = 48 + 8 * (1 - states[:, PLAYER].long())
    states[rows[:, None], start[:, None] + torch.arange(8, device=device)] = (
        initial.int()
    )
    states[:, SEED] = torch.randint(
        0, 2**31 - 1, (n,), device=device, generator=generator, dtype=torch.int32
    )
    return states


@torch.no_grad()
def rollout_actions(batch, model, particles, horizon, generator, confidence=1.0):
    """Evaluate every legal root action on common worlds; later actors see only
    their own observations. Horizon is decisions, and leaves use a shared critic.
    No hidden-state critic, perfect-information optimizer, or original score bound.
    """
    if particles < 1 or horizon < 1 or confidence < 0 or (confidence and particles < 2):
        raise ValueError("Positive rollout budgets required")
    obs, legal = batch.observe()
    logits, _ = model(obs, legal)
    fallback = logits.argmax(1)
    active = (legal.sum(1) > 1).nonzero().flatten()
    if not len(active):
        return fallback, {"rollouts": 0, "transitions": 0, "roots": 0}
    roots = view(batch.state[active], batch)
    _, root_mask = roots.observe()
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
        o, mask = current.observe()
        scores, _ = model(o, mask)
        result = current.step(scores.argmax(1))
        sim.state[alive] = current.state
        transitions += len(alive)
        utility[alive] = result[:, 0].float() * (1 - 2 * root_player[alive])
        completed[alive] = result[:, 2].bool()
    alive = (~completed).nonzero().flatten()
    if len(alive):
        current = view(sim.state[alive], sim)
        o, mask = current.observe()
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
