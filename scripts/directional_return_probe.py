"""Bounded public-policy score probes and complete-match return contributions.

No policy fitting or simulation. Probabilities are those of a frozen policy;
the exponential-tilt features stay fixed when differentiating a probe.
"""

import numpy as np

PROBES = (
    "koi_continue",
    "discard_preference",
    "pick_hand_preference",
    "pick_draw_preference",
    "koi_preference",
)


def directional_scores(weights, actions, phases, legal, valid):
    """d log pi_theta(a|o)/d theta at zero for five fixed exponential tilts."""
    weights = np.asarray(weights, dtype=np.float64)
    actions, phases = np.asarray(actions), np.asarray(phases)
    legal, valid = np.asarray(legal, dtype=bool), np.asarray(valid, dtype=bool)
    if weights.ndim < 2 or weights.shape[-1] != 50:
        raise ValueError("Expected probabilities with a 50-action final axis")
    shape = weights.shape[:-1]
    if legal.shape != weights.shape or any(
        a.shape != shape for a in (actions, phases, valid)
    ):
        raise ValueError("Mismatched public input shapes")
    if actions.dtype.kind not in "iu" or phases.dtype.kind not in "iu":
        raise ValueError("Integer actions and phases required")
    active = weights[valid]
    if not np.isfinite(active).all() or (active < 0).any():
        raise ValueError("Invalid probability weights")
    if (
        (active[~legal[valid]] != 0).any()
        or (active.sum(-1) <= 0).any()
        or not np.isfinite(active.sum(-1)).all()
    ):
        raise ValueError("Weights must have positive mass on legal actions only")
    if ((phases[valid] < 0) | (phases[valid] > 3)).any():
        raise ValueError("Invalid active phase")
    selected = actions[valid]
    if ((selected < 0) | (selected >= 50)).any():
        raise ValueError("Invalid active action")
    if len(active) and not (active[np.arange(len(active)), selected] > 0).all():
        raise ValueError("Selected action has zero probability")
    # multinomial samples relative weights. Renormalize recorded float32 weights
    # in float64 so the analytical probe has an exactly stated base measure.
    probability = np.zeros_like(weights)
    probability[valid] = active / active.sum(-1, keepdims=True)
    chosen = np.zeros(shape)
    chosen[valid] = probability[valid][np.arange(len(active)), selected]
    result = np.zeros((*shape, len(PROBES)), dtype=np.float64)
    koi = valid & (phases == 3)
    result[..., 0] = np.where(koi, (actions == 49) - probability[..., 49], 0)
    centered_preference = chosen - np.square(probability).sum(-1)
    for phase in range(4):
        result[..., phase + 1] = np.where(
            valid & (phases == phase), centered_preference, 0
        )
    return result


def estimate_advantages(reward0, value, player, valid, done, offsets, horizon=128):
    """Terminal-match MC and signed traces in each current actor's payoff units.

    Nonterminal cutoffs reset eligibility, but retain the next value bootstrap.
    A column holds exactly one full match; padding never contributes.
    """
    reward0, value = np.asarray(reward0, dtype=float), np.asarray(value, dtype=float)
    player = np.asarray(player)
    valid, done = np.asarray(valid, dtype=bool), np.asarray(done, dtype=bool)
    offsets = np.asarray(offsets)
    if value.ndim != 2 or min(value.shape) < 1:
        raise ValueError("Expected nonempty time-by-match matrices")
    t_count, count = value.shape
    if any(x.shape != value.shape for x in (reward0, player, valid, done)):
        raise ValueError("Mismatched trajectory shapes")
    if type(horizon) is not int or horizon < 1 or offsets.shape != (count,):
        raise ValueError("Invalid cutoff schedule")
    if offsets.dtype.kind not in "iu" or ((offsets < 0) | (offsets >= horizon)).any():
        raise ValueError("Integer cutoff offsets must be in [0,horizon)")
    lengths = valid.sum(0)
    time = np.arange(t_count)[:, None]
    if (lengths < 1).any() or not np.array_equal(valid, time < lengths):
        raise ValueError("Each column must have a nonempty contiguous valid prefix")
    terminal = time == lengths - 1
    if not np.array_equal(done & valid, terminal):
        raise ValueError("Each match must terminate exactly at its final valid row")
    if not np.isfinite(value).all() or not np.isfinite(reward0).all():
        raise ValueError("Nonfinite value or reward")
    if player.dtype.kind not in "iu" or not np.isin(player, (0, 1)).all():
        raise ValueError("Invalid player")
    player = player.astype(np.int64)
    if (reward0[~terminal] != 0).any() or not np.isin(
        reward0[terminal], (-1, 0, 1)
    ).all():
        raise ValueError("Expected terminal-only signed win rewards")
    sign = 1 - 2 * player
    v0 = np.where(valid, value * sign, 0)
    outcome0 = reward0.sum(0)
    mc = np.where(valid, outcome0[None] * sign - value, 0)
    result = {"mc": mc}
    for name, lam, use_cutoff in (
        ("gae95_full", 0.95, False),
        ("gae95_window", 0.95, True),
        ("gae1_window", 1.0, True),
        ("gae1_full", 1.0, False),
    ):
        estimate = np.zeros_like(value)
        carry = np.zeros(count)
        for t in reversed(range(t_count)):
            next_value = v0[t + 1] if t + 1 < t_count else np.zeros(count)
            alive = valid[t] & ~done[t]
            delta = reward0[t] + np.where(alive, next_value, 0) - v0[t]
            cut = ((t + offsets) % horizon == horizon - 1) if use_cutoff else False
            carry = np.where(
                valid[t], delta + lam * np.where(alive & ~np.asarray(cut), carry, 0), 0
            )
            estimate[t] = carry * sign[t]
        result[name] = estimate
    if not np.allclose(result["gae1_full"], mc, rtol=0, atol=1e-12):
        raise AssertionError("Complete lambda-one traces must equal Monte Carlo")
    return result


def seat_contributions(scores, advantage, player, valid):
    """Sum decisions within each match/seat; never count decisions as replicates."""
    scores, advantage = (
        np.asarray(scores, dtype=float),
        np.asarray(advantage, dtype=float),
    )
    player, valid = np.asarray(player), np.asarray(valid, dtype=bool)
    if scores.ndim != 3 or scores.shape[:2] != advantage.shape:
        raise ValueError("Expected time,match,probe scores and time,match advantages")
    if player.shape != advantage.shape or valid.shape != advantage.shape:
        raise ValueError("Mismatched trajectory shapes")
    if player.dtype.kind not in "iu" or not np.isin(player[valid], (0, 1)).all():
        raise ValueError("Invalid active player")
    if not np.isfinite(scores).all() or not np.isfinite(advantage).all():
        raise ValueError("Nonfinite contribution input")
    return np.stack(
        [
            np.where(
                (valid & (player == seat))[..., None], scores * advantage[..., None], 0
            ).sum(0)
            for seat in (0, 1)
        ],
        axis=1,
    )
