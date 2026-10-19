"""Correct a cheap round-boundary proxy with paired terminal residuals.

This is a two-level Monte Carlo estimator, not a new unbiasedness theorem.
Its target is the declared fixed64 rollout execution, not strict CPU policies.
"""

from dataclasses import asdict

import numpy as np
import torch

from koikoi.layout import DEALER, PHASE, POINTS, ROUND, TERMINAL
from scripts.round_transition_values import RoundWorkspace


def corrected_estimate(cheap, high, proxy, alpha=1.0):
    """Independent cheap sample and coupled high/proxy sample; fixed alpha."""
    cheap, high, proxy = [np.asarray(x, dtype=np.float64) for x in (cheap, high, proxy)]
    if (
        cheap.ndim != 1
        or high.ndim != 1
        or proxy.shape != high.shape
        or min(len(cheap), len(high)) < 2
        or not all(np.isfinite(x).all() for x in (cheap, high, proxy))
        or any((np.abs(x) > 1).any() for x in (cheap, high, proxy))
        or not np.isfinite(alpha)
    ):
        raise ValueError("Expected finite bounded action differences and >=2 samples")
    residual = high - alpha * proxy
    variance = alpha**2 * cheap.var(ddof=1) / len(cheap) + residual.var(ddof=1) / len(
        high
    )
    mean = alpha * cheap.mean() + residual.mean()
    return {
        "mean": float(mean),
        "standard_error": float(np.sqrt(variance)),
        "cheap_mean": float(cheap.mean()),
        "residual_mean": float(residual.mean()),
        "high_mean": float(high.mean()),
        "cheap_variance": float(cheap.var(ddof=1)),
        "residual_variance": float(residual.var(ddof=1)),
        "high_variance": float(high.var(ddof=1)),
        "high_proxy_covariance": float(np.cov(high, proxy, ddof=1)[0, 1]),
        "alpha": float(alpha),
    }


def boundary_proxy(endpoints, labels, seat, tables):
    """Fixed A/B average; only public predeal fields and sampled labels are used."""
    end = np.asarray(endpoints)
    if end.shape != (2 * len(labels), 208):
        raise ValueError("Expected paired endpoint states")
    arguments = (
        end[:, ROUND],
        end[:, POINTS],
        end[:, DEALER],
        end[:, PHASE] == TERMINAL,
        seat,
        np.repeat(labels, 2),
    )
    # This is a fixed surrogate. Its fitting error is corrected by X-Y and must
    # not be separately added as shared-table error in this conditional estimator.
    values = (tables.lookup(*arguments, 0)[0] + tables.lookup(*arguments, 1)[0]) / 2
    paired = values.reshape(len(labels), 2)
    return paired, np.diff(paired, axis=1).ravel() / 2


class BoundaryResidualWorkspace:
    """First stop every lane at its round boundary, then continue the same states.

    The explicit barrier makes the cheap prefix identical in both fidelities.
    Actual next-round cards/RNG are preserved inside simulation, never exported
    as policy features or used by the boundary-value lookup.
    """

    def __init__(self, particles, device="cuda"):
        self.particles = particles
        self.rounds = RoundWorkspace(particles * 2, device)

    @torch.no_grad()
    def run(
        self,
        worlds,
        base,
        alternative,
        continuation,
        seat,
        labels,
        tables,
        *,
        terminal=True,
    ):
        if worlds.shape != (self.particles, 208) or len(labels) != self.particles:
            raise ValueError("Particle/workspace mismatch")
        continuation.prepare(seat, labels)
        states = worlds.repeat_interleave(2, dim=0)
        actions = torch.tensor([base, alternative], device=worlds.device).repeat(
            self.particles
        )
        endpoints, round_work = self.rounds.run(states, continuation, actions)
        boundary = endpoints.cpu().numpy().copy()
        values, proxy = boundary_proxy(boundary, labels, seat, tables)
        work = {
            "decisions": round_work.decisions,
            "round_endings": round_work.round_endings,
            "terminal_endings": round_work.terminal_match_endings,
        }
        first_work = asdict(round_work)
        high = points = None
        if terminal:
            env = self.rounds.env
            for _ in range(512):
                active = env.state[:, PHASE] != TERMINAL
                if not active.any():
                    break
                result = env.step(continuation(env), freeze_done=True)
                work["decisions"] += int(active.sum())
                work["round_endings"] += int(result[:, 1].sum())
                work["terminal_endings"] += int(result[:, 2].sum())
            if (env.state[:, PHASE] != TERMINAL).any():
                raise RuntimeError("Terminal continuation exceeded fixed bound")
            assert work["terminal_endings"] == self.particles * 2
            points = (
                ((env.state[:, POINTS] - env.state[:, POINTS + 1]) * (1 - 2 * seat))
                .cpu()
                .numpy()
                .reshape(self.particles, 2)
            )
            high = np.diff(np.sign(points), axis=1).ravel().astype(np.float64) / 2
        return {
            "proxy": proxy.tolist(),
            "high": None if high is None else high.tolist(),
            "paired_boundary_values": values.tolist(),
            "boundary_fields": boundary[:, [ROUND, POINTS, DEALER, PHASE]]
            .reshape(self.particles, 2, 4)
            .tolist(),
            "field_order": ["round", "points0", "dealer", "phase"],
            "paired_points": None if points is None else points.tolist(),
            "first_round_work": first_work,
            "work": work,
        }
