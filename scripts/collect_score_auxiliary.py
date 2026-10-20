"""Complete frozen-policy self-play labels for auxiliary representation learning."""

import hashlib
import math

import torch

from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from scripts.score_auxiliary import margin_classes


class IncompleteScoreCollection(RuntimeError):
    def __init__(self, evidence):
        super().__init__("Auxiliary collection did not finish every requested match")
        self.evidence = evidence


def tensor_sha(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def compact_evidence(batch):
    """Keep actions and row identities for replay, without duplicating large inputs."""
    return {key: value for key, value in batch.items() if key != "features"} | {
        "features_sha256": tensor_sha(batch["features"]),
    }


@torch.no_grad()
def collect_outcomes(
    actor, *, count, deal_seed, action_seed, reservoir_seed, max_steps=448
):
    """One complete match per lane; one uniform row per player/phase stratum.

    Action and reservoir generators are local, so this does not advance PPO's
    RNG streams. A match's final public margin labels its retained observations
    in each observer's own perspective. Raw hidden state is never passed to the
    actor or returned as an auxiliary input. Inactive lanes are frozen.
    """
    if (
        type(count) is not int
        or count < 1
        or type(max_steps) is not int
        or max_steps < 1
    ):
        raise ValueError("Expected positive integer count and step bound")
    for seed in (deal_seed, action_seed, reservoir_seed):
        if type(seed) is not int or not 0 <= seed < 2**31 - count:
            raise ValueError("Invalid auxiliary seed")
    device = next(actor.parameters()).device
    if any(module.training for module in actor.modules()):
        raise ValueError("Frozen-policy collection requires disabled dropout")
    env = BatchEnv(count, device, "paper8", deal_seed)
    action_rng = torch.Generator(device=device).manual_seed(action_seed)
    reservoir_rng = torch.Generator(device="cpu").manual_seed(reservoir_seed)
    visits = torch.zeros(count, 2, 4, dtype=torch.long)
    rows = {}
    trace = []
    decisions, rounds, matches = 0, 0, 0
    for step in range(max_steps):
        active = env.state[:, PHASE] != TERMINAL
        ids = torch.where(active)[0]
        if not len(ids):
            break
        features, legal = env.published_observe()
        phases = env.state[:, PHASE].clone()
        seats = env.state[:, PLAYER].clone()
        solved = env.solved_actions()
        legal &= (solved[:, None] < 0) | (
            torch.arange(50, device=device)[None] == solved[:, None]
        )
        scores = actor(features[ids], phases[ids], legal[ids])
        log_probs = (scores / 0.05).log_softmax(-1)
        chosen = torch.multinomial(log_probs.exp(), 1, generator=action_rng).flatten()
        actions = torch.zeros(count, dtype=torch.int32, device=device)
        actions[ids] = chosen.int()
        public_indices = (
            torch.stack((ids, seats[ids], phases[ids]), dim=1).cpu().tolist()
        )
        for position, (lane, seat, phase) in enumerate(public_indices):
            key = (lane, seat, phase)
            visits[key] += 1
            if int(torch.randint(int(visits[key]), (), generator=reservoir_rng)) == 0:
                rows[key] = (
                    features[lane].cpu().clone(),
                    legal[lane].cpu().clone(),
                    log_probs[position].cpu().clone(),
                    step,
                )
        result = env.step(actions, freeze_done=True)
        if (result[ids, 0] == 99).any():
            raise RuntimeError("Illegal auxiliary action")
        decisions += len(ids)
        rounds += int(result[ids, 1].sum())
        matches += int(result[ids, 2].sum())
        trace.append(actions.masked_fill(~active, -1).cpu())
    work = {"decisions": decisions, "rounds": rounds, "matches": matches}
    evidence = {
        "format": "score_auxiliary_collection_v1",
        "count": count,
        "deal_seed": deal_seed,
        "action_seed": action_seed,
        "reservoir_seed": reservoir_seed,
        "max_steps": max_steps,
        "temperature": 0.05,
        "actions": torch.stack(trace),
        "visits": visits,
        "work": work,
    }
    if not (env.state[:, PHASE] == TERMINAL).all() or matches != count:
        raise IncompleteScoreCollection(evidence)
    keys = sorted(rows)
    points = env.state[:, POINTS : POINTS + 2].cpu().long()
    margins = torch.tensor(
        [int(points[lane, seat] - points[lane, 1 - seat]) for lane, seat, _ in keys],
        dtype=torch.float32,
    )
    selected_steps = torch.tensor([rows[key][3] for key in keys], dtype=torch.long)
    weights = [1 / int(visits[key]) for key in keys]
    assert all(math.isfinite(weight) and 0 < weight <= 1 for weight in weights)
    return evidence | {
        "features": torch.stack([rows[key][0] for key in keys]),
        "legal": torch.stack([rows[key][1] for key in keys]),
        "reference_log_probs": torch.stack([rows[key][2] for key in keys]),
        "identities": torch.tensor(keys, dtype=torch.long),
        "phases": torch.tensor([key[2] for key in keys], dtype=torch.long),
        "selected_steps": selected_steps,
        "inclusion_probabilities": torch.tensor(weights, dtype=torch.float64),
        "final_points": points,
        "margins": margins,
        "targets": margin_classes(margins),
    }
