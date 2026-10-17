"""Training roots and public continuation against a fixed WP/Point mixture.

Learner seats and opponent identities are experiment metadata, never scorer
inputs. The unchanged two-expert bank supplies the learner's candidate actions.
"""

import torch

from koikoi.layout import EPISODE, PLAYER, SEED
from scripts.paired_rollout_targets import PublicContinuation


def source_roles(env):
    """Cross both seats and both opponents over each four-match lane cycle."""
    lane = torch.arange(env.count, device=env.device)
    episode = env.state[:, EPISODE].long()
    return (lane + episode) % 2, (lane // 2 + episode // 2) % 2


class OpponentContinuation:
    """Frozen parent on the learner seat; specified public opponent otherwise."""

    def __init__(self, parent, wp, point, learner_seats, opponent_ids):
        if (
            learner_seats.ndim != 1
            or opponent_ids.shape != learner_seats.shape
            or opponent_ids.device != learner_seats.device
        ):
            raise ValueError("Expected aligned per-branch role vectors")
        if not (
            ((learner_seats == 0) | (learner_seats == 1)).all()
            and ((opponent_ids == 0) | (opponent_ids == 1)).all()
        ):
            raise ValueError("Roles must be zero or one")
        self.policies = [PublicContinuation(actor) for actor in (parent, wp, point)]
        self.learner_seats = learner_seats.clone()
        self.opponent_ids = opponent_ids.clone()

    @torch.no_grad()
    def __call__(self, env):
        if (
            len(self.learner_seats) != env.count
            or self.learner_seats.device != env.state.device
        ):
            raise ValueError("Continuation roles do not match environment")
        proposals = torch.stack([policy(env) for policy in self.policies], dim=1)
        indices = torch.where(
            env.state[:, PLAYER] == self.learner_seats, 0, self.opponent_ids + 1
        )
        return proposals.gather(1, indices[:, None]).squeeze(1)


@torch.no_grad()
def collect_roots(env, bank, wp, count, generator, max_steps=10000):
    """Collect only learner disagreements, before either branch outcome exists.

    Source learner chooses the parent90% and Point10% at disagreements. Each
    match's other seat follows its assigned deterministic WP or Point opponent.
    Returned states are training-only simulator workspaces, not policy inputs.
    """
    if count < 1:
        raise ValueError("Root count must be positive")
    states, public_rows, groups, opponents = [], [], [], []
    total = 0
    work = dict.fromkeys(
        (
            "decisions",
            "learner_decisions",
            "opponent_decisions",
            "completed_matches",
            "round_endings",
        ),
        0,
    )
    wp_policy = PublicContinuation(wp)
    for _ in range(max_steps):
        learner_seats, opponent_ids = source_roles(env)
        learner = env.state[:, PLAYER] == learner_seats
        public, _ = bank.observe(env)
        proposals = public[:, 560:].long()
        different = proposals[:, 0] != proposals[:, 1]
        indices = torch.where(learner & different)[0]
        if len(indices):
            order = torch.randperm(len(indices), device=env.device, generator=generator)
            indices = indices[order[: count - total]]
            states.append(env.state[indices].clone())
            public_rows.append(public[indices].clone())
            groups.append(env.state[indices][:, [SEED, EPISODE]].clone())
            opponents.append(opponent_ids[indices].clone())
            total += len(indices)
        choose_point = (
            torch.rand(env.count, device=env.device, generator=generator) < 0.1
        )
        learner_actions = proposals.gather(1, choose_point.long()[:, None]).squeeze(1)
        opponent_actions = torch.where(
            opponent_ids == 0, wp_policy(env), proposals[:, 1]
        )
        actions = torch.where(learner, learner_actions, opponent_actions)
        result = env.step(actions)
        work["decisions"] += env.count
        work["learner_decisions"] += int(learner.sum())
        work["opponent_decisions"] += int((~learner).sum())
        work["round_endings"] += int(result[:, 1].sum())
        work["completed_matches"] += int(result[:, 2].sum())
        env.reset_done()
        if total == count:
            return (
                torch.cat(states),
                torch.cat(public_rows),
                torch.cat(groups),
                torch.cat(opponents),
                work,
            )
    raise RuntimeError("Insufficient learner disagreement roots")
