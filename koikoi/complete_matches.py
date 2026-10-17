"""Eager full-match PPO collection with explicit terminal padding removal."""

import torch

from .estimators import native_gae
from .layout import DEALER, PHASE, PLAYER, POINTS, ROUND, TURN
from .model import sample_action


@torch.no_grad()
def collect_complete_matches(trainer):
    """Return active rows, round/match totals and executed padded row count.

    Each environment contributes exactly one complete match. ``horizon`` is
    a hard safety bound, not a bootstrap boundary. On failure, leave state
    intact for diagnosis; callers must restore their last good checkpoint.
    """
    env, config = trainer.env, trainer.config
    if not (
        (env.state[:, PHASE] == 0)
        & (env.state[:, TURN] == 0)
        & (env.state[:, ROUND] == 0)
    ).all():
        raise ValueError("Complete-match collection must start at fresh match states")
    buffers = {
        key: []
        for key in (
            "obs",
            "mask",
            "target",
            "action",
            "old_logp",
            "value",
            "reward",
            "done",
            "player",
        )
    }
    active = []
    if trainer.round_continuation is not None:
        buffers.update(learning_reward=[], learning_done=[])
    totals = torch.zeros(2, dtype=torch.int64, device=env.device)
    for _ in range(config.horizon):
        valid = env.state[:, PHASE] != 4
        if not valid.any():
            break
        obs, mask, target = trainer.observe(labels=True)
        logits, value, _ = trainer.actor(obs, mask)
        action, logp = sample_action(logits)
        player = env.state[:, PLAYER].clone()
        result = env.step(action, freeze_done=True)
        active.append(valid)
        row = {
            "obs": obs,
            "mask": mask,
            "target": target,
            "action": action,
            "old_logp": logp,
            "value": value.masked_fill(~valid, 0),
            "reward": result[:, 0].float() * (1 - 2 * player),
            "done": result[:, 2].bool() | ~valid,
            "player": player,
        }
        if trainer.round_continuation is not None:
            continuation = trainer.round_continuation.value(
                env.state[:, ROUND], env.state[:, POINTS], env.state[:, DEALER]
            )
            # Freeze the target at this round's end. True terminal outcomes take
            # precedence; do not count a learning boundary as a completed match.
            boundary_reward = torch.where(
                result[:, 2].bool(), result[:, 0].float(), continuation
            )
            row["learning_reward"] = torch.where(
                result[:, 1].bool(), boundary_reward * (1 - 2 * player), 0
            )
            row["learning_done"] = result[:, 1].bool() | ~valid
        for key, tensor in row.items():
            buffers[key].append(tensor)
        totals += result[:, 1:3].sum(0)
    if not (env.state[:, PHASE] == 4).all():
        raise RuntimeError("Complete-match horizon exhausted before all matches ended")
    batch = {key: torch.stack(value) for key, value in buffers.items()}
    arguments = (
        batch.get("learning_reward", batch["reward"]),
        batch["value"],
        batch.get("learning_done", batch["done"]),
        batch["player"],
        torch.zeros(env.count, device=env.device),
        env.state[:, PLAYER],
        config.gae_lambda,
    )
    if config.returns_backend == "native":
        advantage, returns = native_gae(env.backend, *arguments)
    else:
        from .train import signed_gae

        advantage, returns = signed_gae(*arguments)
    batch["advantage"], batch["return"] = advantage, returns
    valid = torch.stack(active)
    batch = {key: value[valid] for key, value in batch.items()}
    executed_rows = valid.numel()
    env.reset_done()
    return batch, totals, executed_rows
