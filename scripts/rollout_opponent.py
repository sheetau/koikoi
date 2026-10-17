"""Four fixed-action rollout arms, promoted from the audited diagnostic.

Particle ordering and dispatch preserve the archived routine. This helper is for
controlled value diagnostics, not a deployment policy.
"""

import numpy as np
import torch

from koikoi.agents import EndgameAgent
from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.search import SearchAgent, sample_hidden_batch
from koikoi.upstream import reconstruct
from scripts.diagnose_history_support import reference


@torch.no_grad()
def rollout(neural, published, row, worlds, verify=False):
    """Four paired arms: self base/chosen, published-opponent base/chosen."""
    ident = row["identity"]
    seat, base, chosen = ident["seat"], ident["base_action"], ident["selected_action"]
    particles = len(worlds)
    n = 4 * particles
    env = BatchEnv(n, "cuda", "paper8", debug=True)
    initial = np.tile(worlds, (4, 1))
    env.state.copy_(torch.from_numpy(initial).cuda())
    root_actions = torch.tensor(
        np.repeat([base, chosen, base, chosen], particles), device="cuda"
    )
    alive = torch.ones(n, dtype=torch.bool, device="cuda")
    values = torch.zeros(n, device="cuda")
    mixed = torch.arange(n, device="cuda") >= 2 * particles
    sign = 1 - 2 * seat
    counts = {
        "executed_transitions": 0,
        "active_transitions": 0,
        "published_calls": 0,
        "compared_opponent_actions": 0,
        "opponent_action_disagreements": 0,
        "feature_invariance_checks": 0,
        "serial_checks": 0,
        "phases": {},
    }
    # Verify the first world of every arm against the independent reference engine.
    serial = (
        {arm * particles: reference(initial[arm * particles]) for arm in range(4)}
        if verify
        else {}
    )
    seen_phases = set()
    for depth in range(192):
        obs, mask = env.observe()
        actions = (
            neural.model(obs, mask)[0].argmax(-1) if depth else root_actions.clone()
        )
        solved = env.solved_actions()
        actions = torch.where(solved >= 0, solved, actions)
        if depth:
            indices = (
                torch.where(
                    alive & mixed & (env.state[:, PLAYER] != seat) & (solved < 0)
                )[0]
                .cpu()
                .tolist()
            )
            states = env.state.cpu().numpy() if indices or serial else None
            for index in indices:
                ref = reference(states[index])
                actual = published.act(ref)
                counts["published_calls"] += 1
                counts["compared_opponent_actions"] += 1
                counts["opponent_action_disagreements"] += actual != int(actions[index])
                assert ref.legal()[actual]
                phase = int(ref.state[PHASE])
                phase_counts = counts["phases"].setdefault(
                    str(phase), {"calls": 0, "disagreements": 0}
                )
                phase_counts["calls"] += 1
                phase_counts["disagreements"] += actual != int(actions[index])
                if verify and phase not in seen_phases:
                    alternate = reference(
                        sample_hidden_batch(
                            ref, np.random.default_rng(91001 + phase), 1
                        )[0]
                    )
                    np.testing.assert_array_equal(ref.observe(), alternate.observe())
                    torch.testing.assert_close(
                        reconstruct(ref).feature_tensor,
                        reconstruct(alternate).feature_tensor,
                        rtol=0,
                        atol=0,
                    )
                    assert actual == published.act(alternate)
                    seen_phases.add(phase)
                    counts["feature_invariance_checks"] += 1
                actions[index] = actual
        if serial:
            for index, ref in serial.items():
                if ref.state[PHASE] == TERMINAL:
                    continue
                if depth == 0:
                    expected = base if index // particles in (0, 2) else chosen
                elif index // particles >= 2 and ref.state[PLAYER] != seat:
                    expected = EndgameAgent(published).act(ref)
                else:
                    expected = EndgameAgent(neural).act(ref)
                assert int(actions[index]) == expected, ("serial policy", depth, index)
                ref.step(expected)
                counts["serial_checks"] += 1
        counts["active_transitions"] += int(alive.sum())
        result = env.step(actions, freeze_done=True)
        counts["executed_transitions"] += n
        ended = result[:, 2].bool()
        values = torch.where(alive & ended, result[:, 0].float() * sign, values)
        alive &= ~ended
        if serial:
            states = env.state.cpu().numpy()
            for index, ref in serial.items():
                np.testing.assert_array_equal(ref.state, states[index])
        if not bool(alive.any()):
            break
    obs, mask = env.observe()
    tail = neural.model(obs, mask)[1] * (1 - 2 * env.state[:, PLAYER]) * sign
    values = torch.where(alive, tail, values)
    counts["logical_transitions"] = n * 192
    counts["unfinished_per_arm"] = alive.reshape(4, particles).sum(1).cpu().tolist()
    counts["steps"] = depth + 1
    if verify:
        gold = SearchAgent(
            neural,
            particles=particles,
            horizon=192,
            use_belief=False,
            endgame=True,
            koi_only=False,
        )
        gold_env = BatchEnv(n, "cuda", "paper8")
        gold_env.state.copy_(torch.from_numpy(initial).cuda())
        expected = gold.rollout(gold_env, root_actions, sign)
        torch.testing.assert_close(
            values[: 2 * particles], expected[: 2 * particles], rtol=0, atol=0
        )
        counts["gold_executed_transitions"] = n * 192
        counts["selfplay_exact"] = True
        for index, ref in serial.items():
            if ref.state[PHASE] == TERMINAL:
                diff = int(ref.state[POINTS + seat]) - int(ref.state[POINTS + 1 - seat])
                assert float(values[index]) == np.sign(diff)
    return values.reshape(4, particles).cpu().numpy(), counts
