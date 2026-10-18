import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from koikoi.layout import DEALER, PHASE, PLAYER, POINTS, ROUND, TERMINAL
from koikoi.search import sample_hidden, sample_hidden_batch
from scripts.diagnose_history_support import reference
from scripts.round_value_search import BoundaryTables, RoundValueSearch
from scripts.run_arbitration_study import PARENTS


def fixture_tables():
    values = np.empty((2, 8, 2, 2, 59, 2))
    for t, r, seat, model, p, d in np.ndindex(values.shape):
        values[t, r, seat, model, p, d] = (
            (p - 29) / 60 * (1 - 2 * seat)
            + 0.005 * r
            + 0.01 * model
            + 0.015 * d
            + 0.02 * t
        )
    boot = np.repeat(values[:, None], 7, axis=1)
    for b in range(7):
        boot[:, b] += (
            (b - 3)
            * 0.01
            * ((np.arange(59) - 29) / 30)[None, None, None, None, :, None]
        )
    return BoundaryTables(
        values, boot, PARENTS[0][1], "synthetic-functionality-fixture"
    )


def test_public_lookup_exact_terminal_precedence_and_axes():
    tables = fixture_tables()
    rounds = np.array([1, 3, 7, 7, 0])
    points = np.array([1, 30, 59, 30, -2])
    dealers, labels = np.array([1, 0, 1, 0, 1]), np.array([0, 1, 1, 0, 1])
    terminal = np.array([False, False, False, True, True])
    for seat in (0, 1):
        for table in (0, 1):
            values, boot = tables.lookup(
                rounds, points, dealers, terminal, seat, labels, table
            )
            for i in range(5):
                if terminal[i]:
                    expected = np.sign(points[i] - 30) * (1 - 2 * seat)
                    assert values[i] == expected and (boot[:, i] == expected).all()
                else:
                    key = (rounds[i], seat, labels[i], points[i] - 1, dealers[i])
                    assert values[i] == tables.values[table][key]
                    for b in range(7):
                        assert boot[b, i] == tables.bootstrap[table, b][key]
    with pytest.raises(ValueError, match="later live"):
        tables.lookup(rounds, points, dealers, np.zeros(5, bool), 0, labels, 0)
    assert not tables.values.flags.writeable and not tables.bootstrap.flags.writeable


def search(device, mode="uniform"):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    return RoundValueSearch(
        PARENTS[0][0],
        fixture_tables(),
        mode=mode,
        particles=2,
        confirmation_particles=2,
        device=device,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_paired_boundaries_and_joint_statistics_match_reference(device):
    agent = search(device)
    continuation = agent.continuation
    states = np.load(
        "runs/observer_seat_information_v2/parent1_states.npy", allow_pickle=False
    )
    for index in (12, 13):
        root = reference(states[index])
        root.state[ROUND] = 0
        root.state[POINTS : POINTS + 2] = 30
        seat = int(root.state[PLAYER])
        base = int(agent.parent.act(root))
        alternative = next(int(a) for a in np.flatnonzero(root.legal()) if a != base)
        seed = 40210001 + index
        rng = np.random.default_rng(seed)
        worlds = sample_hidden_batch(root, rng, 2)
        labels = rng.integers(0, 2, 2, dtype=np.int32)
        refs = [reference(world) for world in worlds for _ in range(2)]
        done = [False] * 4
        work = dict.fromkeys(
            ("decisions", "round_endings", "terminal_match_endings"), 0
        )

        def advance(actions, refs=refs, done=done, work=work):
            for i, (ref, action) in enumerate(zip(refs, actions)):
                if done[i]:
                    continue
                result = ref.step(int(action))
                work["decisions"] += 1
                work["round_endings"] += int(result[1])
                work["terminal_match_endings"] += int(result[2])
                done[i] = bool(result[1])

        advance([base, alternative] * 2)

        class Checked:
            def prepare(self, *args):
                continuation.prepare(*args)

            def __call__(self, env, refs=refs, done=done, advance=advance):
                for i, ref in enumerate(refs):
                    expected = ref.state.copy()
                    if done[i]:
                        expected[PHASE] = TERMINAL
                    np.testing.assert_array_equal(env.state[i].cpu().numpy(), expected)
                actions = continuation(env)
                advance(actions.tolist())
                return actions

        agent.continuation = Checked()
        record = agent._compare_actions(
            root, base, [alternative], 2, np.random.default_rng(seed)
        )[0]
        agent.continuation = continuation
        np.testing.assert_array_equal(
            agent.workspaces[2].env.state.cpu().numpy(),
            np.stack([r.state for r in refs]),
        )
        assert record["work"] == work and work["round_endings"] == 4
        values, boot = [], []
        for i, ref in enumerate(refs):
            s = ref.state
            if s[PHASE] == TERMINAL:
                value = np.sign(s[POINTS] - 30) * (1 - 2 * seat)
                values.append(value)
                boot.append([value] * 7)
            else:
                key = (s[ROUND], seat, labels[i // 2], s[POINTS] - 1, s[DEALER])
                values.append(agent.tables.values[0][key])
                boot.append([agent.tables.bootstrap[0, b][key] for b in range(7)])
        differences = np.diff(np.array(values).reshape(2, 2), axis=1).ravel() / 2
        boot_changes = np.diff(np.array(boot).T.reshape(7, 2, 2), axis=2)[:, :, 0] / 2
        weights = np.random.default_rng(
            agent.last_sampling["proposal"]["bootstrap_seed"]
        ).multinomial(2, [0.5, 0.5], size=7)
        means = np.array(
            [
                sum(boot_changes[b, i] * weights[b, i] for i in range(2)) / 2
                for b in range(7)
            ]
        )
        np.testing.assert_allclose(
            record["joint_bootstrap_means"], means, rtol=0, atol=1e-15
        )
        assert record["win_change"] == pytest.approx(differences.mean())
        assert record["lower_margin"] == pytest.approx(
            differences.mean() - agent.confidence * means.std(ddof=1)
        )
        assert record["table_se"] == pytest.approx(boot_changes.mean(1).std(ddof=1))
        assert record["table"] == "A"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_history_confirmation_uses_B_fresh_worlds_and_hidden_invariance(
    device, monkeypatch
):
    agent = search(device, "history")
    saved = json.loads(
        Path("runs/rollout_opponent_roots.json").read_text(encoding="utf-8")
    )["roots"][32]["state"]
    root = reference(saved)
    before = root.state.copy()
    # Only force routing for this test, never use this margin in research/play.
    monkeypatch.setattr(agent, "_joint_lower", lambda mean, samples: 1.0)
    agent.reset_seed(40220001)
    result = agent.act(root)
    assert agent.confirmation_decisions == 1 and agent.last_decision["changed"]
    assert agent.last_decision["proposal"]["table"] == "A"
    assert agent.last_decision["confirmation"]["table"] == "B"
    decisions, sampling, context = (
        agent.last_decision,
        agent.last_sampling,
        agent.last_context,
    )
    assert (
        sampling["proposal"]["worlds_sha256"]
        != sampling["confirmation"]["worlds_sha256"]
    )
    assert (
        sampling["proposal"]["bootstrap_seed"]
        != sampling["confirmation"]["bootstrap_seed"]
    )
    assert asdict(agent.confirmation_work)["round_endings"] == 4
    np.testing.assert_array_equal(root.state, before)
    alternate = reference(sample_hidden(root, np.random.default_rng(61)))
    agent.reset_seed(40220001)
    assert agent.act(alternate) == result
    assert (
        agent.last_decision == decisions
        and agent.last_sampling == sampling
        and agent.last_context == context
    )
