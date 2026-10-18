import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from koikoi.agents import solved_koi_action
from koikoi.layout import PHASE, PLAYER, POINTS, ROUND, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.search import sample_hidden, sample_hidden_batch
from koikoi.upstream import reconstruct
from scripts.diagnose_history_support import reference
from scripts.history_conditioned_search import HistoryConditionedSearch
from scripts.history_conditioned_worlds import enumerate_support
from scripts.run_arbitration_study import PARENTS


def archived_root():
    rows = json.loads(
        Path("runs/rollout_opponent_roots.json").read_text(encoding="utf-8")
    )["roots"]
    return reference(rows[32]["state"])


def search(device="cpu", mode="history", **kwargs):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    return HistoryConditionedSearch(
        PARENTS[0][0],
        device=device,
        mode=mode,
        particles=2,
        confirmation_particles=2,
        **kwargs,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mixture_routes_full_branches_across_seats_and_rounds(device):
    agent = search(device, mode="uniform")
    original = agent.continuation
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
        rng = np.random.default_rng(31110001 + index)
        worlds = sample_hidden_batch(root, rng, 2)
        labels = np.array([0, 1], dtype=np.int32)
        # Exercise BOTH opponent routes even if a small random sample would not.
        original.prepare(seat, labels)
        refs = [reference(world) for world in worlds for _ in range(2)]
        work = {"decisions": 0, "round_endings": 0, "completed_branches": 0}
        seen_routes, seen_rounds = set(), set()

        def step(ref, action, work=work):
            if ref.state[PHASE] == TERMINAL:
                return
            result = ref.step(int(action))
            work["decisions"] += 1
            work["round_endings"] += int(result[1])
            work["completed_branches"] += int(result[2])

        for i, ref in enumerate(refs):
            step(ref, (base, alternative)[i % 2])

        def checked(
            env,
            refs=refs,
            seat=seat,
            labels=labels,
            seen_rounds=seen_rounds,
            seen_routes=seen_routes,
            step=step,
        ):
            for actual, expected in zip(env.state.cpu().numpy(), refs):
                np.testing.assert_array_equal(actual, expected.state)
            actual_actions = original(env)
            expected_actions = [-1] * len(refs)
            # Independently build original published features from reference states.
            for route in range(3):
                indices, features, phases, masks = [], [], [], []
                for i, ref in enumerate(refs):
                    if ref.state[PHASE] == TERMINAL:
                        continue
                    seen_rounds.add(int(ref.state[ROUND]))
                    expected_route = (
                        2 if int(ref.state[PLAYER]) == seat else int(labels[i // 2])
                    )
                    if expected_route != route:
                        continue
                    seen_routes.add(route)
                    x = reconstruct(ref).feature_tensor
                    padded = torch.zeros((300, 50), dtype=torch.float32)
                    padded[:, : x.shape[1]] = x
                    mask = ref.legal().astype(bool)
                    solved = solved_koi_action(ref)
                    if solved is not None:
                        mask[:] = False
                        mask[solved] = True
                    indices.append(i)
                    features.append(padded)
                    phases.append(int(ref.state[PHASE]))
                    masks.append(mask)
                if indices:
                    predicted = (
                        original.actors[route](
                            torch.stack(features).to(device),
                            torch.tensor(phases, device=device),
                            torch.tensor(np.array(masks), device=device),
                        )
                        .argmax(-1)
                        .tolist()
                    )
                    for i, action in zip(indices, predicted):
                        expected_actions[i] = action
            assert actual_actions.tolist() == expected_actions
            for ref, action in zip(refs, expected_actions):
                step(ref, action)
            return actual_actions

        wins, points, measured = agent.workspaces[2](
            torch.from_numpy(worlds).to(device),
            torch.tensor([[base, alternative]] * 2, device=device),
            checked,
        )
        assert asdict(measured) == work
        assert measured.completed_branches == 4
        assert seen_routes == {0, 1, 2} and len(seen_rounds) > 1
        expected_points = []
        for actual, ref in zip(agent.workspaces[2].env.state.cpu().numpy(), refs):
            np.testing.assert_array_equal(actual, ref.state)
            expected_points.append(
                int(ref.state[POINTS] - ref.state[POINTS + 1]) * (1 - 2 * seat)
            )
        np.testing.assert_array_equal(points.cpu().numpy().ravel(), expected_points)
        np.testing.assert_array_equal(
            wins.cpu().numpy().ravel(), np.sign(expected_points)
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_conditioned_action_confirmation_and_hidden_invariance(device, monkeypatch):
    agent = search(device)
    root = archived_root()
    before = root.state.copy()
    # Force the confirmation branch as a routing test, not a strength result.
    monkeypatch.setattr(agent, "_lower", lambda differences: 1.0)
    agent.reset_seed(31120001)
    action = agent.act(root)
    assert agent.last_context["status"] == "searched"
    assert agent.support.model_names == ("point", "wp")
    assert agent.last_context["supported_model_counts"][1] == 8
    assert agent.confirmation_decisions == 1
    assert (
        agent.last_sampling["proposal"]["worlds_sha256"]
        != agent.last_sampling["confirmation"]["worlds_sha256"]
    )
    decisions, context, sampling = (
        agent.last_decision,
        agent.last_context,
        agent.last_sampling,
    )
    assert {r["alternative"] for r in decisions["alternatives"]} == set(
        np.flatnonzero(root.legal())
    ) - {decisions["base"]}
    alternate = reference(sample_hidden(root, np.random.default_rng(51)))
    agent.reset_seed(31120001)
    assert agent.act(alternate) == action
    assert agent.last_decision == decisions
    assert agent.last_context == context
    assert agent.last_sampling == sampling
    np.testing.assert_array_equal(root.state, before)


def test_refusals_preserve_parent_and_do_not_simulate(monkeypatch):
    agent = search()
    early = ReferenceEnv(31130001, "paper8")
    assert agent.act(early) == agent.parent.act(early)
    assert agent.last_context["status"] == "too_many_hands"
    assert not agent.history_work and agent.work.decisions == 0
    root = archived_root()
    support = enumerate_support(root, agent.opponents, max_hands=100)
    from dataclasses import replace

    empty = replace(support, compatible=((False, False),) * support.valid_hands)
    monkeypatch.setattr(
        "scripts.history_conditioned_search.enumerate_support", lambda *args: empty
    )
    assert agent.act(root) == agent.parent.act(root)
    assert agent.last_context["status"] == "empty_support"
    assert agent.work.decisions == 0 and agent.searched_decisions == 0
    assert agent.refusals == {"too_many_hands": 1, "empty_support": 1}


def test_uniform_control_never_uses_history_likelihood(monkeypatch):
    agent = search(mode="uniform")
    monkeypatch.setattr(
        "scripts.history_conditioned_search.enumerate_support",
        lambda *args: pytest.fail("Uniform control scored history"),
    )
    root = archived_root()
    action = agent.act(root)
    assert root.legal()[action]
    assert agent.last_context["status"] == "searched"
    assert not agent.history_work
    assert agent.work.completed_branches >= 4
