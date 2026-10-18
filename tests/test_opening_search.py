import copy

import numpy as np
import pytest
import torch

from koikoi.agents import solved_koi_action
from koikoi.layout import DEALER, PHASE, ROUND
from koikoi.reference import ReferenceEnv, valid_deal
from koikoi.search import sample_hidden, sample_hidden_batch
from scripts.diagnose_history_support import reference
from scripts.opening_search import OpeningRoundSearch, OpeningTerminalSearch, is_opening
from scripts.round_value_search import BoundaryTables
from scripts.run_arbitration_study import PARENTS


def tables():
    values = np.zeros((2, 8, 2, 2, 59, 2))
    for seat in (0, 1):
        values[:, :, seat] = ((np.arange(59) - 29) / 60 * (1 - 2 * seat))[
            None, None, None, :, None
        ]
    return BoundaryTables(
        values, np.repeat(values[:, None], 7, axis=1), PARENTS[0][1], "opening-fixture"
    )


def test_opening_sampler_conditions_only_on_own_public_deal():
    for dealer in (0, 1):
        root = next(
            env
            for seed in range(80100001, 80100051)
            if (env := ReferenceEnv(seed, "paper8")).state[DEALER] == dealer
        )
        assert is_opening(root)
        sampled = sample_hidden_batch(root, np.random.default_rng(80110001), 128)
        for state in sampled:
            env = reference(state)
            np.testing.assert_array_equal(env.observe(), root.observe())
            assert valid_deal(state[48:96])
        altered = reference(sample_hidden(root, np.random.default_rng(80110002)))
        np.testing.assert_array_equal(
            sampled, sample_hidden_batch(altered, np.random.default_rng(80110001), 128)
        )
        later = root.clone()
        later.step(int(np.flatnonzero(later.legal())[0]))
        assert not is_opening(later)
        later_round = root.clone()
        later_round.state[ROUND] = 1
        assert not is_opening(later_round)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("kind", ["terminal", "round"])
def test_all_opening_cards_confirmation_and_parent_fallback(device, kind, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    common = {"particles": 2, "confirmation_particles": 2, "device": device}
    agent = (
        OpeningTerminalSearch(PARENTS[0][0], **common)
        if kind == "terminal"
        else OpeningRoundSearch(PARENTS[0][0], tables(), **common)
    )
    monkeypatch.setattr(
        "scripts.history_conditioned_search.enumerate_support",
        lambda *a, **k: pytest.fail("An opening has no opponent-action evidence"),
    )
    if kind == "terminal":
        monkeypatch.setattr(agent, "_lower", lambda values: 1.0)
    else:
        monkeypatch.setattr(agent, "_joint_lower", lambda mean, samples: 1.0)
    root = ReferenceEnv(80120001, "paper8")
    before = root.state.copy()
    agent.reset_seed(80130001)
    action = agent.act(root)
    assert root.legal()[action] and agent.last_context["status"] == "opening_prior"
    assert agent.last_context["candidate_hands_before_validity"] == 10518300
    assert not agent.history_work and agent.confirmation_decisions == 1
    assert {r["alternative"] for r in agent.last_decision["alternatives"]} == set(
        np.flatnonzero(root.legal())
    ) - {agent.last_decision["base"]}
    decision, context, sampling = copy.deepcopy(
        (agent.last_decision, agent.last_context, agent.last_sampling)
    )
    assert (
        sampling["proposal"]["worlds_sha256"]
        != sampling["confirmation"]["worlds_sha256"]
    )
    if kind == "round":
        assert (
            decision["proposal"]["table"] == "A"
            and decision["confirmation"]["table"] == "B"
        )
        assert agent.work.round_endings == 32
    else:
        assert agent.work.completed_branches == 32
    alternate = reference(sample_hidden(root, np.random.default_rng(80140001)))
    agent.reset_seed(80130001)
    assert agent.act(alternate) == action
    assert (agent.last_decision, agent.last_context, agent.last_sampling) == (
        decision,
        context,
        sampling,
    )
    np.testing.assert_array_equal(root.state, before)
    later = root.clone()
    later.state[ROUND] = 1
    work_before = copy.deepcopy(agent.work)
    expected = solved_koi_action(later)
    expected = int(agent.parent.act(later)) if expected is None else expected
    assert agent.act(later) == expected and agent.work == work_before
    assert (
        agent.last_decision is None
        and agent.last_context["status"] == "outside_opening"
    )
    assert later.state[PHASE] == 0
