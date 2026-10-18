import copy

import numpy as np
import pytest
import torch

from koikoi.layout import PHASE, ROUND, TURN
from koikoi.reference import ReferenceEnv
from koikoi.search import sample_hidden
from scripts.diagnose_history_support import reference
from scripts.midround_search import MidroundRoundSearch, is_midround
from scripts.run_arbitration_study import PARENTS
from tests.test_history_conditioned_worlds import Extremal
from tests.test_round_value_search import fixture_tables


def roots():
    env = ReferenceEnv(31010001, "paper8")
    actor = Extremal()
    result = []
    while env.state[ROUND] == 0 and len(result) < 2:
        if is_midround(env):
            result.append(env.clone())
        env.step(actor.act(env))
    assert [r.state[TURN] for r in result] == [8, 9]
    return result


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_both_seats_conditioned_worlds_all_actions_and_confirmation(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    agent = MidroundRoundSearch(
        PARENTS[0][0], fixture_tables(), particles=2,
        confirmation_particles=2, device=device,
    )
    # Deterministic observation-only fixture models keep this routing test cheap.
    agent.opponents = {"point": Extremal(), "wp": Extremal(True)}
    monkeypatch.setattr(agent, "_joint_lower", lambda mean, samples: 1.0)
    for root in roots():
        agent.reset_seed(82110001)
        before = root.state.copy()
        action = agent.act(root)
        assert root.legal()[action] and agent.last_context["status"] == "searched"
        assert agent.last_context["candidate_hands"] in (4845, 816)
        assert len(agent.last_decision["alternatives"]) == 3
        assert agent.last_decision["proposal"]["table"] == "A"
        assert agent.last_decision["confirmation"]["table"] == "B"
        expected = copy.deepcopy((agent.last_decision, agent.last_context, agent.last_sampling))
        alternate = reference(sample_hidden(root, np.random.default_rng(82110002)))
        agent.reset_seed(82110001)
        assert agent.act(alternate) == action
        assert expected == (agent.last_decision, agent.last_context, agent.last_sampling)
        np.testing.assert_array_equal(root.state, before)
    assert agent.confirmation_decisions == 4


def test_outside_midround_and_public_cap_keep_parent():
    torch.set_num_threads(2)
    agent = MidroundRoundSearch(
        PARENTS[0][0], fixture_tables(), particles=2,
        confirmation_particles=2, device="cpu", max_hands=100,
    )
    initial = ReferenceEnv(31010001, "paper8")
    for root in (initial, *roots()):
        assert agent.act(root) == agent.parent.act(root)
        assert agent.last_decision is None and agent.support is None
    assert agent.outside_midround == 1 and agent.refusals["too_many_hands"] == 2
    later = roots()[0]
    later.state[ROUND] = 1
    assert not is_midround(later)
    later.state[ROUND], later.state[PHASE] = 0, 1
    assert not is_midround(later)
