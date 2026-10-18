import copy

import numpy as np
import pytest
import torch

from koikoi.reference import ReferenceEnv
from koikoi.search import sample_hidden
from scripts.diagnose_history_support import reference
from scripts.fixed_midround_terminal import FixedMidroundTerminal
from scripts.run_arbitration_study import PARENTS
from tests.test_history_conditioned_worlds import Extremal
from tests.test_midround_search import roots


@pytest.mark.parametrize("mode", ["uniform", "history"])
def test_terminal_search_routing_fresh_confirmation_and_hidden_invariance(
    mode, monkeypatch
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    agent = FixedMidroundTerminal(
        PARENTS[0][0], mode=mode, particles=2, confirmation_particles=2, device="cuda"
    )
    agent.opponents = {"point": Extremal(), "wp": Extremal(True)}
    monkeypatch.setattr(agent, "_lower", lambda values: 1.0)
    for root in roots():
        before = root.state.copy()
        agent.reset_seed(84110001)
        action = agent.act(root)
        assert root.legal()[action] and agent.last_context["status"] == "searched"
        assert len(agent.last_decision["alternatives"]) == 3
        assert agent.last_decision["confirmation"] is not None
        assert (
            agent.last_sampling["proposal"]["worlds_sha256"]
            != agent.last_sampling["confirmation"]["worlds_sha256"]
        )
        expected = copy.deepcopy(
            (agent.last_decision, agent.last_sampling, agent.last_context)
        )
        alternate = reference(sample_hidden(root, np.random.default_rng(84110002)))
        agent.reset_seed(84110001)
        assert agent.act(alternate) == action
        assert expected == (
            agent.last_decision,
            agent.last_sampling,
            agent.last_context,
        )
        np.testing.assert_array_equal(root.state, before)
    assert agent.work.completed_branches == 64
    assert agent.confirmation_work.completed_branches == 16
    assert agent.continuation.microbatch == 64 and not hasattr(agent, "tables")
    initial = ReferenceEnv(31010001, "paper8")
    assert agent.act(initial) == agent.parent.act(initial)
    assert (
        agent.last_decision is None
        and agent.last_context["status"] == "outside_midround"
    )
    agent.max_hands = 100
    root = roots()[0]
    assert agent.act(root) == agent.parent.act(root)
    assert (
        agent.last_decision is None and agent.last_context["status"] == "too_many_hands"
    )
