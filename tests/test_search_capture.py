from copy import deepcopy

import numpy as np
import pytest
import torch

from koikoi.agents import GreedyAgent, NeuralAgent
from koikoi.layout import BANKRUPT, KOI, PHASE, PLAYER, ROUND, ROUNDS
from koikoi.reference import ReferenceEnv
from koikoi.search import SearchAgent, sample_hidden
from koikoi.train import Config, Trainer


def roots():
    states = []
    for seed in (11, 32, 57, 93):
        env = ReferenceEnv(seed)
        for _ in range(200):
            if env.state[PHASE] == KOI and env.legal()[48] and env.legal()[49]:
                states.append(env.clone())
                break
            env.step(GreedyAgent().act(env))
            if env.state[PHASE] == 4:
                break
    assert len(states) == 4
    assert {int(s.state[PLAYER]) for s in states} == {0, 1}
    return states


@pytest.mark.cuda
@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("endgame", [True, False])
def test_search_capture_matches_values_actions_games_rng_and_hidden_invariance(
    tmp_path, deterministic, endgame, monkeypatch
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    path = tmp_path / "actor.pt"
    Trainer(Config(device="cpu", envs=2, width=32)).save(path)
    neural = NeuralAgent(path, device="cuda", deterministic=deterministic)
    options = {
        "particles": 4,
        "horizon": 64,
        "use_belief": False,
        "endgame": endgame,
        "seed": 15,
        "koi_only": False,
    }
    eager = SearchAgent(neural, **options)
    captured = SearchAgent(neural, cuda_graph=True, **options)
    expected_values = []
    original_rollout = eager.rollout

    def record(*args):
        values = original_rollout(*args)
        expected_values.append(values.detach().clone())
        return values

    monkeypatch.setattr(eager, "rollout", record)
    late = roots()[0]
    late.state[ROUND] = late.state[ROUNDS] - 1
    for root in roots() + [ReferenceEnv(112), ReferenceEnv(223), late]:
        # Different sampled hidden worlds must never alter identical observations.
        alternate = root.clone()
        alternate.state = sample_hidden(root, np.random.default_rng(72))
        np.testing.assert_array_equal(root.observe(), alternate.observe())
        before = root.state.copy()
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        proposal_rng = deepcopy(eager.rng.bit_generator.state)
        previous_searches = eager.searched_decisions
        expected = eager.act(root)
        after_cpu, after_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        captured.rng.bit_generator.state = proposal_rng
        actual = captured.act(alternate)
        assert actual == expected and root.legal()[actual]
        np.testing.assert_array_equal(root.state, before)
        assert torch.equal(after_cpu, torch.get_rng_state())
        assert torch.equal(after_cuda, torch.cuda.get_rng_state())
        assert eager.rng.bit_generator.state == captured.rng.bit_generator.state
        assert eager.searched_decisions == captured.searched_decisions
        assert eager.simulated_decisions == captured.simulated_decisions
        if eager.searched_decisions == previous_searches:
            continue  # A public endgame dominance decision bypasses search.
        key = (
            int(root.legal().sum()) * 4,
            "paper8" if root.state[BANKRUPT] else "research12",
        )
        graph = captured.graph_cache[key][0]
        assert torch.equal(graph.batch["values"], expected_values[-1])
        if root is late:
            # Every particle finishes this final round within the horizon; values
            # must retain its actual match outcome despite subsequent reset work.
            assert torch.isin(
                expected_values[-1], torch.tensor([-1.0, 0.0, 1.0], device="cuda")
            ).all()
        assert torch.equal(eager.cache[key].state, captured.cache[key].state)
        assert eager.simulated_decisions == captured.simulated_decisions
        assert eager.searched_decisions == captured.searched_decisions
        assert eager.changed_decisions == captured.changed_decisions
        # Captured inference must read current tensors, not stale copied weights.
        with torch.no_grad():
            neural.model.koi.bias.add_(torch.tensor([0.1, -0.1], device="cuda"))
    assert eager.preparation_decisions == 0
    assert captured.preparation_decisions == 3 * 64 * sum(
        key[0] for key in captured.graph_cache
    )


def test_capture_rejects_cpu():
    class Neural:
        device = "cpu"

    with pytest.raises(ValueError, match="CUDA inference device"):
        SearchAgent(Neural(), use_belief=False, cuda_graph=True)
