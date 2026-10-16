from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from koikoi.agents import GreedyAgent, NeuralAgent
from koikoi.layout import KOI, PHASE, PLAYER, ROUND, ROUNDS
from koikoi.reference import ReferenceEnv
from koikoi.search import SearchAgent, sample_hidden
from koikoi.train import Config, Trainer


@pytest.mark.cuda
@pytest.mark.parametrize("profile", ["paper8", "research12"])
@pytest.mark.parametrize("endgame", [False, True])
@pytest.mark.parametrize(
    "horizon,chunk_size", [(1, 32), (7, 7), (8, 7), (15, 7), (65, 16), (193, 32)]
)
def test_chunked_values_actions_rng_tails_and_reuse(
    tmp_path, monkeypatch, profile, endgame, horizon, chunk_size
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    torch.manual_seed(845)
    path = tmp_path / "actor.pt"
    Trainer(Config(device="cpu", envs=2, width=32, profile=profile)).save(path)
    neural = NeuralAgent(path, device="cuda", deterministic=True)
    options = {
        "particles": 4,
        "horizon": horizon,
        "seed": 15,
        "use_belief": False,
        "koi_only": False,
        "endgame": endgame,
    }
    eager = SearchAgent(neural, **options)
    chunked = SearchAgent(neural, cuda_graph=True, chunk_size=chunk_size, **options)
    captured_values = []
    original = eager.rollout

    def record(*args):
        result = original(*args)
        captured_values.append(result.clone())
        return result

    monkeypatch.setattr(eager, "rollout", record)
    early = ReferenceEnv(112, profile=profile)
    late = early.clone()
    late.state[ROUND] = late.state[ROUNDS] - 1
    opponent = early.clone()
    for _ in range(100):
        if (
            opponent.state[PLAYER] == 1
            and opponent.state[PHASE] != KOI
            and opponent.legal().sum() > 1
        ):
            break
        opponent.step(GreedyAgent().act(opponent))
    assert opponent.state[PLAYER] == 1 and opponent.legal().sum() > 1
    # Same cache entry must work after a terminated root and model weight edits.
    expected_executed = 0
    saw_early_stop = False
    for root in (early, late, opponent, early):
        alternate = root.clone()
        alternate.state = sample_hidden(root, np.random.default_rng(72))
        np.testing.assert_array_equal(root.observe(), alternate.observe())
        unchanged = alternate.state.copy()
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        proposal_rng = deepcopy(eager.rng.bit_generator.state)
        expected = eager.act(root)
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
        chunked.rng.bit_generator.state = proposal_rng
        actual = chunked.act(alternate)
        assert actual == expected
        np.testing.assert_array_equal(alternate.state, unchanged)
        assert eager.rng.bit_generator.state == chunked.rng.bit_generator.state
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
        n = int(root.legal().sum()) * 4
        key = (n, profile)
        graph = chunked.graph_cache[key][0]
        assert torch.equal(graph.batch["values"], captured_values[-1])
        expected_executed += n * graph.last_steps
        assert chunked.executed_decisions == expected_executed
        assert eager.executed_decisions == eager.simulated_decisions
        assert eager.simulated_decisions == chunked.simulated_decisions
        assert eager.changed_decisions == chunked.changed_decisions
        if graph.last_steps == horizon:
            assert torch.equal(eager.cache[key].state, chunked.cache[key].state)
        else:
            saw_early_stop = True
            assert not graph.alive.any()
            assert torch.isin(
                graph.batch["values"], torch.tensor([-1.0, 0.0, 1.0], device="cuda")
            ).all()
        if horizon == 1:
            assert graph.alive.all()  # Explicit unfinished-tail bootstrap case.
        with torch.no_grad():
            neural.model.koi.bias.add_(torch.tensor([0.1, -0.1], device="cuda"))
            neural.model.value.bias.add_(0.1)
    if horizon == 193:
        assert saw_early_stop
        assert chunked.executed_decisions < chunked.simulated_decisions
    lengths = [
        min(chunk_size, horizon - start) for start in range(0, horizon, chunk_size)
    ]
    preparation_depth = lengths[0] + sum(set(lengths[1:]))
    assert chunked.preparation_decisions == 3 * preparation_depth * sum(
        k[0] for k in chunked.graph_cache
    )


@pytest.mark.parametrize(
    "options",
    [
        {"chunk_size": -1},
        {"chunk_size": 1.5},
        {"chunk_size": 16},
        {"chunk_size": 16, "cuda_graph": True, "deterministic": False},
        {"chunk_size": 16, "cuda_graph": True, "training": True},
    ],
)
def test_chunked_rejects_unsupported_modes(options):
    options = options.copy()
    neural = SimpleNamespace(
        device="cuda",
        deterministic=options.pop("deterministic", True),
        model=SimpleNamespace(training=options.pop("training", False)),
    )
    with pytest.raises(ValueError, match="[Cc]hunk"):
        SearchAgent(neural, use_belief=False, **options)
