from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from koikoi.agents import EndgameAgent, GreedyAgent
from koikoi.evaluate import seed_agents
from koikoi.layout import PHASE, PLAYER, TERMINAL
from koikoi.model import PolicyValue
from koikoi.reference import ReferenceEnv
from koikoi.search import SearchAgent, sample_hidden


def neural(device="cpu"):
    torch.manual_seed(84)
    return SimpleNamespace(
        device=device,
        deterministic=True,
        belief_trained=False,
        model=PolicyValue(width=32).to(device).eval(),
    )


@pytest.mark.parametrize("outcome", ["accept", "reject", "tie", "no_proposal"])
def test_confirmation_only_checks_selected_deviation(monkeypatch, outcome):
    actor = neural()
    root = ReferenceEnv(112)
    legal = np.flatnonzero(root.legal())
    with torch.no_grad():
        logits = actor.model(
            torch.from_numpy(root.observe())[None], torch.from_numpy(root.legal())[None]
        )[0]
    base = int(logits.argmax())
    proposed = next(int(a) for a in legal if a != base)
    search = SearchAgent(
        actor,
        particles=4,
        horizon=1,
        use_belief=False,
        koi_only=False,
        confirmation_particles=8,
    )
    initial_proposal_rng = deepcopy(search.rng.bit_generator.state)
    initial_confirmation_rng = deepcopy(search.confirmation_rng.bit_generator.state)
    calls = []

    def returns(reference, actions, worlds):
        calls.append((actions.copy(), worlds.copy()))
        if len(calls) == 1:
            values = torch.zeros(len(actions), 4)
            if outcome != "no_proposal":
                values[int(np.flatnonzero(actions == proposed)[0])] = 1
            return values
        assert actions.tolist() == [
            base,
            proposed,
        ]  # No other candidate can win validation.
        values = torch.zeros(2, 8)
        values[1] = {"accept": 1, "reject": -1, "tie": 0}[outcome]
        return values

    monkeypatch.setattr(search, "_evaluate_candidates", returns)
    action = search.act(root)
    assert action == (proposed if outcome == "accept" else base)
    assert search.rng.bit_generator.state != initial_proposal_rng
    assert search.proposed_decisions == int(outcome != "no_proposal")
    assert search.confirmation_decisions == int(outcome != "no_proposal")
    assert search.changed_decisions == int(outcome == "accept")
    assert (
        search.confirmation_rng.bit_generator.state != initial_confirmation_rng
    ) == (outcome != "no_proposal")
    if len(calls) == 2:
        assert not np.array_equal(calls[0][1], calls[1][1][:4])
        # Confirmation must not consume the original proposal stream.
        control = SearchAgent(
            actor, particles=4, horizon=1, use_belief=False, koi_only=False
        )
        monkeypatch.setattr(
            control, "_evaluate_candidates", lambda *args: torch.zeros(len(legal), 4)
        )
        control.act(root)
        assert control.rng.bit_generator.state == search.rng.bit_generator.state


@pytest.mark.parametrize(
    "options",
    [
        {"confirmation_particles": -1},
        {"confirmation_particles": 1.5},
        {"confirmation_particles": 1},
        {"confirmation_particles": 8, "deterministic": False},
        {"confirmation_particles": 8, "training": True},
    ],
)
def test_confirmation_rejects_unsupported_modes(options):
    options = options.copy()
    actor = neural()
    actor.deterministic = options.pop("deterministic", True)
    actor.model.train(options.pop("training", False))
    with pytest.raises(ValueError, match="Confirmation"):
        SearchAgent(actor, use_belief=False, **options)


def roots(profile):
    result = [ReferenceEnv(112, profile), ReferenceEnv(223, profile)]
    other = result[0].clone()
    while other.state[PLAYER] != 1 or other.legal().sum() < 2:
        assert other.state[PHASE] != TERMINAL
        other.step(GreedyAgent().act(other))
    result.append(other)
    return result


@pytest.mark.cuda
@pytest.mark.parametrize("profile", ["paper8", "research12"])
@pytest.mark.parametrize("chunk", [0, 7])
def test_confirmation_graph_returns_streams_hidden_invariance_and_work(
    monkeypatch, profile, chunk
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    actor = neural("cuda")
    options = {
        "particles": 4,
        "horizon": 15,
        "seed": 17,
        "use_belief": False,
        "koi_only": False,
        "confidence": 0,
        "endgame": True,
        "confirmation_particles": 8,
    }
    eager = SearchAgent(actor, **options)
    captured = SearchAgent(actor, cuda_graph=True, chunk_size=chunk, **options)
    traces = [[], []]
    for search, trace in zip((eager, captured), traces):
        original = search._evaluate_candidates

        def record(reference, actions, worlds, original=original, trace=trace):
            values = original(reference, actions, worlds)
            trace.append((actions.copy(), worlds.copy(), values.clone()))
            return values

        monkeypatch.setattr(search, "_evaluate_candidates", record)
    for index, root in enumerate(roots(profile) * 2):
        alternate = root.clone()
        alternate.state = sample_hidden(root, np.random.default_rng(72))
        unchanged = alternate.state.copy()
        np.testing.assert_array_equal(root.observe(), alternate.observe())
        # Match-level resets must reset both streams even after unrelated consumption.
        for search in (eager, captured):
            search.rng.random(index * 11)
            search.confirmation_rng.random(index * 17)
            seed_agents(EndgameAgent(search), GreedyAgent(), 4321 + index % 3)
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        assert eager.act(root) == captured.act(alternate)
        assert eager.rng.bit_generator.state == captured.rng.bit_generator.state
        assert (
            eager.confirmation_rng.bit_generator.state
            == captured.confirmation_rng.bit_generator.state
        )
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
        np.testing.assert_array_equal(alternate.state, unchanged)
    assert eager.confirmation_decisions > 0
    assert len(traces[0]) == len(traces[1])
    for first, second in zip(*traces):
        for a, b in zip(first[:2], second[:2]):
            np.testing.assert_array_equal(a, b)
        assert torch.equal(first[2], second[2])
    for key in (
        "simulated_decisions",
        "confirmation_simulated_decisions",
        "searched_decisions",
        "proposed_decisions",
        "changed_decisions",
        "confirmation_decisions",
    ):
        assert getattr(eager, key) == getattr(captured, key)
    assert eager.simulated_decisions == sum(
        len(a) * len(w) * 15 for a, w, _ in traces[0]
    )
    assert (
        eager.confirmation_simulated_decisions
        == eager.confirmation_decisions * 2 * 8 * 15
    )
    assert eager.preparation_decisions == eager.confirmation_preparation_decisions == 0
    assert 0 < captured.executed_decisions <= captured.simulated_decisions
    assert captured.confirmation_preparation_decisions <= captured.preparation_decisions
    prep_depth = 15 if not chunk else 7 + 7 + 1
    assert captured.preparation_decisions == 3 * prep_depth * sum(
        k[0] for k in captured.graph_cache
    )
    # Repeat roots in another order: candidate values and actions must replay exactly.
    for root_index in (2, 0, 1):
        root = roots(profile)[root_index]
        seed_agents(EndgameAgent(captured), GreedyAgent(), 4321 + root_index)
        start = len(traces[1])
        captured.act(root)
        previous = [x for x in traces[0] if np.array_equal(x[1], traces[1][start][1])]
        assert previous and torch.equal(previous[0][2], traces[1][start][2])
