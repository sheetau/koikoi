import copy

import numpy as np
import pytest
import torch

from koikoi.agents import EndgameAgent, solved_koi_action
from koikoi.batch_evaluate import paired_batch
from koikoi.env import BatchEnv
from koikoi.evaluate import duel, validate_split
from koikoi.layout import PHASE, TERMINAL
from koikoi.model import PolicyValue
from koikoi.reference import ReferenceEnv


class ModelAgent:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def act(self, env):
        obs = torch.from_numpy(env.observe())[None]
        mask = torch.from_numpy(env.legal())[None]
        return int(self.model(obs, mask)[0].argmax(-1))


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("profile", ["paper8", "research12"])
def test_batched_duels_match_reference_and_chunking(device, profile):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    torch.manual_seed(17)
    a, b = (PolicyValue(width=32).eval() for _ in range(2))
    for guard in (False, True, (True, False), (False, True)):
        first, second = ModelAgent(a), ModelAgent(b)
        flags = (guard, guard) if isinstance(guard, bool) else guard
        if flags[0]:
            first = EndgameAgent(first)
        if flags[1]:
            second = EndgameAgent(second)
        expected, decisions = [], 0
        for seed in range(100, 103):
            one = duel(first, second, seed, profile)
            two = duel(second, first, seed, profile)
            expected.append(
                {
                    "seed": seed,
                    "seat0_score": one[0],
                    "seat1_score": 1 - two[0],
                    "seat0_point_difference": one[1],
                    "seat1_point_difference": -two[1],
                }
            )
            decisions += one[3] + two[3]
        models = [copy.deepcopy(model).to(device) for model in (a, b)]
        records, counters = paired_batch(*models, 100, 3, profile, device, guard)
        assert records == expected
        assert counters["decisions"] == decisions
        assert counters["matches"] == 6
        if profile == "research12":
            assert counters["rounds"] == 72
        chunked = []
        for seed, size in ((100, 1), (101, 2)):
            chunked.extend(paired_batch(*models, seed, size, profile, device, guard)[0])
        assert chunked == expected


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_native_endgame_and_frozen_terminal_rows(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    env = BatchEnv(24, device=device)
    refs = [ReferenceEnv(1 + i) for i in range(24)]
    rng = np.random.default_rng(25)
    for _ in range(200):
        env.state.copy_(torch.from_numpy(np.stack([r.state for r in refs])).to(device))
        expected = [solved_koi_action(r) for r in refs]
        assert env.solved_actions().cpu().tolist() == [
            -1 if a is None else a for a in expected
        ]
        actions = []
        for ref in refs:
            if ref.state[PHASE] == TERMINAL:
                actions.append(-1)
            else:
                actions.append(int(rng.choice(np.flatnonzero(ref.legal()))))
                ref.step(actions[-1])
        result = env.step(torch.tensor(actions, device=device), freeze_done=True)
        assert not (result[:, 0] == 99).any()
        np.testing.assert_array_equal(
            env.state.cpu().numpy(), np.stack([r.state for r in refs])
        )
    assert all(r.state[PHASE] == TERMINAL for r in refs)
    before = env.state.clone()
    assert not env.step(torch.full((24,), -1, device=device), freeze_done=True).any()
    assert torch.equal(before, env.state)


def test_development_and_test_seed_ranges_are_disjoint():
    validate_split(1_000_000, 10000, "development")
    validate_split(100_000_000, 10000, "test")
    with pytest.raises(ValueError, match="reserved"):
        validate_split(99_999_999, 2, "development")
    with pytest.raises(ValueError, match="Final test"):
        validate_split(99_999_999, 2, "test")
