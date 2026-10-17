import numpy as np
import pytest
import torch

from koikoi.agents import solved_koi_action
from koikoi.arbitration import ExpertBank, read_checkpoint
from koikoi.env import BatchEnv
from koikoi.layout import EPISODE, PHASE, PLAYER, POINTS, SEED, TERMINAL
from koikoi.published_actor import PublishedActor
from koikoi.reference import ReferenceEnv
from scripts.collect_reusable_paired_targets import PairedWorkspace
from scripts.opponent_paired_roots import OpponentContinuation, collect_roots
from scripts.run_arbitration_study import PARENTS
from scripts.train_paired_arbitration import setup


class OrderedLegalActor(torch.nn.Module):
    def __init__(self, order):
        super().__init__()
        self.order = order

    def forward(self, features, phase, legal):
        ranks = legal.long().cumsum(-1)
        scores = (
            ranks.float() if self.order == -1 else -(ranks - self.order).abs().float()
        )
        return scores.masked_fill(~legal, -1e9)


def reference_outcome(state, first, opponent_id):
    game = ReferenceEnv(1, "paper8")
    game.state[:] = state
    learner = int(state[PLAYER])
    decisions, rounds = 1, 0
    result = game.step(first)
    rounds += int(result[1])
    while game.state[PHASE] != TERMINAL:
        legal = np.flatnonzero(game.legal())
        solved = solved_koi_action(game)
        if solved is not None:
            action = solved
        elif game.state[PLAYER] == learner:
            action = legal[0]
        elif opponent_id == 0:
            action = legal[-1]
        else:
            action = legal[min(1, len(legal) - 1)]
        result = game.step(int(action))
        rounds += int(result[1])
        decisions += 1
    margin = int(game.state[POINTS] - game.state[POINTS + 1]) * (1 - 2 * learner)
    return np.sign(margin), margin, decisions, rounds


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_opponent_continuation_matches_independent_reference(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    setup(18968001)
    env = BatchEnv(4, device, "paper8", seed=18968001)
    for _ in range(23):
        env.step(env.observe()[1].int().argmax(-1))
    states = env.state.clone()
    legal = env.observe()[1]
    proposals = torch.stack(
        [legal.int().argmax(-1), 49 - legal.flip(1).int().argmax(-1)], dim=1
    )
    ids = torch.tensor([0, 1, 0, 1], device=device)
    policy = OpponentContinuation(
        OrderedLegalActor(1),
        OrderedLegalActor(-1),
        OrderedLegalActor(2),
        states[:, PLAYER].repeat_interleave(2),
        ids.repeat_interleave(2),
    )
    wins, points, work = PairedWorkspace(4, device)(states, proposals, policy)
    decisions, rounds = 0, 0
    for index, state in enumerate(states.cpu().numpy()):
        for branch in (0, 1):
            expected = reference_outcome(
                state, int(proposals[index, branch]), int(ids[index])
            )
            assert wins[index, branch] == expected[0]
            assert points[index, branch] == expected[1]
            decisions += expected[2]
            rounds += expected[3]
    assert work.decisions == decisions
    assert work.round_endings == rounds
    assert work.completed_branches == 8
    assert torch.equal(states, env.state)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_mixture_collects_only_learner_disagreements(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    seed = 18969001
    setup(seed)
    parent = read_checkpoint(PARENTS[0][0])[0]
    point = PublishedActor.from_published("point", "cpu").checkpoint()
    wp = PublishedActor.from_published("wp", device)
    bank = ExpertBank([parent["actor"], point], device)
    env = BatchEnv(16, device, "paper8", seed=seed)
    states, public, groups, opponents, work = collect_roots(
        env, bank, wp, 32, torch.Generator(device=device).manual_seed(seed)
    )
    lane = states[:, SEED] - seed
    assert torch.equal(states[:, PLAYER].long(), (lane + states[:, EPISODE]) % 2)
    assert torch.equal(opponents, (lane // 2 + states[:, EPISODE] // 2) % 2)
    assert opponents.unique().tolist() == [0, 1]
    assert states[:, PLAYER].unique().tolist() == [0, 1]
    assert public.shape == (32, 562)
    assert (public[:, 560] != public[:, 561]).all()
    assert torch.equal(groups, states[:, [SEED, EPISODE]])
    assert work["decisions"] == work["learner_decisions"] + work["opponent_decisions"]
    # Stored branch roots must reproduce only the usual public scorer inputs.
    root_env = BatchEnv(32, device, "paper8")
    root_env.state.copy_(states)
    rebuilt, _ = bank.observe(root_env)
    torch.testing.assert_close(rebuilt, public, rtol=0, atol=0)
