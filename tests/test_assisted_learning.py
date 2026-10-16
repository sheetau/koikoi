import numpy as np
import pytest
import torch

from koikoi.agents import NeuralAgent
from koikoi.layout import CONTINUE, KOI, PHASE, PLAYER, POINTS, ROUND, ROUNDS, STOP
from koikoi.reference import ReferenceEnv
from koikoi.search import SearchAgent
from koikoi.train import Config, Trainer


def final_koi_states():
    reference = ReferenceEnv(47)
    rng = np.random.default_rng(12)
    for _ in range(1000):
        if reference.state[PHASE] == KOI:
            break
        reference.step(int(rng.choice(np.flatnonzero(reference.legal()))))
    else:
        raise AssertionError("No koi decision found")
    states = []
    for desired_score in (31, 29):
        env = reference.clone()
        player = int(env.state[PLAYER])
        own = desired_score - env.yaku(player)
        env.state[ROUND] = env.state[ROUNDS] - 1
        env.state[POINTS + player] = own
        env.state[POINTS + 1 - player] = 60 - own
        states.append(env)
    return states


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_on_policy_mask_and_checkpoint_deployment(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = Config(
        device=device,
        profile="paper8",
        envs=2,
        horizon=1,
        width=32,
        epochs=1,
        minibatch=2,
        endgame=True,
    )
    trainer = Trainer(config)
    states = final_koi_states()
    trainer.env.state.copy_(
        torch.from_numpy(np.stack([env.state for env in states])).to(device)
    )
    batch = trainer.collect()
    assert batch["action"].cpu().tolist() == [STOP, CONTINUE]
    assert batch["mask"].sum(1).cpu().tolist() == [1, 1]
    assert batch["old_logp"].cpu().tolist() == [0, 0]
    path = tmp_path / "assisted.pt"
    trainer.save(path)
    agent = NeuralAgent(path, device=device)
    assert agent.requires_endgame
    assert [agent.act(env) for env in states] == [STOP, CONTINUE]
    search = SearchAgent(agent, particles=2, horizon=2, use_belief=False)
    assert [search.act(env) for env in states] == [STOP, CONTINUE]
    assert search.simulated_decisions == 0


def test_point_reward_experiment_cannot_assume_match_dominance():
    with pytest.raises(ValueError, match="match outcome"):
        Config(reward="round", endgame=True)
