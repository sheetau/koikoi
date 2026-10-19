import math

import numpy as np
import torch

from koikoi.layout import PHASE
from koikoi.published_actor import PublishedActor
from koikoi.reference import ReferenceEnv
from scripts.koi_temperature_mechanism import (
    CORPUS,
    describe,
    public_inputs,
    score_public,
)
from scripts.koi_temperature_ppo import (
    KoiTemperatureAgent,
    KoiTemperatureConfig,
    KoiTemperatureTrainer,
)


def test_probability_transformation_and_learned_gap_are_separate():
    scores = np.full((3, 50), -np.inf, dtype=np.float32)
    scores[:, 48] = 0
    scores[:, 49] = 0.05 * math.log(3)
    legal = np.isfinite(scores)
    legal[2, 49] = False
    phases = np.full(3, 3)
    actions = np.array([49, 49, -1])
    original = np.array([48, 49, -1])
    result = describe(scores, legal, phases, actions, original, 0.2)["3"]
    assert result["all_rows"] == 3 and result["unforced_rows"] == 2
    assert result["changed_greedy_actions"] == result["stop_to_continue"] == 1
    assert result["continue_to_stop"] == 0
    expected = -0.75 * math.log(0.75) - 0.25 * math.log(0.25)
    assert abs(result["common_temperature"]["mean_entropy"] - expected) < 1e-7
    assert result["declared_training_temperature"]["mean_entropy"] > expected
    scaled = describe(scores * 4, legal, phases, actions, original, 0.2)["3"]
    assert (
        abs(scaled["declared_training_temperature"]["mean_entropy"] - expected) < 1e-7
    )
    assert scaled["changed_greedy_actions"] == 1
    assert result["common_temperature"]["rows"] == 2


def test_actual_public_features_hidden_invariance_and_deployment_actions(tmp_path):
    torch.set_num_threads(2)
    with np.load(CORPUS, allow_pickle=False) as corpus:
        masks = corpus["masks"] & (
            (corpus["guards"][:, None] < 0)
            | (np.arange(50)[None] == corpus["guards"][:, None])
        )
        indices = [
            np.flatnonzero(
                (corpus["sources"] >= 0)
                & (corpus["states"][:, PHASE] == phase)
                & (masks.sum(1) > 1)
            )[0]
            for phase in range(4)
        ]
        states = corpus["states"][indices].copy()
        hidden = corpus["states"][corpus["hidden_pairs"][:3].flatten()].copy()
    public, phases, mask, _, _ = public_inputs(states)
    variants, vp, vm, _, vg = public_inputs(hidden)
    for i in range(0, 6, 2):
        assert torch.equal(variants[i], variants[i + 1])
        assert (
            torch.equal(vm[i], vm[i + 1]) and vp[i] == vp[i + 1] and vg[i] == vg[i + 1]
        )
    trainer = KoiTemperatureTrainer(
        KoiTemperatureConfig(device="cpu", koi_temperature=0.2)
    )
    path = tmp_path / "initial.pt"
    trainer.save(path)
    actor = PublishedActor(trainer.model.actor.checkpoint(), "cpu").eval()
    scores, actions, calls = score_public(actor, public, phases, mask)
    agent = KoiTemperatureAgent(path)
    assert calls == 4
    for i, state in enumerate(states):
        env = ReferenceEnv(1, "paper8")
        env.state[:] = state
        assert actions[i] == agent.act(env)
        assert np.isfinite(scores[i, mask[i].numpy()]).all()
        assert np.isneginf(scores[i, ~mask[i].numpy()]).all()
