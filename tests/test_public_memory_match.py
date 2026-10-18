"""Complete, paired CPU matches preserve prefixes and public-event timing."""

import numpy as np
import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.env import BatchEnv
from koikoi.layout import POINTS
from koikoi.reference import ReferenceEnv
from koikoi.upstream import PublishedAgent
from scripts.public_memory_match import play_memory_match
from scripts.public_memory_tail import (
    MemoryTailAgent,
    MemoryTailTrainer,
    candidate_code,
)
from scripts.revealed_opponent_models import StrictPublishedPrediction
from tests.test_final_round_training import PARENT, natural_bank
from tests.test_public_memory_tail import config


def test_initial_full_matches_prefix_and_memory_lifecycle(tmp_path):
    trainer = MemoryTailTrainer.from_parent(
        config("cpu"), PARENT, *natural_bank(), torch.tensor([-1, 0, 1, 0])
    )
    path = tmp_path / "initial.pt"
    trainer.save(path)
    models = {n: PublishedAgent(n) for n in ("wp", "point")}
    actor = MemoryTailAgent(
        path, {n: StrictPublishedPrediction(a) for n, a in models.items()}
    )
    parent = NeuralAgent(PARENT, deterministic=True)
    native = BatchEnv(1, "cpu", "paper8")
    for seat in (0, 1):
        for name, model in models.items():
            case = {"seat": seat, "seed": 31010001, "opponent": name}
            opponent = EndgameAgent(model)
            original = play_memory_match(case, parent, opponent, native)
            actual = play_memory_match(case, actor, opponent, native)
            # All decisions, including final-round card/capture/koi decisions,
            # agree with the original parent before any training.
            assert actual["actions"] == original["actions"]
            assert actual["terminal_points"] == original["terminal_points"]
            assert actual["work"] == original["work"]
            ref = ReferenceEnv(case["seed"], "paper8")
            for action in actual["actions"]:
                ref.step(action)
            np.testing.assert_array_equal(
                actual["terminal_points"], ref.state[POINTS : POINTS + 2]
            )
            for event in actual["memory"]["events"]:
                assert event["round"] < 7
                assert event["available_after_decision"] <= (
                    actual["prefix_length"] or len(actual["actions"])
                )
            if actual["memory"]["tail_code"] is not None:
                assert actual["memory"]["tail_code"] == candidate_code(
                    actual["memory"]["candidates"]
                )
