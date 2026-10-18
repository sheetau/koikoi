"""Fresh comparison grid and independently verified public stopping times."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from koikoi.agents import EndgameAgent
from koikoi.env import BatchEnv
from koikoi.upstream import PublishedAgent
from scripts.audit_public_continuation_study import audit_roots, interval, replay_match
from scripts.public_memory_continuation_match import (
    ContinuationAgent,
    play_continuation_match,
)
from scripts.revealed_opponent_models import StrictPublishedPrediction
from scripts.run_public_continuation_study import (
    cases,
    primary_success,
    stratified_interval,
)


def test_fresh_grid_and_stratified_conjunction():
    rows = [r for name in ("wp", "point") for r in cases(1, name)]
    other = [r for name in ("wp", "point") for r in cases(2, name)]
    assert len(rows) == 4096 and len({r["seed"] for r in rows}) == 2048
    assert [(r["seed"], r["seat"]) for r in rows] == [
        (r["seed"], r["seat"]) for r in other
    ]
    assert min(r["seed"] for r in rows) == 93110001
    assert max(r["seed"] for r in rows) == 93121024
    values = np.array([[-0.25, 0, 0.25, 0.5], [0.25, 0.5, -0.5, -0.25]])
    assert stratified_interval(values) == interval(values)
    assert stratified_interval(np.array([[1] * 8, [-1] * 8])) == [0, 0]
    contrasts = [
        {
            "parent": p,
            "scope": "mixture",
            "comparison": c,
            "win": {"interval95": [0.01, 0.03]},
        }
        for p in (1, 2)
        for c in ("public-minus-masked", "public-minus-parent")
    ]
    assert primary_success(contrasts)
    contrasts[0]["win"]["interval95"][0] = 0
    assert not primary_success(contrasts)


def test_independent_all_source_root_reconstruction():
    total = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    for parent in (1, 2):
        for key, value in audit_roots(parent).items():
            total[key] += value
    assert total == {
        "decisions": 212212,
        "round_endings": 14101,
        "completed_matches": 192,
    }


def test_independent_replay_rejects_changed_public_timing_and_terminal():
    torch.set_num_threads(2)
    path = Path("runs/public_memory_continuation_capability/public/initial.pt")
    models = {n: PublishedAgent(n) for n in ("wp", "point")}
    actor = ContinuationAgent(
        path, {n: StrictPublishedPrediction(a) for n, a in models.items()}
    )
    native = BatchEnv(1, "cpu", "paper8")
    row = play_continuation_match(
        {"seed": 88110001, "seat": 0}, actor, EndgameAgent(models["wp"]), native
    )
    prefix, work, checks = replay_match(row)
    assert work["completed_matches"] == 1 and checks > 0
    assert 0 < len(prefix) < len(row["actions"])
    assert row["memory"]["activation"]["round"] < 7
    for mutation in ("activation", "first", "event", "input", "terminal"):
        changed = copy.deepcopy(row)
        if mutation == "activation":
            changed["memory"]["activation"]["decision"] -= 1
        elif mutation == "first":
            changed["memory"]["first_identification"]["available_after_decision"] -= 1
        elif mutation == "event":
            changed["memory"]["events"][0]["available_after_decision"] -= 1
        elif mutation == "input":
            changed["memory"]["events"][0]["checks"][0]["features_sha256"] = "0" * 64
        else:
            changed["terminal_points"][0] += 1
        with pytest.raises((AssertionError, KeyError)):
            replay_match(changed)
