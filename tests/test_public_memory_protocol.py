"""Fresh deal grid, stratified estimand and independent public-event replay."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from koikoi.agents import EndgameAgent
from koikoi.env import BatchEnv
from koikoi.upstream import PublishedAgent
from scripts.audit_public_memory_study import interval, replay_match
from scripts.public_memory_match import play_memory_match
from scripts.public_memory_tail import MemoryTailAgent
from scripts.revealed_opponent_models import StrictPublishedPrediction
from scripts.run_public_memory_study import cases, primary_success, stratified_interval


def test_fresh_balanced_grid_and_independent_bootstrap():
    first = [r for n in ("wp", "point") for r in cases(1, n)]
    second = [r for n in ("wp", "point") for r in cases(2, n)]
    assert len(first) == len(second) == 4096
    assert len({r["seed"] for r in first}) == 2048
    assert [(r["seed"], r["seat"]) for r in first] == [
        (r["seed"], r["seat"]) for r in second
    ]
    assert min(r["seed"] for r in first) == 91110001
    assert max(r["seed"] for r in first) == 91121024
    values = np.array([[-0.25, 0, 0.25, 0.5], [0.25, 0.5, -0.5, -0.25]])
    assert stratified_interval(values) == interval(values)
    # Opposite constant strata must cancel, unlike a pooled unstratified draw.
    assert stratified_interval(np.array([[1] * 8, [-1] * 8])) == [0, 0]
    rows = [
        {"scope": "mixture", "comparison": c, "win": {"interval95": [0.01, 0.03]}}
        for p in (1, 2)
        for c in ("public-minus-masked", "public-minus-parent")
    ]
    assert primary_success(rows)
    rows[0]["win"]["interval95"][0] = 0
    assert not primary_success(rows)


def test_independent_match_replay_rejects_future_information():
    torch.set_num_threads(2)
    path = Path("runs/public_memory_tail_capability/public/initial.pt")
    models = {n: PublishedAgent(n) for n in ("wp", "point")}
    actor = MemoryTailAgent(
        path, {n: StrictPublishedPrediction(a) for n, a in models.items()}
    )
    native = BatchEnv(1, "cpu", "paper8")
    row = play_memory_match(
        {"seed": 31010001, "seat": 0}, actor, EndgameAgent(models["wp"]), native
    )
    prefix, work, checks = replay_match(row)
    assert prefix and work["completed_matches"] == 1 and checks > 0
    changed = copy.deepcopy(row)
    changed["memory"]["events"][0]["available_after_decision"] -= 1
    with pytest.raises((AssertionError, KeyError)):
        replay_match(changed)
    changed = copy.deepcopy(row)
    changed["terminal_points"][0] += 1
    with pytest.raises(AssertionError):
        replay_match(changed)
    changed = copy.deepcopy(row)
    changed["memory"]["events"][0]["checks"][0]["features_sha256"] = "0" * 64
    with pytest.raises(AssertionError):
        replay_match(changed)
