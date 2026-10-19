import copy

import numpy as np
import pytest

from scripts.audit_published_duration_evaluation import replay
from scripts.evaluate_published_duration import cases, play_match, prepare


class FirstLegal:
    def act(self, env):
        return int(np.flatnonzero(env.legal())[0])


def test_seat_deal_schedule_and_independent_native_replay():
    schedule = cases(2, "point", 2, 99040001)
    assert [(r["seed"], r["seat"]) for r in schedule] == [
        (99040001, 0),
        (99040001, 1),
        (99040002, 0),
        (99040002, 1),
    ]
    agent = FirstLegal()
    row = play_match(agent, agent, schedule[0])
    assert replay(row) == row["work"]
    for field in ("points", "endpoint", "work", "actions"):
        bad = copy.deepcopy(row)
        if field == "points":
            bad[field] += 1
        elif field == "endpoint":
            bad[field][0] += 1
        elif field == "work":
            bad[field]["round_endings"] += 1
        else:
            bad[field][0] = 50
        with pytest.raises(AssertionError):
            replay(bad)


def test_training_completion_is_required_before_main_plan(tmp_path, monkeypatch):
    import scripts.evaluate_published_duration as module

    def incomplete(path):
        if path.name == "protocol.json":
            return {"inputs": {}}
        raise FileNotFoundError("Training is still active")

    monkeypatch.setattr(module, "read", incomplete)
    monkeypatch.setattr(module, "sha", lambda _: "test")
    with pytest.raises(FileNotFoundError, match="still active"):
        prepare(tmp_path / "not_created")
    assert not (tmp_path / "not_created").exists()
