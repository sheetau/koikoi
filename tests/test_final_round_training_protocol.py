import numpy as np

from koikoi.env import BatchEnv
from scripts.audit_final_round_training_study import grid, interval, key, replay
from scripts.final_round_actual_match import play_final_round_match
from scripts.final_round_training import prefix_to_final_round
from scripts.run_final_round_training_study import cases
from tests.test_final_round_actual_match import ContinueFirst, StopFirst


def test_two_parent_shared_pairs_and_disjoint_source_evaluation():
    source, evaluation = set(), set()
    for parent in (1, 2):
        for split, count, destination in (
            ("source", 1024, source),
            ("evaluation", 4096, evaluation),
        ):
            rows = cases(parent, split)
            assert len(rows) == count
            assert [key(row) for row in rows] == grid(parent, split == "source")
            for opponent in ("wp", "point"):
                left = [
                    r["seed"]
                    for r in rows
                    if r["opponent"] == opponent and r["seat"] == 0
                ]
                right = [
                    r["seed"]
                    for r in rows
                    if r["opponent"] == opponent and r["seat"] == 1
                ]
                assert left == right and len(set(left)) == len(left)
            destination.update(r["seed"] for r in rows)
    assert len(source) == 512 and len(evaluation) == 2048
    assert source.isdisjoint(evaluation) and max(source | evaluation) < 100000000
    assert interval(np.zeros(1024)) == [0.0, 0.0]
    assert interval(np.ones(1024)) == [1.0, 1.0]


def test_independent_prefix_replay_and_early_terminal():
    native = BatchEnv(1, "cpu", "paper8")
    for seed, actor in ((31010001, StopFirst()), (19, ContinueFirst())):
        root, row = prefix_to_final_round(
            {"seed": seed, "seat": 0}, actor, actor, native
        )
        state, actions = replay(row, prefix=True)
        assert actions == row["actions"]
        if root is not None:
            np.testing.assert_array_equal(state, root)
        _, completed = play_final_round_match(
            {"seed": seed, "seat": 0}, actor, actor, native
        )
        _, earlier = replay(completed, prefix=False)
        assert earlier == row["actions"]
