import json

import numpy as np

from koikoi.layout import PHASE, POINTS, TERMINAL
from koikoi.reference import ReferenceEnv
from scripts.koi_temperature_analysis import analyse, cases
from scripts.koi_temperature_ppo import KoiTemperatureConfig, KoiTemperatureTrainer
from scripts.run_arbitration_study import read
from scripts.run_koi_temperature_study import evaluate_cell


def test_paired_allocation_and_strict_primary_criterion():
    a, b = cases(1, "wp"), cases(2, "wp")
    assert len(a) == len(b) == 4096
    assert {(r["seed"], r["seat"]) for r in a} == {(r["seed"], r["seat"]) for r in b}
    assert not {r["seed"] for r in a} & {r["seed"] for r in cases(1, "point")}
    assert all(r["seed"] < 100000000 for r in a + cases(1, "point"))
    points = np.zeros((2, 3, 2, 16, 2), dtype=int)
    points[:, 2] = 2
    result = analyse(points, bootstrap_replicates=50)
    assert result["replicated_strength_gain"] and len(result["contrasts"]) == 18
    assert all(
        r["win_change"] == 0.5 for r in result["contrasts"] if r["left"] == "calibrated"
    )
    points[1, 2, 1] = -2
    assert not analyse(points, bootstrap_replicates=50)["replicated_strength_gain"]


def test_point_gains_cannot_replace_win_gain():
    points = np.full((2, 3, 2, 16, 2), 2, dtype=int)
    points[:, 2] = 10
    result = analyse(points, bootstrap_replicates=50)
    assert not result["replicated_strength_gain"]
    assert all(
        r["point_change"] == 8 for r in result["contrasts"] if r["left"] == "calibrated"
    )


def test_actual_evaluation_pipeline_and_independent_replay(tmp_path):
    trainer = KoiTemperatureTrainer(
        KoiTemperatureConfig(device="cpu", koi_temperature=0.21989584747027907)
    )
    path = tmp_path / "initial.pt"
    trainer.save(path)
    stem = evaluate_cell(
        tmp_path, 1, "calibrated", "wp", path, pairs=2, seed_base=88110001
    )
    report = read(tmp_path / "evaluation" / (stem + ".json"))
    rows = [
        json.loads(line)
        for line in (tmp_path / "evaluation" / (stem + ".jsonl"))
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    totals = dict.fromkeys(("decisions", "round_endings", "full_matches"), 0)
    assert len(rows) == 4
    for row in rows:
        env = ReferenceEnv(row["seed"], "paper8")
        rounds = 0
        for action in row["actions"]:
            assert env.state[PHASE] != TERMINAL and env.legal()[action]
            rounds += int(env.step(action)[1])
        assert env.state[PHASE] == TERMINAL and env.state.tolist() == row["endpoint"]
        assert row["points"] == int(env.state[POINTS] - env.state[POINTS + 1]) * (
            1 - 2 * row["seat"]
        )
        assert row["work"] == {
            "decisions": len(row["actions"]),
            "round_endings": rounds,
            "full_matches": 1,
        }
        for key, value in row["work"].items():
            totals[key] += value
    assert totals == report["work"]
