"""Research-contract checks; synthetic arrays are not strength evidence."""

from copy import deepcopy

import numpy as np
import pytest

from scripts import export_public_stock_robustness as exporter
from scripts.public_stock_robustness import (
    ALL_THREE,
    CONTROL,
    FIXED,
    PREVIEW_CHOICE,
    SOURCES,
    calibrate_particles,
    validate_result,
    work_admissibility,
)


def test_calibration_uses_work_and_rounds_half_up_exactly():
    assert calibrate_particles(4000 * 64 * 32, 64) == [0, 15, 60]
    assert calibrate_particles(24000 * 64 * 32, 64) == [0, 3, 10]
    # Floating-point division must not change an exact half-up tie.
    for per_particle, low, high in ((120000, 2, 2), (24000, 3, 10)):
        if low == high:
            with pytest.raises(ValueError, match="distinct"):
                calibrate_particles(per_particle * 64 * 32, 64)
        else:
            assert calibrate_particles(per_particle * 64 * 32, 64) == [0, low, high]


@pytest.mark.parametrize("transitions,matches", [(0, 64), (-1, 64), (1, 64), (1000, 32), (float("inf"), 64)])
def test_invalid_or_saturated_calibration_does_not_launch_holdout(transitions, matches):
    with pytest.raises(ValueError):
        calibrate_particles(transitions, matches)


def example_endpoint(tmp_path):
    label = f"{PREVIEW_CHOICE}_parent0"
    folder = tmp_path / "calibration"
    system = {
        "profile": PREVIEW_CHOICE, "parent": 0,
        "checkpoint": str(tmp_path / "parent0.pt"), "checkpoint_sha256": "actor0",
        "opponent": str(tmp_path / "parent1.pt"), "opponent_sha256": "actor1",
        "table": str(tmp_path / "table.npz"), "round_table_sha256": "table",
        "round_table_metadata_sha256": "table-meta",
    }
    study = {**deepcopy(FIXED), "systems": {label: system}, "sources": {p: "source" for p in SOURCES}}
    receipt_sources = (
        "scripts/evaluate_public_stock_rounds.py", "koikoi/public_stock_residual.py",
        "koikoi/public_stock_round_value.py", "koikoi/public_stock_search.py",
        "koikoi/public_stock.py", "koikoi/public_stock_native.py",
        "koikoi/native/public_stock.h", "koikoi/native/core.h",
    )
    result = {
        **{k: v for k, v in system.items() if k not in ("parent", "checkpoint", "opponent", "table")},
        "config": {
            "command": "evaluate", "particles": 32, "pairs": 32,
            "deal_seed": 1200000000, "out": str(folder),
            **{k: system[k] for k in ("checkpoint", "opponent", "table")},
            **{k: FIXED[k] for k in ("batch", "horizon", "confidence", "search_seed")},
        },
        "source_hashes": {p: "source" for p in receipt_sources},
        "matches": 64, "rounds": 256, "decisions": 2048,
        "python_replay_decisions": 2048, "roots": 10,
        "rollouts": 640, "transitions": 6400,
    }
    return result, study, label, folder


def test_correct_changed_opponent_receipt_is_accepted(tmp_path):
    result, study, label, folder = example_endpoint(tmp_path)
    validate_result(result, study, label, "calibration", 32, folder)


@pytest.mark.parametrize("field,value", [
    ("opponent_sha256", "actor0"),
    ("round_table_metadata_sha256", "other-table-meta"),
    ("round_table_sha256", "other-table"),
    ("checkpoint_sha256", "other-actor"),
    ("matches", 128),
    ("rollouts", 641),
    ("transitions", 1000000),
])
def test_wrong_identities_or_work_are_rejected(tmp_path, field, value):
    result, study, label, folder = example_endpoint(tmp_path)
    result[field] = value
    with pytest.raises(ValueError):
        validate_result(result, study, label, "calibration", 32, folder)


@pytest.mark.parametrize("field,value", [
    ("deal_seed", 1210000000), ("particles", 128), ("horizon", 32),
    ("confidence", 1.0), ("opponent", "self"), ("search_seed", 971),
])
def test_wrong_cohort_or_search_settings_are_rejected(tmp_path, field, value):
    result, study, label, folder = example_endpoint(tmp_path)
    result["config"][field] = value
    with pytest.raises(ValueError):
        validate_result(result, study, label, "calibration", 32, folder)


def test_incomplete_source_receipt_is_rejected(tmp_path):
    result, study, label, folder = example_endpoint(tmp_path)
    del result["source_hashes"]["koikoi/public_stock_search.py"]
    with pytest.raises(ValueError, match="source"):
        validate_result(result, study, label, "calibration", 32, folder)


def test_pre_run_receipt_cannot_be_replaced_by_final_metadata(tmp_path, monkeypatch):
    result, study, label, folder = example_endpoint(tmp_path)
    monkeypatch.setattr(exporter, "reduce_endpoint", lambda *args: (np.zeros(32), result))
    monkeypatch.setattr(exporter, "read", lambda *args: {"opponent_sha256": "actor0"})
    with pytest.raises(ValueError, match="Pre-run"):
        exporter.reduce_checked(folder, study, label, "calibration", 32, {})


def test_paired_comparison_keeps_shared_stream_noise_cancelled():
    base = np.array([0, 0.25, 0.5, 0.75])
    control = np.stack([base, base, base])
    candidate = np.stack([base, base + 0.125, base + 0.25])
    data = {
        CONTROL: np.stack([control, control]),
        PREVIEW_CHOICE: np.stack([candidate, candidate]),
        ALL_THREE: np.stack([candidate, candidate]),
    }
    indices = np.random.default_rng(31).integers(0, 4, (500, 4))
    comparisons, passed, ranking = exporter.compare(data, indices)
    assert len(comparisons) == 14
    row = comparisons[f"{PREVIEW_CHOICE}_parent0_gain"]
    assert row == {"mean": 0.25, "lower": 0.25, "upper": 0.25}
    assert all(passed.values())
    assert ranking == "unresolved"


def test_opposite_parent_results_are_not_pooled_or_used_to_force_a_ranking():
    data = {
        CONTROL: np.full((2, 3, 4), 0.5),
        PREVIEW_CHOICE: np.repeat(np.array([[0.5, 0.625, 0.875], [0.5, 0.5, 0.375]])[:, :, None], 4, 2),
        ALL_THREE: np.repeat(np.array([[0.25, 0.375, 0.625], [0.5, 0.625, 0.75]])[:, :, None], 4, 2),
    }
    indices = np.tile(np.arange(4), (100, 1))
    comparisons, passed, ranking = exporter.compare(data, indices)
    assert not passed[PREVIEW_CHOICE]
    assert passed[ALL_THREE]
    assert comparisons["preview_minus_all_parent0"]["mean"] == 0
    assert comparisons["preview_minus_all_parent1"]["mean"] == -0.375
    assert ranking == "unresolved"


def test_work_failures_cannot_cancel_across_endpoints():
    assert work_admissibility(110 * 64, 64, 100)["passed"]
    checks = [work_admissibility(n * 64, 64, 100) for n in (89, 111)]
    assert sum(r["actual_per_match"] for r in checks) / 2 == 100
    assert not any(r["passed"] for r in checks)
