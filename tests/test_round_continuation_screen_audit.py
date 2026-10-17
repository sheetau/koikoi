"""Independent auditor fixtures; no model inference or CUDA work."""

import copy
import hashlib
import json

import numpy as np
import pytest

from scripts.audit_round_continuation_screen import (
    interval,
    raw_pairs,
    statistics,
    verify_metrics,
    verify_table_identity,
)


def table_fixture():
    spec = {
        "coefficients": [[2.0, 0.1] for _ in range(8)],
        "source": {"parent_sha256": "0" * 64},
    }
    digest = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"round_continuation": spec}, {"round_continuation_sha256": digest}


def test_embedded_table_and_control_identities():
    cfg, prov = table_fixture()
    verify_table_identity(cfg, prov, "0" * 64)
    verify_table_identity({"round_continuation": None}, {}, "0" * 64)


@pytest.mark.parametrize("fault", ["coefficients", "parent", "fingerprint", "control"])
def test_embedded_table_identity_drift_rejected(fault):
    cfg, prov = table_fixture()
    if fault == "coefficients":
        cfg["round_continuation"]["coefficients"][1][0] += 0.1
    elif fault == "parent":
        cfg["round_continuation"]["source"]["parent_sha256"] = "1" * 64
    elif fault == "fingerprint":
        prov["round_continuation_sha256"] = "1" * 64
    else:
        cfg["round_continuation"] = None
    with pytest.raises(AssertionError):
        verify_table_identity(cfg, prov, "0" * 64)


def report():
    return {
        "pairs": 2,
        "records": [
            {
                "seed": 1000001,
                "seat0_score": 0,
                "seat1_score": 0,
                "seat0_point_difference": -4,
                "seat1_point_difference": -2,
            },
            {
                "seed": 1000000,
                "seat0_score": 1,
                "seat1_score": 1,
                "seat0_point_difference": 2,
                "seat1_point_difference": 6,
            },
        ],
    }


def test_bootstrap_known_two_pair_endpoints_and_zero_difference():
    score, points = raw_pairs(report(), 2)
    np.testing.assert_array_equal(score, [1, 0])
    np.testing.assert_array_equal(points, [4, -3])
    assert statistics(score, points) == {
        "score_mean": 0.5,
        "score_ci95": [0, 1],
        "point_mean": 0.5,
        "point_ci95": [-3, 4],
    }
    assert interval(score - score) == [0, 0]
    assert interval(np.array([0.25] * 5)) == [0.25, 0.25]


@pytest.mark.parametrize(
    "fault", ["duplicate", "missing", "outcome", "odd_points", "nan"]
)
def test_invalid_raw_match_records_rejected(fault):
    value = copy.deepcopy(report())
    if fault == "duplicate":
        value["records"][0]["seed"] = 1000000
    elif fault == "missing":
        value["records"].pop()
    elif fault == "outcome":
        value["records"][0]["seat0_score"] = 1
    elif fault == "odd_points":
        value["records"][0]["seat0_point_difference"] = -3
    else:
        value["records"][0]["seat0_point_difference"] = float("nan")
    with pytest.raises(AssertionError):
        raw_pairs(value, 2)


def metric_fixture():
    cfg = {"envs": 2, "horizon": 8, "epochs": 3, "minibatch": 4}
    logs = [
        {"updates": 1, "decisions": 8, "rounds": 2, "matches": 2, "collection_rows": 8},
        {
            "updates": 2,
            "decisions": 17,
            "rounds": 4,
            "matches": 4,
            "collection_rows": 18,
        },
        {
            "updates": 3,
            "decisions": 19,
            "rounds": 6,
            "matches": 6,
            "collection_rows": 20,
        },
    ]
    return cfg, logs


def test_partial_minibatch_steps_are_summed_per_update():
    cfg, logs = metric_fixture()
    # 8, 9, 2 valid rows mean 2, 3, 1 minibatches, NOT ceil(19/4).
    assert verify_metrics(logs, cfg, logs[-1], expected_updates=3) == 18


@pytest.mark.parametrize(
    "fault",
    [
        "missing_update",
        "matches",
        "padding_quantum",
        "padding_too_small",
        "horizon",
        "nonfinite",
        "final_count",
    ],
)
def test_malformed_variable_budgets_rejected(fault):
    cfg, logs = metric_fixture()
    counts = logs[-1].copy()
    if fault == "missing_update":
        logs.pop(0)
    elif fault == "matches":
        logs[0]["matches"] = 1
    elif fault == "padding_quantum":
        logs[1]["collection_rows"] = 19
    elif fault == "padding_too_small":
        logs[0]["collection_rows"] = 6
    elif fault == "horizon":
        cfg["horizon"] = 3
    elif fault == "nonfinite":
        logs[0]["policy_loss"] = float("nan")
    else:
        counts["decisions"] += 1
    with pytest.raises(AssertionError):
        verify_metrics(logs, cfg, counts, expected_updates=3)
