"""Coverage, public-key cache consistency and independent scoring arithmetic."""

import copy
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from scripts.complete_sampled_history_exact import (
    aggregate,
    batch_from_record,
    comparisons,
    insert_cache,
    select_records,
)
from scripts.run_arbitration_study import read


def test_every_existing_eligible_root_is_retained_without_score_selection():
    records = [
        json.loads(s)
        for s in Path("runs/sampled_history_diagnostic/records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    selected = select_records(records)
    assert {(r["parent"], r["case_index"]) for r in selected} == {
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 256),
        (1, 257),
        (1, 512),
        (1, 513),
        (1, 514),
        (1, 515),
        (1, 768),
        (1, 769),
        (1, 770),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 256),
        (2, 257),
        (2, 258),
        (2, 512),
        (2, 513),
        (2, 514),
        (2, 515),
        (2, 768),
        (2, 770),
    }
    for rows in (records[:-1], records[:-1] + [records[0]]):
        with pytest.raises(ValueError):
            select_records(rows)
    changed = copy.deepcopy(records)
    next(r for r in changed if r["status"] == "eligible")["status"] = "ineligible"
    with pytest.raises(ValueError):
        select_records(changed)


def test_real_duplicate_public_tables_agree_and_tamper_is_rejected():
    root = Path("runs/sampled_history_exact_check")
    first, second = [
        batch_from_record(read(root / f"parent{p}_point_support.json")) for p in (1, 2)
    ]
    cache = {}
    insert_cache(cache, first)
    insert_cache(cache, second)
    assert len(cache) == 1 and first.root_key == second.root_key
    damaged = read(root / "parent1_point_support.json")
    damaged["compatible"][0][0] = not damaged["compatible"][0][0]
    with pytest.raises(ValueError, match="Conflicting"):
        insert_cache(cache, batch_from_record(damaged))
    damaged = read(root / "parent1_point_support.json")
    damaged["hands"][0] = damaged["hands"][1]
    with pytest.raises(ValueError, match="Invalid"):
        batch_from_record(damaged)


def test_brier_and_sampling_errors_match_exact_fraction_reference():
    exact, sample, truth = np.zeros(48), np.zeros(48), np.zeros(48)
    exact[:2], sample[:2], truth[0] = [0.75, 0.25], [0.5, 0.5], 1
    estimates = {
        "joint": {
            "status": "estimated",
            "card_probabilities": exact.tolist(),
            "distinct_supported_hands": 2,
        }
    }
    sampled = {
        "joint": {
            "status": "estimated",
            "card_probabilities": sample.tolist(),
            "effective_proposals": 2,
            "card_brier": 0.25,
        }
    }
    result = comparisons(estimates, sampled, [0, 1], truth)["joint"]
    assert result["exact_card_brier"] == float(
        (Fraction(3, 4) - 1) ** 2 / 2 + Fraction(1, 4) ** 2 / 2
    )
    assert (
        result["mean_absolute_sampling_error"]
        == result["maximum_absolute_sampling_error"]
        == 0.25
    )
    sampled["joint"] = {"status": "zero_sampled_mass"}
    missing = comparisons(estimates, sampled, [0, 1], truth)["joint"]
    assert missing["exact_card_brier"] == 1 / 16
    assert (
        missing["sampled_status"] == "zero_sampled_mass"
        and "sampled_card_brier" not in missing
    )


def test_aggregate_reports_explicit_failures_without_imputing_uniform():
    results = []
    for parent, opponent, error in [(1, "wp", 1 / 16), (2, "point", 1 / 4)]:
        result = {"parent": parent, "opponent": opponent, "comparison": {}}
        for name in ("uniform", "joint", "identified"):
            result["comparison"][name] = {
                "status": "estimated",
                "exact_card_brier": error,
                "sampled_card_brier": 0.25,
            }
        results.append(result)
    results[1]["comparison"]["identified"].pop("sampled_card_brier")
    out = aggregate(results)
    assert out["all"]["beliefs"]["identified"] == {
        "exact_estimated": 2,
        "sampled_estimated": 1,
        "exact_mean_card_brier": 5 / 32,
        "sampled_mean_card_brier": 0.25,
    }
    assert out["parent1"]["beliefs"]["uniform"]["exact_mean_card_brier"] == 1 / 16
    assert out["point"]["beliefs"]["uniform"]["exact_mean_card_brier"] == 1 / 4
