from collections import Counter
from fractions import Fraction

import pytest

from scripts.frontier_incremental_design import (
    POPULATIONS,
    THRESHOLDS,
    cases,
    specification,
    summarize,
)


def test_fresh_denominators_and_comparison_are_fixed_before_sources():
    spec, rows = specification(), cases()
    assert len(rows) == spec["source_matches"] == 4096
    assert [r["index"] for r in rows] == list(range(4096))
    assert Counter(r["population"] for r in rows) == dict.fromkeys(POPULATIONS, 1024)
    assert spec["initial_dealer"] == 0 and spec["max_draws"] == 5
    assert spec["phase_decisions"] == spec["hybrid"]["phase_decisions"] == 2000000
    assert "existing symbolic controller" in spec["comparison"]
    assert "practical_mixture_gain" not in spec
    assert tuple(spec["thresholds"]) == THRESHOLDS


def test_fallback_zeros_are_included_and_mixtures_do_not_require_each_matchup():
    values = {p: [Fraction(0)] * 16 for p in POPULATIONS}
    # Only the Point populations improve. Both fixed mixtures still improve.
    for p in ("0:point", "1:point"):
        values[p] = [Fraction(1, 5000)] * 8 + [Fraction(0)] * 8
    report = summarize(values, expected_n=16)
    assert report["criterion_met"] and not report["all_population_lowers_positive"]
    assert report["family_error"] == "1/20"
    for parent in ("0", "1"):
        mixture = report["parent_mixtures"][parent]
        assert Fraction(mixture["mean"]) == Fraction(1, 20000)
        assert Fraction(mixture["lower"]) > 0
        assert (
            Fraction(mixture["lower"])
            == Fraction(report["populations"][parent + ":point"]["lower"]) / 2
        )


def test_zero_gains_and_incomplete_or_invalid_data_cannot_pass():
    values = {p: [Fraction(0)] * 8 for p in POPULATIONS}
    assert not summarize(values, expected_n=8)["criterion_met"]
    values["0:wp"].pop()
    with pytest.raises(ValueError, match="Sample count"):
        summarize(values, expected_n=8)
    values["0:wp"].append(Fraction(-1, 1000))
    with pytest.raises(ValueError, match="nonnegative"):
        summarize(values, expected_n=8)
