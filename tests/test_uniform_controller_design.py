from collections import Counter
from fractions import Fraction

import pytest

from koikoi.layout import DEALER
from koikoi.reference import ReferenceEnv
from scripts import uniform_controller_design as design
from scripts.certified_gain_bounds import binomial_lower


def test_fixed_population_schedule_and_unchanged_initial_dealer():
    config, cases = design.specification(), design.cases()
    assert len(cases) == config["source_matches"] == 8192
    assert [c["index"] for c in cases] == list(range(8192))
    assert Counter(c["population"] for c in cases) == dict.fromkeys(
        design.POPULATIONS, 2048
    )
    assert all("seat" not in c for c in cases)  # Sampled independently by collector.
    assert ReferenceEnv(config["environment_seed"]).state[DEALER] == 0
    assert 4 * Fraction(config["population_error"]) == Fraction(config["family_error"])


def test_full_denominators_and_exact_equal_weight_mixtures():
    values = {name: [Fraction(0), Fraction(1, 25)] for name in design.POPULATIONS}
    values["0:point"] = [Fraction(0), Fraction(0)]
    result = design.summarize(values, expected_n=2)
    # Five thresholds at/below1/25 telescope to1/25 times the same survival bound.
    expected = Fraction(1, 25) * binomial_lower(2, 1, Fraction(1, 560))
    assert Fraction(result["populations"]["0:wp"]["lower"]) == expected
    assert Fraction(result["parent_mixtures"]["0"]["lower"]) == expected / 2
    assert Fraction(result["parent_mixtures"]["0"]["mean"]) == Fraction(1, 100)
    assert Fraction(result["parent_mixtures"]["1"]["mean"]) == Fraction(1, 50)
    assert not result["all_population_lowers_positive"]
    assert not result["criterion_met"]
    with pytest.raises(ValueError, match="count"):
        design.summarize(values, expected_n=3)


def test_practical_effect_gate_requires_both_replicas_and_each_opponent():
    values = {name: [Fraction(1, 25)] * 50 for name in design.POPULATIONS}
    result = design.summarize(values, expected_n=50)
    assert result["criterion_met"]
    values["1:point"] = [Fraction(0)] * 50
    result = design.summarize(values, expected_n=50)
    assert all(x["practical_criterion_met"] for x in result["parent_mixtures"].values())
    assert not result["criterion_met"]  # Mixture alone cannot hide a zero bound.
    values["1:point"] = [Fraction(1, 100000)] * 50
    assert not design.summarize(values, expected_n=50)["criterion_met"]


@pytest.mark.parametrize("invalid", [Fraction(-1, 100), 0.01, True, Fraction(2)])
def test_no_clamping_or_float_gain_inputs(invalid):
    values = {name: [invalid] for name in design.POPULATIONS}
    with pytest.raises(ValueError):
        design.summarize(values, expected_n=1)


def test_missing_or_extra_populations_are_not_silently_pooled():
    values = {name: [Fraction(0)] for name in design.POPULATIONS}
    values["other"] = [Fraction(0)]
    with pytest.raises(ValueError, match="four"):
        design.summarize(values, expected_n=1)
    del values["other"], values["1:point"]
    with pytest.raises(ValueError, match="four"):
        design.summarize(values, expected_n=1)
