"""Prespecified fresh comparison of the selected hybrid with the old controller."""

from fractions import Fraction

from scripts.certified_gain_bounds import certified_mean_lower
from scripts.selected_frontier_controller import configuration
from scripts.uniform_controller_design import POPULATIONS
from scripts.uniform_controller_design import specification as original_specification

SOURCES_PER_POPULATION = 1024
THRESHOLDS = ("1/1000000", "1/100000", "1/10000", "1/1000", "1/100", "1/10", "1/2")


def specification():
    result = original_specification()
    result.pop("practical_mixture_gain")
    result.update(
        version="independent-uniform-frontier-increment-v1",
        sources_per_population=SOURCES_PER_POPULATION,
        source_matches=4 * SOURCES_PER_POPULATION,
        thresholds=list(THRESHOLDS),
        comparison="selected frontier hybrid minus existing symbolic controller",
        criterion="both equal-opponent parent-mixture lower bounds strictly positive",
        secondary="four individual population bounds and paired actual score differences",
        hybrid=configuration(),
        boundary="next learner discard; draw threshold 14 minus public root turn",
        extension="none; failures retain all artifacts and never resample",
    )
    return result


def cases():
    return [
        {"index": 4 * replicate + i, "population": population, "replicate": replicate}
        for replicate in range(SOURCES_PER_POPULATION)
        for i, population in enumerate(POPULATIONS)
    ]


def summarize(values, *, expected_n=SOURCES_PER_POPULATION):
    """All missing/no-change sources contribute zero; source proofs are external.

    The four population intervals form one simultaneous 95% family. Their
    fixed equal-weight mixtures need no additional error allocation. This
    tests an incremental contrast, not the old 0.1 percentage-point criterion.
    """
    if set(values) != set(POPULATIONS):
        raise ValueError("Exactly four declared populations required")
    grid = tuple(map(Fraction, THRESHOLDS))
    bounds, populations = {}, {}
    for name in POPULATIONS:
        bound = certified_mean_lower(
            values[name],
            grid,
            expected_n=expected_n,
            alpha=Fraction(1, 80),
            bits=32,
        )
        bounds[name] = bound
        populations[name] = {
            "sources": bound.n,
            "mean": str(bound.empirical_mean),
            "lower": str(bound.mean_lower),
            "error": str(bound.alpha),
            "thresholds": [
                {
                    "threshold": str(row.threshold),
                    "successes": row.successes,
                    "probability_lower": str(row.probability_lower),
                }
                for row in bound.thresholds
            ],
        }
    mixtures = {}
    for parent in (0, 1):
        pair = [bounds[f"{parent}:{opponent}"] for opponent in ("wp", "point")]
        mean = sum((b.empirical_mean for b in pair), Fraction(0)) / 2
        lower = sum((b.mean_lower for b in pair), Fraction(0)) / 2
        mixtures[str(parent)] = {
            "mean": str(mean),
            "lower": str(lower),
            "criterion_met": lower > 0,
        }
    return {
        "scope": "incremental expected terminal win-score gain over the existing controller",
        "populations": populations,
        "parent_mixtures": mixtures,
        "criterion_met": all(m["criterion_met"] for m in mixtures.values()),
        "all_population_lowers_positive": all(
            b.mean_lower > 0 for b in bounds.values()
        ),
        "family_error": "1/20",
        "requires_complete_source_and_certificate_audit": True,
    }
