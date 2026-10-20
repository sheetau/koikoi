"""Fixed design and statistical reduction; this module collects no matches."""

from fractions import Fraction

from scripts.certified_gain_bounds import certified_mean_lower

POPULATIONS = ("0:wp", "0:point", "1:wp", "1:point")
THRESHOLDS = ("1/1000", "1/250", "1/100", "1/50", "1/25", "2/25", "4/25")
SOURCES_PER_POPULATION = 2048


def specification():
    return {
        "version": "independent-uniform-controller-v1",
        "populations": list(POPULATIONS),
        "sources_per_population": SOURCES_PER_POPULATION,
        "source_matches": 4 * SOURCES_PER_POPULATION,
        "profile": "paper8",
        "environment_seed": 1,
        "initial_dealer": 0,
        "learner_seat": "independent uniform randbelow(2) per source",
        "round_deals": "independent uniform permutations conditional on valid_deal",
        "entropy": "secrets.randbelow",
        "thresholds": list(THRESHOLDS),
        "family_error": "1/20",
        "population_error": "1/80",
        "bound_bits": 32,
        "opponent_mixture_weights": {"wp": "1/2", "point": "1/2"},
        "practical_mixture_gain": "1/1000",
        "criterion": "each population lower > 0 and each parent mixture lower > 1/1000",
        "max_draws": 5,
        "max_hands": 1000,
        "capacity": 4096,
        "phase_nodes": 500000,
        "phase_decisions": 2000000,
        "depth": 32,
        "chunk_size": 16,
        "workers": 4,
        "extension": "none; source/audit failures stop inference, never resample",
    }


def cases():
    return [
        {
            "index": 4 * replicate + offset,
            "population": population,
            "replicate": replicate,
        }
        for replicate in range(SOURCES_PER_POPULATION)
        for offset, population in enumerate(POPULATIONS)
    ]


def summarize(values, *, expected_n=SOURCES_PER_POPULATION):
    """Reduce four independently audited population lists, including all zeros.

    The caller must verify the source law and every input certificate. Returning
    a criterion result here alone is not a passed study. expected_n exists for
    finite statistical fixtures; the production protocol must require 2048.
    """
    if set(values) != set(POPULATIONS):
        raise ValueError("Exactly the four declared populations are required")
    bounds, rows = {}, {}
    grid = tuple(map(Fraction, THRESHOLDS))
    for population in POPULATIONS:
        bound = certified_mean_lower(
            values[population],
            grid,
            expected_n=expected_n,
            alpha=Fraction(1, 80),
            bits=32,
        )
        bounds[population] = bound
        rows[population] = {
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
        mean = sum((x.empirical_mean for x in pair), Fraction(0)) / 2
        lower = sum((x.mean_lower for x in pair), Fraction(0)) / 2
        mixtures[str(parent)] = {
            "mean": str(mean),
            "lower": str(lower),
            "practical_criterion_met": lower > Fraction(1, 1000),
        }
    positive = all(x.mean_lower > 0 for x in bounds.values())
    return {
        "scope": "conditional-expectation improvement over the unchanged parent",
        "populations": rows,
        "parent_mixtures": mixtures,
        "all_population_lowers_positive": positive,
        "criterion_met": positive
        and all(x["practical_criterion_met"] for x in mixtures.values()),
        "family_error": "1/20",
        "requires_complete_source_and_certificate_audit": True,
    }
