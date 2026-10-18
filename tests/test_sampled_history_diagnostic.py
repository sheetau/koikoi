"""Public timing and estimator/denominator checks for the development probe."""

from dataclasses import replace

import numpy as np

from koikoi.layout import DISCARD, PHASE, PLAYER, ROUND
from scripts.diagnose_sampled_history_support import aggregate, beliefs, eligible
from scripts.sampled_history_support import HistoryProposalBatch


def test_eligibility_needs_prior_round_public_identification():
    class PublicFixture:
        def __init__(self):
            self.state = {PHASE: DISCARD, PLAYER: 0, ROUND: 3}

        def legal(self):
            return np.array([True, True])

        def observe(self):
            obs = np.zeros(560)
            obs[541] = 4 / 8
            return obs

    env = PublicFixture()
    prior = {"model": "wp", "round": 2, "available_after_decision": 50}
    assert eligible(env, 0, prior, 51)
    assert not eligible(env, 0, None, 51)
    assert not eligible(env, 0, prior, 49)
    assert not eligible(env, 1, prior, 51)
    assert not eligible(env, 0, dict(prior, round=3), 51)
    assert not eligible(env, 0, dict(prior, round=4), 51)


def test_joint_and_public_singleton_use_same_proposals():
    batch = HistoryProposalBatch(
        "key",
        ("point", "wp"),
        "sampled",
        1000,
        3,
        0,
        ((1,), (2,), (3,)),
        ((True, False), (True, True), (False, False)),
        {},
    )
    result = beliefs(batch, "wp")
    np.testing.assert_allclose(
        result["uniform"]["card_probabilities"][1:4], [1 / 3] * 3
    )
    np.testing.assert_allclose(
        result["joint"]["card_probabilities"][1:4], [1 / 3, 2 / 3, 0]
    )
    assert result["identified"]["card_probabilities"][2] == 1
    unsupported = replace(batch, compatible=((True, False),) * 3)
    assert beliefs(unsupported, "wp")["identified"] == {"status": "zero_sampled_mass"}


def test_descriptive_pairs_keep_failure_and_ineligible_denominators():
    def value(error):
        return {"status": "estimated", "card_brier": error}

    report = aggregate(
        [
            {"status": "ineligible"},
            {
                "status": "eligible",
                "beliefs": {
                    "uniform": value(0.2),
                    "joint": value(0.1),
                    "identified": value(0.15),
                },
            },
            {
                "status": "eligible",
                "beliefs": {
                    "uniform": value(0.3),
                    "joint": value(0.4),
                    "identified": {"status": "zero_sampled_mass"},
                },
            },
        ]
    )
    assert report["attempts"] == 3 and report["eligible_roots"] == 2
    assert report["beliefs"]["identified"]["zero_sampled_mass"] == 1
    paired = report["paired_descriptive_differences"]["identified_minus_uniform"]
    assert paired["roots"] == 1
    np.testing.assert_allclose(paired["mean"], -0.05)
