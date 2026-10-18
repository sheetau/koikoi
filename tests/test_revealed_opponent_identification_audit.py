import copy

import pytest

from scripts.audit_revealed_opponent_identification import elimination_ledger


def events():
    return [
        {
            "round": 0,
            "available_after_decision": 8,
            "before": ["point", "wp"],
            "after": ["wp"],
            "checks": [
                {"decision": 0, "observed": 3, "predictions": {"wp": 3, "point": 4}},
                {"decision": 6, "observed": 2, "predictions": {"wp": 2}},
            ],
        },
        {
            "round": 2,
            "available_after_decision": 30,
            "before": ["wp"],
            "after": [],
            "checks": [{"decision": 22, "observed": 7, "predictions": {"wp": 8}}],
        },
    ]


def test_persistent_elimination_and_first_identification():
    candidates, first, calls, updates = elimination_ledger(events())
    assert candidates == set()
    assert first == {"round": 0, "available_after_decision": 8, "model": "wp"}
    assert calls == {"wp": 3, "point": 1}
    assert updates == {8: {"wp"}, 30: set()}


def test_rejects_resurrection_missing_prediction_and_future_check():
    bad = copy.deepcopy(events())
    bad[1]["before"] = ["point", "wp"]
    with pytest.raises(AssertionError):
        elimination_ledger(bad)
    bad = copy.deepcopy(events())
    del bad[0]["checks"][0]["predictions"]["point"]
    with pytest.raises(AssertionError):
        elimination_ledger(bad)
    bad = copy.deepcopy(events())
    bad[0]["checks"][1]["decision"] = 8
    with pytest.raises(AssertionError):
        elimination_ledger(bad)
