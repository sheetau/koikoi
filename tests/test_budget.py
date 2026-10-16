import pytest

from koikoi.budget import lineage_budget


def counts(matches):
    return {
        "matches": matches,
        "rounds": 12 * matches,
        "decisions": 180 * matches,
        "updates": matches // 1000,
    }


def test_resumes_are_not_added_to_weight_initialization_budget():
    payload = {
        "config": {"profile": "research12"},
        "counters": counts(10_000_000),
        "provenance": {
            "parent_checkpoint": {"counters": counts(9_000_000)},
            "initialized_from": {
                "sha256": "twelve",
                "profile": "research12",
                "counters": counts(1_000_000),
                "provenance": {
                    "parent_checkpoint": {"counters": counts(500_000)},
                    "initialized_from": {
                        "sha256": "eight",
                        "profile": "paper8",
                        "counters": counts(11_000_000),
                        "provenance": {
                            "parent_checkpoint": {"counters": counts(4_000_000)}
                        },
                    },
                },
            },
        },
    }
    result = lineage_budget(payload, "current")
    assert result["by_profile"]["research12"]["matches"] == 11_000_000
    assert result["by_profile"]["paper8"]["matches"] == 11_000_000
    assert len(result["stages"]) == 3


def test_budget_rejects_invalid_counters_and_repeated_ancestor():
    payload = {
        "config": {"profile": "research12"},
        "counters": counts(10),
        "provenance": {},
    }
    payload["counters"]["matches"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        lineage_budget(payload, "same")
    payload["counters"] = counts(10)
    payload["provenance"]["initialized_from"] = {
        "sha256": "same",
        "profile": "research12",
        "counters": counts(5),
        "provenance": {},
    }
    with pytest.raises(ValueError, match="Repeated"):
        lineage_budget(payload, "same")
