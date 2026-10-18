"""Eligibility depends on public time and fresh-round counters only."""

import pytest

from scripts.check_public_memory_activation import eligible_boundary


def test_boundary_is_after_public_identification_and_round_end():
    identified = {"round": 2, "available_after_decision": 41, "model": "wp"}
    assert not eligible_boundary(3, 0, 0, 40, identified)
    assert eligible_boundary(3, 0, 0, 41, identified)
    assert not eligible_boundary(2, 0, 0, 41, identified)
    assert not eligible_boundary(3, 1, 0, 41, identified)
    assert not eligible_boundary(3, 0, 1, 41, identified)
    assert not eligible_boundary(3, 0, 4, 41, identified)
    assert not eligible_boundary(7, 0, 0, 200, None)
    assert not eligible_boundary(8, 0, 0, 200, identified)
    final = {"round": 6, "available_after_decision": 120, "model": "point"}
    assert eligible_boundary(7, 0, 0, 120, final)


def test_rejects_invalid_identification():
    with pytest.raises(ValueError, match="timestamped"):
        eligible_boundary(
            3, 0, 0, 41, {"round": 2, "available_after_decision": 0, "model": "wp"}
        )
    with pytest.raises(ValueError, match="timestamped"):
        eligible_boundary(
            3, 0, 0, 41, {"round": 2, "available_after_decision": 41, "model": "human"}
        )
