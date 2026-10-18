import numpy as np
import pytest

from koikoi.layout import PHASE
from scripts.check_final_round_search import cases, first_per_phase


def test_complete_case_grid_uses32_distinct_duplicate_deals():
    rows = cases()
    assert len(rows) == 128
    assert len({r["seed"] for r in rows}) == 32
    assert len({tuple(r.values()) for r in rows}) == 128
    for parent in (1, 2):
        for opponent in ("wp", "point"):
            selected = [
                r for r in rows if r["parent"] == parent and r["opponent"] == opponent
            ]
            assert len(selected) == 32
            for replicate in range(16):
                pair = [r for r in selected if r["replicate"] == replicate]
                assert {r["seat"] for r in pair} == {0, 1}
                assert pair[0]["seed"] == pair[1]["seed"]
    assert rows[0] == {
        "parent": 1,
        "opponent": "wp",
        "seat": 0,
        "replicate": 0,
        "seed": 87110001,
    }
    assert rows[-1] == {
        "parent": 2,
        "opponent": "point",
        "seat": 1,
        "replicate": 15,
        "seed": 87110116,
    }


def test_first_selection_is_by_phase_and_play_order_with_missing_phases_retained():
    def state(phase):
        out = np.zeros(208, dtype=np.int32)
        out[PHASE] = phase
        return out

    roots = [(100, state(0)), (103, state(2)), (106, state(0)), (109, state(2))]
    selected = first_per_phase(roots)
    assert set(selected) == {0, 2} and 1 not in selected
    assert selected[0][0] == 100 and selected[2][0] == 103
    assert first_per_phase([]) == {}
    with pytest.raises(AssertionError):
        first_per_phase(roots[::-1])
