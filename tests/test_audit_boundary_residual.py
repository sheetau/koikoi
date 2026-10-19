import numpy as np
import pytest

from scripts.audit_boundary_residual import scalar_statistics, verify_part


def test_scalar_blocks_independent_closed_form():
    high = {"high": [-1, 1] * 256, "proxy": [0] * 512}
    cheap = {"proxy": [0] * 2048}
    result = scalar_statistics(high, cheap, [0] * 2048)
    assert result["mean"] == result["reference_mean"] == 0
    assert result["standard_error"] == pytest.approx(1 / np.sqrt(511))
    assert result["corrected_block_mse"] == result["ordinary_block_mse"] == 0
    assert result["high_proxy_covariance"] == 0


def test_terminal_override_and_nonterminal_absolute_axes():
    table = np.zeros((2, 8, 2, 2, 59, 2))
    table[:, 2, 1, 0, 20, 1] = 0.25
    part = {
        "boundary_fields": [[[2, 21, 1, 0], [2, 60, 0, 4]]],
        "field_order": ["round", "points0", "dealer", "phase"],
        "paired_boundary_values": [[0.25, -1]],
        "proxy": [-0.625],
        "first_round_work": {
            "decisions": 4,
            "round_endings": 2,
            "terminal_match_endings": 1,
        },
        "work": {"decisions": 4, "round_endings": 2, "terminal_endings": 1},
        "high": None,
        "paired_points": None,
    }
    assert verify_part(part, [0], 1, table, False) == [-0.625]
    part["paired_boundary_values"][0][1] = 0
    with pytest.raises(AssertionError):
        verify_part(part, [0], 1, table, False)
