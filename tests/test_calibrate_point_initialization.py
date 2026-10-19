import copy

import numpy as np
import pytest

from koikoi.layout import PHASE
from scripts.calibrate_point_initialization import fit


def fixture():
    states = np.zeros((24, PHASE + 1), dtype=np.int64)
    states[:, PHASE] = np.tile(np.arange(4), 6)
    masks = np.zeros((24, 50), dtype=bool)
    masks[:, :3] = True
    wp = np.tile([0.0, 0.03, 0.08] + [0.0] * 47, (24, 1))
    return {
        "states": states,
        "masks": masks,
        "cpu_wp": wp,
        "cpu_point": wp * 7 + 2,
        "sources": np.repeat([0, 1, 3, 7, -1, 4], 4),
        "guards": np.repeat([-1, -1, -1, -1, -1, 0], 4),
    }


def test_scale_recovery_and_group_selection():
    result = fit(fixture())
    assert result["phase_temperatures"] == pytest.approx([0.35] * 4)
    assert result["source_groups"] == {"calibration": [0, 1], "validation": [3, 7]}
    for phase in result["phases"].values():
        assert phase["calibration"]["point_matched"]["rows"] == 2


def test_validation_and_ineligible_scores_cannot_change_fit():
    original = fixture()
    changed = copy.deepcopy(original)
    selected = (
        (changed["sources"] % 4 == 3)
        | (changed["sources"] < 0)
        | (changed["guards"] >= 0)
    )
    changed["cpu_point"][selected] *= 3
    assert fit(original)["phase_temperatures"] == fit(changed)["phase_temperatures"]
    assert (
        fit(original)["phases"]["0"]["validation"]
        != fit(changed)["phases"]["0"]["validation"]
    )


def test_unattainable_target_is_rejected():
    data = fixture()
    data["cpu_point"][:] = 1
    with pytest.raises(ValueError, match="attainable"):
        fit(data)
