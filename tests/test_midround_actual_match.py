import numpy as np
import pytest

from koikoi.env import BatchEnv
from koikoi.layout import PLAYER
from scripts.midround_actual_match import play_intervention
from tests.test_history_conditioned_worlds import Extremal
from tests.test_midround_search import roots


@pytest.mark.parametrize("root_index", [0, 1])
def test_only_declared_root_action_is_changed_and_both_complete(root_index):
    root = roots()[root_index]
    case = {"seed": 31010001, "seat": int(root.state[PLAYER])}
    actor = Extremal()
    native = BatchEnv(1, "cpu", "paper8")
    base = actor.act(root)
    alternative = next(int(a) for a in np.flatnonzero(root.legal()) if a != base)
    a = play_intervention(case, actor, actor, native, root.state, base)
    b = play_intervention(case, actor, actor, native, root.state, alternative)
    assert a["work"]["completed_matches"] == b["work"]["completed_matches"] == 1
    assert len(a["intervention_indices"]) == 1
    index = a["intervention_indices"][0]
    assert b["intervention_indices"] == [index]
    assert a["actions"][:index] == b["actions"][:index]
    assert a["actions"][index] == base and b["actions"][index] == alternative
    with pytest.raises(AssertionError):
        play_intervention(case, actor, actor, native, None, None)
