import pytest

from koikoi.agents import EndgameAgent, solved_koi_action
from koikoi.layout import CONTINUE, KOI, PHASE, PLAYER, POINTS, ROUND, ROUNDS, STOP
from koikoi.reference import ReferenceEnv


def fixture(profile="paper8", points=25, final=True):
    env = ReferenceEnv(100, profile)
    s = env.state
    s[PHASE], s[PLAYER] = KOI, 0
    s[ROUND] = s[ROUNDS] - 1 if final else 0
    s[POINTS : POINTS + 2] = [points, 60 - points]
    # Boar, deer, butterfly = five points; all are visible captured cards.
    s[[20, 24, 36]] = 3
    return env


@pytest.mark.parametrize("profile", ["paper8", "research12"])
def test_final_round_dominance(profile):
    assert solved_koi_action(fixture(profile, 26)) == STOP
    assert solved_koi_action(fixture(profile, 24)) == CONTINUE
    assert (
        solved_koi_action(fixture(profile, 25)) is None
    )  # A draw has nontrivial value.


def test_bankruptcy_is_profile_specific():
    assert solved_koi_action(fixture("paper8", 57, final=False)) == STOP
    assert solved_koi_action(fixture("research12", 57, final=False)) is None


def test_guard_only_resolves_proven_decisions():
    class Base:
        def __init__(self):
            self.calls = 0

        def act(self, env):
            self.calls += 1
            return CONTINUE

    base = Base()
    agent = EndgameAgent(base)
    assert agent.act(fixture(points=26)) == STOP
    assert base.calls == 0
    assert agent.act(fixture(points=25)) == CONTINUE
    assert base.calls == 1
