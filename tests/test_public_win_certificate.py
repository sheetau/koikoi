import numpy as np
import pytest

from koikoi.agents import GreedyAgent, RandomAgent
from koikoi.layout import (
    BEFORE,
    DEALER,
    DISCARD,
    PENDING,
    PHASE,
    PICK_DRAW,
    PICK_HAND,
    PLAYER,
    POINTS,
    ROUND,
    ROUNDS,
    STOCK,
    TERMINAL,
    TURN,
)
from koikoi.reference import ReferenceEnv
from scripts.diagnose_win_certificates import public_legal
from scripts.public_win_certificate import CertifiedWinAgent, certify_turn


def force_draw(env, card):
    """Test-only hidden-world construction; never passed to certificate code."""
    other = env.clone()
    s = other.state
    top = int(s[48 + s[STOCK]])
    location = int(np.flatnonzero(s[48:96] == card)[0])
    s[48 + location], s[48 + s[STOCK]] = top, card
    s[top], s[card] = s[card], s[top]
    np.testing.assert_array_equal(other.observe(), env.observe())
    return other


def oracle_actions(env, root_player=None, root_round=None):
    """Independent quantified reference-engine tree, ending at opponent's turn."""
    s = env.state
    p = int(s[PLAYER]) if root_player is None else root_player
    r = int(s[ROUND]) if root_round is None else root_round
    if s[PHASE] == TERMINAL:
        return {None} if s[POINTS + p] > s[POINTS + 1 - p] else set()
    if s[PLAYER] != p or s[ROUND] != r:
        return set()
    winners = set()
    for action in map(int, np.flatnonzero(env.legal())):
        # The engine automatically draws except when the hand capture branches.
        draws = s[PHASE] == PICK_HAND or (
            s[PHASE] == DISCARD
            and sum(s[c] == 2 for c in range(action // 4 * 4, action // 4 * 4 + 4)) != 2
        )
        worlds = (
            [
                force_draw(env, int(c))
                for c in np.flatnonzero(env.observe()[:528].reshape(48, 11)[:, 4])
            ]
            if draws
            else [env.clone()]
        )
        for world in worlds:
            world.step(action)
            if not oracle_actions(world, p, r):
                break
        else:
            if worlds:
                winners.add(action)
    return winners


def sampled_states():
    result = {phase: [] for phase in range(4)}
    for seed in range(31001, 31081):
        env, agent = ReferenceEnv(seed), RandomAgent(seed)
        while env.state[PHASE] != TERMINAL:
            phase = int(env.state[PHASE])
            if len(result[phase]) < 10:
                result[phase].append(env.clone())
            env.step(agent.act(env))
        if all(len(states) == 10 for states in result.values()):
            return result
    raise AssertionError("Did not collect all phases")


def test_exact_public_tree_matches_independent_reference_all_phases():
    phases = sampled_states()
    nonempty = 0
    for phase, states in phases.items():
        for index, original in enumerate(states):
            for final, points in ((True, 29), (True, 40), (False, 57)):
                env = original.clone()
                s, p = env.state, int(env.state[PLAYER])
                s[ROUND] = s[ROUNDS] - 1 if final else 0
                s[POINTS + p], s[POINTS + 1 - p] = points, 60 - points
                # Exercise both automatic-stop turns as well as ordinary turns.
                if index >= 8 and phase != 3:
                    s[TURN] = 14 + (index % 2)
                expected = oracle_actions(env)
                np.testing.assert_array_equal(public_legal(env.observe()), env.legal())
                actual = certify_turn(env.observe(), env.legal())
                assert set(actual.actions) == expected, (phase, index, final, points)
                nonempty += bool(expected)
    assert nonempty > 10


@pytest.mark.parametrize("profile", ["paper8", "research12"])
def test_pathwise_match_win_nonregression_and_immediate_completion(profile):
    overrides = 0
    for seed in range(31200, 31232):
        for seat in (0, 1):
            outcomes = []
            for guarded in (False, True):
                env = ReferenceEnv(seed, profile)
                base = RandomAgent(seed + seat)
                candidate = CertifiedWinAgent(base) if guarded else base
                opponent = RandomAgent(seed + 789)
                first = None
                while env.state[PHASE] != TERMINAL:
                    p = int(env.state[PLAYER])
                    if first is not None:
                        assert p == seat
                        assert (int(env.state[ROUND]), int(env.state[TURN])) == first
                    old_overrides = candidate.overrides if guarded else 0
                    action = (candidate if p == seat else opponent).act(env)
                    if (
                        guarded
                        and candidate.overrides > old_overrides
                        and first is None
                    ):
                        first = (int(env.state[ROUND]), int(env.state[TURN]))
                    env.step(action)
                outcome = int(
                    np.sign(env.state[POINTS + seat] - env.state[POINTS + 1 - seat])
                )
                outcomes.append(outcome)
                if guarded:
                    overrides += candidate.overrides
                    if first is not None:
                        assert outcome == 1
            assert outcomes[1] >= outcomes[0]
    assert overrides > 0


def test_unknown_assignment_invariance_and_no_private_state_access():
    class PublicOnly:
        def __init__(self, env):
            self.obs, self.mask = env.observe(), env.legal()

        def observe(self):
            return self.obs.copy()

        def legal(self):
            return self.mask.copy()

    for states in sampled_states().values():
        for env in states[:2]:
            env.state[ROUND] = env.state[ROUNDS] - 1
            expected = certify_turn(env.observe(), env.legal())
            unknown = np.flatnonzero(env.observe()[:528].reshape(48, 11)[:, 4])
            for card in unknown:
                changed = force_draw(env, int(card))
                assert certify_turn(changed.observe(), changed.legal()) == expected
            agent = CertifiedWinAgent(GreedyAgent())
            assert env.legal()[agent.act(PublicOnly(env))]


def test_terminal_tie_last_turn_dealer_and_profile_limits():
    # Fixture with no yaku possible: all capture outcomes leave fewer than the
    # scoring thresholds. It exercises the public last-turn dealer award.
    env = ReferenceEnv(13, "research12")
    s = env.state
    s[:48] = -1
    s[PLAYER], s[DEALER], s[TURN], s[PHASE] = 0, 1, 15, PICK_DRAW
    s[PENDING], s[BEFORE] = 3, 0
    s[3], s[1], s[2] = 5, 2, 2
    s[POINTS : POINTS + 2] = [31, 29]
    s[ROUND] = s[ROUNDS] - 1
    assert not certify_turn(env.observe(), env.legal()).actions  # 30:30 draw.
    s[POINTS : POINTS + 2] = [32, 28]
    assert certify_turn(env.observe(), env.legal()).actions == (1, 2)
    s[ROUND] = 0
    assert not certify_turn(env.observe(), env.legal()).actions
    s[PHASE] = TERMINAL
    assert not certify_turn(env.observe(), env.legal()).actions
