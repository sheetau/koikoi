from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest

from koikoi.layout import PHASE, POINTS, ROUND, STOCK, TERMINAL, TURN
from koikoi.reference import BLUE, KASU, LIGHTS, RED, RIBBONS, TANE, ReferenceEnv, score
from scripts.belief_tail import visible_keys
from scripts.public_tail_bounds import (
    ALL_CARDS,
    certified_outcome,
    final_margin_bounds,
    public_tail,
    public_tail_from_key,
    score_ceiling,
)
from tests.test_history_conditioned_worlds import Extremal, late_root


def test_score_set_monotonicity_for_each_yaku_component():
    # All subsets of the categories that could have nontrivial interactions.
    # The kasu contribution separately depends only on a monotone count.
    for category in (LIGHTS, TANE, RIBBONS, RED | BLUE, {8, 28, 32}):
        cards = sorted(category)
        for size in range(len(cards) + 1):
            for subset in combinations(cards, size):
                remaining = category - set(subset)
                for k in (0, 1, 3, 4, 7):
                    value = score(subset, k)
                    assert all(score((*subset, c), k) >= value for c in remaining)
    ordered = sorted(KASU)
    for k in range(8):
        assert all(
            score(ordered[:n], k) <= score(ordered[: n + 1], k)
            for n in range(len(ordered))
        )


def test_score_envelope_checks_nonmonotone_koi_and_exhaustion_award():
    # Only flower viewing is possible: score at three kois is six, at four six.
    # With no scoring cards, the implementation's additive 3 exceeds its 4 score.
    assert score([], 3) == 3 and score([], 4) == 0
    assert score_ceiling(ALL_CARDS, 3, 1) == 3
    assert score_ceiling(ALL_CARDS, 0, 0) == 1
    rng = np.random.default_rng(804)
    for _ in range(100):
        possible = set(map(int, rng.choice(48, rng.integers(1, 49), replace=False)))
        excluded = sum(1 << c for c in set(range(48)) - possible)
        current, future = 1, 5
        upper = score_ceiling(excluded, current, future)
        for k in range(current, current + future + 1):
            subset = {c for c in possible if rng.integers(2)}
            assert score(subset, k) <= upper


def test_projection_ignores_private_state_and_key_viewer():
    root = late_root()
    root.state[ROUND] = 7
    expected = public_tail(root.state)
    changed = root.state.copy()
    # Replace every noncaptured location and all private / history slots.
    changed[:48][changed[:48] < 3] = -99
    changed[48:96] = -77
    changed[110:] = -55
    assert public_tail(changed) == expected
    for viewer in (0, 1):
        key = visible_keys(root.state[None], viewer)[0]
        assert public_tail_from_key(key) == expected
    with pytest.raises(ValueError):
        public_tail_from_key(b"bad")


def test_strict_boundaries_and_scope():
    root = late_root()
    root.state[ROUND] = 7
    context = replace(public_tail(root.state), margin=0)
    low, high = final_margin_bounds(context, 0)
    assert certified_outcome(replace(context, margin=-low), 0) is None
    assert certified_outcome(replace(context, margin=-low + 1), 0) == 1
    assert certified_outcome(replace(context, margin=-high), 0) is None
    assert certified_outcome(replace(context, margin=-high - 1), 0) == -1
    assert certified_outcome(replace(context, margin=-low + 1), 1) == -1
    assert final_margin_bounds(replace(context, round=6), 0) is None
    assert final_margin_bounds(replace(context, phase=TERMINAL), 0) is None
    with pytest.raises(ValueError):
        replace(context, captured=(1, 1))


@pytest.mark.parametrize("margin", [-100, 0, 100])
def test_all_legal_continuations_fit_public_bounds(margin):
    roots = []
    for seed in range(31010001, 31010009):
        root, actor = ReferenceEnv(seed), Extremal()
        for _ in range(2000):
            if root.state[TURN] == 11 and root.state[PHASE] == 0:
                break
            root.step(actor.act(root))
        else:
            raise AssertionError("Late test root not found")
        root.state[ROUND] = 7
        root.state[POINTS : POINTS + 2] = (200 + margin, 200)
        # Exercise every possible next stock card, then all legal choices for
        # both players. This is a fixture check, not exhaustive hidden-deck proof.
        top = 48 + int(root.state[STOCK])
        for slot in range(72, top + 1):
            alternate = root.clone()
            alternate.state[[slot, top]] = alternate.state[[top, slot]]
            roots.append(alternate)
    stack = [(root, []) for root in roots]
    leaves, certificates, phases = 0, 0, set()
    while stack:
        env, ancestors = stack.pop()
        if env.state[PHASE] == TERMINAL:
            final = int(env.state[POINTS] - env.state[POINTS + 1])
            for low, high, certificate in ancestors:
                assert low <= final <= high
                if certificate is not None:
                    assert np.sign(final) == certificate
            leaves += 1
            continue
        context = public_tail(env.state)
        low, high = final_margin_bounds(context, 0)
        certificate = certified_outcome(context, 0)
        assert final_margin_bounds(context, 1) == (-high, -low)
        certificates += certificate is not None
        phases.add(context.phase)
        for action in np.flatnonzero(env.legal()):
            child = env.clone()
            child.step(int(action))
            stack.append((child, [*ancestors, (low, high, certificate)]))
    assert leaves >= 16 and phases == {0, 1, 2, 3}
    if margin:
        assert certificates > 0
