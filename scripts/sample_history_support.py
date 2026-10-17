"""Sample fresh simulator worlds from an audited current-hand support set.

Support must already be certified for this root and known deterministic opponent.
Hand choice is uniform; remaining stock permutations and future-deal seeds are
fresh RNG draws. This function never chooses an action or evaluates a world.
"""

import numpy as np

from koikoi.layout import SEED, STOCK
from scripts.enumerate_history_support import canonical_world


def sample_supported_worlds(state, hands, rng, count):
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Expected a positive integer particle count")
    keys = [tuple(sorted(hand)) for hand in hands]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Support must be nonempty and contain unique hands")
    keys.sort()
    templates = [canonical_world(state, hand) for hand in keys]
    if any(s is None for s in templates):
        raise ValueError("Support contains an invalid initial hand")
    worlds, selected = [], []
    for _ in range(count):
        index = int(rng.integers(len(templates)))
        world = templates[index].copy()
        rng.shuffle(world[72 : 49 + int(world[STOCK])])
        world[SEED] = rng.integers(0, 2**31 - 1)
        worlds.append(world)
        selected.append(index)
    return np.stack(worlds), np.array(selected, dtype=np.int32)
