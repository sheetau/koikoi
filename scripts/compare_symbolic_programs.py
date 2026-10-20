"""Compare completed explicit policies with certified parent-default programs."""

import numpy as np

from koikoi.layout import PHASE, PLAYER, POINTS, ROUND, ROUNDS, TERMINAL, TURN
from scripts.symbolic_final_turn_tail import RULE, SymbolicPlan


def certified_own_win_key(key, viewer):
    """Read the certificate's public fields from the current visible-history key.

    visible_keys stores the viewer, 48 card locations, and state columns96:110
    before its remaining public-history/initial-hand fields. No hidden state is
    reconstructed, and no card location affects this predicate.
    """
    if viewer not in (0, 1):
        raise ValueError("Invalid viewer")
    if not isinstance(key, bytes) or len(key) != 160 * np.dtype(np.int32).itemsize:
        raise ValueError("Expected the current complete own/public-history key")
    row = np.frombuffer(key, dtype=np.int32)
    if int(row[0]) != viewer:
        raise ValueError("Policy key belongs to another viewer")

    def public(column):
        return int(row[1 + 48 + column - 96])

    return (
        public(PHASE) != TERMINAL
        and public(ROUND) + 1 == public(ROUNDS)
        and public(TURN) == 15
        and public(PLAYER) == viewer
        and public(POINTS + viewer) - public(POINTS + 1 - viewer) > 2
    )


def compare_common(old_phases, new_phases):
    """Check exact values, retained actions and every omitted explicit decision.

    This is a structural comparison of already completed programs. It does not
    replace independent full rule replay or establish values for partial roots.
    """
    counts = {
        "common_completed_root_values": 0,
        "retained_explicit_entries": 0,
        "omitted_certified_entries": 0,
    }
    for name in old_phases.keys() & new_phases.keys():
        previous, current = old_phases[name], new_phases[name]
        if previous.binding != current.binding or previous.mode != current.mode:
            raise ValueError("Programs must share the same belief, actors and mode")
        old, new = previous.result.plan, current.result.plan
        if old is None or new is None:
            continue
        if not isinstance(new, SymbolicPlan) or new.leaf_rule != RULE:
            raise ValueError("Expected the recognized symbolic program")
        viewer = current.binding.viewer
        for action in old.values.keys() & new.values.keys():
            if old.values[action] != new.values[action]:
                raise ValueError("Common completed root value changed")
            before, after = old.root_policies[action], new.root_policies[action]
            if after.get(current.binding.root_key) != action:
                raise ValueError("Missing or changed explicit root action")
            if not after.keys() <= before.keys():
                raise ValueError("Compressed policy introduced an explicit history")
            if any(value != before[key] for key, value in after.items()):
                raise ValueError("Retained explicit action changed")
            omitted = before.keys() - after.keys()
            if any(not certified_own_win_key(key, viewer) for key in omitted):
                raise ValueError(
                    "Explicit history omitted outside the certified region"
                )
            counts["common_completed_root_values"] += 1
            counts["retained_explicit_entries"] += len(after)
            counts["omitted_certified_entries"] += len(omitted)
    return counts
