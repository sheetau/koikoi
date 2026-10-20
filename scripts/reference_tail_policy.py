"""Independent weighted policy replay with Python rules and deferred stock sets."""

from fractions import Fraction

import numpy as np

from koikoi.layout import PHASE, PLAYER, POINTS, STOCK, TERMINAL
from scripts.belief_tail import visible_keys
from scripts.check_belief_tail import add_step, empty_work
from scripts.diagnose_history_support import reference


def verify_deferred_policy_values(worlds, plan, opponent_cache):
    """Replay all declared root policies, discovering draws via actual engine steps.

    The planner's chance-expansion predicate and native transitions are not used.
    A probe transition also supplies one actual successor, so it is counted once.
    Frozen observation-keyed opponent actions are reused; no neural replay claim.
    """
    physical, represented = empty_work(), empty_work()
    root_values = {}
    initial_mass = sum(map(int, worlds.weights))

    def visit(state, model, weight, policy, depth):
        if state[PHASE] == TERMINAL:
            return weight * int(
                np.sign(
                    state[POINTS + worlds.viewer] - state[POINTS + 1 - worlds.viewer]
                )
            )
        if depth >= 32:
            raise ValueError("Policy replay exceeded the finite tail depth")
        actor = int(state[PLAYER])
        key = visible_keys(state[None], actor)[0]
        action = policy[key] if actor == worlds.viewer else opponent_cache[(model, key)]
        probe = reference(state)
        result = probe.step(action)
        add_step(physical, result)
        difference = int(state[STOCK]) - int(probe.state[STOCK])
        if difference == 0:
            represented["decisions"] += weight
            represented["round_endings"] += weight * int(result[1])
            represented["completions"] += weight * int(result[2])
            return visit(probe.state, model, weight, policy, depth + 1)
        if difference != 1:
            raise ValueError("Unexpected stock change in a final-round transition")
        top = 48 + int(state[STOCK])
        positions = list(range(72, top + 1))
        if len(positions) < 9 or weight % len(positions):
            raise ValueError("Stock successors must have equal integer mass")
        each = weight // len(positions)
        total = 0
        for position in positions:
            if position == top:
                successor, step_result = probe, result
            else:
                successor = reference(state)
                successor.state[position], successor.state[top] = (
                    successor.state[top],
                    successor.state[position],
                )
                step_result = successor.step(action)
                add_step(physical, step_result)
            assert successor.state[STOCK] == state[STOCK] - 1
            represented["decisions"] += each
            represented["round_endings"] += each * int(step_result[1])
            represented["completions"] += each * int(step_result[2])
            total += visit(successor.state, model, each, policy, depth + 1)
        return total

    for first, policy in plan.root_policies.items():
        signed = sum(
            visit(state, int(model), int(weight), policy, 0)
            for state, model, weight in zip(
                worlds.states, worlds.models, worlds.weights, strict=True
            )
        )
        value = Fraction(signed + initial_mass, 2 * initial_mass)
        assert value == plan.values[first], (
            f"Independent root-value mismatch for {first}"
        )
        root_values[str(first)] = str(value)
    assert represented["completions"] == initial_mass * len(plan.root_policies)
    return {
        "physical_work": physical,
        "represented_work": represented,
        "values": root_values,
    }
