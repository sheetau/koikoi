"""Full independent Python replay of a complete interval-evaluated policy."""

from collections import Counter
from fractions import Fraction
from types import SimpleNamespace

import numpy as np

from koikoi.layout import PHASE, PLAYER, POINTS, STOCK, TERMINAL
from scripts.belief_tail import KoiTailDynamics, visible_keys
from scripts.check_belief_tail import add_step, empty_work
from scripts.diagnose_history_support import reference
from scripts.incumbent_tail_controller import bind_worlds


def verify_receipt(worlds, receipt, parent, opponents, parent_identity):
    receipt.validate()
    program = receipt.program
    if parent_identity != program.binding.parent_identity or set(opponents) != set(
        worlds.model_names
    ):
        raise ValueError("Verification actor identities differ")
    binding = bind_worlds(
        worlds,
        SimpleNamespace(keys=visible_keys, legal=KoiTailDynamics.legal),
        parent_identity,
        dict(program.binding.opponent_identities),
    )
    if (
        binding != program.binding
        or sum(map(int, worlds.weights)) != receipt.total_mass
    ):
        raise ValueError("Verification belief differs")
    policy = program.validate()
    physical, represented, calls = empty_work(), empty_work(), Counter()
    parent_cache, opponent_cache = {}, {}

    def act(state, model):
        env = reference(state)
        actor = int(state[PLAYER])
        key = visible_keys(state[None], actor)[0]
        legal = tuple(map(int, np.flatnonzero(env.legal())))
        if actor == worlds.viewer and key in policy and policy[key] in legal:
            action = policy[key]
        else:
            own = actor == worlds.viewer
            cache = parent_cache if own else opponent_cache
            address = key if own else (model, key)
            if address not in cache:
                agent = parent if own else opponents[worlds.model_names[model]]
                cache[address] = legal[0] if len(legal) == 1 else int(agent.act(env))
                calls["parent" if own else "opponent"] += int(len(legal) > 1)
            action = cache[address]
        if action not in legal:
            raise ValueError("Illegal independent replay action")
        return action

    # The reference must really be the unchanged parent, including its root.
    root = reference(worlds.states[0])
    if int(parent.act(root)) != program.parent_root_action:
        raise ValueError("Unchanged parent root action differs")
    calls["parent_root_check"] += 1

    def visit(state, model, weight, depth):
        if state[PHASE] == TERMINAL:
            return weight * int(
                np.sign(
                    state[POINTS + worlds.viewer] - state[POINTS + 1 - worlds.viewer]
                )
            )
        if depth >= 32:
            raise ValueError("Full replay exceeded finite tail depth")
        action = act(state, model)
        env = reference(state)
        result = env.step(action)
        add_step(physical, result)
        difference = int(state[STOCK]) - int(env.state[STOCK])
        if difference == 0:
            represented["decisions"] += weight
            represented["round_endings"] += weight * int(result[1])
            represented["completions"] += weight * int(result[2])
            return visit(env.state, model, weight, depth + 1)
        if difference != 1:
            raise ValueError("Unexpected stock change")
        top = 48 + int(state[STOCK])
        positions = range(72, top + 1)
        if len(positions) < 9 or weight % len(positions):
            raise ValueError("Invalid exact chance multiplicity")
        each, total = weight // len(positions), 0
        for position in positions:
            child, step = env, result
            if position != top:
                child = reference(state)
                child.state[position], child.state[top] = (
                    child.state[top],
                    child.state[position],
                )
                step = child.step(action)
                add_step(physical, step)
            represented["decisions"] += each
            represented["round_endings"] += each * int(step[1])
            represented["completions"] += each * int(step[2])
            total += visit(child.state, model, each, depth + 1)
        return total

    signed = sum(
        visit(s, int(m), int(w), 0)
        for s, m, w in zip(worlds.states, worlds.models, worlds.weights, strict=True)
    )
    value = Fraction(signed + receipt.total_mass, 2 * receipt.total_mass)
    if (
        not receipt.lower <= value <= receipt.upper
        or represented["completions"] != receipt.total_mass
    ):
        raise ValueError(
            "Exact independent value is outside the interval or mass is incomplete"
        )
    return {
        "value": str(value),
        "physical_work": physical,
        "represented_work": represented,
        "actor_calls": dict(calls),
    }
