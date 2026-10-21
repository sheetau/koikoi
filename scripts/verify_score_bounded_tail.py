"""Full Python replay of score-bounded programs; no outcome shortcuts.

Isolated from the frozen symbolic verifier. This executes all parent actions
and chance branches omitted by the score-bound planner.
"""

from fractions import Fraction

import numpy as np

from koikoi.layout import PHASE, PLAYER, POINTS, STOCK, TERMINAL
from scripts.belief_tail import visible_keys
from scripts.check_belief_tail import add_step, empty_work
from scripts.diagnose_history_support import reference
from scripts.public_score_bounds import RULE, outcome_certificates
from scripts.score_bounded_tail import ScoreBoundPlan


def verify_program(worlds, plan, parent, opponents, opponent_cache=None):
    """Evaluate the total program without replacing any subtree by a value.

    At every absent learner key this independent interpreter executes the parent
    and follows all resulting draws to terminal states. The declared leaf region
    is checked only for runtime admissibility, never to supply an outcome.
    """
    if not isinstance(plan, ScoreBoundPlan) or plan.leaf_rule != RULE:
        raise ValueError("Expected the declared symbolic program")
    if set(opponents) != set(worlds.model_names):
        raise ValueError("Every world model needs its verification actor")
    cache = {} if opponent_cache is None else dict(opponent_cache)
    physical, represented = empty_work(), empty_work()
    initial_mass = sum(map(int, worlds.weights))
    values, parent_calls, opponent_calls = {}, 0, 0

    def visit(state, model, weight, policy, depth):
        nonlocal parent_calls, opponent_calls
        if state[PHASE] == TERMINAL:
            return weight * int(
                np.sign(
                    state[POINTS + worlds.viewer] - state[POINTS + 1 - worlds.viewer]
                )
            )
        if depth >= 32:
            raise ValueError("Program replay exceeded finite depth")
        actor = int(state[PLAYER])
        key = visible_keys(state[None], actor)[0]
        env = reference(state)
        legal = tuple(map(int, np.flatnonzero(env.legal())))
        if actor == worlds.viewer:
            if key in policy:
                action = policy[key]
            else:
                win, loss = outcome_certificates(state[None], worlds.viewer)
                if not (win[0] or loss[0]):
                    raise KeyError("Missing learner action outside the symbolic region")
                action = legal[0] if len(legal) == 1 else int(parent.act(env))
                parent_calls += int(len(legal) > 1)
        else:
            address = (model, key)
            if address not in cache:
                cache[address] = (
                    legal[0]
                    if len(legal) == 1
                    else int(opponents[worlds.model_names[model]].act(env))
                )
                opponent_calls += int(len(legal) > 1)
            action = cache[address]
        if action not in legal:
            raise ValueError("Illegal replay action")
        result = env.step(action)
        add_step(physical, result)
        difference = int(state[STOCK]) - int(env.state[STOCK])
        if difference == 0:
            represented["decisions"] += weight
            represented["round_endings"] += weight * int(result[1])
            represented["completions"] += weight * int(result[2])
            return visit(env.state, model, weight, policy, depth + 1)
        if difference != 1:
            raise ValueError("Unexpected stock change")
        top = 48 + int(state[STOCK])
        positions = list(range(72, top + 1))
        if len(positions) < 9 or weight % len(positions):
            raise ValueError("Expected equal integer draw-successor mass")
        each, total = weight // len(positions), 0
        for position in positions:
            if position == top:
                successor, step_result = env, result
            else:
                successor = reference(state)
                successor.state[position], successor.state[top] = (
                    successor.state[top],
                    successor.state[position],
                )
                step_result = successor.step(action)
                add_step(physical, step_result)
            represented["decisions"] += each
            represented["round_endings"] += each * int(step_result[1])
            represented["completions"] += each * int(step_result[2])
            total += visit(successor.state, model, each, policy, depth + 1)
        return total

    for action, policy in plan.root_policies.items():
        signed = sum(
            visit(state, int(model), int(weight), policy, 0)
            for state, model, weight in zip(
                worlds.states, worlds.models, worlds.weights, strict=True
            )
        )
        q = Fraction(signed + initial_mass, 2 * initial_mass)
        assert q == plan.values[action]
        values[str(action)] = str(q)
    assert represented["completions"] == initial_mass * len(plan.root_policies)
    return {
        "physical_work": physical,
        "represented_work": represented,
        "values": values,
        "new_parent_adapter_calls": parent_calls,
        "new_opponent_adapter_calls": opponent_calls,
    }
