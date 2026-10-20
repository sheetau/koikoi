"""Certify an unavoidable final-turn loss without expanding opponent-only moves."""

from fractions import Fraction

import numpy as np

from koikoi.layout import PHASE, PLAYER, POINTS, ROUND, ROUNDS, STOCK, TERMINAL, TURN
from scripts.belief_tail import visible_keys
from scripts.bounded_belief_tail import BoundedBeliefPlanner, ContinuationValue
from scripts.check_belief_tail import add_step, empty_work
from scripts.diagnose_history_support import reference
from scripts.incumbent_belief_tail import IncumbentBeliefPlanner


def opponent_last_turn_loss(states, viewer):
    """Own/public certificate only; no learner action remains in certified rows.

    The opponent can score or the round can end in the dealer's one-point award.
    At most one point can therefore be transferred TO the viewer. A current
    margin below -2 remains negative under every remaining continuation.
    """
    if viewer not in (0, 1):
        raise ValueError("Invalid viewer")
    margin = states[:, POINTS + viewer] - states[:, POINTS + 1 - viewer]
    return (
        (states[:, PHASE] != TERMINAL)
        & (states[:, ROUND] + 1 == states[:, ROUNDS])
        & (states[:, TURN] == 15)
        & (states[:, PLAYER] != viewer)
        & (margin < -2)
    )


class OpponentLastTurnPlanner(IncumbentBeliefPlanner):
    """Preserve explicit learner maps; certified subtrees contain no learner turn."""

    def continuation(self, states, models, weights, depth, cutoff=None):
        certified = opponent_last_turn_loss(states, self.viewer)
        if not certified.any():
            return super().continuation(states, models, weights, depth, cutoff)
        mass = sum(map(int, weights[certified]))
        self.work["certified_opponent_rows"] = self.work.get(
            "certified_opponent_rows", 0
        ) + int(certified.sum())
        self.work["certified_opponent_mass"] = (
            self.work.get("certified_opponent_mass", 0) + mass
        )
        # Bypass only the incumbent recorder so it sees the recombined value.
        # Recursive calls still use this subclass's continuation.
        rest = BoundedBeliefPlanner.continuation(
            self,
            states[~certified],
            models[~certified],
            weights[~certified],
            depth,
            None if cutoff is None else cutoff + mass,
        )
        result = ContinuationValue(rest.value - mass, rest.exact, rest.policy)
        if depth == 1:
            value = Fraction(result.value + self._mass, 2 * self._mass)
            if result.exact:
                policy = dict(result.policy)
                policy[self._root_key] = self._root_action
                self._exact[self._root_action] = value
                self._policies[self._root_action] = policy
            else:
                self._bounds[self._root_action] = value
        return result


def verify_policy_values(worlds, plan, opponents, opponent_cache=None):
    """Independent Python replay, including opponent moves omitted by the planner.

    Unlike the cache-only oracle, resolve missing opponent actions from the
    supplied strict policies. A missing learner-map key is still an error.
    The certificate is not used in this replay.
    """
    if set(opponents) != set(worlds.model_names):
        raise ValueError("Every world model needs its verification actor")
    cache = {} if opponent_cache is None else dict(opponent_cache)
    physical, represented = empty_work(), empty_work()
    initial_mass = sum(map(int, worlds.weights))
    values, calls = {}, 0

    def visit(state, model, weight, policy, depth):
        nonlocal calls
        if state[PHASE] == TERMINAL:
            return weight * int(
                np.sign(
                    state[POINTS + worlds.viewer] - state[POINTS + 1 - worlds.viewer]
                )
            )
        if depth >= 32:
            raise ValueError("Policy replay exceeded finite tail depth")
        actor = int(state[PLAYER])
        key = visible_keys(state[None], actor)[0]
        if actor == worlds.viewer:
            action = policy[key]
        else:
            address = (model, key)
            if address not in cache:
                env = reference(state)
                legal = tuple(map(int, np.flatnonzero(env.legal())))
                cache[address] = (
                    legal[0]
                    if len(legal) == 1
                    else int(opponents[worlds.model_names[model]].act(env))
                )
                calls += int(len(legal) > 1)
                if cache[address] not in legal:
                    raise ValueError("Illegal verification opponent action")
            action = cache[address]
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
            raise ValueError("Unexpected stock change")
        top = 48 + int(state[STOCK])
        positions = list(range(72, top + 1))
        if len(positions) < 9 or weight % len(positions):
            raise ValueError("Expected equal integer draw-successor mass")
        each, total = weight // len(positions), 0
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

    for action, policy in plan.root_policies.items():
        signed = sum(
            visit(state, int(model), int(weight), policy, 0)
            for state, model, weight in zip(
                worlds.states,
                worlds.models,
                worlds.weights,
                strict=True,
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
        "new_opponent_adapter_calls": calls,
        "opponent_cache_entries": len(cache),
    }
