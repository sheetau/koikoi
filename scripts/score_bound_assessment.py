"""Independent score-bound parent/program verification and actual-tail replay.

Isolated from the frozen uniform-study interpreter. Both value replays execute
every remaining transition; certificates never replace terminal outcomes.
"""

from collections import Counter
from fractions import Fraction

import numpy as np

from koikoi.layout import PHASE, PLAYER, POINTS, STOCK, TERMINAL
from scripts.belief_tail import visible_keys
from scripts.check_belief_tail import add_step, empty_work
from scripts.diagnose_history_support import reference
from scripts.export_belief_tail import replay_actions
from scripts.public_score_bounds import RULE, outcome_certificates
from scripts.score_bounded_tail import ScoreBoundExecutor, ScoreBoundPlan


def in_default_region(state, viewer):
    win, loss = outcome_certificates(state[None], viewer)
    return bool(win[0] or loss[0])


def verify_pair(worlds, selection, parent, opponents):
    """Recompute both Qs with true actors, never importing native policy caches.

    Parent actions come from the unchanged actor, not the fixed-phase saved map.
    The candidate executes its explicit map or its checked parent-default rule.
    Both traverse every remaining stock chance to terminal without certificates
    replacing outcomes. Caches here are built only by these reference replays.
    """
    if set(opponents) != set(worlds.model_names):
        raise ValueError("Expected exactly the declared opponent actors")
    if selection.plan is None:
        return {}
    plan = selection.plan
    if not isinstance(plan, ScoreBoundPlan) or plan.leaf_rule != RULE:
        raise ValueError("Expected the recognized symbolic program")
    parent_cache, opponent_cache = {}, {}
    mass = sum(map(int, worlds.weights))

    def evaluate(label):
        physical, represented, calls = empty_work(), empty_work(), Counter()

        def actor_action(state, model, learner):
            env = reference(state)
            legal = tuple(map(int, np.flatnonzero(env.legal())))
            key = visible_keys(state[None], int(state[PLAYER]))[0]
            cache = parent_cache if learner else opponent_cache
            address = key if learner else (model, key)
            if address not in cache:
                actor = parent if learner else opponents[worlds.model_names[model]]
                cache[address] = legal[0] if len(legal) == 1 else int(actor.act(env))
                calls["parent" if learner else "opponent"] += int(len(legal) > 1)
            return cache[address]

        def visit(state, model, weight, depth):
            if state[PHASE] == TERMINAL:
                return weight * int(
                    np.sign(
                        state[POINTS + worlds.viewer]
                        - state[POINTS + 1 - worlds.viewer]
                    )
                )
            if depth >= 32:
                raise ValueError("Reference continuation exceeded finite depth")
            learner = int(state[PLAYER]) == worlds.viewer
            if learner and label == "selected":
                key = visible_keys(state[None], worlds.viewer)[0]
                if key in plan.policy:
                    action = plan.policy[key]
                elif depth > 0 and in_default_region(state, worlds.viewer):
                    action = actor_action(state, model, True)
                else:
                    raise KeyError(
                        "Missing candidate action outside the symbolic region"
                    )
            else:
                action = actor_action(state, model, learner)
            env = reference(state)
            if not 0 <= action < 50 or not env.legal()[action]:
                raise ValueError("Illegal independent continuation action")
            step = env.step(action)
            add_step(physical, step)
            difference = int(state[STOCK]) - int(env.state[STOCK])
            if difference == 0:
                represented["decisions"] += weight
                represented["round_endings"] += weight * int(step[1])
                represented["completions"] += weight * int(step[2])
                return visit(env.state, model, weight, depth + 1)
            if difference != 1:
                raise ValueError("Unexpected stock change")
            top = 48 + int(state[STOCK])
            positions = range(72, top + 1)
            if len(positions) < 9 or weight % len(positions):
                raise ValueError("Expected equal exact stock-successor weights")
            each, total = weight // len(positions), 0
            for position in positions:
                successor, outcome = env, step
                if position != top:
                    successor = reference(state)
                    successor.state[position], successor.state[top] = (
                        successor.state[top],
                        successor.state[position],
                    )
                    outcome = successor.step(action)
                    add_step(physical, outcome)
                represented["decisions"] += each
                represented["round_endings"] += each * int(outcome[1])
                represented["completions"] += each * int(outcome[2])
                total += visit(successor.state, model, each, depth + 1)
            return total

        signed = sum(
            visit(state, int(model), int(weight), 0)
            for state, model, weight in zip(
                worlds.states, worlds.models, worlds.weights, strict=True
            )
        )
        q = Fraction(signed + mass, 2 * mass)
        expected = selection.parent_q if label == "parent" else selection.selected_q
        if q != expected or represented["completions"] != mass:
            raise ValueError(f"Independent {label} value differs from selection")
        return {
            "value": str(q),
            "physical_work": physical,
            "represented_work": represented,
            "new_actor_calls": dict(calls),
        }

    return {label: evaluate(label) for label in ("parent", "selected")}


def assess_pair(source, selection, parent, opponent, parent_id):
    """Execute saved parent/controller tails and independently replay both traces."""
    root = reference(source["selected_root"]["state"])
    if (
        selection.binding.viewer != source["seat"]
        or selection.binding.root_key
        != visible_keys(root.state[None], source["seat"])[0]
    ):
        raise ValueError("Selection belongs to another root or viewer")
    # Validate identity/program before starting even the parent arm.
    ScoreBoundExecutor(selection, parent, parent_id)
    outcomes, actual, replay, calls = {}, Counter(), Counter(), Counter()
    for label in ("parent", "controller"):
        env = reference(root.state)
        executor = (
            None
            if label == "parent"
            else ScoreBoundExecutor(selection, parent, parent_id)
        )
        actions, work = [], empty_work()
        for _ in range(32):
            if env.state[PHASE] == TERMINAL:
                break
            if env.state[PLAYER] == source["seat"]:
                previous = 0 if executor is None else executor.symbolic_parent_calls
                action = int(parent.act(env)) if executor is None else executor.act(env)
                calls["learner_decisions"] += 1
                if executor is None or executor.policy is None:
                    calls["parent_adapter_calls"] += 1
                else:
                    extra = executor.symbolic_parent_calls - previous
                    calls["symbolic_parent_adapter_calls"] += extra
                    calls["parent_adapter_calls"] += extra
            else:
                action = int(opponent.act(env))
                calls["opponent_adapter_calls"] += 1
            if not 0 <= action < 50 or not env.legal()[action]:
                raise ValueError("Illegal conditional-tail action")
            actions.append(action)
            add_step(work, env.step(action))
        if env.state[PHASE] != TERMINAL:
            raise ValueError("Conditional execution exceeded finite tail")
        final = env.state.tolist()
        outcomes[label] = {
            "actions": actions,
            "work": work,
            "final": final,
            "margin": int(
                final[POINTS + source["seat"]] - final[POINTS + 1 - source["seat"]]
            ),
        }
        actual.update(work)
        checked = replay_actions(reference(root.state), actions, final)
        assert checked == work
        replay.update(checked)
    if (
        outcomes["parent"]["final"] != source["final"]
        or outcomes["parent"]["actions"]
        != source["actions"][source["selected_root"]["decision"] :]
    ):
        raise ValueError("Parent execution differs from the recorded source suffix")
    if selection.plan is None and outcomes["parent"] != outcomes["controller"]:
        raise ValueError("Fallback did not retain the parent")
    return {
        "outcomes": outcomes,
        "actual_work": dict(actual),
        "replay_work": dict(replay),
        "calls": dict(calls),
    }
