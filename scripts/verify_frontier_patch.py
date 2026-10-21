"""Independent Python prefix partition and local/global patch-value checks."""

from collections import Counter
from fractions import Fraction
from types import SimpleNamespace

import numpy as np

from koikoi.layout import KOI, PHASE, PICK_DRAW, PLAYER, POINTS, STOCK, TERMINAL, TURN
from scripts.belief_tail import KoiTailDynamics, visible_keys
from scripts.check_belief_tail import add_step, empty_work
from scripts.diagnose_history_support import reference
from scripts.frontier_policy_patch import FrontierExecutor
from scripts.incumbent_tail_controller import bind_worlds
from scripts.verify_interval_tail import verify_receipt


def canonical_row(state, model):
    """Only unused-stock ordering is irrelevant to an unordered chance row."""
    copy = state.copy()
    copy[72 : 49 + int(copy[STOCK])] = np.sort(copy[72 : 49 + int(copy[STOCK])])
    return int(model), copy.tobytes()


def verify_frontier_patch(
    worlds, patch, parent, opponents, parent_id, *, full_values=False
):
    frontier = patch.frontier
    if not frontier.complete:
        raise ValueError("Only complete frontiers have independently usable beliefs")
    binding = bind_worlds(
        worlds,
        SimpleNamespace(keys=visible_keys, legal=KoiTailDynamics.legal),
        parent_id,
        dict(frontier.parent_program.binding.opponent_identities),
    )
    if (
        binding != frontier.parent_program.binding
        or set(opponents) != set(worlds.model_names)
        or sum(map(int, worlds.weights)) != frontier.total_mass
        or int(parent.act(reference(worlds.states[0])))
        != frontier.parent_program.parent_root_action
    ):
        raise ValueError("Root belief or parent differs")
    executor = FrontierExecutor(patch, parent, parent_id)
    patches = {key: e.entries for key, e in executor.patches.items()}
    cache = {}

    def parent_action(state, model):
        actor = int(state[PLAYER])
        own = actor == worlds.viewer
        key = visible_keys(state[None], actor)[0]
        address = (actor, None if own else model, key)
        if address not in cache:
            env = reference(state)
            legal = tuple(map(int, np.flatnonzero(env.legal())))
            agent = parent if own else opponents[worlds.model_names[model]]
            cache[address] = legal[0] if len(legal) == 1 else int(agent.act(env))
            if cache[address] not in legal:
                raise ValueError("Illegal independent actor action")
        return cache[address]

    def children(state, action, weight):
        env = reference(state)
        result = env.step(action)
        difference = int(state[STOCK]) - int(env.state[STOCK])
        if difference == 0:
            yield env.state, weight, result
            return
        top = 48 + int(state[STOCK])
        count = top - 71
        if difference != 1 or count < 9 or weight % count:
            raise ValueError("Invalid independent prefix chance multiplicity")
        for position in range(72, top + 1):
            child, step = env, result
            if position != top:
                child = reference(state)
                child.state[position], child.state[top] = (
                    child.state[top],
                    child.state[position],
                )
                step = child.step(action)
            yield child.state, weight // count, step

    def count_step(physical, represented, result, weight):
        add_step(physical, result)
        represented["decisions"] += weight
        represented["round_endings"] += weight * int(result[1])
        represented["completions"] += weight * int(result[2])

    prefix_physical, prefix_represented = empty_work(), empty_work()
    arrivals, terminal_mass = {}, 0

    def prefix(state, model, weight, depth):
        nonlocal terminal_mass
        if state[PHASE] == TERMINAL:
            terminal_mass += weight
            return
        draws = 16 - int(state[TURN]) - int(state[PHASE] in (PICK_DRAW, KOI))
        if state[PLAYER] == worlds.viewer and draws <= frontier.draws:
            key = visible_keys(state[None], worlds.viewer)[0]
            arrivals.setdefault(key, Counter())[canonical_row(state, model)] += weight
            return
        if depth >= 32:
            raise ValueError("Independent prefix depth exceeded")
        for child, mass, result in children(state, parent_action(state, model), weight):
            count_step(prefix_physical, prefix_represented, result, mass)
            prefix(child, model, mass, depth + 1)

    for state, model, weight in zip(
        worlds.states, worlds.models, worlds.weights, strict=True
    ):
        prefix(state, int(model), int(weight), 0)
    expected = {}
    for key, node in frontier.nodes:
        rows = Counter()
        for state, model, weight in zip(
            node.states, node.models, node.weights, strict=True
        ):
            rows[canonical_row(state, model)] += int(weight)
        expected[key] = rows
    if arrivals != expected or terminal_mass != frontier.terminal_mass:
        raise ValueError("Independent public frontier posterior or mass differs")
    nodes, local_checks = dict(frontier.nodes), []
    exact_increment = Fraction(0)
    for key, result in patch.comparisons:
        selection = result["selection"]
        if selection.candidate is None:
            continue
        node = nodes[key]
        pair = {
            name: verify_receipt(node, receipt, parent, opponents, parent_id)
            for name, receipt in (
                ("parent", selection.parent),
                ("candidate", selection.candidate),
            )
        }
        gain = Fraction(pair["candidate"]["value"]) - Fraction(pair["parent"]["value"])
        if not gain >= selection.gain_lower > 0:
            raise ValueError("Independent local gain fails")
        reach = Fraction(sum(map(int, node.weights)), frontier.total_mass)
        exact_increment += reach * gain
        local_checks.append({"reach": str(reach), "gain": str(gain), "oracle": pair})
    if exact_increment < patch.gain_lower:
        raise ValueError("Independent weighted gain fails")
    report = {
        "prefix_physical": prefix_physical,
        "prefix_represented": prefix_represented,
        "prefix_terminal_mass": terminal_mass,
        "nodes": len(arrivals),
        "local_checks": local_checks,
        "exact_increment": str(exact_increment),
        "gain_lower": str(patch.gain_lower),
    }
    if full_values:
        values = {}
        for arm in ("parent", "patched"):
            physical, represented = empty_work(), empty_work()

            def visit(
                state,
                model,
                weight,
                depth,
                active=None,
                arm=arm,
                physical=physical,
                represented=represented,
            ):
                if state[PHASE] == TERMINAL:
                    return weight * int(
                        np.sign(
                            state[POINTS + worlds.viewer]
                            - state[POINTS + 1 - worlds.viewer]
                        )
                    )
                if depth >= 32:
                    raise ValueError("Independent full value depth exceeded")
                action = parent_action(state, model)
                if arm == "patched" and state[PLAYER] == worlds.viewer:
                    key = visible_keys(state[None], worlds.viewer)[0]
                    if active is None and key in nodes:
                        active = key if key in patches else False
                    if active is not None and active is not False:
                        override = patches[active].get(key)
                        if override is not None and reference(state).legal()[override]:
                            action = override
                signed = 0
                for child, mass, result in children(state, action, weight):
                    count_step(physical, represented, result, mass)
                    signed += visit(child, model, mass, depth + 1, active)
                return signed

            signed = sum(
                visit(s, int(m), int(w), 0)
                for s, m, w in zip(
                    worlds.states, worlds.models, worlds.weights, strict=True
                )
            )
            value = Fraction(signed + frontier.total_mass, 2 * frontier.total_mass)
            if represented["completions"] != frontier.total_mass:
                raise ValueError("Incomplete full-value mass")
            values[arm] = {
                "value": str(value),
                "physical_work": physical,
                "represented_work": represented,
            }
        if (
            Fraction(values["patched"]["value"]) - Fraction(values["parent"]["value"])
            != exact_increment
        ):
            raise ValueError(
                "Full policy difference disagrees with local decomposition"
            )
        report["full_values"] = values
    return report
