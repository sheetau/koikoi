"""Certify local public-policy replacements without valuing the whole parent."""

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace

import numpy as np

from koikoi.layout import KOI, PHASE, PICK_DRAW, PLAYER, TERMINAL, TURN
from scripts.belief_tail import KoiTailDynamics, visible_keys
from scripts.deferred_belief_tail import DeferredWorldSet, expand_draws
from scripts.incumbent_tail_controller import bind_worlds
from scripts.interval_tail import (
    IntervalEvaluator,
    IntervalExecutor,
    make_program,
    select_interval,
)
from scripts.proposal_interval_tail import propose_then_compare


@dataclass
class PublicFrontier:
    parent_program: object
    total_mass: int
    draws: int
    nodes: tuple
    terminal_mass: int
    pending_mass: int
    work: dict
    refusal: str | None

    @property
    def complete(self):
        return self.refusal is None and self.pending_mass == 0


def remaining_draws(states):
    return 16 - states[:, TURN] - np.isin(states[:, PHASE], (PICK_DRAW, KOI))


def parent_frontier(
    worlds,
    dynamics,
    parent_id,
    opponent_ids,
    *,
    draws=3,
    max_decisions=250000,
    max_rows=65536,
    max_depth=32,
):
    """Partition ALL parent paths at their first eligible learner observation.

    Partial arrival groups are never returned: another unexpanded path could
    belong to the same information set. Refusal therefore disables patching.
    Terminal paths have zero policy difference, without needing their values.
    """
    if (
        type(draws) is not int
        or not 0 <= draws <= worlds.draws
        or type(max_decisions) is not int
        or max_decisions < 0
        or dynamics.capacity < 24
    ):
        raise ValueError("Invalid public frontier or prefix budget")
    program = make_program(worlds, dynamics, parent_id, opponent_ids)
    # Reuse belief, parent, depth and row-cap validation; no transitions run.
    IntervalEvaluator(
        worlds,
        dynamics,
        program,
        max_frontier_rows=max_rows,
        max_depth=max_depth,
        certificates=False,
    )
    total = sum(map(int, worlds.weights))
    pending = deque()
    arrived = defaultdict(list)
    terminal_mass = stored_rows = pending_rows = 0
    work = Counter(decisions=0, round_endings=0, completions=0)

    def enqueue(states, models, weights, depth):
        nonlocal terminal_mass, stored_rows, pending_rows
        terminal = dynamics.terminal(states)
        terminal_mass += sum(map(int, weights[terminal]))
        gate = (
            ~terminal
            & (dynamics.players(states) == worlds.viewer)
            & (remaining_draws(states) <= draws)
        )
        for i in np.flatnonzero(gate):
            key = dynamics.keys(states[i : i + 1], worlds.viewer)[0]
            arrived[key].append((states[i], models[i], weights[i]))
            stored_rows += 1
        live = ~terminal & ~gate
        for player in (0, 1):
            indices = np.flatnonzero(live & (dynamics.players(states) == player))
            chunk = dynamics.capacity // 24
            for start in range(0, len(indices), chunk):
                selected = indices[start : start + chunk]
                pending.append(
                    (states[selected], models[selected], weights[selected], depth)
                )
                pending_rows += len(selected)
        work["peak_rows"] = max(work["peak_rows"], stored_rows + pending_rows)

    enqueue(worlds.states, worlds.models, worlds.weights, 0)
    refusal = None
    while pending:
        states, models, weights, depth = pending[0]
        if depth >= max_depth:
            refusal = "Prefix depth cap exceeded"
            break
        if work["decisions"] >= max_decisions:
            refusal = "Prefix transition cap reached"
            break
        actions = (
            np.asarray([dynamics.baseline_action(s) for s in states], dtype=np.int32)
            if states[0, PLAYER] == worlds.viewer
            else dynamics.opponent_actions(states, models)
        )
        expanded, next_models, masses, expanded_actions = expand_draws(
            states, models, weights, actions, dynamics.capacity
        )
        if work["decisions"] + len(expanded) > max_decisions:
            refusal = "Prefix transition cap reached"
            break
        if stored_rows + pending_rows - len(states) + len(expanded) > max_rows:
            refusal = "Prefix row cap exceeded"
            break
        after, physical = dynamics.advance(expanded, expanded_actions)
        pending.popleft()
        pending_rows -= len(states)
        work.update(physical)
        work["expanded_tasks"] += 1
        work["chance_rows_added"] += len(expanded) - len(states)
        work["represented_decisions"] += sum(map(int, masses))
        ended = sum(map(int, masses[dynamics.terminal(after)]))
        work["represented_completions"] += ended
        work["represented_round_endings"] += ended
        enqueue(after, next_models, masses, depth + 1)
    pending_mass = sum(sum(map(int, row[2])) for row in pending)
    arrival_mass = sum(int(row[2]) for rows in arrived.values() for row in rows)
    if terminal_mass + pending_mass + arrival_mass != total:
        raise ValueError("Parent prefix lost probability mass")
    work["arrival_mass"] = arrival_mass
    work["terminal_mass"] = terminal_mass
    work["pending_mass"] = pending_mass
    nodes = []
    if refusal is None:
        for key, rows in arrived.items():
            states = np.stack([r[0] for r in rows])
            depths = set(map(int, remaining_draws(states)))
            if len(depths) != 1:
                raise ValueError("Frontier observation mixes draw counts")
            node = DeferredWorldSet(
                states,
                np.asarray([r[1] for r in rows], dtype=np.int64),
                np.asarray([r[2] for r in rows], dtype=np.int64),
                worlds.model_names,
                worlds.viewer,
                depths.pop(),
            )
            nodes.append((key, node))
        # Reach mass first, then public integer-history bytes; no actual future.
        nodes.sort(key=lambda pair: (-sum(map(int, pair[1].weights)), pair[0]))
    return PublicFrontier(
        program,
        total,
        draws,
        tuple(nodes),
        terminal_mass,
        pending_mass,
        dict(work),
        refusal,
    )


@dataclass
class FrontierPatch:
    frontier: PublicFrontier
    comparisons: tuple
    gain_lower: Fraction
    decisions: int


def solve_frontier_patch(
    worlds,
    dynamics,
    parent_id,
    opponent_ids,
    *,
    draws=3,
    max_decisions=2000000,
    prefix_decisions=250000,
    per_phase_decisions=50000,
    max_nodes_per_phase=25000,
    max_rows=65536,
):
    """One shared transition budget for prefix and all local comparisons.

    Only positive local lower bounds become patches. Unvisited or unresolved
    frontier nodes keep the parent, contributing exactly zero difference.
    """
    if any(
        type(x) is not int or x < 1
        for x in (max_decisions, prefix_decisions, per_phase_decisions)
    ):
        raise ValueError("Positive shared and phase transition budgets required")
    frontier = parent_frontier(
        worlds,
        dynamics,
        parent_id,
        opponent_ids,
        draws=draws,
        max_decisions=min(max_decisions, prefix_decisions),
        max_rows=max_rows,
    )
    used = frontier.work["decisions"]
    comparisons, gain = [], Fraction(0)
    if frontier.complete:
        for key, node in frontier.nodes:
            # Preassign equal phase allowances from the remaining shared work.
            allowance = min(per_phase_decisions, (max_decisions - used) // 2)
            if allowance == 0:
                break
            result = propose_then_compare(
                node,
                dynamics,
                parent_id,
                opponent_ids,
                proposal_decisions=allowance,
                reference_decisions=allowance,
                max_nodes_per_phase=max_nodes_per_phase,
            )
            used += result["decisions"]
            comparisons.append((key, result))
            gain += (
                Fraction(sum(map(int, node.weights)), frontier.total_mass)
                * result["selection"].gain_lower
            )
    if used > max_decisions:
        raise ValueError("Shared patch budget exceeded")
    return FrontierPatch(frontier, tuple(comparisons), gain, used)


class FrontierExecutor:
    """Follow the parent to one public frontier, then commit to its local patch."""

    def __init__(self, patch, parent, parent_id):
        program = patch.frontier.parent_program
        if program.binding.parent_identity != parent_id:
            raise ValueError("Patch parent identity differs")
        if patch.comparisons and not patch.frontier.complete:
            raise ValueError("Incomplete frontier cannot support local beliefs")
        nodes = dict(patch.frontier.nodes)
        self.patches = {}
        seen = set()
        gain = Fraction(0)
        for key, result in patch.comparisons:
            if key not in nodes or key in seen:
                raise ValueError("Unknown or repeated patch frontier")
            seen.add(key)
            selection = result["selection"]
            expected = bind_worlds(
                nodes[key],
                SimpleNamespace(keys=visible_keys, legal=KoiTailDynamics.legal),
                parent_id,
                dict(program.binding.opponent_identities),
            )
            if (
                selection.parent.program.binding != expected
                or selection.parent.total_mass != sum(map(int, nodes[key].weights))
                or select_interval(
                    selection.parent,
                    [] if selection.candidate is None else [selection.candidate],
                )
                != selection
            ):
                raise ValueError("Local comparison has a different belief or selection")
            if selection.candidate is not None:
                self.patches[key] = IntervalExecutor(selection, parent, parent_id)
                gain += (
                    Fraction(
                        sum(map(int, nodes[key].weights)), patch.frontier.total_mass
                    )
                    * selection.gain_lower
                )
        if gain != patch.gain_lower:
            raise ValueError("Patch gain does not equal weighted local gains")
        self.binding, self.parent = program.binding, parent
        self.started, self.active = False, None
        self.frontier_keys, self.entered = set(nodes), False

    def act(self, env):
        if env.state[PHASE] == TERMINAL or env.state[PLAYER] != self.binding.viewer:
            raise ValueError("Expected a live learner decision")
        key = visible_keys(env.state[None], self.binding.viewer)[0]
        if not self.started and key != self.binding.root_key:
            raise ValueError("Patch execution root changed")
        self.started = True
        if not self.entered and key in self.frontier_keys:
            self.entered = True
            self.active = self.patches.get(key)
        action = (
            int(self.parent.act(env)) if self.active is None else self.active.act(env)
        )
        if not 0 <= action < 50 or not env.legal()[action]:
            raise ValueError("Illegal patch execution action")
        return action
