"""Anytime bounds for complete public policies, including the unchanged parent."""

import heapq
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from koikoi.layout import PHASE, PLAYER, TERMINAL
from scripts.belief_tail import TailBudgetExceeded, visible_keys
from scripts.deferred_belief_tail import DeferredWorldSet, expand_draws
from scripts.incumbent_tail_controller import TailBinding, bind_worlds
from scripts.public_score_bounds import outcome_certificates

RULE = "legal-public-overrides-parent-default-v1"


@dataclass(frozen=True)
class PublicProgram:
    """Use a listed legal public override, otherwise the unchanged parent.

    Legality is part of the frozen program semantics. Thus even an unvisited
    override cannot turn unresolved probability mass into an undefined action.
    """

    binding: TailBinding
    parent_root_action: int
    overrides: tuple[tuple[bytes, int], ...]
    rule: str = RULE

    def validate(self):
        entries = dict(self.overrides)
        if (
            self.rule != RULE
            or len(entries) != len(self.overrides)
            or any(
                not isinstance(k, bytes) or type(a) is not int or not 0 <= a < 50
                for k, a in self.overrides
            )
            or self.parent_root_action not in self.binding.legal
            or entries.get(self.binding.root_key) not in self.binding.legal
        ):
            raise ValueError("Malformed complete public program")
        return entries

    @property
    def is_parent(self):
        return self.overrides == ((self.binding.root_key, self.parent_root_action),)


def make_program(
    worlds, dynamics, parent_identity, opponent_identities, overrides=None
):
    binding = bind_worlds(worlds, dynamics, parent_identity, opponent_identities)
    parent_action = int(dynamics.baseline_action(worlds.states[0]))
    entries = {binding.root_key: parent_action}
    entries.update({} if overrides is None else overrides)
    result = PublicProgram(binding, parent_action, tuple(sorted(entries.items())))
    result.validate()
    return result


@dataclass(frozen=True)
class IntervalReceipt:
    program: PublicProgram
    total_mass: int
    resolved_mass: int
    resolved_signed: int
    work: dict
    refusal: str | None

    def validate(self):
        self.program.validate()
        if (
            any(
                type(x) is not int
                for x in (self.total_mass, self.resolved_mass, self.resolved_signed)
            )
            or not 0 <= self.resolved_mass <= self.total_mass < 2**63
            or self.total_mass == 0
            or abs(self.resolved_signed) > self.resolved_mass
        ):
            raise ValueError("Invalid resolved probability-mass ledger")

    @property
    def lower(self):
        return Fraction(self.resolved_signed + self.resolved_mass, 2 * self.total_mass)

    @property
    def upper(self):
        return self.lower + Fraction(
            self.total_mass - self.resolved_mass, self.total_mass
        )


class IntervalEvaluator:
    """Resumable depth-first evaluation; unfinished mass always remains explicit.

    Complete policies are fixed before expansion. Tasks share learner actions
    by own/public key, and the declared opponent acts on its own observations.
    Chance expansions are atomic: a budget refusal cannot erase frontier mass.
    """

    def __init__(
        self,
        worlds,
        dynamics,
        program,
        *,
        max_frontier_rows=65536,
        max_depth=32,
        certificates=True,
    ):
        if not isinstance(worlds, DeferredWorldSet):
            raise TypeError("Unordered-stock worlds required")
        s, m, w = worlds.states, worlds.models, worlds.weights
        if (
            s.ndim != 2
            or not len(s)
            or m.shape != (len(s),)
            or w.shape != (len(s),)
            or m.dtype != np.int64
            or w.dtype != np.int64
            or np.any(w <= 0)
            or sum(map(int, w)) >= 2**63
            or np.any(m < 0)
            or np.any(m >= len(worlds.model_names))
            or np.any(dynamics.terminal(s))
            or not np.all(dynamics.players(s) == worlds.viewer)
        ):
            raise ValueError("Expected positive live learner-root worlds")
        if (
            type(max_frontier_rows) is not int
            or max_frontier_rows < len(s)
            or type(max_depth) is not int
            or max_depth < 1
        ):
            raise ValueError("Invalid frontier/depth cap")
        expected = bind_worlds(
            worlds,
            dynamics,
            program.binding.parent_identity,
            dict(program.binding.opponent_identities),
        )
        if expected != program.binding or dynamics.model_names != worlds.model_names:
            raise ValueError("Program or dynamics belongs to another belief")
        if dynamics.baseline_action(s[0]) != program.parent_root_action:
            raise ValueError("Program parent root differs from unchanged parent")
        self.entries = program.validate()
        self.program, self.dynamics, self.viewer = program, dynamics, worlds.viewer
        self.max_frontier_rows, self.max_depth = max_frontier_rows, max_depth
        self.certificates = bool(certificates)
        self.total_mass = sum(map(int, w))
        self.resolved_mass = self.resolved_signed = 0
        self.work = Counter(
            decisions=0,
            round_endings=0,
            completions=0,
            represented_decisions=0,
            represented_completions=0,
            represented_round_endings=0,
            expanded_tasks=0,
        )
        self.frontier, self.frontier_rows, self.serial = [], 0, 0
        self.refusal = None
        self._enqueue(s, m, w, 0)

    def _enqueue(self, states, models, weights, depth):
        terminal = self.dynamics.terminal(states)
        win, loss = (
            outcome_certificates(states, self.viewer)
            if self.certificates
            else (np.zeros(len(states), bool), np.zeros(len(states), bool))
        )
        resolved = terminal | win | loss
        signed = np.zeros(len(states), dtype=np.int64)
        signed[terminal] = self.dynamics.outcomes(states[terminal], self.viewer)
        signed[win], signed[loss] = 1, -1
        self.resolved_mass += sum(map(int, weights[resolved]))
        self.resolved_signed += sum(
            int(a) * int(b)
            for a, b in zip(weights[resolved], signed[resolved], strict=True)
        )
        for name, mask in (
            ("terminal", terminal),
            ("certified_win", win),
            ("certified_loss", loss),
        ):
            self.work[name + "_rows"] += int(mask.sum())
            self.work[name + "_mass"] += sum(map(int, weights[mask]))
        live = ~resolved
        states, models, weights = states[live], models[live], weights[live]
        groups = defaultdict(list)
        for i, key in enumerate(self.dynamics.keys(states, self.viewer)):
            groups[key].append(i)
        # A late stock has at most 24 cards. Chunking cannot change a public
        # action, and guarantees that a full next-draw expansion fits capacity.
        chunk = max(1, self.dynamics.capacity // 24)
        for indices in groups.values():
            for start in range(0, len(indices), chunk):
                selected = indices[start : start + chunk]
                s, m, w = states[selected], models[selected], weights[selected]
                heapq.heappush(
                    self.frontier, (-depth, -sum(map(int, w)), self.serial, s, m, w)
                )
                self.serial += 1
                self.frontier_rows += len(s)
        self.work["peak_frontier_rows"] = max(
            self.work["peak_frontier_rows"], self.frontier_rows
        )

    def receipt(self):
        # Checking the actual frontier prevents a refusal from concealing lost
        # mass; only this partition supports the claimed interval.
        pending = sum(-node[1] for node in self.frontier)
        if self.resolved_mass + pending != self.total_mass:
            raise ValueError("Resolved and pending masses do not partition the belief")
        result = IntervalReceipt(
            self.program,
            self.total_mass,
            self.resolved_mass,
            self.resolved_signed,
            dict(self.work),
            self.refusal,
        )
        result.validate()
        return result

    def run(self, max_decisions, max_nodes=500000):
        """Advance to cumulative caps, retaining the entire resumable frontier."""
        if (
            type(max_decisions) is not int
            or max_decisions < self.work["decisions"]
            or type(max_nodes) is not int
            or max_nodes < self.work["expanded_tasks"]
        ):
            raise ValueError(
                "Caps must be integers at least as large as work already spent"
            )
        self.refusal = None
        while self.frontier:
            node = self.frontier[0]
            depth, states, models, weights = -node[0], node[3], node[4], node[5]
            if depth >= self.max_depth:
                self.refusal = "Decision depth cap exceeded"
                break
            if self.work["expanded_tasks"] >= max_nodes:
                self.refusal = "Expanded task cap exceeded"
                break
            if self.work["decisions"] >= max_decisions:
                self.refusal = "Physical transition cap reached"
                break
            players = self.dynamics.players(states)
            if len(set(players)) != 1:
                raise ValueError("Task mixes acting players")
            if players[0] == self.viewer:
                key = self.dynamics.keys(states[:1], self.viewer)[0]
                action = self.entries.get(key)
                if action is None or action not in self.dynamics.legal(states[0]):
                    action = int(self.dynamics.baseline_action(states[0]))
                if action not in self.dynamics.legal(states[0]):
                    raise ValueError("Illegal program action")
                actions = np.full(len(states), action, dtype=np.int32)
            else:
                actions = self.dynamics.opponent_actions(states, models)
            try:
                expanded, models, weights, actions = expand_draws(
                    states, models, weights, actions, self.dynamics.capacity
                )
            except TailBudgetExceeded as error:
                self.refusal = str(error)
                break
            if self.work["decisions"] + len(expanded) > max_decisions:
                self.refusal = "Physical transition cap reached"
                break
            if (
                self.frontier_rows - len(states) + len(expanded)
                > self.max_frontier_rows
            ):
                self.refusal = "Pending frontier-row cap exceeded"
                break
            after, physical = self.dynamics.advance(expanded, actions)
            # Commit the whole expansion only after all checks and transitions.
            heapq.heappop(self.frontier)
            self.frontier_rows -= len(states)
            self.work.update(physical)
            self.work["expanded_tasks"] += 1
            self.work["chance_rows_added"] += len(expanded) - len(states)
            self.work["represented_decisions"] += sum(map(int, weights))
            terminal = self.dynamics.terminal(after)
            mass = sum(map(int, weights[terminal]))
            self.work["represented_completions"] += mass
            self.work["represented_round_endings"] += mass
            self._enqueue(after, models, weights, depth + 1)
        return self.receipt()


@dataclass(frozen=True)
class IntervalSelection:
    parent: IntervalReceipt
    candidate: IntervalReceipt | None

    @property
    def gain_lower(self):
        return (
            Fraction(0)
            if self.candidate is None
            else self.candidate.lower - self.parent.upper
        )


def select_interval(parent, candidates):
    parent.validate()
    if not parent.program.is_parent:
        raise ValueError("Reference must be the unchanged parent everywhere")
    best = None
    for candidate in candidates:
        candidate.validate()
        if (
            candidate.program.binding != parent.program.binding
            or candidate.program.parent_root_action != parent.program.parent_root_action
            or candidate.total_mass != parent.total_mass
        ):
            raise ValueError("Comparison changes belief or parent")
        if candidate.lower > parent.upper and (
            best is None or candidate.lower > best.lower
        ):
            best = candidate
    return IntervalSelection(parent, best)


class IntervalExecutor:
    def __init__(self, selection, parent, parent_identity):
        expected = select_interval(
            selection.parent,
            [] if selection.candidate is None else [selection.candidate],
        )
        if (
            expected != selection
            or parent_identity != selection.parent.program.binding.parent_identity
        ):
            raise ValueError("Uncertified selection or mismatched parent identity")
        self.program = (
            selection.parent if selection.candidate is None else selection.candidate
        ).program
        self.entries = self.program.validate()
        self.parent, self.started = parent, False

    def act(self, env):
        binding = self.program.binding
        if env.state[PHASE] == TERMINAL or env.state[PLAYER] != binding.viewer:
            raise ValueError("Expected live learner decision")
        key = visible_keys(env.state[None], binding.viewer)[0]
        if not self.started and key != binding.root_key:
            raise ValueError("Execution root changed")
        action = self.entries.get(key)
        if action is None or not env.legal()[action]:
            action = int(self.parent.act(env))
        if not 0 <= action < 50 or not env.legal()[action]:
            raise ValueError("Illegal execution action")
        self.started = True
        return action


def compare_programs(
    worlds,
    dynamics,
    programs,
    *,
    max_decisions=2000000,
    max_nodes=500000,
    quantum=4096,
    max_frontier_rows=65536,
    certificates=True,
):
    """Round-robin shared budget; stop early only on a proved selection.

    The first program is the unchanged parent. Each other program is complete
    and frozen before the first transition. Node work is expanded tasks, not
    the predecessor planner's live-information-node count.
    """
    if (
        not programs
        or not programs[0].is_parent
        or any(type(v) is not int or v < 0 for v in (max_decisions, max_nodes))
        or type(quantum) is not int
        or quantum < dynamics.capacity
        or len(set(programs)) != len(programs)
    ):
        raise ValueError(
            "Distinct programs, parent first, and valid shared budgets required"
        )
    evaluators = [
        IntervalEvaluator(
            worlds,
            dynamics,
            p,
            max_frontier_rows=max_frontier_rows // len(programs),
            certificates=certificates,
        )
        for p in programs
    ]
    receipts = [e.receipt() for e in evaluators]
    # Validate common comparison binding even when no expansion is affordable.
    select_interval(receipts[0], receipts[1:])
    first_adoption = None
    blocked = set()
    reason = "shared_budget_exhausted"
    while True:
        selection = select_interval(receipts[0], receipts[1:])
        total = sum(r.work["decisions"] for r in receipts)
        nodes = sum(r.work["expanded_tasks"] for r in receipts)
        if selection.candidate is not None:
            if first_adoption is None:
                first_adoption = {
                    "decisions": total,
                    "parent_lower": str(selection.parent.lower),
                    "parent_upper": str(selection.parent.upper),
                    "gain_lower": str(selection.gain_lower),
                }
            if all(
                r is selection.candidate or r.upper <= selection.candidate.lower
                for r in receipts[1:]
            ):
                reason = "best_candidate_certified"
                break
        if all(r.upper <= receipts[0].lower for r in receipts[1:]):
            reason = "no_strict_gain_possible"
            break
        if total >= max_decisions or nodes >= max_nodes:
            break
        progress = False
        for i, evaluator in enumerate(evaluators):
            r = receipts[i]
            if (
                i in blocked
                or r.lower == r.upper
                or (i > 0 and r.upper <= receipts[0].lower)
            ):
                continue
            remaining = max_decisions - sum(x.work["decisions"] for x in receipts)
            remaining_nodes = max_nodes - sum(
                x.work["expanded_tasks"] for x in receipts
            )
            if remaining <= 0 or remaining_nodes <= 0:
                break
            receipts[i] = evaluator.run(
                r.work["decisions"] + min(quantum, remaining),
                r.work["expanded_tasks"] + remaining_nodes,
            )
            progress |= (
                receipts[i].resolved_mass != r.resolved_mass
                or receipts[i].work != r.work
            )
            if receipts[i].refusal not in (None, "Physical transition cap reached"):
                blocked.add(i)
        if not progress:
            reason = "no_affordable_expansion"
            break
    selection = select_interval(receipts[0], receipts[1:])
    return {
        "receipts": receipts,
        "selection": selection,
        "reason": reason,
        "first_adoption": first_adoption,
        "decisions": sum(r.work["decisions"] for r in receipts),
        "expanded_tasks": sum(r.work["expanded_tasks"] for r in receipts),
    }
