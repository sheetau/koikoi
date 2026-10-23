"""Budgeted refinement with complete subpolicy reuse; frozen solvers stay intact.

The cache is private to one immutable root, actor set and solve mode. Keys bind
the entire weighted belief, including public history, rather than a sampled
hidden world. Only fully returned choices are cached; interrupted choices and
upper bounds alone never become executable policies.
"""

import pickle
from collections import Counter, OrderedDict
from copy import deepcopy
from fractions import Fraction

from scripts.frontier_policy_patch import FrontierPatch, parent_frontier
from scripts.incumbent_tail_controller import EvaluatedPhase, bind_worlds
from scripts.interval_tail import IntervalEvaluator, make_program, select_interval
from scripts.proposal_interval_tail import compile_proposal
from scripts.score_bounded_tail import ScoreBoundPlanner


class CompletedChoices:
    def __init__(self, max_bytes=64 * 1024 * 1024):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("Nonnegative cache byte allowance required")
        self.max_bytes, self.bytes, self.peak_bytes = max_bytes, 0, 0
        self.rows = OrderedDict()

    def get(self, key):
        row = self.rows.get(key)
        if row is None:
            return None
        self.rows.move_to_end(key)
        return deepcopy(row[0]), row[1]

    def put(self, key, result, decisions):
        size = len(pickle.dumps((key, result), protocol=5))
        if size > self.max_bytes:
            return
        if key in self.rows:
            self.bytes -= self.rows.pop(key)[2]
        while self.rows and self.bytes + size > self.max_bytes:
            self.bytes -= self.rows.popitem(last=False)[1][2]
        self.rows[key] = (deepcopy(result), decisions, size)
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)


class ReusablePlanner(ScoreBoundPlanner):
    def __init__(self, dynamics, *, cache=None, **kwargs):
        super().__init__(dynamics, **kwargs)
        self.cache = CompletedChoices() if cache is None else cache
        self.context = None

    def solve(self, worlds, optimize_continuation=True):
        # Actor identity is fixed by this dynamics object. Cross-root or mode
        # reuse is forbidden even if a caller keeps the planner alive.
        context = (
            worlds.viewer,
            worlds.draws,
            worlds.model_names,
            bool(optimize_continuation),
            self.max_depth,
            tuple(
                (a.shape, a.dtype.str, a.tobytes())
                for a in (worlds.states, worlds.models, worlds.weights)
            ),
        )
        if self.context is not None and context != self.context:
            raise ValueError("Reusable planner belongs to a different belief or mode")
        self.context = context
        return super().solve(worlds, optimize_continuation)

    def choose(self, states, models, weights, depth, root=False):
        # Root evaluation also updates IncumbentBeliefPlanner's interrupted-root
        # receipts, so it must always pass through the original implementation.
        if root:
            return super().choose(states, models, weights, depth, root)
        key = (
            id(self),
            depth,
            tuple(
                (a.shape, a.dtype.str, a.tobytes()) for a in (states, models, weights)
            ),
        )
        cached = self.cache.get(key)
        if cached is not None:
            result, saved = cached
            self.work["completed_choice_hits"] = (
                self.work.get("completed_choice_hits", 0) + 1
            )
            self.work["reused_choice_transitions"] = (
                self.work.get("reused_choice_transitions", 0) + saved
            )
            return result
        before = self.work["decisions"]
        result = super().choose(states, models, weights, depth, root)
        self.cache.put(key, result, self.work["decisions"] - before)
        return result


class LocalRefinement:
    def __init__(self, worlds, dynamics, parent_id, opponent_ids, cache, max_nodes):
        self.worlds, self.dynamics = worlds, dynamics
        self.parent_id, self.opponent_ids = parent_id, opponent_ids
        self.binding = bind_worlds(worlds, dynamics, parent_id, opponent_ids)
        self.planner = ReusablePlanner(dynamics, cache=cache, max_nodes=max_nodes)
        self.reference = IntervalEvaluator(
            worlds, dynamics, make_program(worlds, dynamics, parent_id, opponent_ids)
        )
        self.candidate, self.phase = None, None
        self.proposal_work = Counter()
        self.attempts, self.done = 0, False
        self.max_nodes = max_nodes

    @property
    def decisions(self):
        return self.proposal_work["decisions"] + self.reference.work["decisions"]

    def refine(self, allowance):
        """Spend at most allowance NEW transitions, including all failed restarts."""
        before = self.decisions
        parent = self.reference.receipt()
        if parent.lower == 1:
            self.done = True
            return
        proposal_budget = allowance // 2
        if proposal_budget < 1:
            return
        self.planner.max_decisions = proposal_budget
        result = self.planner.solve(self.worlds, True)
        self.attempts += 1
        self.proposal_work.update(result.work)
        phase = EvaluatedPhase(self.binding, "optimized", result)
        candidate = compile_proposal(
            phase, self.worlds, self.dynamics, self.parent_id, self.opponent_ids
        )
        if self.phase is None or (
            candidate is not None
            and (self.candidate is None or candidate.lower > self.candidate.lower)
        ):
            self.phase, self.candidate = phase, candidate
        remaining = allowance - (self.decisions - before)
        if self.candidate is not None:
            cap = self.reference.work["decisions"] + remaining
            while parent.lower < self.candidate.lower <= parent.upper:
                old = self.reference.work["decisions"]
                parent = self.reference.run(min(cap, old + 4096), self.max_nodes)
                if self.reference.work["decisions"] == old or parent.refusal not in (
                    None,
                    "Physical transition cap reached",
                ):
                    break
            selection = select_interval(parent, [self.candidate])
            self.done = selection.candidate is not None or (
                result.plan is not None
                and result.plan.search_complete
                and (
                    parent.lower >= self.candidate.lower
                    or parent.resolved_mass == parent.total_mass
                )
            )
        if self.decisions - before > allowance:
            raise ValueError("Local refinement exceeded its shared allowance")

    def result(self):
        parent = self.reference.receipt()
        return {
            "proposal_phase": self.phase,
            "candidate_certificate": self.candidate,
            "parent_receipt": parent,
            "selection": select_interval(
                parent, [] if self.candidate is None else [self.candidate]
            ),
            "decisions": self.decisions,
            "proposal_work": dict(self.proposal_work),
            "attempts": self.attempts,
        }


def solve_reusable_frontier(
    worlds,
    dynamics,
    parent_id,
    opponent_ids,
    *,
    draws=3,
    max_decisions=2000000,
    prefix_decisions=250000,
    pilot_decisions=2000,
    local_decisions=100000,
    max_nodes_per_phase=25000,
    max_rows=65536,
    cache_bytes=64 * 1024 * 1024,
):
    """Mass-ordered broad pilot, then geometric refinement of unresolved nodes.

    Parent intervals resume in place; completed subpolicies survive proposal
    restarts. Every node keeps the parent unless its same-belief gain is certified.
    This may search more nodes but makes no claim to dominate a different search
    schedule. Fresh paired evaluation must determine the practical effect.
    """
    if any(
        type(x) is not int or x < 1
        for x in (
            max_decisions,
            prefix_decisions,
            pilot_decisions,
            local_decisions,
            max_nodes_per_phase,
            max_rows,
        )
    ):
        raise ValueError("Positive integer budgets required")
    frontier = parent_frontier(
        worlds,
        dynamics,
        parent_id,
        opponent_ids,
        draws=draws,
        max_decisions=min(max_decisions, prefix_decisions),
        max_rows=max_rows,
    )
    used, jobs = frontier.work["decisions"], []
    cache = CompletedChoices(cache_bytes)
    if frontier.complete:
        for key, node in frontier.nodes:
            if max_decisions - used < 2:
                break
            job = LocalRefinement(
                node, dynamics, parent_id, opponent_ids, cache, max_nodes_per_phase
            )
            job.refine(min(pilot_decisions, local_decisions, max_decisions - used))
            used += job.decisions
            jobs.append((key, job))
        allowance = pilot_decisions * 4
        while used < max_decisions and any(
            not j.done and j.decisions < local_decisions for _, j in jobs
        ):
            progressed = False
            for _, job in jobs:
                budget = min(
                    allowance, local_decisions - job.decisions, max_decisions - used
                )
                if job.done or budget < 2:
                    continue
                before = job.decisions
                job.refine(budget)
                used += job.decisions - before
                progressed |= job.decisions > before
            if not progressed:
                break
            allowance *= 4
    comparisons = tuple((key, job.result()) for key, job in jobs)
    masses = {key: sum(map(int, node.weights)) for key, node in frontier.nodes}
    gain = sum(
        (
            Fraction(masses[key], frontier.total_mass) * r["selection"].gain_lower
            for key, r in comparisons
        ),
        Fraction(0),
    )
    patch = FrontierPatch(frontier, comparisons, gain, used)
    patch.cache_peak_bytes = cache.peak_bytes
    return patch
