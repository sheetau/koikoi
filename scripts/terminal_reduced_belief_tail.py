"""Value terminal mass before constructing live information-set decisions."""

import numpy as np

from scripts.belief_tail import TailBudgetExceeded
from scripts.deferred_belief_tail import DeferredBeliefPlanner


class TerminalReducedPlanner(DeferredBeliefPlanner):
    """Preserve exact values/policies while omitting unnecessary terminal keys.

    max_nodes counts LIVE observable groups here. A separate physical transition
    cap makes the computation explicit; it is not the old all-node budget.
    """

    def __init__(self, dynamics, max_nodes=500000, max_depth=32, max_decisions=2000000):
        super().__init__(dynamics, max_nodes, max_depth)
        if type(max_decisions) is not int or max_decisions < 1:
            raise ValueError("Positive integer physical transition cap required")
        self.max_decisions = max_decisions

    def advance(self, states, actions):
        if self.work["decisions"] + len(states) > self.max_decisions:
            raise TailBudgetExceeded("Physical transition cap exceeded")
        return super().advance(states, actions)

    def continuation(self, states, models, weights, depth):
        if depth > self.max_depth:
            raise TailBudgetExceeded("Decision depth cap exceeded")
        terminal = self.dynamics.terminal(states)
        count = int(terminal.sum())
        self.work["terminal_rows_valued"] = (
            self.work.get("terminal_rows_valued", 0) + count
        )
        self.work["terminal_mass_valued"] = self.work.get(
            "terminal_mass_valued", 0
        ) + sum(map(int, weights[terminal]))
        total = (
            int(
                np.dot(
                    weights[terminal],
                    self.dynamics.outcomes(states[terminal], self.viewer),
                )
            )
            if count
            else 0
        )
        if terminal.all():
            return total, {}
        value, policy = super().continuation(
            states[~terminal], models[~terminal], weights[~terminal], depth
        )
        return total + value, policy

    def solve(self, worlds, optimize_continuation=True):
        # Avoid leaving an old solve's work in a failed root validation.
        self.work = {}
        try:
            plan = super().solve(worlds, optimize_continuation)
            plan.work["live_observation_nodes"] = plan.work.pop("observation_nodes")
            assert plan.work["terminal_rows_valued"] == plan.work["completions"]
            assert (
                plan.work["terminal_mass_valued"]
                == plan.work["represented_completions"]
            )
            return plan
        finally:
            if "observation_nodes" in self.work:
                self.work["live_observation_nodes"] = self.work.pop("observation_nodes")
