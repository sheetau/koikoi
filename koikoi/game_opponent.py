"""Explicit twelve-round deployment adapter for the existing frozen opponents."""

from .layout import BANKRUPT, ROUND, ROUNDS
from .play import load_opponent
from .rule_variants import solved_variant_action


class GameOpponent:
    def __init__(self, name):
        # Keep checkpoint identity verification and published source pinning.
        self.agent = load_opponent(name).agent
        # Apply payoff-correct public endgame assistance to the real game below,
        # not to the eight-round feature projection.
        if hasattr(self.agent, "requires_endgame"):
            self.agent.requires_endgame = False
        self.name = name

    def act(self, env):
        forced = solved_variant_action(env)
        if forced is not None:
            return forced
        if self.name == "greedy":
            return self.agent.act(env)
        projected = env.clone()
        projected.state[ROUNDS], projected.state[BANKRUPT] = 8, 1
        # Align the last eight rounds; early rounds use the opening context.
        # This is disclosed feature transfer, not a newly trained twelve-round AI.
        projected.state[ROUND] = max(0, int(env.state[ROUND]) - 4)
        action = int(self.agent.act(projected))
        if not env.legal()[action]:
            raise RuntimeError("Frozen opponent returned an illegal game action")
        return action
