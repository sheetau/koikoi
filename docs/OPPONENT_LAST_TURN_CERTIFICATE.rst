Opponent-only final-turn certificate
===================================

The separate ``OpponentLastTurnPlanner`` omits a provably losing opponent
continuation after the learner's last move. It does not alter the running
prospective comparison or the production agent.

Let D be the learner's current point margin. In the last turn of the final
round, with the opponent acting, only two outcomes remain: the opponent scores,
or the round ends with the dealer's one-point award. At most one point can be
transferred to the learner. Thus the terminal margin is at most D+2. If D < -2,
the learner loses under every remaining draw and opponent choice.

This predicate uses only public round, turn, actor and point fields. No learner
decision remains below a certified node. Its signed value is therefore minus
its integer probability mass, and its learner-policy map is empty. Removing
these rows and adding their exact contribution preserves a complete executable
learner map. No runtime fallback rule or policy-map sentinel is introduced.

The remaining continuation cutoff increases by the certified mass, preserving
the original upper-bound comparison. The retained-incumbent recorder receives
the recombined value, including when the certificate applies immediately after
the root action. This is an exact game-specific search simplification, not a
new general improvement theorem.

Physical terminal transitions and represented simulated completions retain their
existing meanings. ``certified_opponent_rows`` and ``certified_opponent_mass``
count the additional symbolic resolutions separately. They are not added to
physical decisions, terminal branches, source matches or training work.

The independent Python oracle performs all omitted opponent moves. It can
resolve missing observation-keyed opponent actions from the declared strict
actors; new adapter calls are reported separately. Missing learner-map keys
remain errors. The oracle never uses the final-turn certificate.

Capability tests compare both seats, both fixed/optimized continuations,
unequal world weights and the strict -2 boundary. They compare full values and
learner maps against the original planner, then independently replay all
weighted policies. A fifty-transition fixture completes an exact parent value
that needs374 transitions without the shortcut. This is a synthetic capability
and cost observation, not fresh match-win evidence or a measured latency gain.
Actual coverage and cost on natural roots remain to be evaluated after the
fixed512-source study closes.

The four-draw nested-choice fixture reduces physical transitions from698133 to
152994 while retaining the selected root action36, exact value1411/10560,
root-action bound4933/36960 and the full learner map. Both complete policies
pass independent weighted Python replay. These fixtures use a deterministic
synthetic actor and deliberately altered final-round scores, not naturally
sampled research outcomes. Nine tests pass in26.83s with zero skips; Ruff passes.
