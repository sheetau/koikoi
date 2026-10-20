Five-draw belief-tail capability protocol
=======================================

The three-draw gate left little opportunity for improvement: one exact
conditional gain among sixteen eligible natural roots. Deferred chance expansion
preserved all eager policies while reducing executed transitions by 79.7%.
This check uses that representation to begin up to two draws earlier.

Sources and intervention
------------------------

Fix 64 complete unchanged-parent source matches: two retained parents x assisted
original WP/Point x both seats x eight replicates. Seed is
122110001+1000*opponent_index+replicate, with WP=0 and Point=1. Parents and seats
share sixteen distinct deal streams. Use strict singleton CPU inference with
two threads and the same public endgame assistance as the closed comparison.
No training or reserved101M data is used.

At each source, retain the first learner decision in the final paper8 round with
at most five remaining draws, provided public evidence from an earlier round has
already identified one member of the declared deterministic WP/Point model set.
Retain that public identification and its availability index. The planner gets
the identified model, not the scheduler's opponent label. Finish all sources
with the unchanged parent. Keep every missing root; do not replace sources.

At an eligible root, reconstruct complete current-round history-compatible hand
support. Use a shared uniform valid-deal prior and the already identified model.
The maximum candidate-hand count is1000. Deferred hypotheses carry the exact
remaining prefix multiplicity. Caps are4096expanded frontier rows,500000observed
nodes and32decisions of depth. Keep refusals without choosing a later root.
No partial solution is deployed or reported as an exact optimum.

Compare the unchanged parent's conditional value, the best first action followed
by its fixed future policy, and the best complete observation-contingent policy.
Run fixed before optimized with shared policy caches. Preserve all action
values, contingent policies, counters, adapter calls, times and support evidence.
These timings are not isolated cold-cache throughput comparisons.

Independent policy-value replay
--------------------------------

``scripts/reference_tail_policy.py`` replays every root action's complete policy
using only Python reference transitions. It identifies a stock draw by executing
the action and observing the stock pointer change. At a draw, enumerate all
possible top cards, divide the incoming integer mass equally, and continue each
successor. The probe successor is reused as one branch and counted once.
The oracle does not call the planner's draw predicate or native transition code.

Every resulting rational action value must match the planner. Opponent actions
come from its frozen own/public-keyed cache; this verifies transitions, chance
mass, policy execution and terminal values, not a second neural forward pass.
Physical replay transitions and represented prefix work are separate.

Freeze and hash all completed plans before actual-world assessment. Then execute
unchanged parent, best fixed-continuation policy and best optimized policy from
each original root, with fresh strict opponent calls. Verify that the unchanged
parent exactly recovers the source's recorded suffix and terminal state.
Keep all actual actions, deviations and point margins.

Work and interpretation
-----------------------

Count source matches/rounds/decisions, native source verification, support replay,
planning branches, independent policy replay and actual conditional completions
separately. A refused optimized plan may follow a completed fixed plan: retain
both the completed work and the labelled partial-stage work. Report the largest
frontier as a maximum, never a sum. Represented prefixes are probability mass
units, not executed simulations or independent matches.

The check measures opportunity, exact model-conditional gain and computation.
It is not a powered replicated strength evaluation. All64sources, including
missing/refused roots, belong in the report; additional worlds or continuations
do not add independent deals. Do not automatically extend collection after
seeing results or promote a model from a favorable conditional example.

The new oracle passes both-seat, nonunit-mass tests against all fixed/optimized
root policies and rejects a corrupted value. A collection regression reproduces
one closed source twice, keeps its actions/outcome/public identification, and
confirms that the five-draw gate selects an earlier root than the three-draw gate.
Three tests pass, no skips, in4.60seconds. The two source replays use one already
consumed stream and are verification work, not new source evidence.

Commit implementation and protocol before launch::

    .venv/Scripts/python.exe -m scripts.check_earlier_belief_tail

The default output is ``runs/earlier_belief_tail_capability``. The runner refuses
an existing directory; preserve failed or completed runs.
