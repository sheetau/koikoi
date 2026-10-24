Public-stock completed-round rollout protocol v3
================================================

The v2 screen continues with its original settings. Its control results
motivate an independent study of completed-round rollouts with reused
continuation values. V3 does not pool, replace or relabel v2 outcomes.

Reuse all sixteen completed v2 adapted checkpoints, with no further policy
training or checkpoint selection. For each policy/rule, prepare a public
round-boundary table as specified in PUBLIC_STOCK_ROUND_VALUE.rst, using
512 conditional round rollouts per cell. Retain absolute dealer identity.
The table contains 8 x 2 x 59 live cells, hence 483,328 conditional round
rollouts per model. Record these separately from full evaluation matches.
Table seed blocks are 900,000,000 + 2,000,000 * profile_index + 1,000,000 *
parent_index, using v2 profile order. Round r adds r * 131,072. These blocks
are disjoint at the fixed sample count. Store all actions and final public
boundary states. Independently reduce every target/mean, and Python-replay
sample offsets 0, 170, 341 and 511 in every cell (3,776 rounds per table).

Development uses 32 new duplicate streams beginning at 930,000,000, all
eight rules, both parent identities, both seats and budgets 0/8/32/128
particles per legal root action. Each arm faces its own fixed adapted actor
without search. Rollouts finish the current round, with an 80-decision
hard limit; failure to settle is an error, not a neural-value cutoff. They
then reuse the checkpoint-specific public boundary table. Use a paired
advantage guard of 2.5 standard errors, search RNG seed 971 and batches of
32 duplicate streams. The guard remains heuristic because hidden-hand
beliefs and offline table values are estimated. Every scored match is
independently replayed in Python.

Selection maximizes the worse parent's 128-particle gain over no search
among the seven noncontrol rules, with lexicographic tie-breaking. The
development gate requires both gains to be positive and the worse-parent
gain to exceed the control's. Publish every rule and budget, including
nonmonotonic or negative effects. Also report 128-minus-8-particle changes;
a gain over no search alone does not establish benefit from greater depth
or monotonically increasing computation.

If the gate passes, reserve 128 new duplicate streams beginning at
932,000,000 for selected rule and control, both parents, both seats, and
budgets 0/8/128. The four primary inequalities are positive high-budget
gain for each parent and positive selected-minus-control gain for each
parent. Use 20,000 paired deal-cluster bootstrap replicates (seed 981),
one-sided 98.75% lower bounds, and require all four to exceed zero. The
128-minus-8 comparisons are secondary. Do not extend, pool or reselect
after confirmation. The smaller confirmation is a bounded rule-search
study; an inconclusive interval must remain inconclusive.

Keep offline conditional simulation, online planning, evaluation decisions,
rounds, matches and verification work separate. Shared GPU activity makes
elapsed times operational observations, not isolated latency benchmarks.
Particle ratios do not equalize actual work between rules. This study tests
fixed trained systems and sampled continuations; it cannot identify a
globally optimal rule or certify every adopted action.
