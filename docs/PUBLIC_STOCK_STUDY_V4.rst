Public-stock robustness and work calibration: protocol v4
=========================================================

Question and decision boundary
------------------------------

V2 and V3 passed their original independent confirmations. This separate
study asks whether the benefit survives a changed opponent and comparable
physical rollout work, and whether either of the two finalists has a
consistent advantage. It does not extend, pool or reselect either earlier
cohort. The current goal includes this robustness assessment; the 28 prior
confirmation endpoints were not a claim that research was exhausted.

Retain exactly three profiles: preview0_choice0_stakes0 (control),
preview1_choice1_stakes0 (preview and choice), and preview1_choice1_stakes1
(all three). Use both retained V2 parents and their existing V3 continuation
tables. The actual opponent is the other parent's adapted no-search policy
under the same rule. Search is unchanged completed-round search: it uses
its own retained policy for rollouts and its own self-play continuation
cache. It is not given an opponent oracle or the opponent's hidden cards.
This tests robustness of the deployed heuristic to a changed policy; it does
not claim robustness to all strategy classes or correct opponent modelling.

Work calibration, without score selection
-----------------------------------------

Use 32 duplicate streams beginning at 1,200,000,000 for all six policy/rule
systems, both seats, at 32 particles per legal action. These 384 pilot
matches calibrate cost only. Do not select a rule, checkpoint, guard or
budget target using pilot scores. For each system let c be its observed
physical rollout transitions per actual match divided by 32. Freeze
particles at floor(target / c + 0.5), clamped to [2, 256], for targets of
60,000 and 240,000 transitions per match. Require positive c and distinct
increasing particle counts. No repeated calibration or outcome-based tuning.

These are matched expected-work targets, not hard execution deadlines or
exact per-match caps. A matched-work conclusion requires every held-out
nonzero endpoint's observed mean transitions per match to lie within 10%
of its target. Report every actual value and deviation. If this check fails,
retain the score results as changed-opponent evidence but label the
matched-work question unresolved. Do not silently describe nominal particle
counts, table preparation or GPU wall time as equal computational cost.

Independent evaluation and comparisons
--------------------------------------

Use 128 new duplicate streams beginning at 1,210,000,000, both seats, the
three rules and two parents, and zero/calibrated-low/calibrated-high budgets:
4,608 actual matches in 18 endpoints. All arms share the fixed evaluation
stream block for paired comparisons; there are 128 independent duplicate
streams, not 4,608 independent deals. Use batches of 32, the V3 80-decision
round-completion hard limit, guard 2.5, and search RNG seed 1091. Execute
GPU evaluation jobs serially; record elapsed time but do not infer a general
latency benchmark. No new policy training or new continuation preparation
occurs in this stage. Reused preparation retains its original accounting.

For each finalist and each parent compute high-minus-zero gain,
its gain minus the corresponding control gain, and high-minus-low gain.
Also compute preview-plus-choice gain minus all-three gain for each parent.
These are 14 fixed comparisons. Use 20,000 paired duplicate-stream bootstrap
replicates (seed 1093), with two-sided Bonferroni intervals at quantiles
0.05 / 28 and 1 - 0.05 / 28. This nominal familywise target is within V4;
no joint coverage across V2/V3/V4 is asserted.

A finalist supports the stronger computation claim if its gain, excess gain
and high-minus-low lower bounds are positive for both parents, subject to
the work-admissibility check for a matched-work claim. A ranking of the two
finalists requires both simultaneous difference intervals to exclude zero
in the same direction. Otherwise retain an unresolved ranking. No extensions,
pooled retries, extra rules or changed endpoints after evaluation begins.

Integrity and subsequent work
-----------------------------

Bind the original checkpoints, opponent checkpoints, continuation tables,
metadata, protocols and all evaluation/export sources. Save and Python-replay
every scored action; independently reduce final scores, margins, match,
round and scored-decision counts. Rollout counts and physical planning
transitions are recorded work counters, not independently replayed internal
search traces. Keep calibration, evaluation, earlier training, reused cache
preparation and verification work separate.

Longer adaptation is a subsequent study, not evidence supplied by V4.
If a finalist retains a reproducible benefit, compare longer-trained
policies with the retained ones on fresh deals before calling them stronger,
then test the search gain on another reserved cohort. If V4 fails, inspect
the changed-opponent and work effects before choosing that next experiment.
Other rule families require separately justified protocols; they cannot alter
the completed cohorts or the endpoints fixed here.
