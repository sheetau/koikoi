Stronger-policy public-stock search: conditional protocol v5B
===========================================================

Question and launch gate
------------------------

Does the value of additional sampled computation persist after measured
policy improvement, against another improved policy under the same rule?
This is a separate study from V4. Its work design addresses the one-pilot
limitation documented in PUBLIC_STOCK_WORK_ANALYSIS.rst; it does not repair,
extend or relabel V4's closed cohort.

Launch only after V5A has completed all eighteen qualification endpoints,
independent reduction, checkpoint audits and a bound finished receipt.
Control and at least one finalist must qualify under V5A's twelve-contrast
criterion. Retain control and every qualifying finalist, with both parents;
do not rank finalists or choose a checkpoint using qualification scores.
Only the additional-update-384 checkpoints frozen before V5A qualification
are eligible. If the gate fails, do not launch this search study: diagnose
the adaptation result first.

Each actual opponent is the other new parent under the same rule. Rollouts
and continuation values use the searching actor's own fixed policy, as in
V4. No true hidden hand, hidden stock order, future randomness or opponent
oracle enters the policy. Improvement in V5A is relative to its two retained
opponents; it does not imply universal dominance or equal policy maturity.

Policy-specific preparation
---------------------------

Build new tables with scripts.build_auditable_round_values and audit them
with scripts.audit_public_round_values. Never attach an old-policy table to
a new checkpoint. Retain 512 conditional fresh deals for each live public
cell, eight rounds, two absolute dealers and 59 live player-0 point totals:
483328 conditional round rollouts per policy. Reduce all saved targets and
means; Python-replay the same four fixed offsets per cell as V3 (3776 rounds
per policy). Tables estimate the fixed policy's self-play value at a public
round boundary before conditioning on a fresh hidden deal; they are not
certified bounds, and conditional standard errors omit propagated table error.

Use initial seed 1250000000 + 2000000 * profile_index + 1000000 * parent,
where profile indices are fixed as control=0, preview-plus-choice=1 and
all-three=2 even if one finalist is ineligible. Keep the builder's per-round
stride 131072. Preparation counts are separate from evaluated matches.

Two-stage work-only calibration
------------------------------

All online endpoints use batch 32, completed-round limit 80 decisions,
confidence guard 2.5 and search seed 1117. A round that fails to complete is
an error, not an invitation to substitute a neural leaf. Targets remain
60000 and 240000 physical planning transitions per actual match. These are
expected-work targets, not hard deadlines, exact per-match caps or equal
inference/wall-time costs.

1. Initial calibration uses 64 new duplicate streams at 1260000000, both
   seats, 32 worlds per action for each retained system. Set each proposed
   count to round-half-up(target * matches * 32 / transitions), clamped to
   [2,256]. Require strictly increasing low/high counts.
2. Refinement uses 64 separate duplicate streams at 1270000000, both seats,
   at each proposed count. Correct once by
   round-half-up(target * matches * proposed_count / transitions), with the
   same clamp and increasing-count requirement. Do not iterate, interpolate
   scores, optimize strength or run extra calibrations. A count-order failure
   stops launch of the reserved evaluation and is reported.
3. Independently reduce all calibration endpoints and freeze their hashes,
   proposed counts and corrected counts before any reserved evaluation.

Only physical work may determine counts. Calibration outcomes are retained
for audit but cannot select rules, checkpoints, budgets or study extension.
The correction can reduce single-count extrapolation error; it cannot
guarantee the target in a different cohort with different stopping histories.

Reserved evaluation
-------------------

Use 256 new duplicate streams beginning at 1280000000, shared by all retained
systems and budgets, with both seats. Evaluate zero search and the two frozen
corrected counts. With two qualifying finalists this is eighteen endpoints
and 9216 actual matches; with one it is twelve endpoints and 6144 matches.
The independent units are 256 duplicate streams, not the number of matches.
Save and Python-replay every scored decision. Independently reduce match
scores, margins, rounds and decisions, and verify checkpoint, table,
opponent, source and cohort identities. Search work is recorded separately;
do not describe it as independently replayed internal planning.

Scores count ties as one half. For each eligible finalist and each parent,
form three paired-stream contrasts: high minus zero, that gain minus the
control's corresponding gain, and high minus low. If both finalists qualify,
also compare preview-plus-choice gain minus all-three gain for each parent.
There are six contrasts with one finalist and fourteen with two. Use 20000
paired-stream bootstrap replicates, seed 1123, and two-sided quantiles
0.05/(2*m) and 1-0.05/(2*m), where m is the fixed eligible contrast count.
The nominal 95% familywise target applies within this new study only.

A finalist passes its score criterion only if all six of its lower bounds
are positive. A ranking requires both finalist-difference intervals to
exclude zero in the same direction. This ranks additional search gains,
not absolute playing strength across different games. No across-study
familywise guarantee or pooled V2/V3/V4/V5B estimate is claimed.

Every nonzero held-out endpoint, including control, must be within +/-10%
of its target for the full matched-work claim. Retain all actual costs and
deviations whether this condition passes or fails. A finalist's matched-work
criterion requires both the full cost condition and its six score bounds.
Matched-work ranking additionally requires the ranking criterion. Equal
expected transitions still exclude neural inference, sampling, bookkeeping
and offline preparation; no equal-wall-time claim follows.

Completion and interpretation
-----------------------------

Complete all fixed eligible endpoints and report every comparison. Do not
extend deals, change tolerance, choose a different intermediate checkpoint,
drop a failed parent, repeat calibration or launch another study solely to
obtain significance. A failed criterion is a completed result.

If a qualified finalist passes, the study establishes retained computation
value for these measurably improved actors against changed improved opponents.
If it fails, report which of gain, excess, additional-budget benefit and work
admissibility failed. Qualification remains evidence of policy improvement;
search failure does not retroactively invalidate it. Changes from V4 cannot
be causally attributed to learning alone because actors, opponents, tables,
budgets and deal cohorts differ. Neither outcome proves a globally optimal
rule or that further AI improvement is impossible.

The manuscript must distinguish the standard-game verified policy method
from these empirical variant-search results and retain the complete chain
of independent confirmations, robustness tests and negative results.
