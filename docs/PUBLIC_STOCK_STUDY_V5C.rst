Qualified variants with retained control: protocol v5C
====================================================

Why this is a separate study
----------------------------

V5A completes all 36864 qualification matches. Both public-stock finalists
qualify as improved against both retained opponents for both parents;
control does not. Thus V5B's all-improved-policy launch gate fails and V5B
is not executed. We do not change that gate or claim its question answered.

There is still a measurable maturity question for the two improved finalist
actors. This new study tests their search against the other improved parent,
while retaining the original control actors. It measures computation value
for these retained systems. Adaptation budgets differ: control has 128 updates,
finalists 512; original parent training is reused separately. Cross-rule
differences cannot be attributed to the rules alone or described as a
matched-training comparison. Within each finalist, zero/low/high search uses
the same improved actors, opponent and policy-specific continuation table.

Launch only after complete independently reduced V5A evidence and its finish
receipt verify. Recompute all twelve qualification decisions. Require failed
control qualification and at least one qualified finalist; keep every
qualified finalist with both parents. If control qualifies, use V5B instead.
Never select intermediate checkpoints, drop one parent or promote an
unqualified new actor. Qualification data cannot be reused as search-test data.

Systems and preparation
-----------------------

For control, retain both original 128-update actors and their audited V3
tables exactly as identified in V4's protocol and V3's evidence. Recheck
artifact identities and saved audit coverage; do not regenerate those tables
or count their past preparation as new work. Control faces the other retained
old parent. For each eligible finalist use the V5A additional-update-384 final
checkpoint and face the other new parent under the same rule.

Build and independently audit new 512-sample public-boundary tables for every
new finalist policy, using the unchanged builder/auditor and V5B's coverage:
483328 conditional round rollouts, all targets/means reduced, 3776 fixed-stratum
Python round replays per table. Rollouts and continuation values use the
searching actor's own policy; actual opponent identity and hidden actual
cards/futures never enter rollout observations.

New-table seed is 1290000000 + 2000000 * profile_index + 1000000 * parent,
with fixed indices control 0, preview-plus-choice 1, all-three 2 and the
builder's 131072 per-round stride. The control uses its unchanged old table
seeds. Report new preparation, reused preparation and evaluation separately.

Calibration, evaluation and decisions
-------------------------------------

Use the two-stage work-only calibration and all inferential/acceptance rules
specified in PUBLIC_STOCK_STUDY_V5B.rst, with these new seed blocks:

* Initial calibration: 64 duplicate streams at 1300000000, 32 worlds.
* One refinement: 64 separate streams at 1310000000 at the proposed low/high
  counts, followed by one exact half-up work-only correction.
* Reserved evaluation: 256 separate streams at 1320000000, both seats,
  zero search and the two frozen corrected counts.
* Online search seed 1137; paired-bootstrap seed 1151.

Targets remain 60000/240000 physical transitions per match; clamp [2,256],
strictly increasing low/high counts, batch 32, completed-round limit 80 and
guard 2.5 remain fixed. Retain a nominal within-study 95% familywise target:
20000 paired bootstrap replicates; two-sided quantiles 0.05/(2*m) and its
complement for m=6 with one finalist or m=14 with both. With both finalists,
evaluation has 18 endpoints and 9216 matches on 256 independent duplicate
streams; with one, 12 endpoints and 6144 matches. All scored decisions must
pass Python replay; independent reduction verifies outcomes/counts/identities.
Internal planning work is recorded rather than independently replayed.

Each finalist requires positive lower bounds for high-minus-zero, excess
over retained control's corresponding gain, and high-minus-low, for both
parents. When both qualify, ranking requires both paired gain-difference
intervals to exclude zero in the same direction. Every nonzero endpoint,
including retained control, must meet its target within +/-10% for a full
matched-work claim. A finalist's matched-work pass requires both its six
score inequalities and that full work condition. No equal-wall-time or
equal-training claim follows.

Finish the complete fixed study and report all costs, comparisons and failed
criteria. Do not extend cohorts, retune after evaluation, change tolerance
or repeat studies solely to obtain significance. A failed criterion is a
completed result. Update the manuscript with this distinction from the
unexecuted V5B plan. Remaining rule rankings or strength claims must follow
the measured intervals; globally maximal AI strength is not established.
