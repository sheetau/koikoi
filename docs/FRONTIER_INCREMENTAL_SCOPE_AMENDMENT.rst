Gate-scope amendment to the incremental evaluation
================================================

The original source collector selects the first identified learner decision
with at most five draws remaining. A forced transition can skip the learner's
five-draw decision, so the first eligible decision can have four remaining draws.
The selected wrapper checked for exactly five and raised an exception on these
valid original-source roots. This was an integration error in the experimental
harness, not evidence about playing strength.

The first run, runs/frontier_incremental_evaluation, is FAILED and closed.
Protocol SHA256 fa36dd16ce48f436b755ab397aaa660b4c04ae5e23f53e6b34d37ed6929e7737.
No chunk completed independent audit; no strength statistic is accepted.
The pending queue was cancelled after worker failures. Worker 68444 was stopped
while planning source 0 after its complete source ledger had been persisted.
Its partial planning count was not saved and is unknown, not zero. Other saved
planning selections remain in the failed run; they are not used as v2 plans.

All 15 acquired sources have complete seat/deck/record ledgers. Their original
case indices are 0, 16--20, 32, 48 and 64--70. The manifest
runs/frontier_incremental_failure_manifest.json has SHA256
bbc63063d172d9c77525db53745abae274044d5a13a99805dbdc0cc90ac10f19.
The correction retains every one at its original index and generates only the
4,081 missing cases. It does not discard, replace or resample failed cases.
Report these 15 as acquired before the mechanical repair, not as newly sampled
after the corrected freeze. All 4,096 remain in the originally fixed denominator.

The correction adds a dispatch layer. At exactly five draws it calls the
unchanged, integrated selected wrapper. At fewer draws it calls the original
symbolic controller with unchanged caps and returns its exact selection without
a frontier patch. The incremental expected value in this latter branch is zero
by policy identity, including original parent-value refusals. No policy budget,
local optimization, threshold, sample size, opponent or success criterion changes.
The scope error was diagnosed from root turn/phase and remaining draws; no
aggregate strength outcomes were used to select this correction.

Before resuming the same cohort in a separate revision directory, freeze the
corrected source code, tests, original protocol, failure manifest and all retained
source-file hashes. Copy retained ledgers unchanged and independently audit them.
Run planning and actual tails anew; do not import old plans or oracle values.
Keep retained source acquisition/native/sampling work separate from new work.
Previously interrupted work remains explicitly unmeasured. The original failed
run and its frozen files remain unchanged.
