Recovery of the fifteen bounded-tail refusals
=============================================

The fixed Stage 1 comparison is complete. All 18 saved completed roots preserve
the optimal action, exact chosen value and complete chosen policy. Every computed
root value/map matches the saved complete solution, and independent Python
weighted replay agrees. Twenty of 36 ceiling plans omit some root alternatives,
always with a certified chosen value of 1.

The paired control/ceiling times are 65.597451/35.732193 seconds, a 45.5281% observed
reduction. Physical transitions are 5,513,147/3,535,665, a 35.8685% reduction.
The new independent oracle executes 3,404,854 transitions and 3,003,624 terminal
completions. These measurements reuse closed sources and add no strength sample.

Stage 1 protocol SHA256:
``aae5eec7dea9637385180462a96c00ad8b0f059355a5d3305a7b6c62a52bc140``.
Complete SHA256:
``6efe77307aa016c8d6d4329f73540c0bf0dc3928262dfdfdbcdc019ee2d24308``.
Session 46597 / PID 22668 exited 0. The recovery preflight checks 26 input bindings,
the old study artifacts, all 18 completed source indices and the precise refusal
set before starting any recovery computation.

This executes Stage 2 of the already fixed CEILING_BELIEF_TAIL.rst protocol.
The fifteen source indices are 0,1,4,18,20,21,25,26,34,36,50,52,53,55,59.
Use all of them, without replacement or new source collection.

The explicit per-plan limits are 500,000 live nodes, 2,000,000 physical transitions,
4,096 frontier rows and 32 decisions of depth. They differ from the original
500,000-all-node limit. The original five-draw/1,000-candidate-hand gates remain.
All remaining refusals and completed/partial work are retained. No automatic
increase in limits follows a refusal.

Reconstruct support with the publicly identified opponent. Run fixed then
optimized ceiling planning with shared caches. Independently replay every
computed root policy and preserve unevaluated actions explicitly. Compare the
best certified optimized value against fixed continuation and the parent's
fixed-continuation action. All plans are frozen and hashed before actual tails.
Actual tails use fresh strict opponent calls and the complete selected policy;
the parent's recorded source suffix must reproduce exactly.

No source matches are added. The existing 18 completed policies/outcomes and
31 missing roots remain in the original 64-source denominator. Any combined gain
is exploratory because the computational extension follows the original study.
No new prospective win comparison, training, reserved101M data or promotion.

Command after committing this runner::

    .venv/Scripts/python.exe -m scripts.recover_ceiling_tails

The fresh output is ``runs/ceiling_tail_recovery``. Preserve completed or failed
directories. The runner refuses an existing output and an incomplete or changed
Stage 1 verification.
