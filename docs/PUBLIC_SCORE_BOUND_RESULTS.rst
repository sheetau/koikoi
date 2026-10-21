Public score-bound development screen: complete
==============================================

All eight fixed six-draw fixtures passed the artifact, posterior, program and
actual-tail checks. The new bound did not recover a parent reference or adopt
an improvement. Both versions value three parents and leave five unvalued at
the physical transition cap. Every actual action trace and outcome is unchanged.
This screen therefore supplies no additional playing-strength evidence.

.. csv-table:: Physical planning decisions at unchanged budgets
   :header: Source,Phase,Previous,Score bound,Result

   1,fixed,74,74,parent value 1
   19,fixed,73550,66390,parent value 0
   19,optimized,175080,149700,no strict gain
   88,fixed,1999988,1999979,parent unvalued
   112,fixed,1999996,1999996,parent unvalued
   237,fixed,44,44,parent value 1
   251,fixed,2000000,2000000,parent unvalued
   322,fixed,1999987,1999987,parent unvalued
   738,fixed,1999978,1999999,parent unvalued

Source 19's fixed/optimized phases use 9.735/14.496 percent fewer physical
decisions. Across the complete screen, physical decisions fall from 10,248,697
to 10,216,169 (0.317387 percent). The capped cases can explore different amounts
of represented work before refusal, so their near-equal physical counts do not
mean the traversed trees are equal. All eight common completed root values
agree. No newly completed parent or adoption required an independent value
oracle in this natural-fixture screen; the full oracles were exercised in the
correctness tests and separate preflight.

The concurrent operational timings are retained in the review receipt. Several
capped phases take longer, for example source 112 rises from 541.51 to 613.28
seconds. These were not isolated latency trials; no runtime speedup is claimed.

Separate work categories
------------------------

New planning physically simulates 10,216,169 decisions and 6,890,484 completions.
Represented work is 1,338,066,982 decisions and 340,490,250 completions. There
are 1,217,187 live observation nodes and 465,180 own decision nodes. Certified
loss rows/mass are 1,479,710 / 13,969,350; certified win rows/mass are zero.
The maximum frontier is 910 rows in both versions (a maximum, not a sum).

The independent support reconstruction uses 60,508 replay decisions, including
31,954 candidate and 28,554 opponent decisions / teacher calls. Actual execution
and its replay each use 42 decisions and 16 terminal tails. These do not add
new source matches. No fresh chance draws or natural-fixture oracle transitions
occurred. The original eight source/native/audit costs remain in
``GATE_COVERAGE_RESULTS.rst`` and are not charged again as new observations.

A supplementary replay checks all 991 recorded decisions / 64 round endings /
8 matches, reproducing final states and work. At 110 final-round decision states,
the score upper bounds never increase along a nonterminal transition. None of
those actual states is certified; the search reductions occur in hypothetical
continuations. Receipt: ``runs/score_bound_trajectory_checks.json``.

Verification and provenance
---------------------------

Eighteen targeted tests pass. Full Python replays agree with exact predecessor
values on both seats and synthetic positive/negative margins. A separate
independent true-parent preflight gives parent 204/275 and selected 97/110;
its saved program executes to terminal legally. These synthetic checks are
not natural strength observations.

``scripts/summarize_score_bound_screen.py`` refuses an incomplete screen,
rechecks all frozen inputs and saved artifact hashes, reconstructs each saved
selection from its phases, and reduces the complete eight-fixture record.
The incomplete-export refusal and final successful export were both checked.

Protocol: ``runs/score_bound_screen/protocol.json``.
SHA256: ``02215fbaf9206ffceebc01aa0578983beb9ffa56ae1854a1543c66b5c1ecb12d``.

Summary SHA256:
``018b5b0ffaf6f284c9ef2dc72fd835cacc9be14afe6a8a991964d8df310067a6``.

Complete review: ``runs/score_bound_screen_review.json``.
SHA256: ``8053995bfc1f9aef30e11b85479b7e7213095d2fce753a2afad38aa4a3b8787d``.

Session 30518 and queue 72696 exited successfully. This screen is closed; do
not restart it, replace fixtures or raise caps to manufacture an adoption.
The paper's previous strength results, canonical play and dealer are unchanged.

Next decision
-------------

The strict complete-parent prerequisite remains the limiting step. Investigate
policy-value intervals that preserve resolved terminal/certified mass when a
budget ends, and compare a candidate's lower bound to the parent's upper bound.
Any candidate must already define a complete public-information policy, such
as a root override followed by the unchanged parent. A partial explicit map
must not silently become a different, unevaluated policy.

This interval extension is not implemented or validated yet. Its mass ledger,
fixed total-budget allocation, executable policy binding and independent replay
must be tested before another development comparison. It would use established
chance-node bounding principles; ``SCORE_BOUND_RELATED_WORK.rst`` records the
primary-source context. Any selected new controller still needs prospective
evaluation on fresh sources.
