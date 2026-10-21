Shared-budget interval screen: complete
======================================

All eight fixed sources passed. No root-only candidate was adopted and no
previously unvalued parent became exact. Actual traces and match outcomes
remain identical to the unchanged parent. The interval mechanism is working,
but this screen does not establish additional playing strength.

It does establish one additional no-gain decision without completing the
parent value. At source 251, stopping has exact value 0 while the parent's lower
bound is positive (41/56216160). The candidate cannot improve on the parent,
so comparison stops after 4,116 physical transitions. The preceding fixed-phase
evaluation spent 2,000,000 transitions and still lacked the parent value. This
is a root-only dominance conclusion; it does not rule out improvement from
changing later continuation decisions.

.. csv-table:: Complete screen, shared budget per source
   :header: Source,Physical decisions,Parent exact,Decision

   1,74,yes (1),no strict root-only gain possible
   19,66390,yes (0),no strict root-only gain possible
   88,1999998,no,budget ends with overlapping intervals
   112,1999999,no,budget ends with overlapping intervals
   237,44,yes (1),no strict root-only gain possible
   251,4116,no,no strict root-only gain possible
   322,1999995,no,budget ends with overlapping intervals
   738,1999984,no,budget ends with overlapping intervals

Each hard source splits approximately two million transitions across all three
root programs. Parent intervals at budget exhaustion remain wide: roughly
[.001,.933], [.006,.731], [.008,.987] and [.000,.811] for sources 88,112,322,738.
All five previously completed fixed-policy values lie inside the new intervals.
No newly exact parent or adoption required a natural-fixture full-value oracle;
the independent interpreters were exercised in the targeted correctness tests.

Physical planning work totals 8,070,600 decisions, compared with 10,066,469 for the
preceding score-bound fixed phases using the same root-only candidate family.
The reduction is 19.8269 percent, predominantly the source 251 early rejection.
The previous optimized phases are excluded from that comparison because they
searched a richer contingent-policy family. Concurrent operational timings
are saved in the receipt; this is not an isolated latency benchmark.

Accounting and provenance
-------------------------

Planning physically executes 4,960,004 terminal completions. Represented work
is 1,970,844,502 decisions and 359,992,026 completions. Expanded tasks total 982,755;
these are distinct from the preceding planner's live-information-node metric.
Certified loss rows/mass are 1,491,388 / 13,854,420, with no certified win mass.
Per-program frontier peaks are retained separately, not summed as an observed
concurrent maximum.

Support reconstruction uses 60,508 replay decisions:31,954 candidate and 28,554
opponent decisions / teacher calls. Actual execution and its replay each use
42 decisions across 16 terminal tails. Natural-fixture full-value oracle work is
zero; there are no fresh source matches or chance draws in this screen.

The complete-only exporter rechecks frozen inputs, result and program hashes,
reconstructs selections from saved intervals, verifies the shared budgets and
resolved-mass accounting, and checks previously exact values. Incomplete export
was refused; the final complete export passes.

Protocol SHA256:
``b55a1e34bac6f1ff0fa55c6f9deb72166d39d86e8923c00b6151e0bf79d4879d``.
Summary SHA256:
``06f36a1daea59ad522f4b6799e328e9d081297397300d3b33da05b1f2c4c8d9f``.
Review: ``runs/interval_tail_screen_review.json``.
SHA256: ``f 8 e 01985 dedaa 30 ad 18 b 4 f 21540531 dce 84 a 653 cd 7 f 10 c 1 d 97 b 0310 ccc 027 d 6 a``.
Session 37010 and queue 28272 exited successfully. This cohort is closed.

Next implementation boundary
----------------------------

The implemented proposal-first bridge now supports searching a full contingent
candidate before spending the parent-reference budget. Three additional tests
verify a compiled candidate of value 97/110 and a strict lower gain 14/275 while
the true parent is still only partially evaluated; full Python replay checks
both. The synthetic preflight spends 1,306 proposal and 465 reference transitions,
1,771 total, with six public program entries. This is a mechanism check, not a
natural strength result. See ``PROPOSAL_INTERVAL_TAIL.rst``.

The next development screen must freeze the proposal-plus-reference allowance
before running, retain every selected fixture, and independently verify complete
compiled candidates and adopted parent comparisons. Root-only rejection must
not be treated as a verdict on contingent play. No fresh prospective strength
evaluation or manuscript claim should precede that development evidence.
