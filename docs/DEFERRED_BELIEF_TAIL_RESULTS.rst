Deferred chance equivalence results
==================================

Delayed stock expansion reproduces all 32 fixed/optimized contingent policies
at the sixteen completed roots of the closed three-draw capability corpus.
Every rational root-action value, selected action and complete policy map is
identical to eager enumeration. All new native transitions agree with the
independent Python engine. No new source match or strength sample was added.

Across those roots, the initial representation falls from 63,000 ordered-prefix
rows to 80 hand/model rows. The largest expanded frontier has 407 rows.

.. list-table:: Executed versus represented work
   :header-rows: 1

   * - Quantity
     - Eager
     - Deferred
   * - Physical simulator decisions
     - 660,758
     - 134,091
   * - Physical terminal branches
     - 255,492
     - 118,941
   * - Observation nodes
     - 126,181
     - 126,181
   * - Own decision nodes
     - 4,706
     - 4,706

The physical decision reduction is 79.7065%. Deferred weighted counters recover
all 660,758 eager decisions and 255,492 eager terminal branches exactly. They
represent the unused prefix multiplicities; they are not additional executed
simulations or independent samples. Independent Python verification adds
134,091 decisions and 118,941 terminal completions, counted separately.

The instrumented fixed/optimized planning calls took 8.113025 seconds in total,
including 5.686892 seconds of Python transition verification. Shared policy
caches and instrumentation differ from an isolated deployment benchmark.
The 79.7% figure describes physical transition work, not wall-clock speedup.

The earlier capability outcome is unchanged: parent1 has no exact gain at its
seven eligible roots; parent2 gains 1/110 at one of nine roots. Actual match
outcomes are unchanged. Three-draw optimization therefore remains an opportunity
and correctness check, not evidence of stronger match play. The computational
extension makes an earlier-tail check feasible; that check needs a separately
fixed protocol and fresh source streams.

The general construction remains established finite POMDP planning. This result
establishes a verified implementation equivalence and a work reduction in this
Koi-Koi system. It does not establish algorithmic priority.

Command: ``python -m scripts.check_deferred_belief_tail``.
Session14613 exited0; output ``runs/deferred_belief_tail_equivalence`` is closed.
Implementation/protocol commit:95cb717.
Protocol SHA256:
``a451f2605fb4e02fa16f532101d38f2babc79e56857da7cb8d7b3416a38f7de8``.
Complete SHA256:
``7ad4de1a0bd4b1dffff83ec55934cc84bbb9a7975b1a764f224dd05b8e5cd95a``.
The completed record contains all sixteen per-root results. Its input bindings
and aggregate counters were checked after completion. No training, canonical
promotion, reserved101M evaluation or manuscript rebuild occurred.
