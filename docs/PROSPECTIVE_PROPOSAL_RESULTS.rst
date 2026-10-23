Fresh direct-frontier proposal comparison
========================================

The fixed 256-source comparison is complete. Parent-default proposals reduce
physical planning transitions by 22.00% while retaining 93.72% of the optimized
proposal arm's summed exact gain. Gain per transition rises by 20.15%, but the
new arm improves fewer sources and fails the prespecified superiority test.
It is a measured computation/value tradeoff, not a stronger replacement or a
wall-clock speedup.

Scope and complete denominator
------------------------------

There are 64 independently acquired sources in each parent/opponent population.
Both direct-frontier arms use the same public at-most-five-draw gate and the
same two-million-transition maximum, including prefix construction and failed
attempts. Neither uses the original symbolic first phase or its parent-refusal
filter. This comparison is against the unchanged parent, not an incremental
test against the original integrated controller.

All 256 sources remain in the denominator: 141 have no eligible root and 115
are evaluated. No support reconstruction is refused. Every saved program is
followed by independent Python prefix reconstruction, complete evaluation of
each adopted pair, and actual suffix replay. No source is replaced or added.

.. csv-table:: Expected equal-opponent mixture gains
   :header: Parent,Optimized pp,Parent-default pp,Difference pp,Lower bound pp

   0,0.026964,0.023142,-0.003822,-0.015265
   1,0.051574,0.050464,-0.001110,-0.008119

The contrast is parent-default minus optimized. Within-population source
bootstrap uses 20,000 replicates and one-sided 97.5% bounds per parent, a
nominal 95% family. The joint positive-bound criterion FAILED. A separate
reduction reconstructs exact gains from arrival-weighted independent local
values and reproduces the means and bootstrap bounds.

Coverage and resources
----------------------

The optimized arm improves 38/256 sources (14.84%); parent-default improves
35/256 (13.67%). All 35 are also improved by the optimized arm. Original-only
sources are 133, 216 and 241; there is no parent-default-only source. The
remaining 218 improve under neither arm. These frequencies must not be
compared as incremental gains against the old 26/4096 result: both the trigger
and the comparator differ. Neither arm changes any realized match score.

.. csv-table:: Operational planning totals
   :header: Arm,Physical transitions,Local programs,Planning seconds

   Optimized proposal,58939164,2011,2734.750
   Parent-default proposal,45974787,1873,2984.690

The exact gain sums are 30228458822327/300693493900800 and
7990537997747/84810985459200. The corresponding all-source means are
0.0392692 and 0.0368031 percentage points. The new arm saves 12,964,377
transitions, but its operational elapsed-time sum is 9.14% larger. Timings
include two concurrent workers and varying background activity; they are not
isolated latency benchmarks. Physical-transition efficiency and runtime are
different quantities.

Separate verification accounting
--------------------------------

* Source acquisition: 256 matches, 1,974 round endings and 29,530 decisions.
  Independent source replay repeats those same counts and adds no sources.
* Support reconstruction: 204,585 replay decisions per complete pass. The
  second independent reconstruction performs another 204,585.
* Prefix oracles: 231,360 physical transitions per arm. Adopted-pair terminal
  oracles: 5,564,857 for optimized and 4,403,319 for parent-default.
* Actual suffix execution: 913 transitions and 230 suffix completions per arm;
  action replay repeats those counts. These are continuations of existing
  source matches, not additional independent matches.

Proof and numerical evidence
----------------------------

The closed protocol, all 256 source records and program bindings are under
``runs/prospective_frontier_proposals_v1``. The independent reduction and exact
fractions are in ``docs/paper/generated/prospective_proposal_results_evidence.json``;
``scripts/export_research_extension.py`` generates that record and its paper
table. Existing eight-/six-draw development results remain separate.

Post-confirmation feasibility check
-----------------------------------

After closing this comparison, a read-only development check asked whether
the new programs could augment unchanged old frontier nodes using only old
unused budget. In 101 sources the old arm left at least 100,000 transitions.
Among 34,533 available full-allowance local comparisons on those sources,
none adds a certified program at an old unpatched node. This is a limited
check of saved comparisons, not a new prospective controller or an exhaustive
search of unseen comparisons. Simple leftover-budget composition supplies no
observed gain here. Both cohorts and all original evidence remain closed.
