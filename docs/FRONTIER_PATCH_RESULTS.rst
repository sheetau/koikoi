Frontier patch development results
==================================

All eight fixed original-gate parent-refusal fixtures passed. Four acquire
strict conditional improvements through 92 local patches, independently checked
by complete prefix reconstruction and full local terminal replay. The old
controller and the preceding whole-root proposal extension adopted no policy
on these same eight fixtures. No absolute root value is needed for the new
increment: unchanged continuations cancel in the verified decomposition.

.. list-table:: Per-source conditional improvement
   :header-rows: 1

   * - Source
     - Frontier nodes
     - Patches
     - Gain lower bound
     - Exact increment
     - Exact pp
     - New transitions
   * - 16
     - 0
     - 0
     - 0
     - 0
     - 0.000000
     - 75,459
   * - 168
     - 0
     - 0
     - 0
     - 0
     - 0.000000
     - 72,248
   * - 246
     - 1,304
     - 0
     - 0
     - 0
     - 0.000000
     - 80,214
   * - 387
     - 1,286
     - 10
     - 1/5265
     - 1/5265
     - 0.018993
     - 1,999,987
   * - 393
     - 1,263
     - 0
     - 0
     - 0
     - 0.000000
     - 89,037
   * - 433
     - 1,409
     - 26
     - 71/112320
     - 71/112320
     - 0.063212
     - 1,999,988
   * - 867
     - 1,200
     - 22
     - 31/214500
     - 31/214500
     - 0.014452
     - 1,999,981
   * - 1054
     - 1,129
     - 34
     - 3563/10193040
     - 3593/10193040
     - 0.035250
     - 1,999,988

Sources 16 and 168 reach the 65,536-row prefix cap and keep the parent. Their
partial arrival groups are never passed to local planning. The other six
prefixes complete 7,591 observation groups. Exact local replay verifies every
adoption and its full probability mass; source 1054's exact increment exceeds
its certified lower bound because a local parent interval remained incomplete.
All eight actual retained trajectories and outcomes remain identical to the
parent. The positive expectation comes from other supported future branches.

Work and provenance
-------------------

New planning uses 8,316,902 physical transitions: prefix 186,876,
proposal 5,539,459,reference 2,590,567.
Imported original fixed work is 15,999,936 historical
transitions and is not new execution. Each nominal integrated sum staysat or below four million.
Support reconstruction adds 21,499 replay
transitions. Independent prefix replay adds 39,169
physical /76,380,480 represented transitions;
adopted local-pair oracles add 918,431 physical /
3,649,710 represented transitions. Actual and
recorded-action replay each add 96 decisions and 16 completed tails. Fresh chance
sampling is zero. Prefix construction and each local phase retain separate
node/task counts in the complete review; no latency claim is made.

Protocol SHA:6597826f4af37927e2e50d95e9134433cd3ca6eb76a3883df35a18db9b3c5db5.
Summary SHA:4fe10fd513c81eec45b4a2888f73bc8351f8eb05dd082c509f657d33eb678e5d.
Review runs/frontier_patch_screen_review.json SHA:
9e1bdab17379071c0e2f70ba15a5eb771b141dc2a78da5325c68cdf4bf817946.
Session 19067/queue 68776 exited with code 0. The complete-only exporter rehashes artifacts,
recompiles every local candidate, validates all belief bindings/budgets and
reconstructs the independent weighted gains. Incomplete export refused;
complete export passed. This screen is closed and must not be restarted.

These targeted cases demonstrate recovery of improvements after a whole-root
parent evaluation fails. They do not estimate prevalence or gain on new matches.
Next check the integrated controller's original phase and fallback, preserve
completed old decisions exactly, and evaluate the selected method prospectively.
