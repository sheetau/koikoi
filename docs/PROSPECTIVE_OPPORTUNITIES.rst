Where the prospective controller leaves opportunities unresolved
===============================================================

This is a descriptive partition of the completed 512-source comparison.
It adds no simulation, policy inference, source matches or strength test.

.. list-table:: All-source controller outcomes
   :header-rows: 1

   * - Outcome
     - Parent 1
     - Parent 2
     - Total
   * - No eligible root
     - 135
     - 140
     - 275
   * - Verified improvement adopted
     - 41
     - 42
     - 83
   * - Parent conditional value already one
     - 31
     - 24
     - 55
   * - Optimization completed without a gain
     - 38
     - 36
     - 74
   * - Parent evaluation incomplete
     - 5
     - 6
     - 11
   * - Optimization incomplete without a retained gain
     - 6
     - 8
     - 14

There are 25 eligible roots with an unresolved
comparison and no adopted improvement. Of these,
15 retain at least 10,000 unused
native transitions under the original 4M maximum allowance. That threshold
describes remaining work only; it is not a new allocation or test rule.

The fixed/optimized phases originally have separate 2M caps, with no
transfer. Incomplete parent evaluation therefore leaves the optimized
phase unused. Reassigning that allowance would be a different controller.
No such reassignment is made in the original or ongoing certificate study.

Completed no-gain searches and parent values of one should be separated
from budget refusals when assessing the certificate's coverage benefit.
The completed optimization is conditional on the declared finite belief;
this diagnosis supplies neither a broader optimality nor a strength claim.
The first-root gate also leaves missing roots outside this search.

All input chunks and their record hashes match the completed export.
Reproduce with python -m scripts.diagnose_prospective_opportunities.
