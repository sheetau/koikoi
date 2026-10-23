Search reuse and allocation: completed development comparison
===========================================================

All eight fixed five-draw fixtures were retained in a three-arm comparison.
Each arm rebuilt its prefix, used at most two million new physical transitions,
saved the policy before verification, and passed independent Python prefix and
adopted-pair evaluation plus actual suffix replay. These reused development
fixtures are not new population-strength evidence.

.. csv-table:: Complete eight-fixture totals
   :header: Method,Transitions,Local adoptions,Positive fixtures,Sum of exact gains,Planning seconds

   Fixed existing allocation,10826089,92,4,149393/113256000,495.378
   Broad pilot and refinement,11835488,12,3,56039/764478000,535.880
   Same refinement with completed-choice cache,11477095,12,3,56039/764478000,519.029

The cache removed 358,393 actual transitions (3.03%) relative to the same
uncached schedule, with identical exact gains and adoption counts. Its 10,478
hits referenced 2,099,800 previously completed transitions; that reference count
is neither additional simulation nor the observed net saving. Cache capacity
is a 64 MiB serialized-payload allowance, not an RSS bound. Planning times are
descriptive measurements with two concurrent workers, not isolated latency.

Neither adaptive arm satisfies the prespecified selection criterion. They
retain only 5.56% of the fixed schedule's summed improvement and improve three
fixtures instead of four. The cached arm still uses 6.01% more transitions than
the fixed schedule. Broad preliminary attempts can use the entire budget before
completing valuable proposals; cache hits alone do not establish better search.
The existing controller is retained.

Implementation and evidence
---------------------------

``scripts/reusable_frontier.py`` retains only fully completed information-set
choices, with the entire weighted belief, public history, depth and planner
context in the key. Interrupted choices are never installed as policies. Parent
interval evaluation resumes without restarting; proposal attempts and failures
are charged cumulatively. All local planners remain alive during a patch, which
is the lifetime boundary of the shared cache. Twelve focused tests include
uncapped equivalence, interrupted-search recovery, independent full program
evaluation, seat swaps, unequal masses and cache-disabled ablation.

Protocol and complete result bindings:
``runs/reusable_frontier_dev_v2/protocol.json`` and
``runs/reusable_frontier_dev_v2/summary.json``. Every fixture/arm directory holds
the pre-audit program, selection receipt and independent audit. The preceding
v1 directory records a launcher failure before any research worker ran; it was
not overwritten. The original implementations and earlier evidence are intact.

Follow-up
---------

``scripts/resumable_frontier.py`` instead proposes a complete root-action
override followed by the unchanged parent. Parent and candidate intervals can
resume directly; the shared parent comparison serves all root alternatives.
Its separate sixteen-fixture study includes the eight earlier six-draw probes.
That follow-up is not included in the completed totals above.
