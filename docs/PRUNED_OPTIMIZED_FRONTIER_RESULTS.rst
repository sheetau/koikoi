Optimized proposal pruning: closed development result
====================================================

Checking public outcome certificates before optimized proposal construction
reduces planning transitions by 137,628 (1.2713%) on the eight original
five-draw development sources. All 92 adopted programs, every retained
comparison record and the complete prefix beliefs remain identical. The
summed exact increment is unchanged at 149393/113256000; four sources improve.
This is a verified implementation saving, not an increase in playing strength.

.. csv-table:: Complete fixed-fixture comparison
   :header: Fixture,Original transitions,Pruned transitions,Skipped nodes

   0,1241546,1232618,62
   1,1415348,1415348,0
   2,80214,4974,1304
   3,1999987,1999987,0
   4,89037,35577,845
   5,1999988,1999988,0
   6,1999981,1999981,0
   7,1999988,1999988,0
   Total,10826089,10688461,2211

The 2,211 omitted comparisons have policy-independent public win/loss
certificates. Their original proposal work accounts for exactly the saved
137,628 transitions. In these eight cases the remaining local allowances
and comparisons do not change; the savings are pure omission. Exact gain per
transition increases by 1.2876%. This measured identity is not a theorem that
all future budget-limited runs produce identical programs: a different root
can spend the freed allowance on a later unresolved comparison.

Protocol and checks
-------------------

The frozen development criterion is met: total exact gain and positive-source
count do not decrease, while total planning transitions decrease. All eight
programs were saved before independent Python prefix, local-value and actual-
suffix verification. All realized own-point and terminal-win-score differences
remain zero. A separate reduction rechecks program/result identities, public
certificate masks and masses, exact arrival-weighted gains and all work totals.
Another saved-policy check verifies every retained comparison and complete
prefix world, including all 92 executable adopted programs.

The unchanged limits are two million global transitions, 250,000 prefix
transitions and 50,000 for each proposal/reference phase. Original fixed-arm
results are imported from ``runs/reusable_frontier_dev_v2``; they are not new
measurements. No source match, training update or rule change is added.
Independent prefix and local oracles use 49,263 and 918,431 transitions.
Actual suffixes use 96 transitions and separately replay the same 96; these
branches are distinct from planning and from independent source matches.

Planning-time sums are 495.378 seconds for the imported baseline and 492.748
for the new two-worker run. These separate concurrent runs do not establish
an isolated wall-clock speedup. The serial fully certified-root measurements
in ``docs/FRONTIER_TIMING_RESULTS.rst`` are a separate special-case result.

Decision and evidence
---------------------

Keep the isolated pruning implementation as an efficiency improvement with
the stated development scope. No stronger-policy claim or fresh strength
confirmation is warranted by identical retained policies in this screen.
General population work savings and controlled end-to-end latency remain
separate future questions. The original frozen controller, models and cohorts
remain unchanged.

* Contract: ``docs/PRUNED_OPTIMIZED_FRONTIER.rst``.
* Implementation: ``scripts/pruned_optimized_frontier.py``.
* Frozen study: ``runs/pruned_optimized_frontier_dev_v1``.
* Independent reduction:
  ``docs/paper/generated/pruned_optimized_development_evidence.json``.
* Complete retained-policy identity:
  ``docs/paper/generated/pruned_optimized_policy_identity_evidence.json``.
