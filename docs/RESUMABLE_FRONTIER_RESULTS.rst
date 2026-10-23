Complete parent-default proposal development results
===================================================

All sixteen fixed fixtures passed independent Python prefix reconstruction,
adopted-pair terminal evaluation and actual suffix replay. Every public prefix
completed. The new representation recovers improvements in two earlier-gate
fixtures where the previous six-draw feasibility screen adopted no policy.
These are reused development cases, not population-strength evidence.
All sixteen actual suffixes retain the parent's final points and match score.

Original five-draw fixtures
--------------------------

The same four of eight sources improve. Under the same two-million-transition
maximum per source, summed exact improvement rises from 149393/113256000 to
133517/76447800, a 32.40% relative increase. Physical transitions rise from
10,826,089 to 11,602,153 (+7.17%), and local adoptions rise from 92 to 125.
Improvement per actual transition therefore rises by 23.55% on this fixed
development set; total work and elapsed time do not decrease. The operational
planning-time sum is 628.254 seconds versus 495.378 for the original schedule,
with concurrent workers and different background activity. This is not an
isolated latency improvement.

The new planner's summed certified lower bound is 2381/1470150 (22.78% above
the old exact sum). Independent full replay gives the larger exact sum above;
the bound and the verified value are separate quantities.

Earlier six-draw fixtures
------------------------

The eight mechanically selected earlier-gate fixtures are unchanged from the
previous screen. Six still produce zero gain. Two now support strict changes:

.. csv-table:: Independently verified earlier-gate improvements
   :header: Original source,Exact gain,Gain percentage points,Adopted local programs

   112,27271/15135120,0.180184,39
   738,1/8408400,0.000011893,1

Each new patch uses at most two million transitions including its prefix. The
new comparison replaces the previous whole-root fixed/optimized search with
public partitioning and parent-default local programs; it is not merely a
larger draw threshold. Source 112's certified lower bound is 41/29106; full
independent evaluation establishes the larger exact gain shown above. Keeping
that distinction prevents relabelling a bound as an exact policy value.

All eight sources remain in the denominator. The conditional gains and their
frequency cannot be extrapolated to all matches or combined with the old
4,096-source cohort. The old failed whole-root gate screen stays closed.

Evidence and next comparison
----------------------------

``runs/resumable_frontier_dev_v1/protocol.json`` freezes the fixtures, source
bindings, budgets and algorithm. ``summary.json`` records all sixteen results;
each ``fixture_XX`` directory contains the saved pre-audit program, selection
and independent audit. The mathematical and implementation boundary is in
``RESUMABLE_FRONTIER.rst``.

The fixed prospective 256-source study compares direct frontier controllers
at a common five-draw gate, without the previous parent-refusal filter. It
addresses generalization and wider activation separately. Its results must
not be described as an incremental win over the original integrated controller.
