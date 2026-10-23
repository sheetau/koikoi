Complete parent-default proposals with resumable comparisons
==========================================================

The complete public-frontier construction and its conditional-gain proof are
unchanged. This extension changes the candidate representation and allocation
of evaluation work. The original optimized-continuation implementation and
all closed studies remain intact.

At each complete arrival belief, construct one program for each legal action:
use that action at the current own/public history, then follow the unchanged
parent at every later history. Each program is executable before its value is
known. Its definition needs no completed contingent search tree. All worlds
sharing the observation use the same action; hidden-world-specific action
selection is forbidden.

The parent and each candidate have a persistent interval evaluator. Resolved
terminal mass and certified public-outcome mass are retained, as is every
unexpanded task. The reference is shared across all candidate comparisons.
An alternative whose upper bound is no better than the parent's lower bound
is eliminated. Otherwise, the widest relevant interval receives the next
work quantum, with ties favoring the shared parent. Atomic chance expansions
that do not fit one quantum can receive a larger quantum, subject to the same
remaining total allowance. Refused node/depth limits are explicit. Every
physical transition is charged once where it runs; unresolved work is never
treated as an exact result.

The first strict candidate-lower/parent-upper separation is adopted. Search
does not claim the best alternative was found. Frontier nodes remain ordered
by arrival mass and public history; a local comparison has at most 100,000
physical transitions, and prefix plus all local comparisons share two million.
Untouched or unresolved nodes keep the parent. The gain is the sum of local
lower bounds weighted by complete arrival mass, exactly as in the original
frontier theorem. Independent Python replay evaluates every adopted pair.

This construction trades a narrower policy family for cheaply specified,
resumable candidates. It cannot dominate full contingent optimization at
unlimited budget. The relevant question is what useful policies can be
verified within the shared finite allowance. The original eight-fixture
comparison, the earlier six-draw probes, and the new source population are
separate evidence sets, not pooled replications.

Verification and studies
------------------------

Eight focused tests cover both seats, severely interrupted budgets, complete
independent interval values, parent-default execution and full root-gain
decomposition. Implementation: ``scripts/resumable_frontier.py``.

``runs/resumable_frontier_dev_v1`` fixes all eight existing five-draw and all
eight earlier six-draw development fixtures. It retains every zero/refusal
and independently checks the prefix, local adoption values and actual tails.
The six-draw boundary uses a public turn-derived threshold; it does not claim
the old five-draw combinatorial prefix bound applies to the larger scope.

``runs/prospective_frontier_proposals_v1`` freezes 256 new independent source
matches, 64 in each parent/opponent population. Both arms apply a direct
frontier controller at the same public at-most-five-draw gate. This deliberately
removes the original parent-reference failure filter; it measures the local
proposal mechanisms against the same parent, not the entire previously
published symbolic-plus-frontier controller. Each arm has the same two-million
transition maximum. Missing roots and refused supports contribute zero.
The paired expected-gain contrast and nominal simultaneous bootstrap bounds
are prespecified, with no extension or replacement of the cohort. Source
acquisition, support reconstruction, planning, Python value verification and
actual/replayed tails are separate work categories.

No result in these development or direct-frontier studies silently replaces
the selected playable model. A direct-frontier result cannot be relabelled as
an incremental gain over the original integrated controller.
