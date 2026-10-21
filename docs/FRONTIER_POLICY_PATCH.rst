Public frontier policy patches
=============================

The controller can improve a subset of future observation histories without
finishing the parent value at the current root. It follows the unchanged parent
to a fixed public frontier, then commits to a locally certified replacement
where one is available. Other frontier observations retain the parent.

Construction and guarantee
--------------------------

Fix a complete root belief, deterministic parent and opponent models, and the
uniform unordered-stock chance law. Enumerate parent paths to the first learner
decision with at most d remaining draws, or to termination if it occurs first.
Group all arriving worlds by the learner's exact own/public history. Each group
contains its entire conditional belief, with integer reach mass M_h. The total
root mass M equals terminal mass plus the sum of frontier masses.

The prefix must finish before any local belief is usable. An interrupted prefix
can leave further worlds that belong to an already encountered observation;
optimizing that partial group would use the wrong posterior. Prefix refusal
therefore yields parent fallback, even if some groups have already arrived.

For a frontier history h, construct a complete local candidate and evaluate it
against the unchanged parent on the same conditional belief. Adopt only if its
certified lower value L_h exceeds the parent's upper value U_h. The interval
comparison can retain an incomplete local parent evaluation. At execution, the
controller commits to that one local program; it never switches between sibling
frontier programs. Both legal overrides and parent defaults remain public.

Let A be the adopted frontier histories and delta_h = L_h - U_h. Then::

    V(patched) - V(parent) >= sum_{h in A} (M_h / M) delta_h >= 0.

Proof. Both policies take identical actions before the frontier, so they have
the same arrival events, reach probabilities and conditional beliefs. These
first-arrival events are disjoint. A path that terminates before the frontier,
or reaches an unpatched history, has identical continuation under both policies.
On each adopted event, the conditional value difference is at least delta_h.
Conditioning on this partition and summing gives the inequality. Neither the
root parent value nor values of untouched continuations are required. Full
independent local values replace the lower bounds with an exact increment.

This proof assumes the declared opponent model and chance law. The fixed
opponent cannot change its earlier behavior in response to the new controller.
It provides no equilibrium or exploitability guarantee against arbitrary
opponents. Choosing a frontier with actual future information would invalidate
the public execution and source interpretation; this implementation uses only
remaining draw count and own/public history.

Budget and execution
--------------------

One shared physical-transition budget covers the prefix and every local attempt.
The prefix has its own ceiling. Complete frontier groups are attempted in
descending reach mass, breaking ties by exact public-history bytes. Each attempt
preassigns equal proposal/reference caps from the remaining shared allowance;
unused work remains available to later groups. The global budget is never
multiplied by the number of groups. Live information nodes and interval expanded
tasks retain their separate meanings.

scripts/frontier_policy_patch.py contains the prefix collector, shared-budget
solver and committed executor. scripts/verify_frontier_patch.py independently
reconstructs the prefix with the Python transition engine, compares each complete
posterior and its mass, and replays both local policies for every adopted patch.
Its optional full-root replay checks that the sum of local increments equals the
complete policy-value difference. Neither oracle uses outcome shortcuts.

Correctness evidence
--------------------

The synthetic three-draw fixture, cut at two draws, has 226 frontier groups and
five adopted patches. Prefix construction uses 365 physical transitions; total
planning uses 2,906. Independent full-root evaluation gives parent 204/275 and
patched 461/550. Their difference is 53/550, exactly the independently verified
sum of reach-weighted local gains. These are correctness data, not estimates
of natural playing strength or a matched-algorithm speed comparison.

Tests cover both seats, unequal integer masses, serialization, prefix transition/
row/depth refusals, shared budgets including failed local attempts, corrupted
beliefs and missing prefix groups, duplicate patches, altered gains, root binding,
and commitment to either an adopted local program or the unchanged parent.

Prior work and intended contribution
------------------------------------

Policy improvement through choosing among policies is established. Chang,
Givan and Chong's parallel-rollout work proves improvement properties for
parallel rollout and policy switching; see Section II and Appendix A of the
`author-hosted paper <https://engineering.purdue.edu/~givan/papers/deds01.pdf>`_.
The conditional-expectation argument above is not a new general policy-
improvement theorem.

Burch, Johanson and Bowling use opponent counterfactual values to safely
re-solve imperfect-information subgames and preserve whole-game exploitability
bounds; see their Theorem 1 in `AAAI 2014
<https://cdn.aaai.org/ojs/8810/8810-13-12338-1-2-20201228.pdf>`_.
Here the opponent is fixed, and the guarantee concerns expected terminal value
under a complete history-conditioned belief. These different assumptions and
guarantees must remain explicit in Related Work.

The proposed practical contribution is a complete public-history posterior
partition and composable local gain certificates under one finite work budget.
It addresses the concrete failure of a controller that needs a completed root
parent value before retaining any improvement. A natural-case benefit and a
fresh population evaluation remain necessary before elevating this extension
to a manuscript strength claim.
