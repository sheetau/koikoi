Anytime comparisons of complete public policies
===============================================

The interval controller retains the value of resolved belief mass when a
computation budget expires. It can adopt a complete candidate when its lower
value bound exceeds the unchanged parent's upper bound, even if neither tree
is fully evaluated. The candidate's execution does not depend on finishing
the evaluation tree.

Complete policy before evaluation
---------------------------------

``PublicProgram`` binds a parent identity, opponent identities, root belief,
root legal actions and a finite own/public-history override map. At every
learner decision it executes a listed legal override, otherwise the unchanged
parent. The legality guard is part of the program's declared semantics from
the outset. It is applied by the evaluator, runtime executor and independent
Python interpreter. Thus an unvisited override cannot make the unresolved
part of the policy undefined. The root override itself must be legal.

The parent reference is exactly the root parent action followed by the same
parent everywhere. A candidate with additional overrides cannot masquerade as
that reference. Contingent maps are supported, not only root replacements.
Whatever procedure proposes them, they must be frozen before their interval
evaluation; an interrupted search map is not assigned its predecessor's value.

Mass bound
----------

Let M be the total integer belief mass, S the resolved mass, and Z the sum of
resolved signed outcomes (-1, 0, +1) times their masses. Every unresolved
continuation of the complete policy has signed outcome in [-1, 1]. Hence its
match-win score lies in the exact rational interval

    L = (Z + S) / (2M),     U = L + (M - S) / M.

Terminal transitions and valid public outcome certificates can resolve mass.
The remaining queued mass plus S must equal M at every receipt. This actual
frontier check is performed after budget refusals as well as successful runs.
Resolving more mass can only raise L and lower U. A strict comparison
L_candidate > U_parent proves a gain of at least L_candidate - U_parent
under the declared full belief and opponent law. No confidence approximation
or sampled outcome substitutes for an unresolved branch.

An expansion is atomic: first construct all next-draw rows and check physical,
depth, task and frontier caps; then execute and replace the queued parent task.
If it cannot fit, its entire mass stays queued. The frontier can be resumed
without replaying already executed transitions. Chance multiplicities remain
integers. Terminal simulation work and symbolic resolution are separate.

Information and scheduling
--------------------------

Tasks group worlds by learner own/public history. Learner actions are shared
within that group; opponent actors use their own observation/model cache keys.
Hypothetical hands and stock positions are never exposed as actor features.
Small chunks fit the existing native workspace without changing public policy
actions. Tasks are expanded depth first, with larger-mass peers first and a
stable insertion-order tie rule. This is a declared scheduling choice, not an
optimal allocation theorem.

``compare_programs`` shares one physical-transition budget across the parent
and every candidate. Round-robin quanta prevent the parent from consuming the
entire budget before alternatives receive evaluation. The shared task cap
counts expanded tasks, which is distinct from the old live-information-node
counter. The frontier-row budget is divided between programs. A candidate may
be discarded once its upper bound cannot beat the parent's lower bound.
Selection stops early when a strictly improving candidate's lower bound is at
least every other candidate's upper bound, or when no strict gain is possible.
Otherwise it returns the best certified candidate at the fixed budget, with
parent fallback if none is certified. Actor inference calls are counted apart
from physical transitions.

Implementation evidence
-----------------------

``scripts/interval_tail.py`` implements programs, intervals, resumable evaluation,
shared scheduling and execution. ``verify_interval_tail.py`` follows every
Python transition and every remaining stock draw without outcome shortcuts
or native policy caches. It checks that the exact value lies inside the saved
interval and that all represented mass terminates.

Twenty-seven targeted tests cover both seats, unequal integer masses, optional
certificates, resumption, node/depth/frontier refusals, shared physical caps,
chunking, reference/belief identity rejection, and consistent legality guards.
A contingent-map test independently recovers value 97/110. A separate
root-replacement example retains parent interval [159/275, 219/275] at cap525
and candidate value228/275. It certifies gain9/275 before completing the parent;
full Python replay gives parent204/275 and verifies the comparison. These are
synthetic correctness examples, not natural match-strength estimates.

Development screen
------------------

``scripts/screen_interval_tail.py`` retains all eight earlier six-draw fixtures.
It compares the unchanged parent with every other legal root action followed
by the parent. This first natural screen isolates interval comparison from
contingent-policy proposal. It does not replace the broader contingent-policy
research objective or establish that root-only proposals are sufficient.

All programs share 2,000,000 physical transitions, 500,000 expanded tasks,
quanta4,096 and a total pending-row budget65,536; native capacity remains4,096,
depth32. The transition budget equals one old phase, not a separate two-million
allowance per policy. Sources, posterior support, model identities and hidden-
information boundaries are unchanged. Two bounded workers are used.

Selections/programs are saved before actual-tail execution. Every adoption
receives independent full parent and candidate value replay; every newly exact
parent also receives full replay. Actual traces are independently replayed,
and the parent must reproduce the original source suffix. Input/result hashes
are frozen before planning. No fixture may be replaced and no fresh chance is
drawn. This is development feasibility, not fresh strength evidence. A selected
new policy requires subsequent prospective evaluation.

The general chance-node interval principle has prior art; see the primary
sources in ``SCORE_BOUND_RELATED_WORK.rst``. The implemented extension couples
resolved-mass bounds to complete public policy programs and a same-belief,
unchanged-parent adoption gate. No claim of a new general bounding principle
or of equilibrium performance follows from it.
