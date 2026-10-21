Contingent proposal before parent comparison
===========================================

``scripts/proposal_interval_tail.py`` reverses the old fixed-parent-first
schedule. It first searches for a complete contingent policy under the full
belief, then evaluates the unchanged parent only far enough to decide whether
that candidate improves it. The comparison uses the interval implementation
in ``INTERVAL_TAIL.rst``; it is not limited to replacing the root action.

The proposal phase uses the existing score-bound incumbent planner. A policy
completed before a later budget refusal can be retained. An incomplete policy
or a mere action upper bound cannot supply a candidate value. Compilation
checks the full world/parent/opponent binding, optimized-phase invariants,
selected root map/value, original parent root action and the recognized public
score-bound leaf rule. It constructs a complete legal-override/parent-default
program before any parent interval comparison.

For a completed proposal of value q on mass M, the exact signed outcome total
is (2 q - 1)M. The compiler requires an integer total and creates the point
interval [q, q]. This is a transfer of a completed same-belief policy-value
certificate, not a claim that the proposal's aggregate search work represents
one execution tree. Proposal physical decisions/live nodes and reference
interval transitions/tasks remain separate work categories. The full Python
interpreter must independently verify the compiled policy and unchanged parent
before reporting an empirical adoption.

The default schedule reserves one million physical transitions for the
proposal and one million for the reference, without borrowing. Each phase
has a 250,000-node/task cap, with the two meanings reported separately.
If no complete proposal exists, the controller retains its parent. If a
proposal exists, it advances parent evaluation until q exceeds the parent's
upper bound, the parent's lower bound reaches q, or the reference cap is met.
Only the first case adopts. No optimized parent-root continuation is ever
treated as the unchanged parent reference.

Three targeted tests pass, including rejection of incomplete proposals,
changed beliefs/modes/leaf rules, and a verified contingent adoption while the
parent interval remains nontrivial. The synthetic proposal has value 97/110;
at reference cap 500, the parent interval ends at 457/550, proving gain 14/275.
Independent full Python replay gives the unchanged parent's value 204/275 and
checks the entire contingent program. These tests establish correctness, not
a new natural-match strength result.

No natural proposal-first screen has been launched yet. Finish the already
frozen eight-source root-interval screen before selecting the next development
comparison. Any new protocol must predeclare its full proposal-plus-reference
budget and account for both phases. Existing closed strength cohorts must not
be relabeled as prospective evidence or extended until a desired bound passes.
