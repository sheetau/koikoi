Exact value-ceiling pruning
==========================

The tail planner now evaluates the retained parent's action first at every
learner information set. For a set with integer mass M and terminal utility
in [-1,1], no action has signed value above M. Once an evaluated action attains
M, all remaining alternatives can be omitted. The chosen contingent policy
is complete; only alternatives to it are skipped.

Parent-first ordering followed by the remaining legal order preserves the
existing tie rule. If the parent action is optimal it wins the tie. Otherwise
the first maximizing action in legal order wins. Induction over the finite
tree preserves the selected action, value and contingent policy at every
information set. Hidden worlds still share one action whenever their visible
histories agree.

``CeilingPlan.values`` contains exact values only for evaluated root actions.
``unevaluated_actions`` explicitly lists the others. If this list is nonempty,
the chosen root value must be exactly 1 in draw-adjusted win units. It is an
optimality certificate from the known utility ceiling, not an estimate of the
omitted actions' individual values.

This is standard bounded dynamic programming applied to the existing exact
tail controller. It does not establish algorithmic priority. It removes work
while retaining the same finite-model optimum.

Verification and resource limits
---------------------------------

Seven tests pass without skips in 2.76 seconds. Four finite-game cases cover
hidden/revealed information and balanced/unbalanced integer mass. A winning
root checks early omission and preference for a parent action that is not first
in legal order. Both-seat Koi-Koi fixtures compare all computed action values
and policies with complete enumeration and independently replay the resulting
policies using Python rules. Physical-transition and live-node caps are tested.

Retain the 500,000-live-node, 2,000,000-physical-transition, 4,096-frontier-row
and 32-decision-depth limits per plan. These differ from the original five-draw
study's 500,000-all-node budget. All actual work and refusals remain explicit.

Stage 1: all completed roots
---------------------------

Compare terminal-reduced and ceiling-pruned planning on all 18 completed roots
of ``runs/earlier_belief_tail_capability``. Each method receives fresh separate
policy caches; method order alternates by root ordinal, with fixed then optimized
continuation within each method. Use strict singleton CPU actors and two threads.
Both methods have identical live-node and physical-transition limits.

Require the saved optimal action, value and whole chosen policy to match.
Every evaluated root action must match its saved exact value and complete
root-policy map. Omitted actions must be listed and certified by root value 1.
Independently replay every computed ceiling-policy value using the Python oracle.
Measure planning times separately from oracle work and report all roots.
This adds computation on closed data, not source matches or strength samples.

Command, after committing implementation and protocol::

    .venv/Scripts/python.exe -m scripts.check_ceiling_tail

The fresh output is ``runs/ceiling_tail_equivalence``.

Stage 2: all previously refused roots
-------------------------------------

Only after Stage 1 completes and its bindings pass, attempt all 15 previously
refused roots of the same fixed 64-source corpus. Do not add source matches,
replace a refusal or choose roots by potential gain. Reconstruct support from
the saved source's public model identification and current-round history.
Keep the original five-draw and 1,000-candidate-hand gates.

Run fixed then optimized ceiling planning with the explicit limits above.
Verify every computed value with the independent Python chance replay. Keep
any remaining refusal and its partial-stage work. Save all completed plans
before actual-tail assessment. Then execute parent/fixed/optimized continuations
against fresh strict opponent calls and verify that the parent recovers its
original source suffix and outcome.

The existing 18 completed roots retain their already verified policies/outcomes;
31 missing roots retain the parent. Report the original 64-source denominator,
all 15 recovery attempts, remaining refusals and all additional work separately.
The extended computation was selected after observing the original study's
limits: resulting gains are exploratory capability evidence, not a new
prospective replicated strength claim. There is no automatic budget escalation.
No training, reserved101M use, model promotion or manuscript update follows
from a successful conditional example alone.
