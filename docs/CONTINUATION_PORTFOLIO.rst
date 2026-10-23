Complete learned-policy continuations: implementation contract
=============================================================

Status: isolated implementation in scripts/continuation_frontier.py, with
both-seat semantic and interrupted-budget tests. No strength result is implied. The completed 256-source comparison stays closed.
Its parent-default arm saves transitions but loses gain and coverage, while
saved leftover-budget composition adds no observed program. The next question
is whether already executable learned continuations provide useful multi-step
candidates without constructing contingent search trees.

Program semantics
-----------------

A candidate names a frozen, observation-only continuation actor by identity,
and supplies explicit legal public-history overrides including its root action.
At every other history it calls that named actor. The reference remains the
unchanged original parent. The actor identity is part of the serialized program,
not an implicit change to the meaning of existing parent-default programs.
Only the existing two parents and released assisted WP/Point actors are eligible;
no training, new rules, hidden-world-specific actions or privileged inputs.

Keep all frozen implementation modules intact. An isolated program type must
validate root legality, reference identity and continuation identity, and must
never report an alternative continuation as the unchanged-parent reference.
The existing interval mass accounting and same-belief strict selection can be
reused, but candidate evaluation and execution must dispatch to the declared
actor. IntervalEvaluator's reference-root check must still check the original
parent; the candidate's explicit root entry defines its actual action.

An evaluation adapter may return the original parent's action at that root for
the invariant check and use the named continuation at other keys. Actual root
execution always uses the explicit legal entry. This adapter is not sufficient
by itself: independent Python verification and actual suffix execution must
implement the same serialized continuation semantics and reject missing actors.

Execution commits to exactly one adopted local program at first frontier arrival.
The original parent still supplies the complete shared prefix and all untouched
branches. An executor adapter can route the fallback actor once on entry, then
reuse the existing frontier validation and legal override logic. It must not
switch between sibling programs or change the reference used in certificates.

Legacy parent-default evaluators/executors must not silently accept a named
continuation and call the wrong fallback. A guarded entry mapping can refuse
ordinary lookup until the dedicated adapter resolves the actor identity and
copies the entries into an execution mapping. Keep the ordinary reference
program unchanged. The existing frontier constructor can validate bindings and
arrival gains, then a dedicated wrapper installs local executors with the
declared fallback actors before any action is taken.

Verification and development
----------------------------

First test both seats, tiny/interrupted budgets, unknown actor identities,
reference-root checks, alternative fallback actions, immutable program identity,
full independent local values and full root-value decomposition. Existing
public score bounds are policy-independent under paper8; rule-variant bounds
are not used here. No engine transition or original checkpoint is changed.

Independent prefix verification can use the original no-change frontier patch.
For each adopted candidate, independently replay its explicit overrides and
named continuation, alongside the original parent, then recompute the exact
arrival-weighted increment. Actual suffix execution must reproduce those same
semantics and retain the old parent's source trajectory as its reference.
For a local full replay, an explicit adapter can expose the original reference
at the root check and the declared continuation elsewhere, with the actual root
still supplied by its legal override. Trace all reached continuation actions;
for small-fixture full-root tests those traces can form explicit ordinary
programs and use the existing independent full-frontier oracle. Such internal
projections must not be saved or represented as unchanged-parent certificates.

Candidate families to develop separately:

* Existing one-step root alternatives plus the other frozen parent's complete
  policy, using that actor's own root action.
* The same family plus complete assisted WP and Point policies, each with its
  own root action. More candidates may consume the budget without helping;
  do not assume this larger family is better.

Share one persistent parent evaluator. Candidates are complete before evaluation;
resume interval work under the physical-transition allowance. Candidate order,
allocation, local/global limits and selection must be frozen before new sources.
Start with the existing 100,000 local/two-million global limits and the same
complete next-discard prefix. Reuse of evaluation receipts must never skip
mass-conservation checks after a changed evaluator advances.

Use the sixteen existing development fixtures plus the three closed-comparison
sources 133, 216 and 241 (missed by parent-default) as explicitly labelled
development. Imported baseline results are not new measurements. Save new
programs before independent audits, retain every zero/refusal and compare
coverage, exact gain, physical work and elapsed time separately. Only a useful
verified development candidate should proceed to a new, separately frozen
prospective comparison. Do not extend, pool or relabel the closed 256 sources.

The two rule confirmations are also closed failed tests and remain separate.
Update the paper with measured outcomes at each coherent milestone; new
candidate definitions alone are not strength evidence.

Frozen development decision
---------------------------

The runner uses ordinary root alternatives first, then the other parent, WP
and Point in that order where included. It retains the first strict certificate.
An eligible family must add at least one positive source, retain every baseline
positive source, match or exceed the baseline total exact increment, and match
or exceed its exact increment per physical transition. Select the eligible
family with greatest total exact increment; ties prefer the smaller family.
If neither qualifies, do not launch confirmation under this design. The baseline
is the imported parent-default result for each of the nineteen sources.

The one-source 20,000-global/2,000-local smoke checks execution only and cannot
qualify. Development uses two million global/100,000 local transitions, a
250,000-transition prefix cap and two workers. The smaller family always runs
first within each source; elapsed time is descriptive rather than isolated
latency. Programs are serialized before independent Python audits.
