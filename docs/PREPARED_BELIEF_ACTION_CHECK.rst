Exact-belief action diagnostic
=============================

The completed25-root enumeration reduces mean card Brier error from0.155551
under uniform hands to0.136122 with current public history and0.131641 with
earlier public opponent identification as well. This descriptive development
result motivates testing action selection. It does not establish match strength.

Retain all25eligible roots in original order,including both parents and both
opponents,and the7ineligible attempts in coverage reporting. The roots comprise
20distinct public observations from8shared development deals. Do not select
roots by posterior error or outcome. Require the complete enumeration audit.

Use four conditions on the same complete valid-hand set: uniform hand/equal
model prior;uniform hand/earlier public model identification;current-history
joint hand/model posterior/equal prior;current history/earlier public identity.
The uniform identified control isolates model routing from current-hand
information. Identity comes from the audited earlier public event,never from
the actual opponent label. Existing deterministic-policy likelihoods are reused;
their already recorded inference cost is reported separately from new search.

For every condition,compare every legal alternative to the unchanged strict
CPU parent with128 paired particles per alternative. Rank by mean terminal
draw-adjusted win difference,ties in legal-action order. Require a positive
mean-minus2.5-standard-error margin,then confirm only that selected alternative
with512fresh paired particles and the same positive-margin rule. This margin
is a search heuristic,not a population-level confidence guarantee. The root
seed is102100001+10000*parent+case_index,shared across conditions. Confirmation
uses the existing separate SeedSequence stream. These are simulation seeds;
no reserved101M actual deals are used. Report actual work,including conditional
confirmation and different rollout lengths,not merely particle caps.

Continuation uses fixed64 GPU microbatches with TF32 disabled and two CPU
threads. This is explicitly a different numerical execution from the strict
singleton CPU likelihood and fallback policies. A CUDA/reference transition
check must pass before the diagnostic. Future hidden worlds overwrite actual
hidden hand,stock and RNG;all policy inputs remain player-visible observations.

Freeze all100choices and their particle results before outcome assessment.
Then complete each root with each distinct chosen first action (including the
parent),using the actual source hidden state and strict CPU parent/opponent
for subsequent decisions. Reuse identical first-action completions and report
their reuse. Actual hidden state enters only this assessment stage. Record
every action and final state for independent reference replay. These are
conditional completions of old development roots,not new full evaluation
matches or independent outcome samples. Report all four conditions and both
parents descriptively;no significance or promotion decision uses these25roots.

Bind code,design,tests,weights,source records and prior audits before running.
Keep failed runs and never overwrite an existing output directory. A positive
diagnostic would justify a separately fixed prospective full-match study;
generalization and a replicated strength gain would still need that study.
