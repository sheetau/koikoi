Public-stock rule exploration: measured results
==============================================

This report covers the new eight-profile factorial in PUBLIC_STOCK_RULES.rst.
It does not revise the closed nine-rule frozen-weight transfer studies.

At a yaku stop, stakes transfers ``(S_w + S_l) * (2 if k_w > 0 else 1)``
from loser to winner. Each ``S_i`` is that player's current captured-card
score under unchanged paper8, including their own declaration effects;
``k_w`` is the winner's declaration count. Captured sets are scored separately.
The ordinary one-point exhaustion settlement remains unchanged.

V1 qualification failure
------------------------

Two control policies trained from scratch for 33,554,432 decisions each.
They completed 4,382,014 rounds and 558,001 matches in total. Training counts
are not strength measurements. The planned fourteen noncontrol jobs were
not launched after the control-quality failure; the already running second
control was allowed to finish. No rule screen or confirmation was consumed.

On 256 shared new duplicate streams, with both seats, the qualification
results are:

================= ============ =====================
Training seed     Greedy       Released RL-WP
================= ============ =====================
911               63.7695%     31.9336%
912               65.8203%     28.8086%
================= ============ =====================

RL-WP here uses unassisted GPU batching; these are not the historical assisted
strict-CPU comparator endpoints. Both controls fail the predeclared 45% WP
quality gate. Seed 911 also fails the 65% greedy gate. The four endpoints
contain 2,048 matches, 16,171 rounds and 229,470 decisions. Every decision
was replayed in Python and all final states matched. These results justify
changing policy initialization; they say nothing about which new rule is best.

Source implementation is retained at Git revision 0a1c70f; protocols and
checkpoints are under runs/public_stock_training_v1 and qualification traces
under runs/public_stock_qualification_v1. The closed.json record explicitly
marks the unlaunched jobs. The separate eight-update engineering pilots and
eight-match search smoke checks are not added to qualification evidence.

V2 adaptation: independent confirmation complete
-------------------------------------------------

PUBLIC_STOCK_STUDY_V2.rst fixes a separate protocol starting from the two
existing stronger parent policies. It freezes their actors and learns
public-stock corrections and rule-specific values. Initial control logits
preserve the parent; five focused tests verify that behavior, absence of
teacher gradients, bounded encoding batches, and hidden-state invariance of
encoding and root search. The 45 engine/sampling/search checks from the first
milestone remain relevant.

All sixteen adaptations completed: eight rules times two parents, each with
4,194,304 additional decisions. The total is 67,108,864 decisions, 3,462,582
rounds and 498,538 matches across 2,048 updates. These are additional training
counts, not strength evidence; historical parent training remains separate.

A separate CPU audit checks all sixteen saved checkpoints against their parent
actor/value/configuration, declared training identity and budget, source hashes,
and correction tensor shapes. All parent fields match exactly and all 128
correction tensors are finite. Actor corrections are nonzero for every model;
public-card input weights are nonzero for all twelve models with preview or
choice. This is numerical/identity validation, not evidence that the features
improve decisions. The audit is runs/public_stock_model_audit_v2.json, generated
by scripts.audit_public_stock_models.

Both controls pass the predeclared qualification gates on 256 new duplicate
streams, with both seats:

========= ============ =====================
Parent    Greedy       Released RL-WP
========= ============ =====================
0         76.6602%     50.4883%
1         79.6875%     52.1484%
========= ============ =====================

The WP implementation is again unassisted GPU batching. Do not replace the
historical assisted-2023 comparison with these qualification results. The
four endpoints contain 2,048 matches, 16,105 rounds and 225,889 decisions.
All decisions were independently replayed in Python; final-state/action
reductions reproduce every score, margin and evaluation count.

The fixed screen uses 0/2/8/32 sampled worlds per legal root action and a
32-decision horizon. It covers 64 shared duplicate streams, 8,192 actual matches,
55,032 rounds and 1,153,925 policy decisions. Search uses 22,993,192 rollouts
and 665,914,496 physical transitions across 413,372 roots. Independent
reduction reproduces every endpoint's scores and counts; all 1,153,925 scored
decisions were replayed in Python. The 8,192 matches are not 8,192 independent
deal streams.

The table gives win scores (%) for parent 0 / parent 1 against the same
adapted no-search actor. Zero-world scores are 50% for every profile.

================  =================  =================  =================
Rule              2 worlds           8 worlds           32 worlds
================  =================  =================  =================
Control           33.9844 / 27.3438  36.7188 / 32.0312  39.4531 / 40.2344
Stakes            28.9062 / 23.4375  25.7812 / 35.9375  46.8750 / 37.5000
Choice            19.9219 / 27.7344  35.1562 / 50.3906  64.4531 / 57.0312
Choice + stakes   24.6094 / 33.9844  42.9688 / 53.1250  67.1875 / 67.9688
Preview           41.7969 / 35.9375  49.6094 / 47.2656  64.8438 / 61.7188
Preview + stakes  36.7188 / 41.0156  43.3594 / 53.5156  62.5000 / 65.2344
Preview + choice  40.2344 / 31.6406  60.9375 / 52.3438  76.1719 / 79.2969
All three         39.4531 / 27.3438  58.9844 / 56.6406  83.5938 / 78.1250
================  =================  =================  =================

All three factors are selected by the prespecified worse-parent gain:
+33.59375 / +28.125 percentage points at 32 worlds, compared with control
changes of -10.546875 / -9.765625 points. This selection does not establish
superiority to every alternative: preview plus choice scores slightly higher
for parent 1. Every rule loses mean score at two worlds in this screen.

The independently reduced screen is preserved in
``docs/paper/generated/public_stock_v2_development_evidence.json``. Its complete
curves and per-match work table are under ``docs/paper/generated/public_stock_v2``.
The figure's intervals are exploratory paired 95% intervals, without selection
or multiplicity adjustment. They are not the confirmation criterion.

A supplemental observed-work figure plots all eight curves against physical
rollout transitions per match (``public_stock_v2/public_stock_work.pdf`` in
the same generated directory). At 32 worlds, all three uses 230.2 / 227.8
thousand transitions per match, compared with 331.4 / 338.8 for preview plus
choice. The corresponding mean match lengths are 5.16 / 5.10 versus
7.51 / 7.52 rounds. This cross-game work comparison includes match-length
differences and excludes inference and preparation costs; it is not a
matched-cost or elapsed-time result. Source/figure identities and visual QA
are bound by ``docs/paper/generated/public_stock_v2_work_evidence.json``.

The gate passes and the unchanged reserved confirmation completes on 512
new duplicate streams, selected rule and control, all four budgets and both
parents. It contains 16,384 matches, 112,112 rounds, 2,249,011 decisions,
45,492,908 rollouts and 1,313,528,021 online transitions. All scored decisions
are replayed in Python. Independent reduction reproduces all outcomes.

At 32 worlds, all three score 82.4707% / 80.9082% against their own no-search
actors, gains of 32.4707 / 30.9082 percentage points. The one-sided 98.75%
lower bounds are 29.9316 / 28.1250 points. Control scores are 40.6250% /
42.2363%, changes of -9.3750 / -7.7637 points. Selected-minus-control gains
are 41.8457 / 38.6719 points, with lower bounds 37.8418 / 34.6191. All four
registered inequalities pass; the two parents are not pooled.

The selected-rule curves at 0/2/8/32 worlds are 50.0000/30.8594/59.9121/82.4707%
and 50.0000/33.4473/62.1582/80.9082%. Thus very small search budgets still
harm performance. Full confirmation curves, counts and source identities
are in ``docs/paper/generated/public_stock_v2_evidence.json``.

V3 completed-round search: independent confirmation complete
------------------------------------------------------------

The early v2 control results motivate a separate completed-round protocol
(PUBLIC_STOCK_STUDY_V3.rst); v2 continues unchanged. V3 reuses all sixteen
adapted policies without further training. Every candidate rollout finishes
the current round and looks up a sampled future-round value indexed by round,
absolute dealer and player-0 points. This avoids a learned value cutoff within
the round and reuses subsequent-round computation.

All sixteen tables are complete, each using 483,328 conditional round
rollouts. Preparation totals 7,733,248 conditional rounds and 163,572,208
transitions; 1,751,874 conditional rollouts reach a terminal state. These are
not counts of complete evaluation matches.

Every saved target and table mean was independently reduced. Four fixed
samples per live public cell were replayed in Python: 3,776 rounds per table,
60,416 rounds and 1,276,957 decisions in total. All sixteen table hashes match
their policy/rule metadata and saved audits. Aggregating those audit records
is not a new replay or strength evaluation. All scored online matches are
replayed in full. Absolute dealer cannot be collapsed by seat symmetry:
the released parent encoder observes it.

The independent screen uses 32 duplicate streams and 0/8/32/128 particles per
legal action. Its 2.5-standard-error adoption guard is heuristic. Uniform
hidden-world proposals omit action-likelihood conditioning, and table values
are estimated; their conditional standard errors omit propagated future-table
uncertainty. V3 is not a certified search method. Its changed horizon, guard
and particle budgets also prevent a causal attribution to caching alone.
The predeclared rule selection and confirmation are separate from v2.

The complete online screen contains 4,096 matches, 27,752 rounds and
574,620 decisions from 32 shared duplicate streams. Search uses 46,309,552
rollouts and 653,460,580 physical transitions across 211,542 roots. All scored
decisions pass Python replay, and independent reduction reproduces every
score, evaluated-match counter and the registered selection. Preparation counts above remain
separate from these online counts.

Win scores (%) for parent 0 / parent 1 follow; every zero-world score is 50%.

================  =================  =================  =================
Rule              8 worlds           32 worlds          128 worlds
================  =================  =================  =================
Control           48.4375 / 45.3125  53.9062 / 49.2188  46.8750 / 53.1250
Stakes            46.8750 / 51.5625  45.3125 / 48.4375  54.6875 / 54.6875
Choice            59.3750 / 53.1250  64.0625 / 57.0312  63.2812 / 64.8438
Choice + stakes   54.6875 / 51.5625  67.1875 / 59.3750  67.1875 / 64.0625
Preview           60.9375 / 62.5000  64.0625 / 68.7500  75.0000 / 78.9062
Preview + stakes  51.5625 / 64.0625  68.7500 / 71.8750  75.0000 / 79.6875
Preview + choice  64.8438 / 64.0625  78.1250 / 71.0938  86.7188 / 84.3750
All three         58.5938 / 54.6875  75.7812 / 73.4375  85.9375 / 84.3750
================  =================  =================  =================

Preview plus choice is selected. Its high-budget gains are +36.71875 /
+34.375 percentage points, versus control changes of -3.125 / +3.125 points.
The worse-parent gain ties all three at +34.375 points; the registered
lexicographic tie-break chooses preview plus choice. Parent 0 also has a
slightly higher mean under that profile, while parent 1 ties. This screen
does not establish a statistically superior rule among those alternatives.

Within the selected rule, 128 worlds improve over eight worlds by
21.875 / 20.3125 points in development. The full curves retain all
nonmonotonic results. These are exploratory means, not held-out conclusions.
Development results are preserved in
``docs/paper/generated/public_stock_v3_development_evidence.json``; curves,
work tables and observed-work plots are under ``generated/public_stock_v3``
in the paper directory. Figure identities and visual QA are bound by
``docs/paper/generated/public_stock_v3_figures_evidence.json``.

The original gate passes. Reserved confirmation completes on 128 new
duplicate streams, control and preview plus choice, both parents, and
0/8/128 worlds. It contains 3,072 matches, 23,850 rounds, 492,836 decisions,
42,932,832 rollouts and 605,313,887 online transitions across 160,561 roots.
Every scored decision is replayed in Python; independent reduction reproduces
the results and verifies the continuation-table identities.

Preview plus choice scores 85.1563% for both parents at 128 worlds, a gain
of 35.1563 points over no search. One-sided 98.75% lower bounds are 30.6641 /
30.4688 points. Control gains are 3.1250 / 3.5156 points. The excess gains
are 32.0313 / 31.6406 points, with lower bounds 25.7813 / 25.5859. All four
registered inequalities pass. This does not confirm a ranking against all three.

Increasing eight to 128 worlds adds 22.8516 points (paired secondary 95%
interval 17.1875 to 28.5156) for parent 0 and 17.7734 points (11.7188 to
23.8281) for parent 1. These secondary intervals are not part of the primary
familywise claim. The selected-rule curves at 0/8/128 worlds are
50.0000/62.3047/85.1563% and 50.0000/67.3828/85.1563%.
The full evidence is ``docs/paper/generated/public_stock_v3_evidence.json``.
Both protocols retain their own cohorts and inference; they are not pooled.

V4 changed opponents and work calibration: complete
--------------------------------------------------

The fixed protocol PUBLIC_STOCK_STUDY_V4.rst retains control, preview plus
choice, and all three, with both parents. Each actual opponent is the other
parent's retained adapted actor under the same rule. Rollouts and continuation
tables retain the searching actor's own policy; no opponent oracle is supplied.
This study adds no training or table preparation.

Thirty-two separate duplicate streams at 1200000000, both seats, calibrate
work using 32 worlds per action: 384 matches, 2,731 rounds, 61,213 scored and
Python-replayed decisions, 3,664,640 rollouts, 55,378,349 transitions and
30,237 roots. Only work selects the frozen low/high counts for targets of
60,000/240,000 transitions per match: control 24/95 and 23/94; preview plus
choice 10/40 and 10/38; all three 12/49 and 13/50 (parents 0/1).
Counts use the fixed half-up rounding and [2,256] clamp. Calibration identity:
``49e45bc4af0657ac2f5c3a956c37cdb94d5192a8e5cf6ada58042eb478221528``.

The eighteen evaluation endpoints use 128 new duplicate streams at
1210000000, both seats and zero/low/high budgets: 4,608 matches, 32,587 rounds,
725,475 scored decisions, 32,934,469 rollouts, 449,378,614 transitions and
235,771 roots. Every scored decision is Python-replayed. Independent reduction
reproduces scores, margins, rounds and decisions; internal rollout work is
recorded rather than independently replayed. The 128 streams are the
independent units, not the 4,608 matches.

Win-score curves (%) at zero/low/high, with ties worth one half:

================  ===========================  ===========================
Rule              Parent 0                     Parent 1
================  ===========================  ===========================
Control           47.4609 / 50.0000 / 49.2188  52.5391 / 51.5625 / 54.2969
Preview + choice  48.0469 / 58.7891 / 73.4375  51.9531 / 69.1406 / 71.8750
All three         51.9531 / 64.0625 / 78.9063  48.0469 / 63.0859 / 76.7578
================  ===========================  ===========================

Fourteen fixed comparisons use 20,000 paired-stream bootstrap replicates
(seed 1093), two-sided Bonferroni quantiles 0.05/28 and 1-0.05/28, for a
nominal 95% familywise target within V4 only. Each finalist must have positive
lower bounds for high-minus-zero, excess over control's gain, and high-minus-low
for both parents. All three passes all six score inequalities: high-minus-zero
gains are 26.9531/28.7109 points, excess gains 25.1953/26.9531 points, and
high-minus-low gains 14.8438/13.6719 points (simultaneous lower bounds
6.2500/5.0220 for the latter). Preview plus choice has positive gain and excess
bounds for both parents, but parent 1's high-minus-low gain is 2.7344 points
with interval [-6.0547,11.5234], so its full six-inequality criterion fails.
Neither paired finalist-difference interval excludes zero; ranking is unresolved.
All fourteen intervals are retained in the evidence and manuscript table.

Eleven of twelve work checks fall within the registered +/-10% tolerance.
All-three parent 0 high uses 214,281.8086 transitions per match: -10.7159%
from its 240,000 target. The full matched-work criterion fails, and no
matched-work replication or ranking is claimed. Actual work and every
deviation remain reported; the closed cohort is neither retuned nor extended.
These are expected-work targets, not hard per-match deadlines or equal wall
time. Neural inference, bookkeeping and preparation are excluded from the
physical transition count.

Independent evidence is ``docs/paper/generated/public_stock_v4_evidence.json``
(226 input/source bindings; SHA256
``694b550472811cf3f1bb5fc1511eec01f5771420795ebd132d4b4d9d493e089e``).
The runner's finished receipt binds that evidence. All curves, fourteen
intervals and twelve cost checks are rendered under
``docs/paper/generated/public_stock_v4``. Figure generation checks the complete
evidence bindings; it does not generate new strength measurements.

V5A adaptation maturity: qualification complete
----------------------------------------------

All six systems completed 384 additional updates from the retained 128-update
weights, with fresh Adam and the original teacher frozen. Additional totals
are 75,497,472 decisions, 3,536,901 rounds, 485,230 matches and 2,304 updates.
Each system has 16,777,216 cumulative adaptation decisions; original parent
training remains separate. Final checkpoint lineage, finite weights, frozen
teacher identity and progress/counter records all pass the independent audit.
Only final additional-update-384 policies are evaluated; no checkpoint is
selected retrospectively. Training counters are recorded, not replayed.

The original runner omitted the evaluator's required ``evaluate`` subcommand
and stopped at argument parsing before any qualification directory or scored
match existed. The isolated completion driver preserves the failed log and
original sources, verifies all frozen training audits, and binds a recovery
receipt before executing the unchanged evaluation protocol. No cohort is
restarted or extended, and no scientific setting changes.

All eighteen qualification endpoints are complete: 36,864 actual matches,
264,776 rounds and 6,024,229 scored/Python-replayed decisions on 1,024 new
shared duplicate streams at 1240000000. Recorded online search counters
are zero. Independent reduction verifies outcomes, margins, counts, opponent
identities and complementary old-parent baselines. The complete evidence
has 409 source/input bindings and SHA256
``11c74d2ac1cedb8548c436457a352c28fd578c41f94afb40c69690b0f2e7cdb0``:
``docs/paper/generated/public_stock_v5a_evidence.json``.

All twelve paired contrasts use 20,000 bootstrap replicates (seed 1109)
and one-sided lower quantile 0.05/12, retaining the nominal within-study
95% familywise target. Each rule requires positive lower bounds for new
versus its own retained parent and for new-minus-old against the other
retained parent, for both initial parents. Scores count ties as one half.

Both finalist rules qualify. Preview plus choice gains 7.5195/7.0313 points
against its own retained actors (lower bounds 5.0049/4.5410), and
8.4229/5.8838 points over the old actors against the other retained opponents
(lower bounds 4.9805/2.4658). All three gains 7.7148/6.5430 points against
its own retained actors (lower bounds 5.0537/3.8330), and 9.3262/5.2490
points against the other retained opponents (lower bounds 5.8594/1.7334).
Thus every one of the finalists' eight lower bounds is positive.

Control does not qualify. Its own-parent changes are -0.9033/+0.5859 points
(lower bounds -2.7588/-1.3184), and its other-parent changes are
+0.9521/-0.8789 (lower bounds -1.3916/-3.1738). No positive lower bound
establishes improved control strength. These results do not prove control
degradation or explain it causally. More updates alone did not establish
improvement under the retained self-play/teacher-KL training regime.

All score triplets and twelve contrasts are in the evidence and generated
manuscript tables under ``docs/paper/generated/public_stock_v5a``. The
qualification contains no new-policy search evaluation.

Follow-up after adaptation qualification
----------------------------------------

The planned V5B all-improved-policy search study requires improved control
and at least one improved finalist. That joint launch gate fails, so V5B is
not run; its gate is not relaxed. Nevertheless, both finalists now have
measurably stronger policies, making their search robustness a substantive
remaining question. The separate PUBLIC_STOCK_STUDY_V5C.rst now retains the
old control and tests qualified finalist actors against the other new parent.
Its real preflight verifies six systems, 43 sources and 35 inputs. The runner
is preparing four new policy-specific finalist tables and reuses two
unchanged, previously audited control tables. New calibration/evaluation
streams are fixed before use. Unequal adaptation budgets are explicit;
cross-rule differences cannot be attributed to rules alone. This study is
running, not yet search-strength evidence. V4's failed work criterion and
unresolved finalist ranking remain unchanged.

Interpretation limits fixed before results
-----------------------------------------

Particle counts give a 1:4:16 nominal budget ratio within a rule. They do not
equalize actual transitions across rules: legal-action counts, extra stock
choices, terminal timing and rollout lengths differ. A 32-decision horizon
also spans fewer ordinary turns when stock choice is enabled. Report actual
work and the full curves. A superior win-score increment alone does not prove
superiority at equal wall time or equal transition budget. Such a claim needs
a separately specified matched-cost comparison.

Increasing particle counts increases hidden-world sampling, not the fixed
rollout horizon. These curves measure added computation, not a general theorem
about deeper game-tree search. Confirmation familywise targets apply within
each protocol; there is no claimed adjustment across the two protocols.
The reserved confirmations compare selected-rule gains with the control.
Rankings among the seven noncontrol rules remain exploratory.

Each screen opponent is that rule's own adapted no-search actor. This measures
the additional value of computation, not an absolute ranking of strength
between different games. Equal adaptation budgets do not establish equally
mature policies across rules; larger search gains can partly reflect more
remaining policy errors. The conclusions concern this fixed training/search
regime, not intrinsic game difficulty or all possible agents. Winning by more
points under doubled stakes is not strength evidence. Two parent identities, a finite rule family and one fixed
rollout algorithm cannot establish globally maximal AI strength or that all
further improvement has been exhausted.
