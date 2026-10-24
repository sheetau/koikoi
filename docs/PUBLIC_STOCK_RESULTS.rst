Public-stock rule exploration: measured results
==============================================

This report covers the new eight-profile factorial in PUBLIC_STOCK_RULES.rst.
It does not revise the closed nine-rule frozen-weight transfer studies.

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

V2 adaptation: screen complete, confirmation active
--------------------------------------------------

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

The gate passes and the unchanged reserved confirmation is now running on
512 new duplicate streams, comparing the selected all-three rule with control
at all four budgets for both parents (16,384 planned matches). No confirmation
outcome or rule promotion is asserted while that cohort is incomplete.

V3 completed-round search: active
--------------------------------

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
