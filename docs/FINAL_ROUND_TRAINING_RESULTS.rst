Fixed-prefix terminal-outcome adaptation: complete results
=========================================================

The frozen study in FINAL_ROUND_TRAINING_STUDY.rst completed both training
endpoints and all16,384 strict full-match evaluations. Producer19319 exits0.
Independent audit/summary queue54117 exits0. The recorded-action replay,
prefix preservation, checkpoint checks and paired arithmetic all pass.
All89 independent-audit input hashes were reverified after completion.

Outcome
-------

Each cell contains1,024 duplicate-deal pairs per policy. Intervals resample
whole pairs10,000 times. Win changes are percentage points; point changes are
final learner-minus-opponent margin units. Candidate minus its own parent:

===========  ===================  =======================  ======================
Parent/foe   Win change            Win95% interval          Point change [95% CI]
===========  ===================  =======================  ======================
1 / WP       +0.122070             [-0.439453,+0.683594]    +0.166016 [-0.048828,+0.385767]
1 / Point    +0.537109             [+0.048828,+1.049805]    +0.017578 [-0.134766,+0.175781]
2 / WP       -0.122070             [-0.634766,+0.390625]    +0.040039 [-0.134790,+0.210938]
2 / Point    -0.097656             [-0.634766,+0.390625]    -0.127930 [-0.284180,+0.026392]
===========  ===================  =======================  ======================

Only parent1 versus Point has a positive nominal win-change lower bound.
The fixed primary criterion requires all four lower bounds to be positive,
so replicated gain is NOT established. Every point-change interval includes
zero. The individual positive interval is not a multiplicity-adjusted or
two-parent strength certificate. All null and adverse cells remain reported.
No checkpoint selection, extra updates, deal extension or model promotion occurs.

Parent/tail draw-adjusted win rates are50.390625/50.512695% against WP and
51.342773/51.879883% against Point for parent1;50.732422/50.610352% and
51.879883/51.782227% respectively for parent2. These are development-matchup
estimates. They do not replace the preserved100M held-out canonical evaluation.

Accounting
----------

The2,048 source prefix attempts share512 deal streams. They yield938/918
natural final-round roots for the two parents and192 early completed matches.
Source work:212,212 decisions /14,101 round endings /192 early completions;
the other1,856 prefixes stop at the natural final-round boundary.

Each parent receives20 critic-only plus500 policy/critic updates, with128
complete tails per update. Combined training:1,860,647 decisions, comprising
918,437 learner and942,210 opponent decisions;133,120 conditional round endings
and133,120 tail completions. Full training-match count is zero because these
trajectories start from stored roots. Repeated sampling adds no new source deals.

Actual evaluation:1,908,865 decisions /127,476 round endings /16,384 full matches
on2,048 distinct fresh evaluation streams. The independent source/evaluation
replay adds2,121,077 decisions /141,577 round endings /16,576 completed source
or evaluation matches, but no new independent deals and no neural inference.

The public-hand revelation scan and short reconstruction/model capability
tests overlapped evaluation. Runtime is not an isolated deployment benchmark.
The canonical playable model, dealer selection and reserved101M deals remain
unchanged. The manuscript is not yet regenerated for this completed extension.

Identities
----------

All study artifacts are under runs/final_round_training_study/:

* protocol.json:99c47e73db4dbd08374e8defb95a0689d89a5ba4170f954ba5bb60586da32087
* analysis.json:0b3498df45a6cad2a95a751143e06e860af9ee25bb4ddbe1bf58a82021abc4d2
* independent_audit.json:2f5b65a1fbb5e036c757c106cd9ee9294effef878a0688081ad29d987cd10278
* summary.md:f9113c579eb1688883975f64056878d8c0643ce5cc1d318461b3effadd88f42c
* parent1/final.pt:7574187043111a5ed15b1d9c7003590fa18a14bbdc8243dd5b0b8fb16fc9d777
* parent2/final.pt:06346d9c7127537f64b21b8b1a16a1eb16156fb765a8f7e34141c39803888dd8

Completion receipt runs/final_round_training_finish.json:
74a13f71e60c29734b9a2f81017d94b8f8d84088d4d9484dd8bcc77790d0bb7a.
Do not poll/restart terminal sessions19319 or54117. The separate queued
opponent-identification diagnostic81291 uses only the completed training
prefixes; it does not alter or extend this study.
