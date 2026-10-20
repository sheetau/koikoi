Score-auxiliary learning: complete development results
====================================================

The prespecified six-contrast improvement criterion fails.
All six unrounded simultaneous lower bounds must exceed zero.

Seed 1: 106110001; seed 2: 106210001. Each compares none, outcome-only (win)
and full-margin (score) auxiliary learning from original Point initialization.
The win/score arms share 65-logit heads; score adds within-outcome likelihood.
All arms collect 128 complete matches every 5 policy updates; none does not optimize.
Auxiliary rows and compute can differ because learned policies differ.
Retained parents A/B are separate WP-initialized reference policies.

131,072 matches use 8,192 shared initial deals and both seats. Draws count as
half wins; the mixture weights strict assisted WP and Point equally.
4,096 compound blocks retain all policies, learner groups, opponents and seats.
10,000 bootstrap replicates use 109310001. Six primary comparisons form one
simultaneous 95% family; all 120 secondary coordinates have descriptive marginal
95% intervals. These quantify deal variation, not variation across new learner seeds.

Primary comparisons
-------------------

==============================  =================================  =========
Comparison                      Win change (pp), simultaneous 95%  Lower > 0
==============================  =================================  =========
seed1/mixture/score-none/win    -0.1953 [-1.1108, +0.7202]         no
seed1/mixture/score-parent/win  +0.8209 [-0.2281, +1.8699]         no
seed1/mixture/score-win/win     -0.0305 [-0.9607, +0.8997]         no
seed2/mixture/score-none/win    +0.3510 [-0.5326, +1.2345]         no
seed2/mixture/score-parent/win  +0.6805 [-0.3703, +1.7314]         no
seed2/mixture/score-win/win     -0.1709 [-1.0401, +0.6983]         no
==============================  =================================  =========

All absolute results
--------------------

====  =======  =================  ==========================  ==========================
Seed  Scope    Policy/comparison  Win score (%)               Point margin
====  =======  =================  ==========================  ==========================
1     mixture  parent             50.7080 [50.1282, 51.3031]  +0.1971 [-0.1653, +0.5594]
1     mixture  none               51.7242 [51.1292, 52.3164]  +2.3837 [+2.0107, +2.7468]
1     mixture  win                51.5594 [50.9705, 52.1484]  +2.0748 [+1.7069, +2.4479]
1     mixture  score              51.5289 [50.9430, 52.1088]  +2.2357 [+1.8701, +2.6012]
1     wp       parent             51.0498 [50.2380, 51.8616]  +2.2751 [+1.8059, +2.7417]
1     wp       none               52.4597 [51.5747, 53.3447]  +4.5405 [+4.0002, +5.0791]
1     wp       win                52.1179 [51.2390, 52.9907]  +4.0972 [+3.5486, +4.6416]
1     wp       score              52.0813 [51.2146, 52.9419]  +4.5269 [+3.9983, +5.0586]
1     point    parent             50.3662 [49.4812, 51.2695]  -1.8809 [-2.4429, -1.3042]
1     point    none               50.9888 [50.2136, 51.7883]  +0.2268 [-0.2590, +0.7400]
1     point    win                51.0010 [50.2013, 51.8005]  +0.0525 [-0.4595, +0.5603]
1     point    score              50.9766 [50.2197, 51.7822]  -0.0554 [-0.5454, +0.4380]
2     mixture  parent             50.8026 [50.2105, 51.3916]  -0.0548 [-0.4302, +0.3152]
2     mixture  none               51.1322 [50.5341, 51.7334]  +2.1882 [+1.8221, +2.5623]
2     mixture  win                51.6541 [51.0559, 52.2461]  +2.1552 [+1.7857, +2.5239]
2     mixture  score              51.4832 [50.8942, 52.0691]  +2.4670 [+2.1001, +2.8408]
2     wp       parent             51.3428 [50.5492, 52.1545]  +2.1519 [+1.6631, +2.6360]
2     wp       none               51.7029 [50.8299, 52.5757]  +4.4333 [+3.8843, +4.9683]
2     wp       win                52.1240 [51.2451, 52.9968]  +4.3440 [+3.7971, +4.8721]
2     wp       score              51.6418 [50.7629, 52.5085]  +4.6343 [+4.0840, +5.1802]
2     point    parent             50.2625 [49.3835, 51.1414]  -2.2615 [-2.8269, -1.6933]
2     point    none               50.5615 [49.7620, 51.3489]  -0.0569 [-0.5583, +0.4426]
2     point    win                51.1841 [50.3906, 51.9897]  -0.0337 [-0.5271, +0.4873]
2     point    score              51.3245 [50.5675, 52.1059]  +0.2998 [-0.1824, +0.7913]
====  =======  =================  ==========================  ==========================

All paired comparisons
----------------------

====  =======  =================  ==========================  ==========================
Seed  Scope    Policy/comparison  Win change (pp)             Point change
====  =======  =================  ==========================  ==========================
1     mixture  score-none         -0.1953 [-0.8820, +0.4791]  -0.1479 [-0.5535, +0.2625]
1     mixture  score-win          -0.0305 [-0.7324, +0.6623]  +0.1609 [-0.2454, +0.5735]
1     mixture  score-parent       +0.8209 [+0.0183, +1.5930]  +2.0386 [+1.5710, +2.5180]
1     mixture  win-none           -0.1648 [-0.8271, +0.4853]  -0.3088 [-0.6926, +0.0690]
1     mixture  win-parent         +0.8514 [+0.0763, +1.6418]  +1.8777 [+1.4084, +2.3454]
1     mixture  none-parent        +1.0162 [+0.2532, +1.7945]  +2.1865 [+1.7267, +2.6506]
1     wp       score-none         -0.3784 [-1.3672, +0.6104]  -0.0137 [-0.5769, +0.5579]
1     wp       score-win          -0.0366 [-1.0193, +0.9277]  +0.4297 [-0.1426, +0.9927]
1     wp       score-parent       +1.0315 [-0.0671, +2.1179]  +2.2517 [+1.6121, +2.9128]
1     wp       win-none           -0.3418 [-1.2756, +0.5981]  -0.4434 [-0.9834, +0.0996]
1     wp       win-parent         +1.0681 [-0.0366, +2.1973]  +1.8220 [+1.1691, +2.4810]
1     wp       none-parent        +1.4099 [+0.3235, +2.5085]  +2.2654 [+1.6270, +2.9075]
1     point    score-none         -0.0122 [-0.9706, +0.9399]  -0.2822 [-0.8577, +0.2937]
1     point    score-win          -0.0244 [-0.9888, +0.9399]  -0.1079 [-0.7024, +0.4658]
1     point    score-parent       +0.6104 [-0.5006, +1.7274]  +1.8254 [+1.1181, +2.5044]
1     point    win-none           +0.0122 [-0.8972, +0.8972]  -0.1743 [-0.7053, +0.3589]
1     point    win-parent         +0.6348 [-0.4822, +1.7397]  +1.9333 [+1.2519, +2.6045]
1     point    none-parent        +0.6226 [-0.4761, +1.7334]  +2.1077 [+1.4343, +2.7796]
2     mixture  score-none         +0.3510 [-0.3021, +1.0010]  +0.2788 [-0.1181, +0.6825]
2     mixture  score-win          -0.1709 [-0.8301, +0.4761]  +0.3119 [-0.0680, +0.7017]
2     mixture  score-parent       +0.6805 [-0.0977, +1.4679]  +2.5219 [+2.0481, +3.0009]
2     mixture  win-none           +0.5219 [-0.1343, +1.1781]  -0.0331 [-0.4181, +0.3563]
2     mixture  win-parent         +0.8514 [+0.0610, +1.6479]  +2.2100 [+1.7260, +2.6987]
2     mixture  none-parent        +0.3296 [-0.4486, +1.0926]  +2.2430 [+1.7642, +2.7240]
2     wp       score-none         -0.0610 [-1.0193, +0.8789]  +0.2009 [-0.3496, +0.7515]
2     wp       score-win          -0.4822 [-1.4221, +0.4456]  +0.2903 [-0.2527, +0.8369]
2     wp       score-parent       +0.2991 [-0.8545, +1.4404]  +2.4824 [+1.8061, +3.1607]
2     wp       win-none           +0.4211 [-0.5127, +1.3428]  -0.0894 [-0.6289, +0.4592]
2     wp       win-parent         +0.7812 [-0.3479, +1.9226]  +2.1921 [+1.5266, +2.8694]
2     wp       none-parent        +0.3601 [-0.7446, +1.4528]  +2.2815 [+1.6174, +2.9541]
2     point    score-none         +0.7629 [-0.1648, +1.6846]  +0.3567 [-0.1883, +0.9077]
2     point    score-win          +0.1404 [-0.7875, +1.0437]  +0.3335 [-0.2102, +0.8799]
2     point    score-parent       +1.0620 [-0.0610, +2.1667]  +2.5613 [+1.8779, +3.2583]
2     point    win-none           +0.6226 [-0.3114, +1.5442]  +0.0232 [-0.5320, +0.5662]
2     point    win-parent         +0.9216 [-0.2075, +2.0752]  +2.2278 [+1.5305, +2.9409]
2     point    none-parent        +0.2991 [-0.8240, +1.4101]  +2.2046 [+1.5085, +2.9023]
====  =======  =================  ==========================  ==========================

Actual work
-----------

===========  ==============  =========  ======  =======  =============  =======================
Arm          Stream          Decisions  Rounds  Matches  Retained rows  Auxiliary presentations
===========  ==============  =========  ======  =======  =============  =======================
seed1_none   ppo_work        8519680    559940  72763    -              -
seed1_none   auxiliary_work  1490245    98096   12800    96442          0
seed1_score  ppo_work        8519680    562889  73152    -              -
seed1_score  auxiliary_work  1484689    98184   12800    96497          578982
seed1_win    ppo_work        8519680    562218  72935    -              -
seed1_win    auxiliary_work  1489727    98393   12800    96553          579318
seed2_none   ppo_work        8519680    561106  72896    -              -
seed2_none   auxiliary_work  1492804    98476   12800    96697          0
seed2_score  ppo_work        8519680    560337  72961    -              -
seed2_score  auxiliary_work  1490552    98259   12800    96558          579348
seed2_win    ppo_work        8519680    562644  73092    -              -
seed2_win    auxiliary_work  1488898    98510   12800    96745          580470
===========  ==============  =========  ======  =======  =============  =======================

Main training totals (warmup included once): {"auxiliary_work": {"decisions": 8936915, "matches": 76800, "rounds": 589918}, "ppo_work": {"decisions": 51118080, "matches": 437799, "rounds": 3369134}}
Actual evaluation and separately replayed work: {"decisions": 15240199, "full_matches": 131072, "round_endings": 1014387}

Point margins are candidate minus opponent; divide by two for the original
paper's final-minus-initial point convention. Capability and repeated-check
work are recorded separately in the experiment ledger. Export adds no matches.
JSON retains unrounded win fractions. The fixed mechanism report is separate.
Passing identifies the prospectively named seed1_score for a separately
prepared confirmation; it does not promote an endpoint. Reserved 101M is unused.
No better-seed selection or universal-strength claim follows from this report.

Recovery work outside the retained comparison
---------------------------------------------

The original process failed before auxiliary collection29, after seed2_none
PPO update165. The failed prefix executed2,703,360PPO decisions. Its164logged
updates record2,686,976decisions/174,865round endings/22,736matches. Round and
match counts for unlogged update165 are unavailable. The28complete auxiliary
collections contain421,608decisions/27,488rounds/3,584matches. These failed-run
costs are additional to the retained six-arm work above. Repeating the same
prefix adds no independent evidence. Saved states, RNG, optimizers, numerical
logs and collections match exactly; wall time and git provenance may differ.
All original artifacts and failure receipts remain preserved.

Interpretation and evidence
---------------------------

None of the six primary simultaneous intervals excludes zero. Across both
seeds, both opponents and their mixture, all24 descriptive score-versus-none
and score-versus-win win/point intervals also include zero. The auxiliary
margin task changes the actor on fixed public inputs, but this comparison
does not establish better match decisions. Small benefits or harms remain
compatible with these intervals.

All18 descriptive point contrasts of the three Point-initialized conditions
against the retained WP-initialized parents have positive lower bounds. This
pattern includes the no-auxiliary control; it does not identify an auxiliary
learning benefit. No endpoint is promoted and reserved confirmation stays
unused. The stronger-play and substantive-novelty objective remains open.

The complete queue and independent audit exited0. The audit replays every
recorded action through native CPU, checks initial deals, legality, work and
terminal points, and independently reproduces all120 marginal intervals and
six simultaneous bands. It does not rerun neural inference. The131,072 replayed
matches are validation work, not new independent gameplay evidence.

Evaluation protocol:
b49a9f06a9757f54e39f089cac1587f21786c6d7c07c786cbb35cc20d8f24c4b.
Evaluation audit (1,235bound inputs):
cf5f8fc9c5873dca98893ae3abac13722ec4e71f7c091f98464e5ca59d868ceb.
Queue completion:
2d1fc7fe8523d6981e3c5ebf0c6fe16d8d11cc9709419fc7a95698cca762f387.
Export evidence (1,825bound inputs):
a8f3ac919fba4cfe1e10f0e3caa1fb23b5542fcb5e76beac2d10cfa5ecbaf311.
Original complete export:
bcea0e5022aec1ef0052fb90637368b2579dd86e0551c4ac778b335cad1c9abc.

Export command (completed once, session18695, exit0)::

    .venv/Scripts/python.exe -m scripts.report_score_auxiliary_recovery export --audit-sha256 cf5f8fc9c5873dca98893ae3abac13722ec4e71f7c091f98464e5ca59d868ceb

The original unrounded export remains in runs/score_auxiliary_results.
All fixed-state diagnostics are in SCORE_AUXILIARY_MECHANISM_RESULTS.rst.
Do not rerun the completed study, audits, diagnostics or export.
