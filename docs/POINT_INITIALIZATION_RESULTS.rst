Point initialization improves point margins; calibrated sampling gives no replicated win gain
============================================================================================

Starting from the published Point policy improves final point margins over
both the retained agents and matched WP-initialized controls. Across two
training seeds, both Point variants, both opponents and their equal mixture,
all24descriptive paired95% point-change intervals have positive lower bounds.
For the mixture, improvements over the matched WP controls are+1.9663/+2.0848
points with fixed divisors and+1.9724/+1.7689 with calibrated divisors.
These are final candidate-minus-opponent margins, not the upstream half-margin.

The four prespecified simultaneous win-improvement intervals all include zero.
Thus the Point-calibrated policy does not establish replicated improvement over
both WP-initialized and retained controls. The calibrated-minus-fixed Point
mixture win change is+0.7385pp at seed1 and-0.3387pp at seed2; temperature's
win benefit does not replicate. Point calibration also reduces WP point margin
at seed2 by0.8909points with a descriptive interval[-1.4089,-0.3691].

Point-calibrated agents score50.9155% and50.8240% against published Point,
with descriptive marginal95% intervals[50.1219,51.7212] and[50.0305,51.5930].
Those matchup results do not replace the failed primary comparison against
adapted controls. Their Point margins are-0.0498 and+0.0383; both intervals
include zero. Neither equal strength nor win noninferiority is established.
No endpoint is promoted, no better seed is selected, and the reserved101M
confirmation remains unused.

The pre-outcome mechanism diagnostic found higher declared-divisor entropy
but lower common-divisor entropy and larger raw score gaps at both seeds in
every phase. Together with the match results, this shows that preserving
higher sampling entropy on the old input corpus is insufficient to establish
better greedy match play. It does not identify a causal mediator or rule out
other calibrations. The initialization and temperature controls are established
techniques; the study supplies a measured objective-transfer comparison.

The complete-only export binds233inputs, all rehashed after completion.
AuditSHA:6fa668b21f7422677c16892ca7f39429895387955e4e5ff2105760bd833584c8.
SummarySHA:b1ce6a1e59eb7c4aef4494ab3a02ba5cbe7e825bdd52cc4840aead61767a0fd9.
ExportSHA:9de59a0b52764c5a823f63c8dcb0f5cf30bb294417f9298aaa233c9d2a03b647.
QueueSHA:f84db1b43691c1641a04a4c7bbc879d1f1ed8ceaacbf310609a4b50c9f74e5f3.
The following complete tables are copied unchanged from the verified export.

Point initialization: complete development results
=================================================

The prespecified four-contrast improvement criterion fails.
All four unrounded simultaneous lower bounds must exceed zero.
A passing development gate needs separate prospective confirmation.

Group parent1: fresh seed 102110001, retained comparator A.
Group parent2: fresh seed 102210001, retained comparator B.
Three arms per seed: original-WP fixed, original-Point fixed and Point calibrated.
All have 20 critic-only plus 500 policy updates with identical decision budgets.
Calibration matches phase-average entropy on the old calibration corpus only.
It changes both sampling and policy-loss derivatives, not exploration alone.
Initializations and fixed divisors do not constitute a new algorithm.

131,072 matches use 8,192 shared initial deals and both seats.
Draws count as half wins; mixture weights assisted WP and Point equally.
4,096 compound blocks pair WP/Point deals and retain all policies/seats.
10,000 bootstrap replicates use seed 104110001. Primary intervals form one
simultaneous 95% family; secondary intervals are descriptive marginal 95%.
They measure deal variation, not variation across new training seeds.

Primary comparisons
-------------------

==========================================  =================================  =========
Comparison                                  Win change (pp), simultaneous 95%  Lower > 0
==========================================  =================================  =========
parent1/mixture/point_matched-parent/win    +0.4486 [-0.5250, +1.4222]         no
parent1/mixture/point_matched-wp_fixed/win  +0.2350 [-0.7334, +1.2034]         no
parent2/mixture/point_matched-parent/win    -0.0366 [-0.9674, +0.8942]         no
parent2/mixture/point_matched-wp_fixed/win  +0.2716 [-0.6764, +1.2196]         no
==========================================  =================================  =========

All absolute results
--------------------

=====  =======  =================  ==========================  ==========================
Group  Scope    Policy/comparison  Win score (%)               Point margin
=====  =======  =================  ==========================  ==========================
1      mixture  parent             51.0010 [50.3937, 51.6113]  +0.2251 [-0.1431, +0.5874]
1      mixture  wp_fixed           51.2146 [50.6134, 51.8250]  +0.0642 [-0.2925, +0.4275]
1      mixture  point_fixed        50.7111 [50.1312, 51.2909]  +2.0305 [+1.6692, +2.3945]
1      mixture  point_matched      51.4496 [50.8606, 52.0233]  +2.0366 [+1.6650, +2.3986]
1      wp       parent             51.3916 [50.5737, 52.1973]  +2.4956 [+2.0139, +2.9602]
1      wp       wp_fixed           52.0569 [51.2756, 52.8625]  +2.4839 [+2.0083, +2.9653]
1      wp       point_fixed        51.0925 [50.2380, 51.9531]  +4.2104 [+3.6770, +4.7478]
1      wp       point_matched      51.9836 [51.1108, 52.8381]  +4.1230 [+3.5774, +4.6536]
1      point    parent             50.6104 [49.7253, 51.4954]  -2.0454 [-2.5967, -1.4951]
1      point    wp_fixed           50.3723 [49.4934, 51.2573]  -2.3555 [-2.8979, -1.8074]
1      point    point_fixed        50.3296 [49.5361, 51.1292]  -0.1494 [-0.6433, +0.3521]
1      point    point_matched      50.9155 [50.1219, 51.7212]  -0.0498 [-0.5525, +0.4524]
2      mixture  parent             51.0956 [50.5219, 51.6724]  +0.1342 [-0.2120, +0.4899]
2      mixture  wp_fixed           50.7874 [50.2045, 51.3855]  +0.1569 [-0.2190, +0.5421]
2      mixture  point_fixed        51.3977 [50.7995, 51.9867]  +2.2417 [+1.8786, +2.6041]
2      mixture  point_matched      51.0590 [50.4760, 51.6357]  +1.9258 [+1.5612, +2.2953]
2      wp       parent             51.8250 [51.0498, 52.6062]  +2.4680 [+2.0046, +2.9492]
2      wp       wp_fixed           51.3245 [50.5127, 52.1303]  +2.4285 [+1.9221, +2.9358]
2      wp       point_fixed        52.0996 [51.2328, 52.9541]  +4.7041 [+4.1626, +5.2410]
2      wp       point_matched      51.2939 [50.4333, 52.1423]  +3.8132 [+3.2734, +4.3340]
2      point    parent             50.3662 [49.4995, 51.2573]  -2.1997 [-2.7410, -1.6394]
2      point    wp_fixed           50.2502 [49.3896, 51.1169]  -2.1147 [-2.6873, -1.5515]
2      point    point_fixed        50.6958 [49.8840, 51.4893]  -0.2207 [-0.7200, +0.2693]
2      point    point_matched      50.8240 [50.0305, 51.5930]  +0.0383 [-0.4600, +0.5317]
=====  =======  =================  ==========================  ==========================

All paired comparisons
----------------------

=====  =======  =========================  ==========================  ==========================
Group  Scope    Policy/comparison          Win change (pp)             Point change
=====  =======  =========================  ==========================  ==========================
1      mixture  point_matched-wp_fixed     +0.2350 [-0.5463, +1.0132]  +1.9724 [+1.5005, +2.4347]
1      mixture  point_matched-point_fixed  +0.7385 [+0.0885, +1.3855]  +0.0061 [-0.3679, +0.3809]
1      mixture  point_matched-parent       +0.4486 [-0.3388, +1.2299]  +1.8115 [+1.3519, +2.2717]
1      mixture  point_fixed-wp_fixed       -0.5035 [-1.2695, +0.2686]  +1.9663 [+1.5116, +2.4374]
1      mixture  point_fixed-parent         -0.2899 [-1.0712, +0.5005]  +1.8054 [+1.3414, +2.2760]
1      mixture  wp_fixed-parent            +0.2136 [-0.4181, +0.8545]  -0.1609 [-0.5197, +0.2022]
1      wp       point_matched-wp_fixed     -0.0732 [-1.2085, +1.0315]  +1.6392 [+0.9587, +2.2876]
1      wp       point_matched-point_fixed  +0.8911 [-0.0488, +1.8311]  -0.0874 [-0.6426, +0.4627]
1      wp       point_matched-parent       +0.5920 [-0.5127, +1.7029]  +1.6274 [+0.9673, +2.2837]
1      wp       point_fixed-wp_fixed       -0.9644 [-2.0691, +0.1221]  +1.7266 [+1.0630, +2.3850]
1      wp       point_fixed-parent         -0.2991 [-1.3916, +0.8179]  +1.7148 [+1.0640, +2.3750]
1      wp       wp_fixed-parent            +0.6653 [-0.2197, +1.5871]  -0.0117 [-0.5095, +0.4942]
1      point    point_matched-wp_fixed     +0.5432 [-0.5615, +1.6235]  +2.3057 [+1.6460, +2.9629]
1      point    point_matched-point_fixed  +0.5859 [-0.3418, +1.4954]  +0.0996 [-0.4268, +0.6355]
1      point    point_matched-parent       +0.3052 [-0.7751, +1.3918]  +1.9956 [+1.3298, +2.6536]
1      point    point_fixed-wp_fixed       -0.0427 [-1.1354, +1.0437]  +2.2061 [+1.5261, +2.8826]
1      point    point_fixed-parent         -0.2808 [-1.3794, +0.8179]  +1.8960 [+1.2358, +2.5640]
1      point    wp_fixed-parent            -0.2380 [-1.1169, +0.6226]  -0.3101 [-0.8118, +0.1985]
2      mixture  point_matched-wp_fixed     +0.2716 [-0.4852, +1.0254]  +1.7689 [+1.2904, +2.2479]
2      mixture  point_matched-point_fixed  -0.3387 [-0.9705, +0.2839]  -0.3159 [-0.6846, +0.0405]
2      mixture  point_matched-parent       -0.0366 [-0.7843, +0.6989]  +1.7916 [+1.3292, +2.2450]
2      mixture  point_fixed-wp_fixed       +0.6104 [-0.1649, +1.3794]  +2.0848 [+1.6028, +2.5638]
2      mixture  point_fixed-parent         +0.3021 [-0.4578, +1.0559]  +2.1075 [+1.6507, +2.5581]
2      mixture  wp_fixed-parent            -0.3082 [-0.9216, +0.2991]  +0.0227 [-0.3506, +0.3886]
2      wp       point_matched-wp_fixed     -0.0305 [-1.1475, +1.0803]  +1.3848 [+0.7014, +2.0708]
2      wp       point_matched-point_fixed  -0.8057 [-1.6968, +0.0854]  -0.8909 [-1.4089, -0.3691]
2      wp       point_matched-parent       -0.5310 [-1.6357, +0.5188]  +1.3452 [+0.6750, +1.9927]
2      wp       point_fixed-wp_fixed       +0.7751 [-0.3479, +1.8800]  +2.2756 [+1.5927, +2.9739]
2      wp       point_fixed-parent         +0.2747 [-0.8118, +1.3550]  +2.2361 [+1.5754, +2.8855]
2      wp       wp_fixed-parent            -0.5005 [-1.3918, +0.4089]  -0.0396 [-0.5620, +0.4898]
2      point    point_matched-wp_fixed     +0.5737 [-0.5066, +1.6357]  +2.1531 [+1.4673, +2.8223]
2      point    point_matched-point_fixed  +0.1282 [-0.7812, +1.0132]  +0.2590 [-0.2532, +0.7678]
2      point    point_matched-parent       +0.4578 [-0.6165, +1.5198]  +2.2380 [+1.5718, +2.9094]
2      point    point_fixed-wp_fixed       +0.4456 [-0.6226, +1.5320]  +1.8940 [+1.2161, +2.5672]
2      point    point_fixed-parent         +0.3296 [-0.7629, +1.4343]  +1.9790 [+1.3027, +2.6304]
2      point    wp_fixed-parent            -0.1160 [-0.9888, +0.7629]  +0.0850 [-0.4510, +0.6052]
=====  =======  =========================  ==========================  ==========================

Work and interpretation
-----------------------

Main training totals (warmup included once): {"decisions": 51118080, "matches": 439369, "rounds": 3398982, "updates": 3120}
Actual evaluation and separately replayed work: {"decisions": 15229803, "full_matches": 131072, "round_endings": 1016240}
Point margins are candidate minus opponent; divide by two for the
original paper's final-minus-initial point convention.
The audit replays recorded actions and checks statistics; it does not
rerun training or neural inference. This export adds no matches.
Calibration and capability work is separate from these main totals;
the experiment ledger retains those costs and repeated verification work.
JSON preserves unrounded values in fraction units for win scores.
If qualified, the predeclared candidate is seed102110001 point_matched.
Reserved101M prospective confirmation is still required for promotion.
No endpoint is selected or promoted by this report.
