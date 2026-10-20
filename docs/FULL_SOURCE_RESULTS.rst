Complete-source Monte Carlo comparison
======================================

The completed twelve-arm study does not pass its predefined six-contrast
development criterion. Both training and evaluation audits pass.
No canonical model is promoted by this screen.

Both retained parents receive all-source Monte Carlo, one-root Monte
Carlo and shared all-action branches, each at learning rates 1e-5 and
1e-4. Every arm receives 128 updates and 8,192 source matches.
Matching source matches does not match gradient or branch computation.
All twelve endpoints and both parents face assisted WP and Point on
1,024 duplicate pairs per opponent, using GPU batch-128 greedy play.
The 57,344 evaluation matches share 2,048 distinct deal streams.
The reserved 101M block is unused. See FULL_SOURCE_STUDY.rst for the
fixed design and the distinction from the paper's CPU evaluation.

Primary comparisons
-------------------

Win changes are percentage points in the equal-WP/Point mixture,
counting draws as one half. All six lower bounds must exceed zero.
The simultaneous 95% radius uses 10,000 maximum-absolute-centered
bootstrap deviations over compound paired-deal blocks. These are
development comparisons, not a universal strength ranking.

.. list-table:: Every predefined primary contrast
   :header-rows: 1

   * - Candidate
     - Comparator
     - Change (pp)
     - Simultaneous 95% interval
   * - parent1_full_high
     - parent1
     - -1.1719
     - [-3.2227, +0.8789]
   * - parent1_full_high
     - parent1_root_high
     - +2.3315
     - [+0.2808, +4.3823]
   * - parent1_full_high
     - parent1_branch_high
     - +2.5757
     - [+0.5249, +4.6265]
   * - parent2_full_high
     - parent2
     - -0.6592
     - [-2.7100, +1.3916]
   * - parent2_full_high
     - parent2_root_high
     - +3.8940
     - [+1.8433, +5.9448]
   * - parent2_full_high
     - parent2_branch_high
     - +1.4893
     - [-0.5615, +3.5400]

Secondary comparisons
---------------------

The following means are descriptive; no secondary interval is inferred.

.. list-table:: Every predefined secondary contrast
   :header-rows: 1

   * - Candidate
     - Comparator
     - Change (pp)
   * - parent1_full_low
     - parent1_root_low
     - -1.0498
   * - parent1_full_low
     - parent1_branch_low
     - +0.1831
   * - parent1_full_high
     - parent1_full_low
     - -1.3794
   * - parent1_root_high
     - parent1_root_low
     - -4.7607
   * - parent1_branch_high
     - parent1_branch_low
     - -3.7720
   * - parent1_full_low
     - parent1
     - +0.2075
   * - parent2_full_low
     - parent2_root_low
     - -0.0244
   * - parent2_full_low
     - parent2_branch_low
     - -0.0610
   * - parent2_full_high
     - parent2_full_low
     - -0.6714
   * - parent2_root_high
     - parent2_root_low
     - -4.5898
   * - parent2_branch_high
     - parent2_branch_low
     - -2.2217
   * - parent2_full_low
     - parent2
     - +0.0122

.. list-table:: Every endpoint and parent matchup
   :header-rows: 1

   * - Model
     - Opponent
     - Win score (%)
     - Point margin
   * - parent1_full_low
     - WP
     - 49.2920
     - +0.9961
   * - parent1_full_low
     - POINT
     - 52.2461
     - -0.9795
   * - parent1_root_low
     - WP
     - 50.5371
     - +1.9941
   * - parent1_root_low
     - POINT
     - 53.1006
     - -0.0752
   * - parent1_branch_low
     - WP
     - 49.1455
     - +1.2617
   * - parent1_branch_low
     - POINT
     - 52.0264
     - -0.5625
   * - parent1_full_high
     - WP
     - 47.9492
     - -0.9893
   * - parent1_full_high
     - POINT
     - 50.8301
     - -2.7471
   * - parent1_root_high
     - WP
     - 46.4600
     - -2.7734
   * - parent1_root_high
     - POINT
     - 47.6562
     - -6.0010
   * - parent1_branch_high
     - WP
     - 46.3379
     - -2.4893
   * - parent1_branch_high
     - POINT
     - 47.2900
     - -5.1992
   * - parent2_full_low
     - WP
     - 50.5127
     - +1.8848
   * - parent2_full_low
     - POINT
     - 51.3916
     - -0.7451
   * - parent2_root_low
     - WP
     - 49.7559
     - +1.6953
   * - parent2_root_low
     - POINT
     - 52.1973
     - -1.0332
   * - parent2_branch_low
     - WP
     - 50.8301
     - +2.3799
   * - parent2_branch_low
     - POINT
     - 51.1963
     - -0.5518
   * - parent2_full_high
     - WP
     - 49.2188
     - +0.2803
   * - parent2_full_high
     - POINT
     - 51.3428
     - -2.4980
   * - parent2_root_high
     - WP
     - 46.0938
     - -3.5303
   * - parent2_root_high
     - POINT
     - 46.6797
     - -6.1836
   * - parent2_branch_high
     - WP
     - 48.8525
     - +1.1650
   * - parent2_branch_high
     - POINT
     - 48.7305
     - -2.3916
   * - parent1
     - WP
     - 49.3164
     - +1.4551
   * - parent1
     - POINT
     - 51.8066
     - -0.7520
   * - parent2
     - WP
     - 50.4150
     - +2.1211
   * - parent2
     - POINT
     - 51.4648
     - -1.0039

Work and evidence
-----------------

Work triples below are decisions / round endings / completed matches
or branches. Gradient rows count all source decisions for full-source
learning and selected roots for the two root estimators. Branch action
values contribute within each selected root's loss.

.. list-table:: Actual training work
   :header-rows: 1

   * - Condition
     - Source work
     - Additional branch work
     - Gradient rows
   * - parent1_full_low
     - 950,976 / 64,195 / 8,192
     - 0 / 0 / 0
     - 907,342
   * - parent1_root_low
     - 939,050 / 64,128 / 8,192
     - 0 / 0 / 0
     - 8,192
   * - parent1_branch_low
     - 955,097 / 63,954 / 8,192
     - 2,440,654 / 177,814 / 39,142
     - 8,192
   * - parent1_full_high
     - 959,781 / 64,604 / 8,192
     - 0 / 0 / 0
     - 911,461
   * - parent1_root_high
     - 881,495 / 65,253 / 8,192
     - 0 / 0 / 0
     - 8,192
   * - parent1_branch_high
     - 915,305 / 64,779 / 8,192
     - 2,336,805 / 179,217 / 39,928
     - 8,192
   * - parent2_full_low
     - 955,016 / 63,735 / 8,192
     - 0 / 0 / 0
     - 909,347
   * - parent2_root_low
     - 953,641 / 64,415 / 8,192
     - 0 / 0 / 0
     - 8,192
   * - parent2_branch_low
     - 966,060 / 63,631 / 8,192
     - 2,405,556 / 173,091 / 38,927
     - 8,192
   * - parent2_full_high
     - 962,322 / 64,156 / 8,192
     - 0 / 0 / 0
     - 917,428
   * - parent2_root_high
     - 915,632 / 65,250 / 8,192
     - 0 / 0 / 0
     - 8,192
   * - parent2_branch_high
     - 918,396 / 64,162 / 8,192
     - 2,353,841 / 178,065 / 39,586
     - 8,192

Source: 11,272,771 decisions, 772,262 round endings, 98,304 completions.

Branches: 9,536,856 decisions, 708,187 round endings, 157,583 completions.

Evaluation: 6,598,037 decisions, 446,749 round endings, 57,344 completions.

Total: 1,536 optimizer updates and 3,711,114 gradient rows.
Training elapsed: 1806.783383 seconds including I/O;
this is not an isolated throughput benchmark.

The training audit replays source and branch transitions, every stored
full-source state/action/return, deals, selected roots and work counts.
The evaluation audit replays trajectories and independently constructs
the simultaneous bootstrap. Neither repeats all neural inference or
optimizer updates. This exporter checks the completed receipt chain
and metrics hashes; it does not rerun those audits.

* ``runs\full_source_study\protocol.json``: ``d878e4c8307f711e4fb1a3460502bf1fa6de353ede85b168f81a1d725ab00447``.
* ``runs\full_source_study\audit.json``: ``58279e548d54d872dca9495c246a265d181999430199fe6591a570d339612525``.
* ``runs\full_source_study\evaluation\summary.json``: ``ff6b9b7cff08a4dc517d10d577286666b1a24f6d98f551ccbba12c35ce0e59fb``.
* ``runs\full_source_study\evaluation\audit.json``: ``e30fd79fe943c27d4ee1b24612571b456f688809de5b2c797373ec86e7ddfb38``.
* ``runs\full_source_study\complete.json``: ``fa6eec799809b5b709169f9318654a354a6c8323579156f84f2ff0792fc77fa2``.
* ``runs\full_source_study\finished.json``: ``60c7cf919e41a443039a2ef7e362ee4b699d4dfaa7279b80d32abbba673de840``.
* ``runs\full_source_study\evaluation\points.npy``: ``3d736b17ab5b747de8b5ce54ba178452465665861baeabe89f6ea846c1a56358``.
* ``runs\full_source_study\parent1_full_low\metrics.jsonl``: ``8254e3015af6e52fb1c2c4604c00ff953da256e86264f5b87f36974a8c716442``.
* ``runs\full_source_study\parent1_root_low\metrics.jsonl``: ``bcd76ab841283c64a100f9df10a356803b6f490b395a09c3360a55a27bb22a28``.
* ``runs\full_source_study\parent1_branch_low\metrics.jsonl``: ``7db4c0314ccfba5347aea5d558d8c3a6821ec33d79ad3a3a8f5d903d2790db99``.
* ``runs\full_source_study\parent1_full_high\metrics.jsonl``: ``d0f3c55934eaca8fbd05c20fea0d9d5bed64a0d7ea08f7fbb5a9d0bcc9d47a43``.
* ``runs\full_source_study\parent1_root_high\metrics.jsonl``: ``48f89e520577e5457c5887b08d912b29e71f58532c82601bccadff20cda0cf65``.
* ``runs\full_source_study\parent1_branch_high\metrics.jsonl``: ``f0c92804eed208982adccdded397b492d0cde4f2a47b2001944dfda4eb21b5a6``.
* ``runs\full_source_study\parent2_full_low\metrics.jsonl``: ``f8e1cd055ed4f8d42081f66de1be68196854e1c2d8754bb747153cfd4df884aa``.
* ``runs\full_source_study\parent2_root_low\metrics.jsonl``: ``9f71b9650b1918b07452ea4eff5cc1ae338c8291ba354e7d6cfd164c74cd05c3``.
* ``runs\full_source_study\parent2_branch_low\metrics.jsonl``: ``718fd87a2fc7ef85d37d66d2603e01895617e1fd738de6f9144af11e07da712e``.
* ``runs\full_source_study\parent2_full_high\metrics.jsonl``: ``8b28994c06702a3823c85fb4d19b665edadf0c16e78b5344fe9de400948b0f29``.
* ``runs\full_source_study\parent2_root_high\metrics.jsonl``: ``9591be38760199a25499006d1dfd717b4596706b453d2e2ea8204cc90b3e95a6``.
* ``runs\full_source_study\parent2_branch_high\metrics.jsonl``: ``1bf9db7333eabbdf1fc1c163143e9b0671cd9a4abbfb7dcb5aa0b5870da3fd44``.
