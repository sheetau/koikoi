Root allocation and step-size results
=====================================

The eight-arm study is complete and independently audited. Its predefined
progression criterion fails. The priority/high-rate candidate does not establish
an improvement over either retained parent; the second parent-relative simultaneous interval
is wholly negative. No model is promoted and the canonical player is retained.

Both parents were trained with uniform/priority roots and Adam rates 1e-5/1e-4.
Every arm received 128 updates and 8,192 complete source matches. Source and
update budgets match; additional branch costs differ. All eight endpoints and
both parents were evaluated against assisted original WP and Point, using the
GPU batch-128 greedy path. There are 1,024 distinct paired streams per opponent,
2,048 streams in total, and 40,960 actual matches. This development evaluation
is distinct from the original paper's strict CPU singleton final comparison.
The reserved 101M block remains unused.

Primary results
---------------

Changes are equal-WP/Point-mixture draw-adjusted win scores in percentage points.
One simultaneous 95% family covers the four predefined contrasts. Its radius
comes from 10,000 maximum-absolute-centered-deviation bootstrap replicates over
1,024 compound blocks, retaining both seats, opponents and all models together.
Progression requires all four lower bounds above zero.

.. list-table:: All primary comparisons
   :header-rows: 1

   * - Candidate
     - Comparator
     - Change (pp)
     - Simultaneous 95% interval
   * - parent1_priority_high
     - parent1
     - -0.3906
     - [-2.2583, +1.4771]
   * - parent1_priority_high
     - parent1_uniform_high
     - +3.2349
     - [+1.3672, +5.1025]
   * - parent2_priority_high
     - parent2
     - -2.0874
     - [-3.9551, -0.2197]
   * - parent2_priority_high
     - parent2_uniform_high
     - -0.6226
     - [-2.4902, +1.2451]

Priority selection exceeds uniform selection at the high learning rate for
parent 1, but that comparator itself has a low observed win score. This does
not establish improvement over the starting parent, and the selection effect
does not replicate at parent 2. The high-rate versus low-rate means below are
also nonpositive in all four within-sampler comparisons. Increasing the update
size is therefore not a supported improvement from this screen.

Secondary results
-----------------

The following contrasts were predefined descriptive means, with no confidence
intervals or separate success criterion. All are reported; none is promoted
from a favorable single comparison.

.. list-table:: All secondary contrast means
   :header-rows: 1

   * - Candidate
     - Comparator
     - Win change (pp)
   * - parent1_priority_low
     - parent1_uniform_low
     - -0.2930
   * - parent1_uniform_high
     - parent1_uniform_low
     - -3.5278
   * - parent1_priority_high
     - parent1_priority_low
     - +0.0000
   * - parent1_priority_high
     - parent1_uniform_low
     - -0.2930
   * - parent2_priority_low
     - parent2_uniform_low
     - -0.9155
   * - parent2_uniform_high
     - parent2_uniform_low
     - -1.8066
   * - parent2_priority_high
     - parent2_priority_low
     - -1.5137
   * - parent2_priority_high
     - parent2_uniform_low
     - -2.4292

Absolute matchup means are descriptive. Win scores count a draw as one half.
Point margins are candidate minus opponent.

.. list-table:: All endpoint and parent matchup means
   :header-rows: 1

   * - Model
     - Opponent
     - Win score (%)
     - Point margin
   * - parent1_uniform_low
     - WP
     - 51.2451
     - +2.2871
   * - parent1_uniform_low
     - POINT
     - 49.8535
     - -1.5420
   * - parent1_priority_low
     - WP
     - 51.5381
     - +2.2725
   * - parent1_priority_low
     - POINT
     - 48.9746
     - -1.9609
   * - parent1_uniform_high
     - WP
     - 48.1934
     - -1.3975
   * - parent1_uniform_high
     - POINT
     - 45.8496
     - -6.5479
   * - parent1_priority_high
     - WP
     - 51.1230
     - +1.0869
   * - parent1_priority_high
     - POINT
     - 49.3896
     - -3.6445
   * - parent2_uniform_low
     - WP
     - 50.8057
     - +1.5654
   * - parent2_uniform_low
     - POINT
     - 50.4883
     - -2.2217
   * - parent2_priority_low
     - WP
     - 49.8047
     - +0.8408
   * - parent2_priority_low
     - POINT
     - 49.6582
     - -2.8379
   * - parent2_uniform_high
     - WP
     - 48.6816
     - -0.5938
   * - parent2_uniform_high
     - POINT
     - 48.9990
     - -4.4072
   * - parent2_priority_high
     - WP
     - 48.0469
     - +1.1816
   * - parent2_priority_high
     - POINT
     - 48.3887
     - -3.0430
   * - parent1
     - WP
     - 52.1484
     - +2.3662
   * - parent1
     - POINT
     - 49.1455
     - -1.6865
   * - parent2
     - WP
     - 50.3418
     - +1.8564
   * - parent2
     - POINT
     - 50.2686
     - -1.9756

Training diagnostics
--------------------

These are logged training-batch summaries over all 128 updates. A confident
root assigns at least 0.99 probability to one action. A constant-outcome root
has identical sampled terminal outcomes for all legal actions; it need not
have identical true action values. Action changes count only greedy choices
before/after each individual update on that update's own 64 sampled roots.
They are not endpoint disagreement rates on a shared held-out state set.

Priority selection reduces confident-root frequency from approximately 30%
to 5--6%, while constant sampled outcomes remain at approximately 68--70%.
The larger learning rate produces larger local changes, but those changes do
not establish stronger play. The observations distinguish working allocation
mechanics from the failed playing-strength hypothesis; they do not identify
which source of estimation or generalization error causes the failure.

.. list-table:: All logged training diagnostics
   :header-rows: 1

   * - Arm
     - Confident roots (%)
     - Constant outcomes (%)
     - Within-update action changes
     - Mean within-update KL
   * - parent1_uniform_low
     - 29.529
     - 68.811
     - 1
     - 0.00003104
   * - parent1_priority_low
     - 5.347
     - 69.080
     - 7
     - 0.00002699
   * - parent1_uniform_high
     - 33.423
     - 69.910
     - 52
     - 0.00130635
   * - parent1_priority_high
     - 6.067
     - 68.384
     - 84
     - 0.00204460
   * - parent2_uniform_low
     - 29.651
     - 68.335
     - 5
     - 0.00003925
   * - parent2_priority_low
     - 5.408
     - 69.897
     - 10
     - 0.00002721
   * - parent2_uniform_high
     - 27.722
     - 70.105
     - 43
     - 0.00094143
   * - parent2_priority_high
     - 6.287
     - 68.811
     - 66
     - 0.00184413

Work and verification
---------------------

Source training totals 7,619,431 decisions, 512,651 round endings and 65,536
matches. Additional branches total 19,943,108 decisions, 1,458,051 round endings
and 326,506 completions. The branches share source worlds and are not additional
independent source deals. There are 1,024 optimizer updates in total. Training
elapsed time is 1,641.024 seconds including collection, optimization, recording
and checkpoint I/O; it is not an isolated throughput benchmark.

.. list-table:: Per-arm work; each has 8,192 source matches
   :header-rows: 1

   * - Arm
     - Source decisions
     - Source rounds
     - Branch decisions
     - Branch rounds
     - Branch completions
   * - parent1_uniform_low
     - 958738
     - 64010
     - 2391379
     - 174167
     - 39304
   * - parent1_priority_low
     - 948094
     - 64010
     - 2562401
     - 186831
     - 42234
   * - parent1_uniform_high
     - 966260
     - 64699
     - 2455373
     - 180021
     - 39409
   * - parent1_priority_high
     - 895955
     - 65115
     - 2517751
     - 196254
     - 43491
   * - parent2_uniform_low
     - 955446
     - 64279
     - 2410800
     - 176895
     - 39386
   * - parent2_priority_low
     - 949820
     - 64313
     - 2532416
     - 185712
     - 42077
   * - parent2_uniform_high
     - 979784
     - 63242
     - 2469145
     - 174870
     - 38817
   * - parent2_priority_high
     - 965334
     - 62983
     - 2603843
     - 183301
     - 41788

Evaluation adds 4,783,517 decisions, 318,537 round endings and 40,960 matches.
Training and evaluation audits replay every recorded transition and verify
initial deals, selected roots, recorded reservoir probabilities, corrections,
branch labels, counters and all reported match statistics. The primary bootstrap
is independently reconstructed. These audits do not repeat all neural inference
or optimizer calculations. Replay is not new independent playing evidence.

Session 92368 / PID 33408 exited successfully. The run is closed; do not restart
or overwrite it. Fixed design: PRIORITY_BRANCH_GRADIENT.rst.
Artifacts: ``runs/priority_branch_study``.

Training protocol SHA256 4543224c95c1ec5e9c24cfa0ed6f4450eda956e3d8e5bf79f125bd9fa105e79b.
Training completion SHA256 29ab49b2d19ed122db8c397f0fa229300142a19ec30aca4b034577af3531d425.
Training audit SHA256 8fb3967cfb3af9e4fd0c50e7e9b4e2cb407faa0fb059aff89c94826fda19c5d7.
Evaluation protocol SHA256 78c133875b4efd9d8d5037d2fa7683053fb9a0e8d68f1b0fb45f4d91d0056d8f.
Evaluation summary SHA256 c824076bb5af7e745dd22c1b284de28c41ac45d652a1a0d55be8dd22077a5b78.
Evaluation audit SHA256 f6dafc4c4d7609026822ce268ae8efd13b8710ebc778e9656f72674887da01c8.
Overall completion SHA256 397eab0cc34387e96a8e09e42f439f6bcae7009eaf08025d128db819a16e8626.
