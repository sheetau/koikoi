Bounded continuation comparison
===============================

All 28 completed roots / 56 fixed and optimized plans preserve the
chosen action, exact value and entire chosen policy. Every computed
root value/policy matches the saved independently verified solution.
Partial upper bounds contain the control's exact values and cannot
beat the incumbent. Fourteen roots run each method first; caches are
fresh and separate, with fixed then optimized within each method.

Physical decisions fall from 13,896,524 to 12,422,529 (10.6069% reduction).
Observed total planning time is 155.243126s versus 139.910682s (9.8764% reduction).
The largest individual plan takes 21.064108s versus 21.528018s. These are paired observations, not a
latency distribution or a prospective strength evaluation.

There are 26 root-action upper bounds below1; 2,330 candidate actions are
discarded after partial evaluation across the full trees.

.. list-table:: Every completed plan
   :header-rows: 1

   * - Source
     - Mode
     - Action
     - Exact chosen Q
     - Ceiling decisions
     - Bounded decisions
     - Ceiling seconds
     - Bounded seconds
     - Root bounds
   * - 1
     - fixed
     - 29
     - 3339/5720
     - 895459
     - 792149
     - 15.081687
     - 13.439084
     - 2
   * - 1
     - optimized
     - 29
     - 51767/85800
     - 1500136
     - 1207086
     - 18.357607
     - 12.805783
     - 2
   * - 2
     - fixed
     - 43
     - 1
     - 72
     - 72
     - 0.005279
     - 0.006220
     - 1
   * - 2
     - optimized
     - 43
     - 1
     - 72
     - 72
     - 0.001346
     - 0.001947
     - 1
   * - 6
     - fixed
     - 37
     - 1
     - 72
     - 72
     - 0.005626
     - 0.004941
     - 1
   * - 6
     - optimized
     - 37
     - 1
     - 72
     - 72
     - 0.002247
     - 0.001873
     - 1
   * - 11
     - fixed
     - 23
     - 1487/3861
     - 162036
     - 151525
     - 3.204073
     - 2.857810
     - 2
   * - 11
     - optimized
     - 23
     - 30451/77220
     - 290819
     - 262396
     - 3.911468
     - 3.785823
     - 2
   * - 12
     - fixed
     - 7
     - 1
     - 71975
     - 71975
     - 0.859692
     - 0.815117
     - 1
   * - 12
     - optimized
     - 7
     - 1
     - 71975
     - 71975
     - 0.403627
     - 0.348558
     - 1
   * - 18
     - fixed
     - 21
     - 23433/28600
     - 611509
     - 533876
     - 10.335674
     - 9.274688
     - 1
   * - 18
     - optimized
     - 0
     - 25249/28600
     - 997803
     - 644251
     - 9.345834
     - 6.285843
     - 2
   * - 19
     - fixed
     - 26
     - 1
     - 35205
     - 35205
     - 0.472785
     - 0.486093
     - 1
   * - 19
     - optimized
     - 26
     - 1
     - 35205
     - 35205
     - 0.369671
     - 0.375029
     - 1
   * - 21
     - fixed
     - 48
     - 1
     - 50
     - 50
     - 0.000400
     - 0.000480
     - 1
   * - 21
     - optimized
     - 48
     - 1
     - 50
     - 50
     - 0.000411
     - 0.000324
     - 1
   * - 25
     - fixed
     - 48
     - 1
     - 32
     - 32
     - 0.000328
     - 0.000325
     - 1
   * - 25
     - optimized
     - 48
     - 1
     - 32
     - 32
     - 0.000272
     - 0.000297
     - 1
   * - 26
     - fixed
     - 36
     - 14177/77220
     - 927229
     - 895182
     - 13.742766
     - 13.655905
     - 2
   * - 26
     - optimized
     - 36
     - 26051/123552
     - 1807939
     - 1700861
     - 21.064108
     - 21.528018
     - 2
   * - 27
     - fixed
     - 15
     - 0
     - 381507
     - 381507
     - 4.576034
     - 4.683511
     - 0
   * - 27
     - optimized
     - 15
     - 0
     - 381543
     - 381543
     - 1.942893
     - 1.955028
     - 0
   * - 28
     - fixed
     - 17
     - 3293/4158
     - 180490
     - 176613
     - 3.918577
     - 3.929605
     - 1
   * - 28
     - optimized
     - 17
     - 3293/4158
     - 180490
     - 176613
     - 0.846350
     - 0.761297
     - 1
   * - 29
     - fixed
     - 42
     - 2335/2376
     - 18042
     - 15305
     - 0.306292
     - 0.280237
     - 1
   * - 29
     - optimized
     - 42
     - 2335/2376
     - 18042
     - 15305
     - 0.183904
     - 0.165522
     - 1
   * - 34
     - fixed
     - 41
     - 1
     - 274820
     - 274820
     - 3.723894
     - 3.714361
     - 2
   * - 34
     - optimized
     - 41
     - 1
     - 274820
     - 274820
     - 2.648419
     - 2.755951
     - 2
   * - 38
     - fixed
     - 37
     - 1
     - 48
     - 48
     - 0.004441
     - 0.003664
     - 1
   * - 38
     - optimized
     - 37
     - 1
     - 48
     - 48
     - 0.001680
     - 0.001525
     - 1
   * - 39
     - fixed
     - 15
     - 5273/5280
     - 417758
     - 394919
     - 3.374528
     - 3.211526
     - 1
   * - 39
     - optimized
     - 15
     - 5273/5280
     - 417758
     - 394919
     - 2.157015
     - 2.101800
     - 1
   * - 41
     - fixed
     - 10
     - 1/1320
     - 5228
     - 5228
     - 0.126990
     - 0.134255
     - 0
   * - 41
     - optimized
     - 10
     - 1/1320
     - 5242
     - 5242
     - 0.063198
     - 0.065469
     - 0
   * - 44
     - fixed
     - 7
     - 1
     - 71975
     - 71975
     - 0.760546
     - 0.798560
     - 1
   * - 44
     - optimized
     - 7
     - 1
     - 71975
     - 71975
     - 0.350212
     - 0.355732
     - 1
   * - 46
     - fixed
     - 46
     - 1
     - 32090
     - 32090
     - 0.417509
     - 0.405361
     - 1
   * - 46
     - optimized
     - 46
     - 1
     - 32090
     - 32090
     - 0.260441
     - 0.277402
     - 1
   * - 50
     - fixed
     - 21
     - 70903/85800
     - 609799
     - 556800
     - 9.910104
     - 9.183589
     - 1
   * - 50
     - optimized
     - 0
     - 25249/28600
     - 999217
     - 648205
     - 9.141691
     - 5.987357
     - 2
   * - 51
     - fixed
     - 26
     - 1
     - 35205
     - 35205
     - 0.480185
     - 0.508235
     - 1
   * - 51
     - optimized
     - 26
     - 1
     - 35205
     - 35205
     - 0.351741
     - 0.354566
     - 1
   * - 53
     - fixed
     - 48
     - 1
     - 97
     - 97
     - 0.000402
     - 0.000674
     - 1
   * - 53
     - optimized
     - 48
     - 1
     - 97
     - 97
     - 0.000435
     - 0.000416
     - 1
   * - 55
     - fixed
     - 48
     - 1
     - 10
     - 10
     - 0.000342
     - 0.000298
     - 1
   * - 55
     - optimized
     - 48
     - 1
     - 10
     - 10
     - 0.000250
     - 0.000249
     - 1
   * - 57
     - fixed
     - 48
     - 1
     - 8
     - 8
     - 0.000305
     - 0.000464
     - 1
   * - 57
     - optimized
     - 48
     - 1
     - 8
     - 8
     - 0.000239
     - 0.000271
     - 1
   * - 59
     - fixed
     - 15
     - 0
     - 730817
     - 730817
     - 5.330639
     - 5.460291
     - 0
   * - 59
     - optimized
     - 15
     - 0
     - 730933
     - 730933
     - 2.818998
     - 2.821298
     - 0
   * - 60
     - fixed
     - 25
     - 135319/146520
     - 236507
     - 236507
     - 2.125900
     - 2.157528
     - 0
   * - 60
     - optimized
     - 25
     - 135319/146520
     - 236507
     - 236507
     - 0.972414
     - 0.985754
     - 0
   * - 61
     - fixed
     - 42
     - 2335/2376
     - 18042
     - 15305
     - 0.314841
     - 0.266414
     - 1
   * - 61
     - optimized
     - 42
     - 2335/2376
     - 18042
     - 15305
     - 0.186411
     - 0.173691
     - 1
   * - 62
     - fixed
     - 44
     - 1
     - 37156
     - 37156
     - 0.442741
     - 0.445248
     - 1
   * - 62
     - optimized
     - 44
     - 1
     - 37156
     - 37156
     - 0.337664
     - 0.289377
     - 1

.. list-table:: Additional planning work, kept separate from source matches
   :header-rows: 1

   * - Method
     - Counter
     - Count
   * - ceiling
     - decisions
     - 13896524
   * - ceiling
     - round_endings
     - 12333528
   * - ceiling
     - completions
     - 12333528
   * - ceiling
     - live_observation_nodes
     - 1127819
   * - ceiling
     - own_decision_nodes
     - 778233
   * - ceiling
     - chance_rows_added
     - 12312978
   * - ceiling
     - represented_decisions
     - 202121109
   * - ceiling
     - represented_round_endings
     - 97646760
   * - ceiling
     - represented_completions
     - 97646760
   * - ceiling
     - terminal_rows_valued
     - 12333528
   * - ceiling
     - terminal_mass_valued
     - 97646760
   * - ceiling
     - ceiling_cutoffs
     - 7753
   * - ceiling
     - pruned_actions
     - 7755
   * - bounded
     - decisions
     - 12422529
   * - bounded
     - round_endings
     - 11003741
   * - bounded
     - completions
     - 11003741
   * - bounded
     - live_observation_nodes
     - 988289
   * - bounded
     - own_decision_nodes
     - 677707
   * - bounded
     - chance_rows_added
     - 11001884
   * - bounded
     - represented_decisions
     - 188749692
   * - bounded
     - represented_round_endings
     - 92806947
   * - bounded
     - represented_completions
     - 92806947
   * - bounded
     - terminal_rows_valued
     - 11003741
   * - bounded
     - terminal_mass_valued
     - 92806947
   * - bounded
     - ceiling_cutoffs
     - 6714
   * - bounded
     - pruned_actions
     - 6716
   * - bounded
     - continuation_cutoffs
     - 2330
   * - bounded
     - bounded_actions
     - 2330

Both methods use500k live groups,2M physical transitions,4096 frontier
rows and depth32. No source match, training update or actual-tail
assessment was added. Existing Python oracle evidence is reused by
exact equality of full policies and values; no new full Python or
neural replay is claimed. This is an exact computational improvement,
not a new match-win result or a claim of general algorithm priority.

Protocol SHA256: 110dd1c58a9d89e5073dd0835852fc9930d8b5777b13141afb1e90dd58268df5.
Complete SHA256: 7770f4879c856119ffcca0547544fcaf547b3d39241396375a5f449eef9938c2.
