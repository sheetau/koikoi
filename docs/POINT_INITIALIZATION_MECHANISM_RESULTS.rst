Final-policy effects of Point phase calibration
==============================================

All six fixed 520-update learners are complete. This diagnostic evaluates
how their action scores changed on the existing validation inputs. The separate
131,072-match comparison is running; no partial match outcomes are used here.

Findings
--------

At both seeds and in all four phases, calibrated Point endpoints retain higher
normalized entropy at their declared divisors than fixed-divisor Point controls.
The intervention therefore remains visible after learning on these inputs.
The final calibrated entropy is nevertheless below its initial calibrated
value in every phase at both seeds. The fixed divisors do not keep validation entropy at its initial calibrated
value. These validation entropies were not exact WP matches at initialization.

The calibrated endpoints also have larger mean raw top-two score gaps than
fixed Point controls in all eight phase/seed comparisons. Applying the common
0.05 divisor reverses the entropy ordering: calibrated endpoints are then more
concentrated than fixed controls. Both the declared divisor and the learned
scores matter when describing the final action probabilities.

Greedy koi/stop choices differ between calibrated and fixed Point endpoints on
only 3 and 6 of the same 215 unforced inputs, respectively. Discard choices
differ on 232 and 200 of 2,578 inputs. More probability entropy does not itself
imply different greedy actions: deployment uses the same positive 0.05 divisor,
and changes between final greedy policies arise from their learned scores.
Neither disagreement nor agreement says which action is better.

Population and interpretation
-----------------------------

The old corpus supplies 3,258 rows from 25 random-policy source matches whose
nonnegative IDs are 3 modulo 4. These are the same descriptive validation groups
excluded from fitting the four divisors. After unchanged public endgame guards,
there are 3,045 unforced choices:2,578 discard,122 hand-pick,130 draw-pick and215
koi/stop. No new states or final training snapshots are added.

Six endpoint policies make 18,270 singleton CPU2 neural calls. Original WP and
Point scores are reused from the archive. There are zero simulator transitions,
new matches or training updates. Initial Point fixed/calibrated rows have the
same scores and greedy actions, with different declared probability divisors.
The current results describe reused correlated inputs, not the agents' complete
training or deployment visitation. No confidence intervals, causal mediation,
playing-strength gain or new algorithm is inferred from these measurements.
The full-match comparison remains the separate source of strength evidence.

All phases and reference settings
--------------------------------

Entropy is normalized by the logarithm of the legal action count. H common uses
0.05 for every policy. H declared uses each policy's fixed training divisors.
The complete JSON also retains entropy in nats and mean/median maximum-action
probability. Scores and full-precision statistics are preserved in the run root.

.. csv-table:: Probability and score summaries
   :header: "Phase", "Policy", "Choices", "H common", "H declared", "Raw top-two gap"

   Discard,initial_wp,2578,0.452271,0.452271,0.153437
   Discard,initial_point_fixed,2578,0.153910,0.153910,0.484228
   Discard,initial_point_matched,2578,0.153910,0.379238,0.484228
   Discard,seed1_wp_fixed,2578,0.373732,0.373732,0.192790
   Discard,seed1_point_fixed,2578,0.227718,0.227718,0.401501
   Discard,seed1_point_matched,2578,0.119722,0.295504,0.722551
   Discard,seed2_wp_fixed,2578,0.383323,0.383323,0.188803
   Discard,seed2_point_fixed,2578,0.215481,0.215481,0.409419
   Discard,seed2_point_matched,2578,0.116508,0.285362,0.722064
   Hand pick,initial_wp,122,0.710939,0.710939,0.085824
   Hand pick,initial_point_fixed,122,0.349895,0.349895,0.561340
   Hand pick,initial_point_matched,122,0.349895,0.640632,0.561340
   Hand pick,seed1_wp_fixed,122,0.646431,0.646431,0.108885
   Hand pick,seed1_point_fixed,122,0.362173,0.362173,0.520067
   Hand pick,seed1_point_matched,122,0.354437,0.605402,0.683066
   Hand pick,seed2_wp_fixed,122,0.635008,0.635008,0.116314
   Hand pick,seed2_point_fixed,122,0.428305,0.428305,0.506637
   Hand pick,seed2_point_matched,122,0.317546,0.541775,0.796513
   Draw pick,initial_wp,130,0.611028,0.611028,0.132800
   Draw pick,initial_point_fixed,130,0.303543,0.303543,0.565785
   Draw pick,initial_point_matched,130,0.303543,0.579449,0.565785
   Draw pick,seed1_wp_fixed,130,0.544914,0.544914,0.153856
   Draw pick,seed1_point_fixed,130,0.307235,0.307235,0.523691
   Draw pick,seed1_point_matched,130,0.287482,0.558090,0.607190
   Draw pick,seed2_wp_fixed,130,0.508912,0.508912,0.168523
   Draw pick,seed2_point_fixed,130,0.335721,0.335721,0.448368
   Draw pick,seed2_point_matched,130,0.242945,0.470123,0.818264
   Koi/stop,initial_wp,215,0.199169,0.199169,0.787767
   Koi/stop,initial_point_fixed,215,0.028503,0.028503,2.908384
   Koi/stop,initial_point_matched,215,0.028503,0.130684,2.908384
   Koi/stop,seed1_wp_fixed,215,0.182826,0.182826,0.620821
   Koi/stop,seed1_point_fixed,215,0.028373,0.028373,2.354113
   Koi/stop,seed1_point_matched,215,0.021430,0.129434,2.684670
   Koi/stop,seed2_wp_fixed,215,0.179292,0.179292,0.611138
   Koi/stop,seed2_point_fixed,215,0.026054,0.026054,2.458732
   Koi/stop,seed2_point_matched,215,0.021532,0.117826,2.728571

Greedy action changes
---------------------

Own initialization means original WP for WP endpoints and original Point for
Point endpoints. Stop/continue counts use that same reference.

.. csv-table:: Changes on identical public inputs
   :header: "Phase", "Policy", "Own initial", "Original WP", "Original Point", "Stop to continue", "Continue to stop"

   Discard,initial_wp,0,0,822,0,0
   Discard,initial_point_fixed,0,822,0,0,0
   Discard,initial_point_matched,0,822,0,0,0
   Discard,seed1_wp_fixed,317,317,774,0,0
   Discard,seed1_point_fixed,381,695,381,0,0
   Discard,seed1_point_matched,423,659,423,0,0
   Discard,seed2_wp_fixed,333,333,756,0,0
   Discard,seed2_point_fixed,374,678,374,0,0
   Discard,seed2_point_matched,371,674,371,0,0
   Hand pick,initial_wp,0,0,21,0,0
   Hand pick,initial_point_fixed,0,21,0,0,0
   Hand pick,initial_point_matched,0,21,0,0,0
   Hand pick,seed1_wp_fixed,13,13,18,0,0
   Hand pick,seed1_point_fixed,7,22,7,0,0
   Hand pick,seed1_point_matched,7,24,7,0,0
   Hand pick,seed2_wp_fixed,11,11,20,0,0
   Hand pick,seed2_point_fixed,6,21,6,0,0
   Hand pick,seed2_point_matched,8,19,8,0,0
   Draw pick,initial_wp,0,0,14,0,0
   Draw pick,initial_point_fixed,0,14,0,0,0
   Draw pick,initial_point_matched,0,14,0,0,0
   Draw pick,seed1_wp_fixed,11,11,17,0,0
   Draw pick,seed1_point_fixed,4,18,4,0,0
   Draw pick,seed1_point_matched,8,18,8,0,0
   Draw pick,seed2_wp_fixed,9,9,17,0,0
   Draw pick,seed2_point_fixed,7,19,7,0,0
   Draw pick,seed2_point_matched,4,16,4,0,0
   Koi/stop,initial_wp,0,0,38,0,0
   Koi/stop,initial_point_fixed,0,38,0,0,0
   Koi/stop,initial_point_matched,0,38,0,0,0
   Koi/stop,seed1_wp_fixed,22,22,34,9,13
   Koi/stop,seed1_point_fixed,13,27,13,4,9
   Koi/stop,seed1_point_matched,16,26,16,4,12
   Koi/stop,seed2_wp_fixed,20,20,28,14,6
   Koi/stop,seed2_point_fixed,19,25,19,4,15
   Koi/stop,seed2_point_matched,21,25,21,4,17

.. csv-table:: Disagreement between final settings
   :header: "Pair", "Discard", "Hand pick", "Draw pick", "Koi/stop"

   seed1_wp_fixed__seed1_point_fixed,637,21,21,23
   seed1_wp_fixed__seed1_point_matched,612,19,19,20
   seed1_point_fixed__seed1_point_matched,232,8,12,3
   seed2_wp_fixed__seed2_point_fixed,608,20,22,19
   seed2_wp_fixed__seed2_point_matched,610,16,17,23
   seed2_point_fixed__seed2_point_matched,200,6,7,6

Verification
------------

Three focused tests pass with no skips, covering all four divisors, scalar
probability agreement, own-initialization references, score rescaling, guarded
choices, negative-source exclusion, paired disagreements and exclusive writes.
Ruff passes. The completed producer exits0. Independent verification reconstructs
the population/masks, matches archived initial probability statistics to the
frozen calibration, checks stored float32 greedy choices, and recomputes all
nine settings across four phases with scalar arithmetic. It adds no neural
calls or simulator steps and does not independently replay the networks.

- Root:runs/point_initialization_mechanism.
- Protocol:cb4fdf6236f7e6cb74210ce663bac29a6b2536e5aff41fbc320ad21290a83b04 (194 inputs).
- Report:36e66a1aaea7f0179a505558ca2d254dcb827bbe31b4060cc76c03d910373f6a.
- Scores:1a3ec3324dee6bae91cf69d60162e3e1113d11ee8b892a17ac8c7b02ff0aff54.
- Audit:892486fa2f44efcdca5d93978cfce913fc8e8ff2bdbeccd2573fd3ad1aca257f (198 inputs).
- Test XML:0d423bb7580bf08489d92d2170b3bd4738787f51a828649c96033b4038538fab.

All diagnostic sources, test receipts and inputs are now frozen. Main evaluation
inputs and the canonical human-play model remain unchanged.
