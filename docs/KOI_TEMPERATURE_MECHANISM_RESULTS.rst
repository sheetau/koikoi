Final-policy temperature diagnostic
==================================

All four learners completed the fixed 520-update budget. Their full-match
strength evaluation is still running. This diagnostic describes how their
scores and greedy choices differ on common public inputs.

The seven evaluated policies are original WP, retained A/B, and the two fixed
and two calibrated learners. The old validation population contains 3,258 rows
from 25 random-policy source matches, of which 3,045 are unforced. The second
population pools 512 saved final environment states, of which 481 are unforced.
Those snapshots follow the final training rollout and precede its optimizer
update; they are not a fresh rollout of the final policy. These populations
are descriptive and do not estimate complete training/deployment visitation.

Koi/stop on common old validation states
---------------------------------------

There are 215 unforced koi decisions. Entropy below is normalized by log(2).
``H common`` applies temperature 0.05 to every model. ``H declared`` uses each
model's declared training temperature:0.21989584747027907 for the calibrated
learners and 0.05 otherwise. Changes compare deployment greedy actions to
original WP on exactly the same inputs.

================  ========  ==========  ========  =======  ==============  ==============
Policy            H common  H declared  Raw gap   Changes  Stop->continue  Continue->stop
================  ========  ==========  ========  =======  ==============  ==============
Original WP       0.199169  0.199169    0.787767  0        0               0
Retained A        0.198772  0.198772    0.609497  20       7               13
Retained B        0.180346  0.180346    0.638053  18       9               9
Seed1 fixed       0.200316  0.200316    0.590015  20       7               13
Seed1 calibrated  0.085556  0.253215    1.462182  17       6               11
Seed2 fixed       0.165775  0.165775    0.618418  22       7               15
Seed2 calibrated  0.080724  0.253104    1.388870  24       10              14
================  ========  ==========  ========  =======  ==============  ==============

The calibrated learners have higher entropy at their declared temperatures
than their matched fixed controls. Their learned raw score gaps are also much
larger. An additional scalar calculation applies the already-fixed calibrated
temperature to original WP's stored singleton scores: normalized entropy is
0.446049 on these inputs, versus about0.253 at the final calibrated endpoints.
Thus the initial probability change is partly offset by subsequent score
adaptation. This calculation uses no new neural call or selected temperature.

The number of changed greedy koi actions does not consistently increase under
calibration: 17 versus 20 at seed 1, and 24 versus 22 at seed 2. The 2,578 unforced
discard states also change:fixed/calibrated counts 346/293 at seed 1 and 302/339
at seed2. An intervention in koi sampling can alter later learning trajectories
and therefore card-policy weights. These counts do not establish better choices.

Saved final environment states
------------------------------

The pooled snapshot population contains 39 unforced koi decisions. Both seeds
again have higher entropy under their declared calibrated temperatures, with
few changes to greedy decisions relative to original WP.

================  ========  ==========  ========  =======
Policy            H common  H declared  Raw gap   Changes
================  ========  ==========  ========  =======
Original WP       0.101381  0.101381    0.990986  0
Retained A        0.107902  0.107902    0.773039  2
Retained B        0.109601  0.109601    0.772597  2
Seed1 fixed       0.114088  0.114088    0.740050  2
Seed1 calibrated  0.053409  0.161931    1.650555  2
Seed2 fixed       0.079056  0.079056    0.778294  2
Seed2 calibrated  0.088652  0.201813    1.585158  3
================  ========  ==========  ========  =======

All these koi changes are stop-to-continue. The small snapshot sample is not a
population frequency estimate. Each source arm and all four decision phases
remain separately reported in the complete JSON; no confidence intervals or
causal mediation effect are inferred from these correlated states.

Verification and interpretation
-------------------------------

The producer performs 24,682 singleton CPU neural calls: seven policies times
3,526 common unforced inputs. There are zero simulator steps, new matches or
training updates. Saved old-corpus features, masks and public endgame guards
match their original oracle records. The completed scorer session 10737 exits 0.

Independent verification checks all 7 models across 6 populations and 4 phases, regenerates
public masks and guards, checks recorded-score float32 argmax, and recomputes scalar
entropy, gaps, choice changes and work. It adds zero neural calls or simulator
steps and does not independently rerun the score networks. All checks pass.

- Protocol: a25292a03b0d343870fd2e3b3c829bd2834aa41a3cdf6f467bb6877fd571c941, 98 inputs.
- Report: ad65e505430c175d78a51e32455f2be43824d18158be2ee1dd482f4ae22b04c4, 108 inputs.
- Scores: f548966b5560d312568987c84da63df63d760ab28b3bda9733285cd5bbd90b79.
- Audit: bcda39c6f5abef07a79332676a33df841cd344072aff9a27966bd50b985ae214, 113 inputs.

Files reside in runs/koi_temperature_mechanism. Preserve completed receipts.
This establishes a probability/score effect on the declared common populations.
It supplies no win-rate result and does not isolate visitation from gradient
rescaling. The frozen 49,152-match comparison must determine whether the
intervention improves playing strength. No model or temperature is selected
from this diagnostic; the canonical human-play player remains unchanged.
