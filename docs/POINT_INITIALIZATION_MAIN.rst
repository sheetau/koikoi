Fixed Point-initialization learning and development comparison
=============================================================

This specification follows the completed phase-temperature capability. It fixes
the main comparison before learning. It does not reopen the duration study or
change the canonical player, dealer, rules or reserved101M confirmation set.

Learning
--------

Two fresh seeds102110001/102210001 each receive three arms in this fixed order:
wp_fixed,point_fixed,point_matched. The first two use divisor0.05 in all phases.
The last uses the four frozen divisors from point_initialization_calibration.json
(SHA869ab1e45e7e8a5569ff20edaea55fdc5e2a2464d6d6a32a9c50df188bd701f3).
Initial weights are original WP for wp_fixed and original Point for both Point
arms. Every arm has the same initial critic, environment and RNG within a seed;
the two Point arms also have identical initial actor parameters.

Each arm uses20critic-only warmup and500policy updates,128environments,
128decision horizons,512minibatches,threeepochs,float32CUDA,actorLR3e-6,
valueLR1e-4,entropy0.02,lambda0.95,public560criticwidth256,andpaper8with public
endgame assistance. No intermediate strength evaluation,optimizer reset,training
extension or endpoint selection. The single endpoint is totalupdate520.
Each arm processes8,519,680decisions;all six together51,118,080decisions,
including1,966,080warmup decisions once. Report actual rounds and matches.

The phase-temperature code changes rollout probabilities and policy-loss
derivatives. It is not a pure exploration-only intervention. Equal calibration
entropy does not imply equal action distributions,gradient scale,visitation or
later entropy. Report these distinctions when interpreting mechanism results.

Evaluation
----------

Evaluate all eight policies:three endpoints per fresh seed and the two retained
parents A/B. Each plays assisted original WP and Point at both seats on4,096
paired deals per opponent. WP uses103110001..103114096;Point uses
103210001..103214096. There are16cells,131,072actual full matches and8,192shared
initial deal streams. Use strict singleton greedy CPU inference with two threads,
the existing deployment divisor0.05 and unchanged public endgame assistance.
The temperature wrapper is used in learning only. No search is introduced.

For each fresh seed, the primary comparisons are point_matched-minus-wp_fixed
andpoint_matched-minus-retained-parent,using equally weighted WP/Point win score
(win1,draw0.5,loss0). Combine the four comparisons into one simultaneous95%
family. A block combines opponent-specific distinct deal i,allpolicies andboth
seats. Use10,000whole-block multinomial bootstrap replicates,seed104110001,
fixed original standard errors,andthe higher95th percentile of maximum absolute
centered standardized mean deviation. Zero-SE coordinates cannot qualify.
Every one of the four unrounded lower bounds must exceed zero.

Report all120secondary coordinates with descriptive marginal95% intervals:
four policies andsix pairwise differences,at both seeds,for mixture/WP/Point,
for both win score andpoint margin. The six differences are point_matched minus
each of wp_fixed,point_fixed,parent;point_fixed minus wp_fixed andparent;and
wp_fixed minus parent. Opponent-specific superiority or a temperature mechanism
requires its own evidence; neither follows from the mixture qualification.

If the four-primary gate passes,the fixed candidate is seed102110001's
point_matched endpoint. Do not select the stronger seed or switch to point_fixed
after seeing outcomes. Promotion still requires separately frozen prospective
101Mconfirmation. Otherwise retain the current playable model. This is an
initialization/control study,not a new algorithm or world-strength certificate.

Preservation and checks
-----------------------

Prepare a new directory only after the completed production capability and
saved-state audit pass. Bind all calibration,source,weight,capability and test
inputs and this specification before main training. Preserve initial,20warmup,
periodic and520endpoint checkpoints and append-only update records. Record an
exclusive process-start receipt and a terminal completion or failure receipt;
do not periodically replace a file opened by monitors. Never restart merely
because a wait expires. No automatic continuation after a failed main run.

The seed inventory scans3,642existingJSONfiles,956withseedfields,andfinds no
recorded scalar seed collisions,including128initialenvironmentstreams per
training seed. It is not a proof about unrecorded work or implicit schedules.
Receipt:point_initialization_main_seed_scan.json,
dfa5688894853738b608b0c79c530027cc6a0e7dc0854d54b1b932244aa98e68.

Freeze and capability-test the evaluator/statistics code against this schedule
before any main evaluation. Independently verify completed training identities,
work and optimizer state,then replay all recorded evaluation actions through
native CPU and reproduce statistics before interpreting strength. Evaluation
code may be implemented during training;training sources and this design stay
fixed. Administrative repeats remain separate from independent evidence.
