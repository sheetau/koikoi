Public-stock frozen-parent correction protocol v2
================================================

Reason for a separate protocol
------------------------------

The v1 control trained from scratch failed its qualification gate (31.9336%
and 28.8086% against released, unassisted GPU RL-WP). Both control models
finished 33,554,432 training decisions. The remaining fourteen training jobs
were not launched; no v1 rule screen or confirmation was consumed. V1's
implementation, checkpoints and results are preserved. This failure concerns
policy initialization, not the desirability of the proposed rules.

Adaptation
----------

Use the two already retained strong parent checkpoints, with their recorded
SHA256 identities. Freeze each published actor. Add a 660-input, 256-unit
two-layer correction network to its temperature-scaled logits. Initialize
the correction to zero, copy the parent's critic, and initialize weights
from newly public inputs to zero. Thus ordinary control decisions preserve
the parent's actor at initialization. At the new stock-choice phase, use a
small immediate-capture logit prior and learn the correction. The original
actor does not see the new public stock; the correction does. Apply only
rule-correct public stop/continue dominance, including in rollout policies.

Train all eight rules separately for each parent: 128 updates x 512
environments x 64 decisions = 4,194,304 additional decisions per model,
67,108,864 for the matrix. Use sampling seeds 931 and 932 and initial deal
blocks 823,000,000 and 824,000,000. Keep v1's match reward, PPO, signed GAE,
minibatches and entropy setting, but use learning rate 0.0001 and an added
0.02 KL(base || corrected) penalty. Optimize only correction/critic weights.
Use the final checkpoint, without checkpoint selection. The pretrained
parents' historical work is reused and is not counted as new training.

Qualification and rule screen
-----------------------------

After training both controls, repeat v1's fixed quality gates on 256 new
duplicate streams beginning at 825,000,000: at least 65% against greedy and
45% against unassisted released GPU RL-WP for both parents. If this fails,
stop before training the other rules. The threshold is an engineering gate,
not a hypothesis test or a claim of strongest status.

If it passes, finish the fourteen other models and screen all eight rules
on 64 duplicate streams beginning at 826,000,000. Use each rule's own frozen
adapted policy as the no-search opponent, both seats, both parent identities,
and 0/2/8/32 particles per legal root action. Keep v1's 32-decision horizon,
one-standard-error paired advantage guard and selection criterion. Search
RNG seed is 941; evaluation batches contain 64 duplicate streams. Frozen
actor encoding uses chunks of at most 512 rows. All sampled rollout actors
receive only their own/public features. These are approximate policy
rollouts, not certified frontier improvements or equilibrium solutions.

Select the noncontrol rule maximizing the worse parent's 32-particle gain
over its no-search score; break ties lexicographically. The screen gate
requires that gain to be positive for both parents and larger than the
control's worse-parent gain. Report all curves, including failed trends.
Do not use amplified point margins to select rules.

Independent confirmation and limits
-----------------------------------

Reserve 512 duplicate streams beginning at 827,000,000 for the selected rule
and control, both parents and all four budgets. Launch only if the screen
gate passes. Keep v1's four inequalities and 20,000-replicate paired bootstrap
with one-sided 98.75% lower bounds. No pooling, extension or reselection.
Cross-opponent and stronger-adaptation replication remain separate questions.

Bind sources and checkpoints, save all actions and final states, and replay
every scored trajectory in Python. Distinguish actual decisions, rounds,
matches, rollout count, planning transitions and verification work. Model
adaptation, rule screening and confirmation use disjoint deal blocks. The
fact that computation helps a policy does not establish reduced intrinsic
luck, superiority over humans, or a globally optimal Koi-Koi rule.
