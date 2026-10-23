Nine-rule transfer screen and first held-out confirmation
=======================================================

The first confirmation did not meet the prespecified joint criterion. The
selected both-yaku settlement has positive mean gains for both parent seeds,
but only one parent's simultaneous-family lower bound exceeds zero. This
completed cohort remains closed. Neither the playable rule nor its checkpoint
has been replaced.

The nine versioned profiles are defined in ``RULE_VARIANTS.rst``. Both players
use the same rules, frozen neural weights and rule-correct public endgame
assistance. Inference uses fixed GPU batch arithmetic and the original public
feature encoder. These transferred policies differ from the strict-CPU agents
in the earlier paper8 final test; the control is measured within this study.
No variant-specific training or universal-strength claim is involved.

Development
-----------

Each rule was evaluated on the same 256 duplicate-deal streams for both parents
and both opponents. Selection maximized the worse of the two parents' equal-
opponent mean win scores, with insertion-order ties fixed in advance.

.. csv-table:: Selection statistic (minimum parent-mixture win score)
   :header: Profile,Percent

   paper_control,50.8789
   opening_six,50.3906
   seven_double,50.5371
   koi_cover,51.1230
   own_koi_double,51.4648
   winner_takes_yaku,51.8066
   common_mix,50.1465
   single_koi_mix,50.4395
   combined_mix,50.4395

The selected ``winner_takes_yaku`` transfers both players' current yaku scores
to the player who ends the round. It is the explicitly defined experimental
variant, not a claim about every local rule called total-taking.

Held-out confirmation
---------------------

The selected rule and control were frozen before consuming 2,048 new duplicate
streams. Within each seed, both seats are averaged; the same streams couple
opponents and rules. The bootstrap resamples these seed clusters, preserving
the paired contrast and equal-opponent mixture. Each parent receives a one-
sided 97.5% bootstrap bound, giving a nominal 95% Bonferroni family. This is a
sampling approximation, distinct from the exact conditional search certificates.

.. csv-table:: Confirmatory win scores
   :header: Parent,Opponent,Control percent,Both-yaku percent

   0,WP,51.4160,51.4038
   0,Point,49.9390,50.7813
   1,WP,50.9644,50.6714
   1,Point,48.1445,50.0366

.. csv-table:: Selected-minus-control equal-opponent mixture
   :header: Parent,Mean gain pp,Lower bound pp

   0,0.415039,-0.274658
   1,0.799561,0.085449

Joint criterion (both lower bounds positive): FAILED. The positive means
motivate a separate precision replication, not a reclassification of this
test. Protocol and complete result bindings are under ``runs/rule_variants_v1``.
Every saved action trace was replayed through the independent Python engine;
all terminal states, decisions, round counts and match counts agreed.

Accounting
----------

* Development: 18,432 matches, 124,255 round endings and 1,829,605 decisions,
  on 256 shared unique streams. Of the round endings, 743 are automatic
  opening awards (128 occur at match initialization).
* Confirmation: 32,768 matches, 253,249 round endings and 3,782,656 decisions,
  on 2,048 shared unique streams. Neither selected arm has automatic opening
  awards. Python replay repeats existing trajectories and adds no fresh matches.

Independent precision replication
---------------------------------

``runs/rule_variants_replication_v1/protocol.json`` fixes one new comparison of
the same selected rule against the same control on 8,192 previously unused
duplicate streams. The purpose is roughly halving sampling uncertainty while
retaining both parent seeds and both opponents. The previous development
summary is an imported selection record, not a newly executed screen. The
failed first confirmation remains separate, is not pooled into the new test,
and cannot be extended or overwritten. No further rule-selection pass or
sample extension is authorized by this protocol.
