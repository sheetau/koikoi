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

Replication outcome and final rule decision
------------------------------------------

The independent replication is complete: 131,072 matches, 1,011,437 round
endings and 15,110,348 decisions on 8,192 new shared duplicate streams. All
saved trajectories passed independent Python replay. A separate reduction
reconstructed win scores from saved terminal states, aligned exact seed IDs,
and reproduced the paired means and all bootstrap bounds.

.. csv-table:: Independent replication win scores
   :header: Parent,Opponent,Control percent,Both-yaku percent

   0,WP,51.525879,51.602173
   0,Point,50.344849,51.818848
   1,WP,51.580811,51.052856
   1,Point,50.585938,51.611328

.. csv-table:: Independent selected-minus-control mixture
   :header: Parent,Mean gain pp,Lower bound pp

   0,0.775146,0.439415
   1,0.248718,-0.090027

The exact sample mean differences are 127/16384 and 163/65536. The joint
positive-lower-bound criterion FAILED again: this time parent 1 has a lower
bound below zero. Both cohort means are positive for each parent, but neither
cohort establishes the prespecified two-parent result. The first failed test
remains separate; no pooling, sample extension or new selection was performed.
The nominal confidence family concerns the two parents within each cohort,
not all sequential research decisions in the project.

The transferred policies' mean gains are concentrated against the point-trained
opponent. Parent 1's mean against WP decreases in both confirmations. Thus the
screen identifies a potentially useful rule-dependent advantage, not a robust
strength-maximizing rule. No rule is promoted to the playable default. Neither
rule-specific policy training nor an exhaustive search of possible variants
was performed; the nine-profile transferred-policy question is now closed.

The replication protocol and complete bound results are in
``runs/rule_variants_replication_v1/protocol.json`` and
``runs/rule_variants_replication_v1/confirmation_summary.json``. Python replay
repeats the same 131,072 trajectories; it does not add independent matches.
