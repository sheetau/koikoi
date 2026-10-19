Published initialization comparison
===================================

The existing pretrained-PPO runs start from RL-WP. RL-Point remains a distinct
initialization to investigate, but copying the current temperature would also
change exploration substantially. This is a diagnostic of a possible follow-up,
not a new learning algorithm or evidence that Point initialization improves wins.

Evidence
--------

The inventory finds 28 existing config.json records with kind=published, all
initial_variant=wp. It includes capabilities, inherited configurations and the
live duration study; it is not a count of independent experiments or a proof
that no unrecorded Point adaptation has ever been run.

The score comparison reuses the archived CPU outputs on the same 25 old source
groups (source ID modulo four equals three), comprising 3,258 public input rows.
Endgame-forced rows are excluded from choice statistics. No outcomes are used.
At temperature 0.05, mean entropy divided by log(number of legal actions) is:

==================== ===== ======== ======== ===================
Decision             Rows  RL-WP    RL-Point Different argmaxes
==================== ===== ======== ======== ===================
Discard              2,578 0.452271 0.153910 822
Discard capture        122 0.710939 0.349895 21
Draw capture           130 0.611028 0.303543 14
Stop / continue        215 0.199169 0.028503 38
==================== ===== ======== ======== ===================

Of the 38 differing stop/continue decisions, 25 are WP-stop/Point-continue and
13 the reverse. The Point mean maximum sampling probability at those 215
decisions is 0.990467, compared with WP's 0.931126. These are descriptive
statistics on reused correlated public inputs, not population estimates or
optimal-action labels. Different actions alone do not identify the better actor.
Argmaxes use archived raw scores, not a new deployment-float inference replay.

An independent scalar softmax/entropy computation reproduces every normalized
entropy mean to 1e-12. Source corpus SHA256:
e49004ce6795da3c23fbcb036c4d23712b6676a8d96cde1b43f6fca28d17d322.
Report runs/published_initialization_comparison.json binds 33 inputs:
ee8a8ffb512e8fb769bbaf09a12eec3973b32b2ac1ae7e48eb959a3012a2b29b.
Zero new simulator transitions, neural calls, matches or training updates.

Decision
--------

Finish the frozen longer-WP experiment and its evaluation first. A subsequent
Point-initialization comparison must distinguish initialization from the
sampling-distribution change caused by different score scales. A fixed-scale
arm can answer the literal replacement question; a separately specified
exploration-matched control can answer whether low exploration explains its
behavior. Neither temperature calibration nor pretrained transfer is novel by
itself, and the earlier negative koi-temperature result remains reported.
No temperature is fitted or adopted here and no extra training is launched.

A second candidate, averaging within-month equivalent-card relabelings, is not
prioritized: the existing actor already uses pointwise convolutions and
position-free self-attention. Exact-arithmetic token permutation equivariance
is already present; floating-point/tie changes would need separate treatment
and are not a fresh architectural contribution. The current GPU training,
completion queue, canonical player, dealer and final confirmation remain fixed.

Reproduction::

    python -m scripts.compare_published_initializations --output runs/published_initialization_comparison.json

The script refuses to overwrite an existing output. Preserve the completed
report rather than rerunning it under the same path.
