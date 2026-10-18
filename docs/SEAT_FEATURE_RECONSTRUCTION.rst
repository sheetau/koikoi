Actor-input reconstruction from the public-seat critic
======================================================

Question
--------

The seat-label diagnostic identified a public distinction used by the actor
but absent from the 561-input critic. The 562-input public-seat extension
restores that bit. Does its active-viewer input also retain enough information
to reconstruct the rest of the published actor input?

``scripts/check_seat_feature_reconstruction.py`` answers this on the retained
probes using an explicit decoder. ``reconstruct_actor`` accepts only one
562-element feature vector. It receives no simulator state, opponent hand,
stock permutation, RNG state, actor output or learned parameters.

Construction
------------

The first 528 entries contain 48 card records. Their eleven channels identify
current own cards, field cards, both public capture piles, unknown cards, the
pending card, the initial field and four public action timestamps. The next
32 entries contain phase, turn, relative dealer, round/match context, points,
koi counts and public continuation history. The last two entries identify
whether the viewer is acting and the viewer's absolute seat.

The decoder requires an active viewer under paper8 and constructs:

* Card groups, month indicators and combination counts from the card records.
* Score transforms from public points, capture piles and koi counts, using the
  independent Python scoring oracle.
* Round, turn and continuation indicators from the public scalar/history data.
* Absolute dealer from the viewer seat and relative-dealer indicator.
* The chronological field changes and capture alternatives from the initial
  field and the timestamped played/chosen cards.
* The original phase-dependent padding and stop/continue token columns.

Thus the check concerns the actor's actual input representation, including
its inherited duplicate rows and conventions, rather than a selected list of
game concepts. It does not replace the deployed actor or either feature builder.

Measured result
---------------

The decoder exactly reproduces all 300 x 50 floating-point actor features for
512 retained source roots and their 512 seat relabelings. The 1,024 active-viewer
probes comprise 862 discard, 52 hand-capture, 28 draw-capture and 82 koi decisions.
Native feature extraction leaves every saved state unchanged. No action is
sampled and no state is stepped.

For each of the 512 paired probes, the first 561 critic features are equal
while the actor's rows 50 and 51 differ. The new seat bit distinguishes those
rows. The old independent diagnostic already checked the native actor features
against the pinned upstream Python implementation on these same saved states;
this new decoder supplies the missing observation-to-observation construction.

Report: ``runs/seat_feature_reconstruction/audit.json``.
SHA256: ``a35a5a3d79626b5eca84edd3e9bdb9207dc394e9784e3f1322e164e8d48d42b6``.
Command::

   .venv/Scripts/python.exe -m scripts.check_seat_feature_reconstruction --output runs/seat_feature_reconstruction/audit.json

The command completed successfully in 2.31 seconds. Ruff passes. It adds zero
decisions, round endings, matches, training updates and policy inferences.
Environment construction and observation extraction are diagnostic work.

Interpretation
--------------

This result strengthens the implementation rationale for the planned seat
comparison: the additional bit closes the demonstrated actor-input ambiguity,
and the remaining published features can be recovered on every checked probe.
The 562-input MLP still has to learn a useful value function; the decoder is not
its learned representation. Recovering the current actor input does not imply
that the critic preserves the full history needed for an exact conditional
value, nor that a future bootstrap retains an earlier policy score. Consequently
this check supplies no unbiased-gradient or strength guarantee.

The fixed public/masked comparison, its gates and its training/evaluation
budgets remain unchanged. The manuscript has not yet incorporated this
diagnostic, and no interim strength result was inspected.
