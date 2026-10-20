Score-auxiliary evaluation and evidence
======================================

The main design remains SCORE_AUXILIARY_MAIN.rst. This document describes its
evaluation implementation and completed capability check, not playing results.
The six main learners are still running. Main preparation requires complete
training and an independent training audit that has not yet been implemented.

Implementation
--------------

scripts/evaluate_score_auxiliary.py uses the existing NeuralAgent for every
standard published_ppo_v1 endpoint, greedy CPU2 inference and public endgame
assistance. Strict original assisted WP/Point opponents are unchanged. The
existing reference play_match/cases helpers preserve deal pairing and seat
assignment. Every match stores all actions, initial-state digest, final state,
candidate-relative point margin and separate work counts. Two owned workers
execute cells; failure stops the remaining owned worker and preserves receipts.
Exclusive output creation prevents silent restarts/overwrites.

Main preparation requires the frozen training-protocol digest, all six complete
arm receipts, full-study completion, endpoint manifests and a passed
score_auxiliary_training_audit_v1 artifact. That artifact must bind the complete
receipt, all six endpoints and its implementation source. The audit producer is
the next required component; there is no bypass that enables a main evaluation
while training/auditing is incomplete. The completed capability audit and its
dependencies must also remain unchanged.

scripts/score_auxiliary_analysis.py computes all120absolute/paired win/point
coordinates. Draws receive half win credit; duplicate seats are averaged within
deals, then the two opponents receive equal weight. One family contains all six
score-minus-none/win/parent mixture-win comparisons at both learner seeds.
Qualification requires six strictly positive lower bounds. A degenerate empirical
coordinate receives no sign claim. Secondary point or opponent-specific gains
cannot replace this criterion.

scripts/audit_score_auxiliary_statistics.py independently reconstructs seat and
opponent sums with scalar loops, uses different bootstrap chunk sizes, and
computes sorted quantiles separately. It verifies every mean/marginal interval,
the simultaneous family, flags and work/deal dimensions. It does not import the
producer's analysis function. The established max-standardized bootstrap
primitive is reused by the producer; this is not a new statistical method.

scripts/audit_score_auxiliary_evaluation.py replays every recorded reference
trajectory through native CPU transitions, checks initial deal sharing, legal
actions, final states, margins, row counts and receipts, then invokes the
separate scalar statistics verification. It checks the declared cell schedule
and full completion set. This replay does not rerun either policy network and
does not add independent matches. All outputs and input digests are retained.

Checks completed before main evaluation
--------------------------------------

Ten focused tests pass,zero skips,3.61s. They include an actual saved score-
auxiliary actor against strict Point, native replay and altered-margin/work
rejection; incomplete-training gating; owned-worker cleanup/restart rejection;
all120numeric coordinates; six-comparison success/failure including a missing
outcome-only gain; draws,point-only improvements and invalid input dimensions.
Test XML: runs/score_auxiliary_evaluation_tests.xml.

The actual two-match test uses one110030001deal stream in both seats:
217decisions/13round endings/2complete matches. Native replay verifies those
actions and adds no neural action selection or independent deals. Synthetic
bootstrap tests add no simulator or learning work. The initial standalone seven
analysis tests also pass(0.25s),with no simulated matches.

Capability commands::

    .venv/Scripts/python.exe -m scripts.evaluate_score_auxiliary capability --output runs/score_auxiliary_evaluation_capability
    .venv/Scripts/python.exe -m scripts.evaluate_score_auxiliary run --protocol runs/score_auxiliary_evaluation_capability/evaluation_protocol.json
    .venv/Scripts/python.exe -m scripts.audit_score_auxiliary_evaluation --input runs/score_auxiliary_evaluation_capability

All16cells complete and replay successfully:64actual matches,7,089decisions,
482round endings. The four distinct deal streams are110010001/110010002 against
WP and110020001/110020002 against Point,shared across policies/groups/seats.
The three small capability-trained endpoints are reused for both learner groups;
this does not create independent training replicates. Both retained parents
are also exercised. No main evaluation or101Mconfirmation deal is used.

The capability evaluator session2640 exits0; its native/statistical auditor
also exits0. All123protocol inputs and158final audit bindings verify. All107
active-learning input hashes remain unchanged. Bootstrap10,000replicates
reproduce all120marginal coordinates and six simultaneous primary intervals.
Capability qualification is false and cannot authorize promotion. Timing
overlaps ongoing GPU training and is not an isolated performance benchmark.

Protocol911f74110a9850ab421f402069a90f5cb29f5ee9bef9e11b00ae65a4c451a1ed.
Completiond51a5ca053889c170d39ff446f13785779b8ceaa18936c22f399ca15617c4114.
Auditcdcd8942c20b8bb6e6a49c527d06289899b0d17e3f15828d78be891d975e99a4.
Summary4b0e40c458ee10c0f2e03155119a1232cda93d690c43d5bdf60d7c178e960e69.

Next implement independent collection/checkpoint/optimizer accounting for the
main learner,then the completion queue and fixed mechanism diagnostic. Preserve
these completed capability outputs and their bound sources. The canonical
player,dealer and verified manuscript are unchanged; stronger-play novelty is
still unestablished.
