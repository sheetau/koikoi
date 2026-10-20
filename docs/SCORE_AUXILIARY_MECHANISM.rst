Fixed score-auxiliary mechanism diagnostics
===========================================

The diagnostic implements the mechanism section of SCORE_AUXILIARY_MAIN.rst.
It cannot be prepared until the complete six-arm training audit passes. It does
not depend on favorable strength results and does not change any endpoint or
hyperparameter. Main learning is still active; no final diagnostic result exists.

Retain every one of the100auxiliary training-phase records per arm,including
mean outcome NLL,within-outcome NLL and cloning KL. None retains null losses and
zero presentations. These are averages observed during training,not held-out
prediction accuracy or fixed-endpoint losses. All cycle rows,retained-row counts,
example presentations and collection work accompany the curves. Do not select
a favorable subset of cycles or interpret these losses as causal mediation.

The input population is the old25validation source groups,source ID modulo4==3,
in runs/published_batch_validation.npz. All3,258rows are retained for coverage;
3,045unforced rows enter action/entropy summaries and213forced rows are counted
separately. Source groups and row indices are fixed before new outcomes. The
corpus is reused public input,not deployment visitation or a new independent
evaluation sample.

Compare all six final actors and the cached original Point outputs. Report per
decision phase greedy disagreements against original Point and between all three
conditions within each seed,plus common-temperatureT0.05 entropy and normalized
entropy. Koi stop-to-continue and continue-to-stop changes are explicit. The
existing scalar-division/singleton CPU scoring helper is reused. Only public
features,phase and legal/public-endgame masks reach the actors. Feature digests,
original legal masks and endgame guards are checked against the corpus.

Expected execution is18,270singleton endpoint forward calls on CPU2; original
Point scores are cached. No new matches,simulator steps or learning updates.
Raw scores/actions,all training-phase rows and every input digest are saved.
The separate auditor rebuilds the population/masks,independently computes scalar
entropy and action counts,and checks all100log phases against their fixed update
schedule. It does not rerun neural inference,gradients or training losses.

Commands after the full training audit has completed::

    .venv/Scripts/python.exe -m scripts.score_auxiliary_mechanism prepare --output runs/score_auxiliary_mechanism
    .venv/Scripts/python.exe -m scripts.score_auxiliary_mechanism run --output runs/score_auxiliary_mechanism
    .venv/Scripts/python.exe -m scripts.audit_score_auxiliary_mechanism --input runs/score_auxiliary_mechanism

Every output is exclusive; preserve partial/failed runs. This diagnostic is
separate from the already-live training/evaluation completion queue. No main
diagnostic has been prepared or run yet.

Four focused tests pass,zero skips,1.50s. Analytic3:1choice probabilities reproduce
normalized entropy; both koi directions and forced exclusion are checked. Tests
also check all six within-seed comparisons,negative-source exclusion,and rejection
of missing/duplicate training cycles. Cached original Point statistics agree
with the previously audited diagnostic on all3,258rows. These tests add no
network calls,simulation or learning. End-to-end main execution remains pending.

Milestone digests:

* Producerddc23c6706f5a719093405d87b513c4b47b73eb11549826097890df78ec128d1.
* Auditor14a3f2176adc019ad7020a2d02943e339add42910bee8f9ecf5945a0cad323d3.
* Tests867c86550a7cce4fc63030bcbf688648951ec51be42559fcd441daa3216fcab5.
* XML55bfca318d13a9f224696d38d99d621975e37613c437d49117abdd19ed297840.

These measurements can describe how the intervention changed stored public-input
decisions. They do not independently establish stronger play,novelty or a causal
explanation for any eventual win-rate difference.
