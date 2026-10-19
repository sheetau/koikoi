Point initialization: final-policy mechanism diagnostic
======================================================

This descriptive diagnostic is specified after all six learners finish and
before inspecting main evaluation outcomes. It tests whether the initial
phase-entropy intervention remains visible in the final scores and whether it
changes greedy choices. It selects no model, temperature or training duration.

Use only the existing public-score corpus with nonnegative source IDs congruent
to 3 modulo 4: the same 25 descriptive validation groups excluded from fitting
the four Point divisors. Retain all rows for coverage and summarize only choices
with at least two legal actions after unchanged public endgame assistance.
These reused, correlated random-policy states are not fresh held-out games or
an estimate of training or deployment visitation. Do not add final snapshots.

Score the six fixed 520-update endpoints with the existing singleton CPU2,
float32 public-only actor path. Reuse archived original WP/Point CPU scores;
do not rerun those networks. Verify public feature hashes, masks and guards
against the archived oracle records before scoring. No hidden state enters an
actor, and no simulator transition, new match or training update is performed.

Report every phase for all six endpoints and three initial reference settings
(WP fixed, Point fixed, Point calibrated). For each, report common-0.05 and
declared-temperature entropy, normalized entropy and greedy probability;
the mean raw top-two score gap; greedy changes against original WP, original
Point and the corresponding initialization; and stop/continue changes against
that initialization. Report within-seed greedy disagreements for all three
pairs of final settings. Do not select phases or omit null changes.

The initial reference for a Point endpoint always uses original Point actions;
positive temperatures do not change greedy choices. Its calibrated probability
reference uses the frozen four divisors. Score adaptation can offset the
initial entropy change. The intervention changes both sampling probabilities
and loss derivatives: this diagnostic does not identify their separate causal
effects or establish an improvement in chosen actions.

Freeze the sources, tests, test receipt, corpus, calibration, all final models
and completed training evidence before running. Keep exclusive start/failure/
completion receipts and preserve partial outputs. Independently reconstruct
the selected population and recompute probability statistics with scalar
arithmetic, gaps and action counts from stored scores. This verifies recorded
scores and arithmetic, not an independent rerun of the neural networks.
No intervals or playing-strength claims are inferred from these states.

Main evaluation and its frozen inputs remain unchanged. Its completed,
audited full-match result is the separate source of strength evidence.
