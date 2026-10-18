Prospective fixed-prefix final-round training study
==================================================

Question and fixed policies
---------------------------

Does complete-tail adaptation improve actual match wins over each retained
parent when the deployed prefix remains fixed? Use the two canonical
update_00000520.pt parents identified in scripts/run_arbitration_study.py.
The final-round tail actor starts from its corresponding parent's weights.
No actor input, game rule, dealer choice or playable default changes.
FINAL_ROUND_TRAINING.rst defines the method and its prior-art boundary.

Source banks
------------

For each parent, collect 1,024 natural prefixes: opponents WP/Point, learner
seats 0/1 and 256 replicates per cell. Case order is opponent, seat, replicate.
Seed = 88110001 + 10000*opponent_index + replicate. Both parents and both seats
share the same 512 distinct source deal streams. Stop at the untouched natural
round-eight start or earlier termination. Retain every source record, including
bankruptcies. Do not manufacture score/round contexts or select roots by outcome.
Check each transition against the independent reference implementation.

Use strict CPU greedy parent and assisted published opponent for collection.
Only surviving roots enter the bank. Uniformly sample all those roots with
replacement; do not balance the surviving cells. Record source attempts,
survivors, early-terminal full matches and prefix work separately. No held-out
evaluation source enters training.

Training
--------

Each parent receives 520 updates: 20 critic-only warmup and 500 actor/critic
updates. Every update samples 128 roots and completes all tails: 66,560 tails
per parent, 133,120 in total. The actual simulator/learner decision counts vary
with tail lengths and must be reported. These are not new complete matches.

Use TailConfig: CUDA float32, horizon safety bound 64, three epochs,
minibatch 512, actor learning rate 3e-6, value learning rate 1e-4, score
temperature 0.05 and the inherited PublishedConfig loss settings. Seed is
88210001 + 10000*(parent_index-1); the root sampler uses seed+1. Fresh optimizer
and RNG are intentional; initialize actor/value weights from the parent.
Frozen WP/Point opponents use fixed64 inference and the original source label.
No truncated bootstrap, hidden resampling, later-round reset or opponent critic.

Keep initial, every100-update, update520 and final checkpoints. Only update520
is evaluated; do not select a checkpoint using training or evaluation outcomes.
Record per-update counters/metrics and verify prefix/opponent parameter identity,
optimizer steps and input hashes at completion. A failed execution is preserved;
it does not authorize new seeds, hyperparameters or extra training updates.

Actual evaluation
-----------------

For each parent and each of WP/Point, evaluate both unchanged parent and the
fixed-prefix adapted candidate on 1,024 duplicate-deal pairs. Seed = 89110001
+ 10000*opponent_index + replicate. Both parents share evaluation deals and each
pair exchanges learner seats while retaining the same seat-assigned deal.
There are 2,048 distinct evaluation streams and 16,384 complete policy matches.
All policies use strict CPU greedy inference and the same public assistance.
No cached source roots or conditional branches substitute for actual matches.

Save every match's actions, final points and separate decision/round/match work.
Check every transition against the Python reference. Earlier bankruptcies and
matches without an effective final-round change remain in every comparison.
Verify exact prefix-action agreement between parent and adapted candidates.

For each of the four parent/opponent cells, report draw-adjusted match-win and
final point-margin changes, candidate minus parent. Resample 1,024 whole pairs
10,000 times, using shared bootstrap seed89310001. The primary replicated-gain
criterion requires all four win-change lower95% bounds to be positive. Report
all cells regardless of sign. Point margins are secondary. No threshold change,
checkpoint selection, deal-pool extension or automatic model promotion follows
this study. Positive development results require independent confirmation.

Execution and audit
-------------------

Run one GPU trainer at a time. Complete training before actual evaluation;
evaluation may use at most three CPU workers with two Torch threads each.
Timing is experiment throughput, not an isolated deployment latency measurement.
Use fresh output paths; bind sources, core provenance, capability evidence,
parent checkpoints and upstream model files before collecting source deals.

An independent auditor replays recorded source/evaluation actions through the
reference engine, checks natural roots, prefixes, endpoints, all work totals,
checkpoint identities and raw-pair outcome arithmetic. It does not independently
rerun every neural inference or stochastic optimizer update. CPU/CUDA trajectory
and exact-restart tests supply separate execution evidence. Keep all raw data,
null/adverse results and any failed attempts. Reserved101M streams remain unused.
