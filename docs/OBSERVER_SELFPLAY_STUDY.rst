Fixed-observer self-play: fixed development comparison
=====================================================

Question and conditions
-----------------------

Test whether fixed-observer primitive-step policy advantages improve the two
retained pretrained parents, and outperform a matched signed-return control.
The implementation/design is in OBSERVER_SELFPLAY.rst. Ordinary per-agent
critics are established prior work; this study tests the return-conditioning
intervention, not a first general actor-critic method or universal optimality.

Each parent has two new arms: observer and signed. Both use the same 561-input
row-wise critic, independent observer features, per-observer value targets,
mean value-loss normalization, optimizer, gradient clipping and minibatches.
Only the policy advantages differ. The original actor is unchanged initially;
the parent value first layer is extended with a zero role column. Both arms
fit the two observers' targets at every primitive decision. No incomplete-tail
filter or fixed public opponent is used during self-play training.

Include unchanged parents and archived ordinary-self-play continuations from
the completed response study as controls. They are published_ppo_v1 checkpoints,
not response-trained policies. This historical control retains the original
560-input active-observer critic and signed/native returns. Its comparison is
a full intervention; it is not the isolated return-conditioning contrast.

Fixed budgets and seeds
----------------------

Four new runs: two retained parents x two return modes. Each has 128 environments,
128 primitive steps per rollout,512-row minibatches,3 epochs,width256,
temperature0.05,actor LR3e-6,value LR1e-4,lambda0.95 and fresh Adam state.
Use20 critic-only warmup updates followed by500 policy updates:8,519,680
decisions per arm and34,078,720 in total. Each primitive decision supplies
two separate value rows,so total new value rows are68,157,440; they are not
extra games or simulation decisions. Match/round endings are counted as executed.

Training seeds19910001 and20010001 match the corresponding archived ordinary
self-play controls. All new arms begin with fresh environments and optimizers.
New-arm warmup model/optimizer/environment/counter/CPU+CUDA RNG states must match
within each parent after20 updates. Save and compare the actual warmup
checkpoints. Initial actor/environment/RNG equality to archived ordinary
self-play is checked and recorded; no later-trajectory equality is assumed.

Eight players x assisted RL-WP/RL-Point x5,000 duplicate pairs:160,000 evaluation
matches. All16 cells share the same5,000 fresh development deal streams
27,000,000--27,004,999 with both seat assignments. This is5,000 independent
deal units,not80,000. Greedy singleton CPU2 inference,public endgame assistance,
paper8 rules and no search match prior evaluation. Record decisions separately;
evaluation round counts are unavailable in the existing duel interface.

Archived two-run ordinary-self-play training contributes17,039,360 decisions
as a reused dependency,not new work. The capability's720,896 decisions and
reduced pipeline checks remain separate. No budget extension,intermediate
outcome selection or reranking. The reserved101M final test remains unused.

Analysis fixed before running
-----------------------------

Compute all16 absolute outcomes and all24 pairwise contrasts among each
parent's four conditions (observer,signed,ordinary selfplay,unchanged parent)
against both public opponents. Report draw-adjusted win score and final point
margin in their original units. Use5,000 whole-deal paired bootstrap resamples
with seed99 and percentile95% intervals. Shared seed99 gives joint resampling
across reported contrasts. This describes deal uncertainty for fixed trained
endpoints,not uncertainty over a population of training seeds.

Primary criterion: all eight win-change lower95 bounds must exceed zero for
observer minus unchanged parent AND observer minus matched signed control,
for both parents against both opponents. This conjunction requires all named
comparisons; do not select a subset after observing results. Ordinary-self-play
contrasts and point margins are secondary and remain fully visible. Passing
the primary gate is development evidence,not a final independent confirmation.
Failure is retained as failure. No automatic change to the human-play model.

Execution and verification
--------------------------

Bind source,parents,archived self-play checkpoints/audits,capability inputs and
the protocol before starting. One GPU training worker and at most three CPU2
evaluation workers. Evaluate a new endpoint only after its training audit.
Use a new output directory; failed runs are inspected rather than overwritten.

The launcher audits every update's counters,initial weights,full endpoint Adam
state and within-parent matched warmup. Evaluation retains every pair's seat
outcomes and scores. A later independent auditor must recompute all outcome
statistics and contrasts from those raw rows before manuscript inclusion.
No inference/trajectory replay is implied by a raw-result audit.

Reduced pipeline mode uses one parent,two new arms,two updates with one warmup,
four environments x16 steps,32-row minibatches and one epoch. It evaluates all
four players against both opponents on two duplicate pairs (32 matches),
using development-only seeds26060001/26061000. It validates scheduling and
analysis plumbing; its results never determine the full-study configuration.
