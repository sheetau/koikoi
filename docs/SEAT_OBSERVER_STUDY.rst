Public seat context: planned matched development comparison
==========================================================

Question and decision timing
---------------------------

The actor distinguishes absolute dealer seats, whereas the 561-input critic
merges the corresponding seat-relabeled observations. The saved diagnostic
finds different actor choices at some such collisions. Test whether restoring
this public policy context improves the fixed-observer learning procedure.
The finding and CPU capability are in OBSERVER_SEAT_INFORMATION.rst.

This design is written before inspecting the running observer study's outcome
summaries. The return mode is fixed to observer, for its consistent per-player
conditioning, not selected by whichever arm wins that study. The two starting
parents remain the original retained adaptations. No replacement with a better
endpoint from the ongoing comparison is allowed. The running study must finish
and receive its independent audit before launching this strength comparison.

Adding a public feature is established methodology. The question is the effect
of this identified actor/critic information mismatch, not a new general critic
architecture or a guarantee of improved play. The CPU capability establishes
that the added column is learned; it does not establish useful prediction or
policy improvement. A failed comparison remains a failed comparison.

Treatment, matched control and historical control
------------------------------------------------

For each retained parent, train two new continuations using
SeatObserverTrainer and policy_returns=observer:

* public: 562 inputs comprising the existing 561 independent observer features
  and that viewer's public seat bit.
* masked: the same 562-input architecture with its last input always zero.

The public seat and existing relative-dealer feature together recover absolute
dealer identity. Each value row sees only that observer's own cards and public
information. Rows are evaluated separately; there is no joint-hand critic.
The actor and its input remain unchanged. Both critics import the original
560-input parent with zero role/seat columns and fresh Adam state.

Architecture, initialization, return construction, value targets, loss
normalization, clipping, optimizer, minibatch order and simulation budget are
matched. Initial actor/critic tensors, environments and CPU/CUDA RNG must be
identical within each parent. During critic-only warmup, actor tensors,
environments and RNG must continue to match. Critic parameters, moments and
targets may differ because the additional information is the intervention.
Do not impose the previous observer/signed study's full-critic warmup equality
on these different-input conditions. Later policy trajectories may diverge.

Include each unchanged parent and its completed 561-input observer continuation
as controls. The latter is the fixed parentN_observer endpoint of
runs/observer_selfplay_study, not a selected winner. It is a historical
architecture control; public-minus-masked is the information contrast. Keep
its original measured training work separate from the new continuation work.

Fixed settings and work
-----------------------

Four new runs: two retained parents x public/masked. Each uses128 environments,
128 primitive steps,512-row minibatches,3 epochs,width256,temperature0.05,
actor LR3e-6,value LR1e-4,lambda0.95 and fresh Adam. Twenty critic-only updates
precede500 policy updates. Each arm executes8,519,680 primitive decisions and
17,039,360 independent value rows. Totals are34,078,720 new decisions and
68,157,440 value rows. Count executed round/match endings separately.

Use training seeds19910001 and20010001 for parents1 and2, matching the archived
observer controls. Check initial actor/environment/RNG identities against those
controls; the old561-column critic is padded by one zero column for comparison.
The two archived observer continuations contribute17,039,360 reused training
decisions, not new work. CPU/GPU capability and pipeline runs remain separate.

Eight players (two parents x public/masked/archived-observer/unchanged) face
assisted RL-WP and RL-Point on5,000 shared fresh duplicate deals,
28,000,000--28,004,999. This gives16 cells and160,000 matches on5,000 independent
deal units. Use paper8,greedy singleton CPU2 inference,public endgame assistance
and no search. Evaluation decisions are recorded; the existing duel interface
does not supply round-ending counts. Reserved101M tests remain unused.

Analysis and strength decision
------------------------------

Report all16 absolute outcomes and all24 within-parent pairwise contrasts
among the four conditions against each opponent. Report draw-adjusted win
score and final point margin separately, using their original units. Joint
whole-deal bootstrap uses5,000 resamples,seed99,percentile95% intervals.
These intervals describe deal uncertainty for fixed endpoints,not variation
over a population of training seeds.

The primary strength criterion requires all eight paired win-change lower
95% bounds to exceed zero: public minus unchanged parent AND public minus
matched masked control,for both parents against both opponents. No subset may
replace this conjunction after results are known. Archived-observer contrasts
and all point-margin results are secondary. A primary pass is development
evidence requiring an independently specified confirmation; it does not
automatically replace the playable model. No budget extension,intermediate
checkpoint selection,reranking or automatic deployment.

Prerequisites and execution
---------------------------

Before launch:

* Complete and independently audit the current observer study; retain all its
  outcomes regardless of whether its primary criterion passes.
* Validate the new feature/model/collector/restart path on CUDA after the
  current study's four GPU training workers have completed. Run one GPU worker
  at a time. A small CPU check is not a substitute for this capability.
* Verify public/masked initial identities and actor/environment/RNG warmup
  equality at20 production-size updates,then execute policy updates. Audit
  actual optimizer states and count any restart replay as additional work.
* Implement and run a reduced real training/evaluation pipeline,including the
  new checkpoint adapter,raw duplicate rows,paired analysis and tamper checks.
* Bind source,parents,archived observer checkpoints/audits,capability evidence
  and this protocol in a new directory before starting full training.

The full scheduler permits one GPU trainer and three CPU2 evaluators. Evaluate
an endpoint only after its training audit. Preserve failed directories. After
all cells terminate,an independent auditor must recompute raw paired outcomes,
contrasts and the fixed primary criterion before manuscript inclusion. None of
these checks substitutes for the measured strength result.
