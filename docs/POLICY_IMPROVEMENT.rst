Policy-improvement positioning and next controls
===============================================

Evidence boundary
-----------------

The replicated positive result is additional self-play adaptation of the released
RL-WP actors under the pinned game rules. It is an improvement over specified
weights, with inherited representations. The new selector studies have not yet
established replicated win improvement over those adapted parents. A stronger
method claim needs that comparison plus a control identifying which intervention
caused the improvement. More named losses or more citations alone do not provide
such evidence.

The completed static-hybrid control adds a useful distinction: using Point only
for koi decisions improves expected point margin over both parents against both
public opponents, while all corresponding win-change intervals include zero.
The study therefore supports an objective-specific gain, not a match-win gain.
Exact outcomes and matched differences are in runs/static_expert_controls.

Relevant established methods
---------------------------

DAgger addresses the dependence of future observations on earlier learned actions
by iteratively collecting data under policies influenced by the learner. Its
analysis connects online learning to performance under the induced observation
distribution. Our current one-shot fixed-source fitting does not perform that
aggregation. Source: Ross, Gordon and Bagnell (2011),
https://proceedings.mlr.press/v15/ross11a.html.

AggreVaTe extends interactive imitation learning with expert cost-to-go estimates.
Its rollout data follows the current learner before exploration and expert
continuation afterward, and is aggregated across iterations. The paper explicitly
allows featurized partial observations; partial observability alone is not a
reason to dismiss its framework. Our fixed90/10 source, restricted proposal set
and one fitting stage differ from that iterative procedure. A squared-error loss
on rollout outcomes does not by itself inherit its no-regret conditions.
Source: Ross and Bagnell (2014), https://arxiv.org/html/1406.5979v1.

SPIBB preserves baseline behavior in uncertain regions and provides guarantees
for its specified batch-RL construction. A threshold on our neural scorer is
neither an uncertainty certificate nor an implementation of SPIBB. Retaining the
parent more often is a diagnostic intervention requiring its own match evaluation.
Source: Laroche, Trichelair and Tachet des Combes (2019),
https://proceedings.mlr.press/v97/laroche19a.html.

Existing domain work also separates point and win objectives. Guan et al.'s
Transformer/Monte-Carlo framework is the inherited baseline, not a contribution
of this project: https://ieeexplore.ieee.org/document/10032777/.
A targeted English/Japanese search on18 October2026 did not identify a new
comparable benchmark beyond the already recorded primary studies. That search
is not exhaustive and cannot prove priority or worldwide superiority.

Two distinct distribution questions
-----------------------------------

The active opponent-target study holds source roots and features identical while
changing the continuation opponent. It tests target construction conditional on
that common source. It does not establish that the learned selector visits the
same states as the source policy during repeated execution.

The source switches from the parent to Point with probability0.1 at disagreements.
Several fitted scorers instead switch on about half the held-out disagreements.
This measured action-frequency difference motivates a separate fixed diagnostic.
It is not, by itself, a measurement of occupancy-distribution distance or proof
that distribution shift caused the negative full-match results.

Fixed switch-budget diagnostic
------------------------------

For the existing larger-data feature/paired_win endpoint at each parent, compute
the90th percentile of predicted Point-minus-parent scores on training roots only,
using the higher order statistic. Set the threshold to the larger of this value
and zero. Switch only above the threshold; ties retain the parent. Thus at most
10% of training disagreements switch, although the held-out/deployment fraction
is free to differ. This is one fixed budget motivated by source behavior, not a
validation sweep, calibrated probability or confidence bound.

Retain both endpoints. On their existing held-out source matches, compare this
rule with unrestricted greedy selection, always-parent and always-Point choices.
Also report an independent randomized-switch comparator at the observed held-out
switch rate using its exact per-root expected outcome; that comparator is not a
simulated repeated-policy match result. Intervals resample source matches.
No new simulation or automatic deployment follows. The diagnostic is exploratory
on consumed development data. If it produces useful evidence, freeze the rule
before a fresh repeated-policy match comparison. The active opponent-target
study keeps its original four endpoints and complete protocol unchanged.

Implementation: scripts.diagnose_switch_budget. Analytic threshold/tie fixtures
and an actual collect-fit-diagnose round trip pass. The first application is
limited to the two existing larger-data feature/paired_win checkpoints; no other
loss, budget or threshold is selected from held-out outcomes.


Completed diagnostics and next intervention
------------------------------------------

Both the fixed switch-budget diagnostic and opponent-preserving-target study
are complete. Neither establishes replicated win improvement. Their exact
outcomes and separate source/branch counts are recorded in EXPERIMENTS.md.
The existing published features already contain score difference and round
identity; simply adding such context is not a new observation capability.

The next capability is actual pretrained actor learning against the two fixed
public opponents, rather than one-shot supervised proposal selection. Existing
published PPO was restricted to pure self-play; scripts/published_response.py
extends it experimentally with learner-only policy losses and a separate
public opponent-role critic. Seven CPU/CUDA/deployment tests and the actual-size
capability check pass. A fixed mixture is established opponent training, not an
original RL algorithm. Any study must compare unchanged parents and fresh
self-play continuation at both parent seeds, report actual learner counts under
equal simulator budgets, and use separate development deals. No bounded
strength-study protocol or final-test promotion has yet been launched.
