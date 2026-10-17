# Paired rollout supervision for expert disagreements

This is a second improvement hypothesis prepared while the fixed arbitration
PPO study runs. Target generation, direct advantage regression, embedded deployment and a
queued comparative runner are implemented. No trained-policy strength result
is claimed before the new study finishes.

## Motivation

Terminal-reward PPO assigns credit through a learned critic over a long match.
At an expert disagreement, the simulator can instead evaluate both first
actions in the same sampled world. This provides a direct target for the
relative usefulness of the two proposals. Search work is paid during training;
deployment would still run the two frozen experts and a small public controller.

The world comes from an actual behavior-policy trajectory. This differs from
the earlier inference-time searches, which resampled hidden cards from an
approximate distribution. The training generator can copy the simulator state
to run counterfactual branches, but neither continuation policy receives it as
a model input. Both seats use their own public/own-card observations.

## Target

At a root observation `o`, the adapted win expert proposes `a_W` and RL-Point
proposes `a_P`. A training trajectory samples a hidden simulator world `s`.
Two branches force these different first actions, then use the same frozen
public continuation policy for both seats until match termination. Their
root-player win/draw/loss returns are `G_W` and `G_P`, in `{-1,0,1}`. The target
is `delta = G_P - G_W`. Both branches share the current hidden cards and all
future round-deal streams. Identical first actions produce identical returns.

For roots sampled from a fixed behavior distribution, the conditional mean
`E[delta | o]` is the difference of the two continuation values averaged over
that distribution's hidden worlds. Regression can estimate this mean using
public features only. This statement describes the target; finite samples,
function approximation and deployment distribution shifts can still produce
wrong decisions. Simultaneously changing both players' policies adds another
distribution shift. A full-match evaluation is required.

The variance identity is
`Var(delta) = Var(G_P) + Var(G_W) - 2 Cov(G_P,G_W)`.
Shared randomness reduces difference variance only when the induced covariance
is favorable. It is not an automatic guarantee for these game trajectories.

## Implementation and accounting

`scripts/paired_rollout_targets.py` contains the prototype:

- Root collection samples expert disagreements under a90% parent/10% Point
  behavior policy, before looking at any branch outcomes.
- `paired_outcomes` copies each root twice, freezes completed rows and uses
  public observations for every subsequent action. It returns both match
  outcomes, both point differences and explicit rollout-work counters.
- Source decisions/matches/rounds and branch decisions/completions/rounds are
  separate. A branch endpoint is not a newly sampled independent match.
- The intended dataset exports only the562 public/proposal features and target
  values. Source seed/episode identifiers are grouping metadata only. Every
  decision from the same source match belongs to the same train/validation split.

CPU and CUDA branches agree with independent Python-reference trajectories,
including root-player signs, completed-branch counts and decision counts.
Input roots remain unchanged. CPU feasibility:64 roots,128 completed branches,
15,018 branch decisions,1,003 round endings,14.29 seconds for rollouts. Twelve
roots had nonzero outcome differences. This tiny execution check neither
estimates strength nor establishes a useful prediction signal.

## Fixed comparative study

`scripts/run_paired_arbitration_study.py` queues the new study after the current
PPO work. GPU collection starts only when all six earlier training jobs finish;
CPU evaluations start only when the earlier study is complete. It writes a
source-bound protocol before waiting. Do not change its bound scripts while it
is queued or active.

Each parent supplies65,536 disagreement roots, in batches of256, under collection
seeds19210001 and19220001. The public controller fits the return difference by
MSE using Adam with learning rate3e-4, weight decay1e-4, minibatches of1,024,
and exactly20 epochs. Validation holds out whole source matches using a fixed
hash partition. The final epoch is always selected. Full and no_match controllers
share the same dataset, architecture, initialization seed and fitting budget.
The no_match arm removes only explicit round/score features from the controller;
its expert proposals still depend on match context.

Fresh deals22,000,000--22,000,999 compare both regression arms, the earlier PPO
gate, continued Transformer PPO and the unchanged parent against RL-WP and
RL-Point. Regression endpoints also play their own parent directly. This gives
24 matchups of1,000 duplicate pairs each. Results remain development evidence;
any later confirmation needs fresh deals and a new frozen protocol.

The distinct checkpoint format identifies regression explicitly and embeds both
experts. A tested tiny CPU pipeline covers collection, all three supported
controller contexts, independent-match validation splitting, checkpoint loading,
hidden-state invariance and a four-match evaluator/statistics round trip.
These checks establish execution correctness, not playing strength.

## Prior work and research question

Rollout-based approximate policy iteration already treats policy improvement
as supervised learning; see [Dimitrakakis and Lagoudakis (2008)](https://arxiv.org/abs/0805.2027).
[Expert Iteration](https://arxiv.org/abs/1705.08439) similarly separates planning
from policy generalization. Common-random-number rollout coupling is also prior
work; [Yadav et al. (2026)](https://arxiv.org/abs/2605.04732) analyze conditions
and constructions for reducing relative-utility variance.

The proposed contribution is therefore the particular combination of public
expert disagreements, actual self-play hidden worlds, complete-match paired
targets and inexpensive deployment, supported by controlled Koi-Koi evidence.
The fixed protocol above compares direct outcome-difference regression with
arbitration PPO and continued full-policy PPO, counts all branch work, holds out
entire source matches, and tests explicit match context. Dataset size, fitting
budget and endpoints are fixed before any new-controller strength results are
inspected.


## Held-out decision diagnostics

After both fixed fits finish, run, for example:

```powershell
.venv/Scripts/python.exe -m scripts.diagnose_paired_targets --dataset runs/paired_arbitration_study/parent1/targets.pt --checkpoints runs/paired_arbitration_study/parent1_full/final.pt runs/paired_arbitration_study/parent1_no_match/final.pt --output runs/paired_arbitration_study/parent1/held_out_diagnostic.json
```

The diagnostic uses only source matches excluded from fitting. It reports MSE
against zero and training-mean predictors, fixed calibration bins, actual native
action choices (including ties), and the observed selected-branch win-score gain
relative to each always-expert choice. It resamples whole source matches5,000
times, retaining root weighting instead of counting correlated roots as independent.
It also reports the four action phases separately. A single source match has no
estimated confidence interval. This measures changing one action before frozen
continuation, not repeatedly deploying the new selector throughout a match.
It selects no epoch, threshold or model; full-match comparisons remain decisive.

Four analytical fixtures check signs, win-score scaling, unequal match root counts,
source grouping and native action tie breaking. The actual tiny collection/fit
pipeline also runs the diagnostic and checks held-out counts and independently
recomputed prediction loss. Combined diagnostic/pipeline tests:5 passed.

## Koi-Koi-specific antecedents

[Sato and Ikeda (2017)](https://dspace.jaist.ac.jp/dspace/handle/10119/16089)
compare policy gradients and Neural Fitted Q Iteration with268 input features
against a rule-based opponent under simplified scoring.
[Nagamine and Suzuki (2021)](https://www.ipsj.or.jp/event/taikai/83/ipsj_web2021/data/pdf/6P-07.html)
construct fast-finishing and blocking play through Monte Carlo reward changes.
Their two-page paper disables koi-koi continuation and proposes switching
scoring/fast-finishing strategies across a match as future work. Thus, the broad
motivation for switching objectives is prior work. The present question concerns
learning a selector from complete-match outcomes over frozen neural proposals,
with both card and continuation decisions and explicit no-learning controls.
The manuscript now cites these direct domain antecedents in addition to general
policy switching, rollout policy iteration and common-random-number work.


## Score-distribution method (no full-match strength study yet)

The completed PPO selectors did not establish replicated win gains over their
parents. `scripts/train_score_distribution.py` prepares a separate follow-up
using the existing paired dataset's final point differences. A shared action
scorer receives560 public features and one50-way proposal encoding. It evaluates
each proposed action with the same610->128->128 encoder. This makes equal
proposals equal-valued and removes dependence on expert ordering.

At a root let L be own points minus opponent points, and D_a the final margin
after proposal a and frozen continuation. The training target is X_a=D_a-L.
Active paper8 matches have positive integral scores summing to60, so L is even
and lies in[-58,58]. Every X_a is even. Map X_a to65 classes at -64,-62,...,64,
with the two endpoint classes including their respective tails. For every valid
root, sign(L+clip(X_a,-64,64))=sign(D_a). Thus this tail compression preserves
all three terminal outcomes without requiring a bound on individual round scores.
The categorical head learns these classes by cross entropy and evaluates an
action as sum_x p(x|o,a)*sign(L+x), equivalent to twice win score minus one.

Two controls use the same shared encoder: a scalar head learns the unbounded
increment/64 by MSE and chooses expected points; a scalar win head uses binary
cross entropy with half credit for draws. These controls distinguish distributional
supervision from simply sharing an action-conditioned encoder. Output heads and
parameter counts differ; they are not equal-capacity architectures. Common seeds
produce identical initial encoder hashes and a separate fixed shuffle generator
keeps minibatch order identical despite different output-head sizes.

Fitting reuses source-match train/validation groups, fixed20 epochs, Adam3e-4,
weight decay1e-4 and1,024 roots/minibatch. Checkpoints embed frozen experts,
dataset/source identities, initial encoder hash, head size and reused-work counts.
New simulation decisions are zero. Actors and heads use only own/public inputs.
The categorical and scalar losses fit marginal branch labels; they do not inherit
a variance-reduction theorem merely because the dataset uses paired worlds.

[Bellemare, Dabney and Munos (2017)](https://proceedings.mlr.press/v70/bellemare17a.html)
establishes the broader distributional-value perspective. This prototype fits
complete Monte Carlo returns under a frozen continuation, not their distributional
Bellman algorithm. The research question is whether score-distribution supervision
and explicit win-threshold decoding improve public neural expert selection.
At initial implementation no full-scale fit or matchup had been launched. The paired regression
and static controls finish first; their evidence will determine whether to run
this follow-up. It is not included as a measured contribution in the manuscript.


## Stored public inputs and pretrained feature reuse

The stored560-dimensional observation is player-relative. The pinned300x50
published encoder additionally uses the absolute dealer label in rows50/51.
Consequently, the original features or Q scores cannot generally be recovered
exactly from the saved public observation and two proposal IDs alone.
`scripts.diagnose_public_feature_information` demonstrates this by exchanging
player labels, initial-hand deck segments, ownership, scores and koi counts in
2,048 cached states. Relative observations, legal masks and public guards stay
identical; only published rows50/51 change. Both parent expert models and Point
change some actions. Even when both proposals stay identical, their scores can
change. This is public information omitted from the compact representation,
not privileged information or a simulator-rule error.

Reports: `runs/public_feature_information/parent1.json` and `parent2.json`.
The diagnostic has no training or new match work. Exact future caching of
pretrained scores/embeddings must retain the required public seat metadata;
a seat-averaged representation would instead be a new method to evaluate.
The current score-distribution prototype operates directly on the saved public
inputs and requires no such reconstruction. Existing frozen experiments and
deployment remain unchanged.


Exact public features can instead be recovered by replaying the source stream,
using its recorded seed, device, embedded experts and dedicated behavior RNG.
`scripts.cache_paired_expert_features` verifies every root's public observation,
proposals and source-match ID before exporting features. It adds source-replay
work and records it separately; branch outcomes are reused. It exports the public
absolute dealer bit, two experts'256-wide encoder vectors for both proposals and
their raw scores. It never exports packed states or private card labels.

Initial small CPU/CUDA replay tests used32 roots each, exact source alignment,
checkpoint reload, overwrite/misalignment rejection, and original-head equality
across all four action phases. Four tests pass with no skips. No full dataset
cache, feature-based fit or strength comparison has been launched. The helper
provides a precise alternative to guessing missing public seat metadata.


## Full feature caches and fixed fitting screen

Both full65,536-root feature replays now completed. Each root's public inputs,
proposals and source group exactly matched the original stream. GPU replay took
33.7605/34.7620 seconds, adding311,552/311,680 source decisions and zero branch
simulation decisions. `runs/paired_public_features/audit.json` checks retained
identities, sizes, finite values and reconstruction of every cached raw score
from its256-wide expert vector. The CPU check differs from stored GPU head scores
by at most3.43e-5/2.67e-5, within relative1e-5 plus absolute1e-5 tolerance. This is
not a second replay of all source trajectories.

The feature-based shared action scorer takes512 latent features, two expert
scores scaled by1/16, and32 public context features. Categorical, mean-margin and
direct-win heads share the same encoder initialization and minibatch order within
a representation. The one-pass inference adapter matches the original expert
heads and cached feature path across all four phases on CPU/CUDA; hidden-world
perturbations preserve its inputs. Proposal permutation and duplicate consistency,
actual fits/reloads, cache identity rejection and held-out diagnostics are tested.
Raw and feature models differ in input dimension/parameter count; comparisons do
not assert equal capacity.

Sequence revision: the first paired-regression full-match endpoint and both
held-out prediction diagnostics are now available, without a consistent win gain.
While the remaining paired/static CPU experiments continue unchanged, prepare
fixed final-epoch supervised fits on the idle GPU. `scripts.run_score_fit_screen`
freezes12 cases:2 parents x2 representations x3 losses,20 epochs, seeds19410001 /
19420001. Every endpoint is retained; source-match validation is development data
and may inform a future method choice. There is no early stopping or automatic
selection. New score-selector match evaluation still waits for the current
paired/static comparisons to finish. This revises the earlier plan to delay all
full-scale fitting, not any bound current-study protocol or confirmation set.
