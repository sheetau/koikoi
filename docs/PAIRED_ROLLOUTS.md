# Paired rollout supervision for expert disagreements

This is a second improvement hypothesis prepared while the fixed arbitration
PPO study runs. Only target generation and correctness checks are implemented
so far. No trained-policy strength result is claimed.

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

## Prior work and prospective comparison

Rollout-based approximate policy iteration already treats policy improvement
as supervised learning; see [Dimitrakakis and Lagoudakis (2008)](https://arxiv.org/abs/0805.2027).
[Expert Iteration](https://arxiv.org/abs/1705.08439) similarly separates planning
from policy generalization. Common-random-number rollout coupling is also prior
work; [Yadav et al. (2026)](https://arxiv.org/abs/2605.04732) analyze conditions
and constructions for reducing relative-utility variance.

The proposed contribution is therefore the particular combination of public
expert disagreements, actual self-play hidden worlds, complete-match paired
targets and inexpensive deployment, supported by controlled Koi-Koi evidence.
The next protocol should compare direct outcome-difference regression with the
current arbitration PPO and continued full-policy PPO, count all branch work,
hold out entire source matches, and evaluate fresh duplicate deals. It should
also test whether explicit match context contributes beyond expert proposals.
Dataset size, fitting budget, endpoints and comparisons must be fixed before
strength results are inspected.
