# Public expert arbitration study

The previous adapted Transformers improve match wins against RL-WP, but their
RL-Point result does not replicate across both training seeds. Both lose expected
points to RL-Point. This motivates combining their different action preferences
at the decision level, using public match context to choose between them.

## Method and comparison

Two frozen experts propose legal actions: an adapted match-win Transformer and
the published point-oriented Transformer. A small controller receives the usual
560 public/own-card features and the two proposed action identities. It learns
terminal-match-reward PPO on the set of distinct proposals. When the experts
agree, there is only one action and the controller receives no policy gradient.
The public outcome-dominance guard applies before both proposals. Initial greedy
play selects the frozen parent expert proposal; training initially selects the parent
with probability0.9 at disagreements.

Policy switching is established prior work, including Comanici and Precup's
*Optimal Policy Switching Algorithms for Reinforcement Learning* (AAMAS2010).
The research question here is whether restricted exploration over differently
trained Koi-Koi experts improves match play more effectively than further
updating the full pretrained policy. A useful contribution requires measured
benefit, an analysis of when the expert choices differ, and comparisons that
explain the benefit. The architecture alone does not establish novelty.

Two separately adapted parent checkpoints each initialize three fresh runs:

| Arm | Trainable action policy | Actor inputs |
| --- | --- | --- |
| full | 128-wide two-hidden-layer controller | Public observation and proposals |
| phase | Linear controller | Four public decision-phase indicators |
| ppo | Continued pretrained Transformer | Existing published features |

All arms inherit the same corresponding parent public critic, use fresh Adam
state, and receive20 critic-only plus500 policy updates at128 environments by128
steps. This is8,519,680 additional simulator decisions per arm. Original expert
training remains an inherited dependency. Equal decisions do not imply equal
FLOPs. Full versus phase changes both information and capacity; it is not an
isolated test of score information.

Training seeds19010001 and19020001 and development deals21000000--21000999 are
fresh relative to the previous tracked protocols. Only final endpoints are
evaluated. Every endpoint plays1000 duplicate pairs against assisted RL-WP,
assisted RL-Point and its own parent. Both unchanged parents also play the two
published opponents on these same deals. Comparisons resample entire deal pairs.
All evaluation uses greedy CPU inference with two threads and no search. Three
evaluation workers may overlap one GPU training worker; timing is not a latency
benchmark. No previously consumed final-test deals are reused.

The machine-readable protocol, written before work begins, binds source, runner,
parents, configs, endpoints and selection criteria. A small feasibility run used
seed18990001 for five updates, completed81,920 decisions, and tested execution
only. It did not evaluate strength or select hyperparameters.

## Execution

```
.venv/Scripts/python.exe -m scripts.run_arbitration_study
```

Artifacts go to `runs/arbitration_study`, which must not already exist. The runner
retains all endpoints, raw paired results and training counters. A failure does
not authorize overwriting completed cases. Analysis and any later confirmation
use separately recorded scripts and protocols. The existing playable model and
dealer determination remain unchanged until a new model is supported by evidence.

## First-parent behavior diagnostics

On2,048 fixed cached random-play states, the full controller selects Point in
188 of523 expert disagreements. Observed Point-selection fractions are40.6%
when trailing,37.8% when tied and30.6% when leading. Holding the proposals and
all other inputs fixed while varying own points over10/20/30/40/50 and round
index over0/3/6 changes choices in only8 of523 disagreements. The observed
association does not establish that explicit score inputs caused the point
improvement. This is an input-sensitivity diagnostic, not a counterfactual game
or a deployment-distribution estimate. Raw public inputs and actions are in
`runs/arbitration_diagnostic_parent1`; plotting code produces PNG/SVG figures.

The phase-only controller favors the parent proposal in all four phases. Its
parent-minus-Point logit margins are0.783/1.157/1.171/1.649. Thus its greedy
controller does not exercise its learned stochastic Point probabilities. To
compare learned selection with a useful no-learning alternative, the separate
`run_static_expert_controls` study fixes two card/koi substitutions at both
parents and uses the same22,000,000--22,000,999 development deals as paired
regression. Its8 matchups run after that study, with three CPU workers. These
shared deals support paired comparisons; they are not independent confirmation.

### A measured inference-path near-tie

The parent expert's frozen inference path and the original deployed parent
wrapper use identical weights but different requires-grad settings. On the
completed parent1 Point comparison, one of1,000 pairs differs in recorded points
and decision count: seed21,000,447, candidate seat1, decision82, round index4,
koi phase. Both feature tensors, masks, rules/guard decisions and weights agree.
Under no-grad execution, setting parameters' requires_grad=False produces
stop/continue scores6.9657087326/6.9657096863, whereas requires_grad=True produces
6.9657087326/6.9657087326. Swapping only that flag in each in-memory model swaps
the result, repeatably. The frozen expert chooses continue; the deployed parent
chooses stop on its tie. The cause is an inference-path rounding difference,
not a different game rule or hidden input, and is not attributable solely to
temperature scaling.

The aggregate paired win score is unchanged and the phase-only point difference
is-1.601 versus the parent's-1.603. `scripts/diagnose_parent_inference_tie.py`
reproduces the first divergence and factorial flag check. Initial controller
selection of its frozen expert proposal is exact; universal action identity
with the older deployed wrapper is not claimed. Active studies retain their
frozen paths. Full and phase controllers use the same expert path, allowing
comparison of learned selection without changing this numerical convention.
