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
play exactly follows the adapted parent; training initially selects the parent
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
