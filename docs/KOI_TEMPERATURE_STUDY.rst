Koi-temperature training comparison
==================================

The original released WP scores yield much more deterministic sampling at
koi/stop decisions than at discard decisions on the old random-policy corpus.
The completed score-only calibration sets koi temperature to
0.21989584747027907; the common baseline temperature is 0.05. This experiment
tests that fixed intervention without changing learning rate or actor inputs.
It is a sampling-temperature intervention, not an isolated causal test of
visitation: temperature also changes entropy and log-probability gradients.

Train four agents from the original published WP initialization: two seeds
(96110001 and 96210001), each with all-0.05 and calibrated-koi conditions. Both
capture phases and discard retain 0.05. Use the unchanged actor architecture,
public critic, signed GAE/PPO returns, endgame assistance, 128 environments,
128-step horizon, three epochs, minibatch512, actor LR3e-6, critic LR1e-4,
20 critic-only plus500 policy updates. Keep every arm at520 updates and select
only that endpoint. Sampling-induced trajectories and critic warmup may diverge;
initial weights, optimizer initialization and random seeds are matched.

Require actual production-size capability before starting. All-0.05 wrapping
must reproduce the original trainer's updates; verify actor stationarity during
warmup, subsequent actor updates, full optimizer/environment/RNG exact restart
and greedy inference through the original singleton CPU path. The learned agent
uses ordinary inference temperature0.05, independently of its sampling setting.
Do not alter the canonical human-play model or initial dealer rule.

After all four endpoints finish, evaluate each and both retained parents against
assisted WP and Point. Pair each learned seed with the correspondingly indexed
retained parent for the practical improvement contrast. All six policies share
2,048 duplicate deals per opponent, both seats, greedy CPU2 execution. Evaluation
seeds are97110001+rep for WP and97120001+rep for Point (rep0..2047). This yields
49,152 complete matches on4,096 shared actual deal streams. These are development
deals; old100M final outcomes are not reused and reserved101M remains untouched.

The primary criterion requires all four balanced-opponent mixture win-change
95% lower bounds to be positive: calibrated-minus-fixed and calibrated-minus-
retained-parent, for each of the two training seeds. Draws count as half wins.
Bootstrap10,000 blocks with seed97310001; a block retains both seats and both
fixed opponent strata at the same replicate index. Report every absolute result,
all18 paired contrasts (including fixed-minus-parent), individual opponents and
point margins. Secondary findings do not substitute for the primary criterion.

Write actions, terminal states, work and checkpoint identities for every match.
An independent audit must replay the recorded trajectories, verify endpoint
ancestry/training counters/optimizer state and recompute statistics before any
qualification decision. Recorded-action replay is not a second neural evaluation.
No intermediate checkpoint evaluation, outcome-based extension, temperature
retuning or final-confirmation run is authorized by this protocol. A positive
development result would require a separate prospective confirmation design.

Run one GPU training worker at a time, then up to three strict CPU evaluation
workers. Preserve failures and incomplete output directories. Bind calibration,
capability, code, tests, reference weights and this design before execution.
This comparison can establish a useful phase-temperature effect; calibration
and Boltzmann sampling alone are not claimed as a new general algorithm.
