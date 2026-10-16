# Koi-Koi research

GPU-parallel Koi-Koi simulation, self-play PPO and Q-boosting, published-baseline evaluation, and belief-sampled inference rollouts. This is a research implementation; no strongest-AI or equilibrium claim has been established. See [current evidence](docs/STATUS.md), [rules](docs/RULES.md), and [architecture](docs/DESIGN.md).

## Setup (Windows / NVIDIA)

Requires Python 3.12, a recent NVIDIA driver, Git, and Visual Studio C++ build tools. The project-local environment leaves system Python unchanged. NVRTC is loaded from the CUDA PyTorch wheel; a system CUDA toolkit is unnecessary.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
```

The first native CPU build is cached under ignored `build/`; kernels compile at process startup. Linux uses `c++` and the CUDA PyTorch NVRTC library path, but has not been validated on this machine.

## Validate and measure

```powershell
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe -m koikoi.benchmark --envs 4096 --steps 1000 --policy
```

Tests compare independent Python rules with C++ and CUDA, scoring and full matches with the pinned original implementation, model inputs with replayed original inputs, hidden-information invariance, sampling constraints, signed returns, and exact CPU checkpoint continuation. CUDA tests skip explicitly on CPU-only hosts. Upstream tests skip unless fetched.

## Train

```powershell
./.venv/Scripts/python.exe -m koikoi.train --config configs/smoke.json --run runs/smoke --updates 10
./.venv/Scripts/python.exe -m koikoi.train --config configs/ppo8.json --run runs/ppo8 --updates 1000
./.venv/Scripts/python.exe -m koikoi.train --resume runs/ppo8/latest.pt --run runs/ppo8 --updates 2000
```

`--updates` is an absolute cumulative update target. `--target-matches` instead trains until the cumulative completed-match counter reaches the specified number (possibly overshooting by a rollout). `--checkpoint-every` controls completed-update snapshots. Ctrl+C retains the last completed snapshot, never a partial update. Resuming modified source requires explicit `--allow-code-change` and creates a new provenance record; it is not an exact continuation experiment.

PPO uses the native C++/CUDA signed-return scan by default. Set `returns_backend` to `torch` to use the independent PyTorch recurrence. Tests verify bit-identical float32 returns and full PPO updates; `python -m koikoi.benchmark --returns --envs 2048 --horizon 128 --steps 50` measures this component separately from learning.

`cuda_graph=true` captures fixed-size PPO minibatch updates on CUDA. Warmup/recording updates are rolled back, including Adam moments and step counters, so they do not add hidden training steps. Graph checkpoints resume their exact next update; tests also compare graph steps with eager steps using the same capturable Adam settings. Capturable Adam can round differently from the older non-capturable optimizer, so switching modes is a distinct experiment, not a bit-identical continuation. `configs/ppo12_graph.json` enables this path. Dynamic minibatch sizes and CPU/Q-boosting capture are rejected.

`cuda_rollout=true` also captures the fixed-horizon PPO collection and return calculation. Preparation rolls back simulator state and RNG; Python counters advance only on real replay. The captured policy reads current model weights, including BF16 casts. Tests compare full trajectories and optimizer updates with ordinary collection, including match boundaries, and verify checkpoint continuation. This option supports CUDA PPO only. Add `--cuda-rollout` to a full-learning benchmark to measure it.

For an existing CUDA PPO run, first stop cleanly with its run-local STOP file, verify the process exited, then resume its completed checkpoint in a fresh directory with `--allow-code-change --enable-cuda-rollout`. The explicit execution change retains model, optimizer, simulator, RNG and cumulative counters; source lineage records the change. Starting an additional trainer against the live run's directory is not a continuation.

Measure full learning, excluding five warmup updates, with `python -m koikoi.benchmark --training-config configs/ppo12.json --steps 20`; add `--cuda-graph` for captured updates. Unlike simulator-only throughput, this includes rollout, return estimation and gradient updates. Configuration comes from the supplied file.

`--max-seconds` sets a wall-time limit checked between completed updates. Creating `runs/<run>/STOP` also requests a clean checkpoint and exit; remove that file before resuming. Starting a new training run in a directory with existing metrics is rejected. Checkpoints record the parent checkpoint hash when continued.

`--initialize runs/<source>/update_XXXXXXXX.pt` starts a new experiment from actor weights with the architecture specified in `--config`. It resets optimizer, simulator, RNG and counters, recording the source checkpoint hash and its training budget separately. This supports changing rewards or transferring from paper8 to research12 without counting eight-round pretraining toward twelve-round matches. `--resume` instead restores the entire experiment and cannot be combined with `--config`.

Audit an immutable local checkpoint with `python -m koikoi.budget runs/<run>/update_XXXXXXXX.pt --output budget.json`. Totals group completed learning by rule profile and add weight-initialization ancestry without double-counting resumed counters. This is one model lineage; reports with shared ancestors must not be summed.

A verified twelve-round baseline scale run is now active alongside method selection (see STATUS.md). Its command is:

```powershell
./.venv/Scripts/python.exe -m koikoi.train --config configs/ppo12_graph.json --initialize runs/ppo12_transfer_v1/update_00000693.pt --run runs/ppo12_scale_seed30001 --target-matches 100000000 --checkpoint-every 1000
```

The target is not complete; consult recorded counters rather than inferring progress from the command. `paper8` and `research12` are different versioned protocols. Do not compare published eight-round weights on a twelve-round match. Config `reward=round` is a separately labeled point-reward experiment, not the default win-rate objective. `kind=transformer` selects card-token attention; `belief_weight>0` trains the hidden-card auxiliary predictor. Policy inputs never receive its ground-truth labels.

Additional comparison configs: `vrpo8.json` selects a separate centralized Q critic ([implementation scope and deviations](docs/VRPO.md)); `features8.json` adds public role/month counts to the MLP; `transformer8_bf16.json` uses mixed precision with float32 loss calculations. The Q-boosting option is experimental, not a proven improvement. Compare by both decision budget and wall time, with multiple independent training seeds.

`ppo8_wide_bf16.json` screens a width-512 MLP with BF16 and CUDA Graph updates. Match its decision budget to smaller-policy checkpoints; it is not yet a demonstrated strength improvement.

`relational8.json` scores candidate cards with shared weights, their same-month hand/field aggregates and a public global context. Its BF16 training configuration is an architectural experiment; current evidence does not establish a strength advantage.

`ppo8_endgame.json` trains with public match-outcome dominance rules enabled in the policy mask before sampling. Checkpoints retain this behavior automatically during inference and search; reports show the actual assistance settings for each agent. `ppo8_control.json` is identical except for this flag, for controlled weight-initialized comparisons. This option requires match-outcome rewards; point difference is not its objective.

## Evaluate

```powershell
./.venv/Scripts/python.exe -m koikoi.evaluate --agent greedy --opponent rl-wp --pairs 100
./.venv/Scripts/python.exe -m koikoi.evaluate --agent runs/ppo8/latest.pt --opponent rl-wp --pairs 1000 --output runs/ppo8_vs_wp.json
./.venv/Scripts/python.exe -m koikoi.evaluate --agent runs/ppo8/latest.pt --opponent rl-wp-assisted --pairs 1000 --endgame --deterministic --output runs/ppo8_assisted.json
./.venv/Scripts/python.exe -m koikoi.evaluate --agent runs/smoke/latest.pt --opponent greedy --pairs 100 --search belief --particles 32 --search-horizon 32
```

`--search uniform` requires no trained belief head. Search initially focuses on koi/stop decisions; both rollout players use only their own observations. It is a finite-budget policy-rollout heuristic, not PIMC or a Nash-safe subgame solver. Report search time separately. Neural policies sample by default; `--deterministic` is a separate evaluation setting.

`--endgame` applies public-information decisions that guarantee the best possible match outcome, such as stopping when that wins the final round's match. `rl-wp-assisted` and `rl-point-assisted` apply the same rules to the original models, matching upstream's `AgentForTest` assistance. Use these opponents for assisted comparisons; the plain names evaluate the raw networks. The guard optimizes match outcome, not point difference.

`--search-all` extends rollouts to card decisions; `--device cuda` runs neural rollouts on GPU. Search uses the deployment policy's deterministic/sampled action setting and paired hidden worlds. The default `--search-confidence 2.5` switches away from the base action only when the paired estimated improvement exceeds 2.5 standard errors. This controls some sampling noise, not critic bias or multiple-comparison error. It is not a calibrated improvement guarantee. Current pilot evidence does not justify enabling search by default.

Each pair exchanges agent seats while retaining identical seat-assigned deals and initial dealer. Reports include pair-bootstrap confidence intervals, draw-adjusted win rate, point difference, decision latency, exact seeds and code provenance. Small smoke evaluations cannot establish superiority. Development seeds start at 1,000,000; reserve 100,000,000 and above for final tests (`--split test`). Do not tune against final test results.

Evaluation protocol version 2 independently seeds each pair and repeats policy RNG seeds across its two legs. Reports hash the checkpoint bytes actually loaded and record their training counters/provenance. Earlier pilot reports without a protocol version used one continuous policy RNG stream; do not combine their pair records with version 2.

Compare two version-2 reports on their shared pairs with `python -m koikoi.compare first.json second.json --output difference.json`. This reports first-minus-second score and point differences with paired confidence intervals, checking opponent identity and checkpoint hashes. Reports also count decisions by phase and search-induced action changes.

For large checkpoint-versus-checkpoint comparisons, `batch_evaluate` runs independent paired matches on the native CPU/CUDA simulator:

```powershell
./.venv/Scripts/python.exe -m koikoi.batch_evaluate --agent runs/new/latest.pt --opponent runs/old/latest.pt --profile research12 --pairs 10000 --endgame --output runs/new_vs_old.json
```

This path is deterministic and applies `--endgame` to both agents. It freezes completed rows instead of redealing, and reports actual match/round/decision counts plus batch throughput. Throughput is not single-decision latency. Published baselines and stochastic/search policies continue to use `koikoi.evaluate`. Native batched results are checked against the independent sequential reference for both rule profiles and both devices. Floating-point near-ties can still make different devices or batch shapes choose different actions; execution device and batch size are recorded. Development evaluation rejects seeds in the reserved final-test range.

`--koi-agent <agent>` replaces only the candidate's koi/stop policy, retaining its card policy. For example, combine a checkpoint `--agent` with `--koi-agent rl-wp` to diagnose which decisions limit strength. Reports identify both components and their checkpoint metadata; endgame assistance is determined by the effective koi policy and explicit flags. Published components still require paper8. This substitution cannot be combined with search, and is a separate composite policy rather than evidence about either component alone.

In the sequential evaluator, add `--opponent-endgame` to apply the guard to a checkpoint opponent as well. Reports explicitly record whether the opponent receives assistance; comparisons reject a mismatch.

Original models retain their historical input quirks. The loader adapts an obsolete PyTorch class name and loads the original parameter tensors into the same architecture on modern PyTorch. Fetch source with `python -m koikoi.upstream --fetch`; the pin and license are in [third-party notices](THIRD_PARTY_NOTICES.md). Load only trusted local training checkpoints: PyTorch pickle deserialization is used.
