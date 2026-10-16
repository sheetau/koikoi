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

`--max-seconds` sets a wall-time limit checked between completed updates. Creating `runs/<run>/STOP` also requests a clean checkpoint and exit; remove that file before resuming. Starting a new training run in a directory with existing metrics is rejected. Checkpoints record the parent checkpoint hash when continued.

`--initialize runs/<source>/update_XXXXXXXX.pt` starts a new experiment from actor weights with the architecture specified in `--config`. It resets optimizer, simulator, RNG and counters, recording the source checkpoint hash and its training budget separately. This supports changing rewards or transferring from paper8 to research12 without counting eight-round pretraining toward twelve-round matches. `--resume` instead restores the entire experiment and cannot be combined with `--config`.

The eventual scale target, after selecting a validated method:

```powershell
./.venv/Scripts/python.exe -m koikoi.train --config configs/ppo12.json --run runs/selected12 --target-matches 100000000 --checkpoint-every 100
```

This command is a capability, not evidence that the run has been performed. `paper8` and `research12` are different versioned protocols. Do not compare published eight-round weights on a twelve-round match. Config `reward=round` is a separately labeled point-reward experiment, not the default win-rate objective. `kind=transformer` selects card-token attention; `belief_weight>0` trains the hidden-card auxiliary predictor. Policy inputs never receive its ground-truth labels.

Additional comparison configs: `vrpo8.json` selects a separate centralized Q critic ([implementation scope and deviations](docs/VRPO.md)); `features8.json` adds public role/month counts to the MLP; `transformer8_bf16.json` uses mixed precision with float32 loss calculations. The Q-boosting option is experimental, not a proven improvement. Compare by both decision budget and wall time, with multiple independent training seeds.

`relational8.json` scores candidate cards with shared weights, their same-month hand/field aggregates and a public global context. Its BF16 training configuration is an architectural experiment; current evidence does not establish a strength advantage.

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

Original models retain their historical input quirks. The loader adapts an obsolete PyTorch class name and loads the original parameter tensors into the same architecture on modern PyTorch. Fetch source with `python -m koikoi.upstream --fetch`; the pin and license are in [third-party notices](THIRD_PARTY_NOTICES.md). Load only trusted local training checkpoints: PyTorch pickle deserialization is used.
