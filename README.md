# Koi-Koi research

GPU-parallel Koi-Koi simulation, self-play PPO, published-baseline evaluation, and belief-sampled inference rollouts. This is a research implementation; no strongest-AI or equilibrium claim has been established. See [current evidence](docs/STATUS.md), [rules](docs/RULES.md), and [architecture](docs/DESIGN.md).

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

The eventual scale target, after selecting a validated method:

```powershell
./.venv/Scripts/python.exe -m koikoi.train --config configs/ppo12.json --run runs/selected12 --target-matches 100000000 --checkpoint-every 100
```

This command is a capability, not evidence that the run has been performed. `paper8` and `research12` are different versioned protocols. Do not compare published eight-round weights on a twelve-round match. Config `reward=round` is a separately labeled point-reward experiment, not the default win-rate objective. `kind=transformer` selects card-token attention; `belief_weight>0` trains the hidden-card auxiliary predictor. Policy inputs never receive its ground-truth labels.

## Evaluate

```powershell
./.venv/Scripts/python.exe -m koikoi.evaluate --agent greedy --opponent rl-wp --pairs 100
./.venv/Scripts/python.exe -m koikoi.evaluate --agent runs/ppo8/latest.pt --opponent rl-wp --pairs 1000 --output runs/ppo8_vs_wp.json
./.venv/Scripts/python.exe -m koikoi.evaluate --agent runs/smoke/latest.pt --opponent greedy --pairs 100 --search belief --particles 32 --search-horizon 32
```

`--search uniform` requires no trained belief head. Search initially focuses on koi/stop decisions; both rollout players use only their own observations. It is a finite-budget policy-rollout heuristic, not PIMC or a Nash-safe subgame solver. Report search time separately. Neural policies sample by default; `--deterministic` is a separate evaluation setting.

Each pair exchanges agent seats while retaining identical seat-assigned deals and initial dealer. Reports include pair-bootstrap confidence intervals, draw-adjusted win rate, point difference, decision latency, exact seeds and code provenance. Small smoke evaluations cannot establish superiority. Development seeds start at 1,000,000; reserve 100,000,000 and above for final tests (`--split test`). Do not tune against final test results.

Original models retain their historical input quirks. The loader adapts an obsolete PyTorch class name and loads the original parameter tensors into the same architecture on modern PyTorch. Fetch source with `python -m koikoi.upstream --fetch`; the pin and license are in [third-party notices](THIRD_PARTY_NOTICES.md). Load only trusted local training checkpoints: PyTorch pickle deserialization is used.
