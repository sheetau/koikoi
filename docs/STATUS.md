# Current status

First implementation milestone. Repository initially contained only ignored plan.txt; origin is https://github.com/sheetau/koikoi.git and was empty.

Hardware: RTX 5070 Ti (16 GB), Ryzen 9 9900X, about 62 GiB RAM. Project-local Python 3.12 / PyTorch 2.12.0+cu130 / NumPy 2.5.4. Visual Studio C++ toolchain builds the CPU core; PyTorch-bundled NVRTC builds the CUDA core. WSL is unnecessary.

## Implemented and checked

- Versioned paper8 and research12 rules, Python oracle and shared C++/CUDA core.
- Device-resident batched observations, legal actions, labels and transitions.
- MLP/card-token Transformer, PPO match returns, auxiliary belief loss, atomic snapshots and exact CPU continuation.
- Published RL-WP/RL-Point loader, exact upstream feature reconstruction, paired evaluation with confidence intervals and latency.
- Cardinality-constrained hidden-state proposals and observation-only neural rollouts, initially at koi decisions.
- Seeded independent round deals use 64-bit SplitMix-based generation with unbiased bounded draws.

`./.venv/Scripts/python.exe -m pytest -q`: 18 passed on this host, including CUDA and pinned upstream (no skips). Includes 2,000 scoring comparisons, 12 full upstream matches, replayed feature equality, CPU/CUDA/reference trajectory equality, hidden-state invariance, sampler constraints, signed GAE, and exact CPU restart. One upstream Transformer performance warning is expected.

## Initial measurements (development only)

Before upgrading the deal RNG to 64 bits, a 30-update research12 PPO pilot completed 40,247 matches / 492,776 rounds / 7,864,320 decisions in 12.38 seconds including updates (~3,251 matches/s). A 4,096-environment inference-only benchmark measured ~7.66M decisions/s; this is NOT training throughput. These are short warmup-sensitive measurements, not scale guarantees, and must be repeated on the committed core.

The initial greedy versus RL-WP adapter check was 27.5% draw-adjusted score over 40 matches (20 paired seeds), bootstrap CI 15–40%; too small for a research claim. A belief-search smoke run completed successfully but used a nearly untrained model and is not evidence of improvement.

Currently running: paper8 PPO pilot, 500 updates, configs/ppo8.json, runs/ppo8_v1. Run artifacts are ignored; durable findings will be summarized here after evaluation.

## Next work

Evaluate the pilot against greedy and RL-WP on development seeds, repeat stable throughput measurement, then compare alternative estimators/networks and belief/search. VRPO, archived-opponent training, exact reduced-game exploitability, multi-seed strength experiments, 100M-match selected training, and manuscript remain pending. Do not describe the implemented rollout heuristic as equilibrium search.

No superiority result, 100M-match run, or publication is complete.
