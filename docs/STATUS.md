# Current status

Implementation and pilot comparisons in progress. The verified first milestone was pushed as 077dc59 to https://github.com/sheetau/koikoi.git.

Hardware: RTX 5070 Ti (16 GB), Ryzen 9 9900X, about 62 GiB RAM. Project-local Python 3.12 / PyTorch 2.12.0+cu130 / NumPy 2.5.4. Visual Studio C++ toolchain builds the CPU core; PyTorch-bundled NVRTC builds the CUDA core. WSL is unnecessary.

## Implemented and checked

- Versioned paper8 and research12 rules, Python oracle and shared C++/CUDA core.
- Device-resident batched observations, legal actions, labels and transitions.
- Plain MLP, public-feature MLP, and card-token Transformer; PPO match returns, auxiliary belief loss, mixed precision, atomic snapshots, and exact CPU/CUDA continuation.
- Separate centralized Q critic and Q-boosting experiment; see VRPO.md for mathematical scope and engineering deviations.
- Published RL-WP/RL-Point loader, exact upstream feature reconstruction, paired evaluation with confidence intervals and latency.
- Cardinality-constrained hidden-state proposals and observation-only neural rollouts, initially at koi decisions.
- Seeded independent round deals use 64-bit SplitMix-based generation with unbiased bounded draws.

`./.venv/Scripts/python.exe -m pytest -q`: 27 passed on this host, including CUDA and pinned upstream (no skips). Includes 2,000 scoring comparisons, 12 full upstream matches, replayed feature equality, CPU/CUDA/reference trajectory equality, hidden-state invariance (including original model inputs), sampler constraints, signed GAE, exact CPU/CUDA restart for both estimators, matching-pennies Q-boosting, pair replay, and BF16 training checks. Ruff also passed. One upstream Transformer performance warning is expected.

## Initial measurements (development only)

Before upgrading the deal RNG to 64 bits, a 30-update research12 PPO pilot completed 40,247 matches / 492,776 rounds / 7,864,320 decisions in 12.38 seconds including updates (~3,251 matches/s). A 4,096-environment inference-only benchmark measured ~7.66M decisions/s; this is NOT training throughput. These are short warmup-sensitive measurements, not scale guarantees, and must be repeated on the committed core.

The initial greedy versus RL-WP adapter check was 27.5% draw-adjusted score over 40 matches (20 paired seeds), bootstrap CI 15–40%; too small for a research claim. A belief-search smoke run completed successfully but used a nearly untrained model and is not evidence of improvement.

## Completed learning pilots

Both below used 500 updates / 131,072,000 decisions, seed 1, paper8, a width-128 MLP, and match-outcome rewards. These are single-seed development pilots; they do not isolate centralized information from the advantage estimator or establish a method ranking.

| Method | Completed matches | Training seconds | Score vs RL-WP | 95% pair-bootstrap CI | Mean point difference |
| --- | ---: | ---: | ---: | --- | ---: |
| PPO | 1,100,095 | 202.70 | 37.5% | 33.125–41.75% | -7.565 |
| Q-boosting variant | 1,096,878 | 350.72 | 34.125% | 30.125–38.25% | -9.45 |

Each evaluation used 200 pairs / 400 matches, development deal seeds 1,000,000–1,000,199. Both used the earlier continuous policy RNG protocol (reports predate `protocol_version: 2`); do not merge their records with newer independently seeded pairs. Q-boosting's run overlapped short Transformer smoke checks; timings are practical pilot measurements, not controlled hardware benchmarks. All checkpoints and detailed JSON reports are in ignored runs/ppo8_v1 and runs/vrpo8_v1.

PPO's 100-update checkpoint scored 27.0% over 100 pairs; the 500-update result shows development progress but remains below RL-WP. No superiority claim is justified. The Q-boosting comparison does not currently justify its extra compute cost.

Transformer smoke: 5 updates / 327,680 decisions worked in float32 and BF16; BF16 completed in 6.44 seconds versus 13.43 seconds for float32 under overlapping workloads. This is a functionality/sizing check, not a controlled speed comparison or strength result.

Currently running:

- `runs/ppo8_v2`: continue PPO from update 500 to 2,000, snapshots every 250 updates. Source changes were explicitly permitted and parent checkpoint lineage is recorded; gameplay rules/representation stayed unchanged.
- `runs/features8_v1`: a fresh 500-update public-feature MLP comparison, snapshots every 100 updates. It adds role progress and month counts computed exclusively from the same actor observation.

Both use GPU; do not treat their concurrent throughput as isolated measurements. Check `metrics.jsonl` / `latest.pt` for actual counters. A run-local STOP file requests a clean completed-update checkpoint. Resume only after removing it. No 100M-match job has started.

## Next work

Evaluate new PPO/features snapshots against RL-WP using protocol version 2 and fixed development seeds. Promote no method on training loss alone. Once a policy exceeds the baseline, validate across independent training seeds and test inference-time search/hidden-card prediction at matched budgets. Centralized-V controls for Q-boosting, archived-opponent training, exact reduced-game exploitability, multi-seed strength experiments, 100M-match selected twelve-round training, and manuscript remain pending. Do not describe the implemented rollout heuristic as equilibrium search.

No superiority result, 100M-match run, or publication is complete.
