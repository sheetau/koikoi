# Current status

Implementation and pilot comparisons in progress. Verified simulator/learning and Q-boosting milestones were pushed as 077dc59 and 07580fe to https://github.com/sheetau/koikoi.git. No superiority claim is established.

Hardware: RTX 5070 Ti (16 GB), Ryzen 9 9900X, about 62 GiB RAM. Project-local Python 3.12 / PyTorch 2.12.0+cu130 / NumPy 2.5.4. Visual Studio C++ toolchain builds the CPU core; PyTorch-bundled NVRTC builds the CUDA core. WSL is unnecessary.

## Implemented and checked

- Versioned paper8 and research12 rules, Python oracle and shared C++/CUDA core.
- Device-resident batched observations, legal actions, labels and transitions.
- Plain MLP, public-feature MLP, relational card policy and card-token Transformer; PPO match returns, auxiliary belief loss, mixed precision, atomic snapshots, exact CPU/CUDA continuation and weight-only experiment initialization with separate counters.
- Separate centralized Q critic and Q-boosting experiment; see VRPO.md for mathematical scope and engineering deviations.
- Published RL-WP/RL-Point loader, exact upstream feature reconstruction, paired evaluation with confidence intervals and latency.
- Cardinality-constrained hidden-state proposals and observation-only neural rollouts, at koi decisions or all card decisions; paired standard-error guard against noisy action changes.
- Public endgame outcome guard and corresponding assisted published opponents; shared-seed report comparisons with paired difference intervals.
- Seeded independent round deals use 64-bit SplitMix-based generation with unbiased bounded draws.

`./.venv/Scripts/python.exe -m pytest -q`: 45 passed on this host, including CUDA and pinned upstream (no skips). Includes 2,000 scoring comparisons, 12 full upstream matches, replayed feature equality, CPU/CUDA/reference trajectory equality, hidden-state invariance (including original model inputs and search actions), sampler constraints, signed GAE, exact CPU/CUDA restart for both estimators, matching-pennies Q-boosting, pair replay, BF16 training, transfer budget/lineage, paired report compatibility and assisted baseline equality with upstream AgentForTest. Native batched duels match sequential reference results on both devices/profiles, with/without endgame assistance, and across chunk sizes; terminal rows remain frozen and evaluation split ranges are enforced. Ruff also passed. The upstream Transformer performance warning is expected.

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

## Later development evaluations (protocol 2, 2026-10-16)

All rows below use paper8 and deterministic action selection, with the public endgame guard applied to both candidate and RL-WP. The assisted opponent matches upstream's AgentForTest endgame rules; earlier raw-network rows above are a different comparison. Each row uses 300 paired seeds / 600 matches, seeds 1,000,000–1,000,299, except the explicitly noted 100-pair early relational check.

| Model / update | Cumulative training matches | Score vs assisted RL-WP | 95% pair-bootstrap CI | Mean point difference |
| --- | ---: | ---: | --- | ---: |
| MLP PPO / 2,000 | 4,443,788 | 45.833% | 42.665–49.167% | -1.220 |
| MLP PPO / 3,000 | 6,675,832 | 49.250% | 45.917–52.583% | +0.750 |
| Relational PPO / 500 | 554,063 | 44.250% | 40.917–47.585% | -2.817 |

Immutable evaluated checkpoints and reports: `runs/ppo8_v2/update_00002000.pt` / `eval_2000_guarded.json`; `runs/ppo8_v3/update_00003000.pt` / `eval_3000_guarded.json`; `runs/relational8_v1/update_00000500.pt` / `eval_500_guarded.json`. Reports contain checkpoint SHA-256, training counters, source hashes and every pair result. For example, the 3,000-update checkpoint hash is `d6a9b15bddabc3da9a3711c92c24cffb5dd4966d50cdaa5d8162f34ab2e22f8c`.

The relational 100-update checkpoint (109,159 training matches) scored 44.25% on 100 pairs, CI 38.25–50.0%. Its 500-update comparison against the MLP 2,000-update checkpoint on 300 shared pairs differs by -1.583 percentage points, CI -6.167–+2.917; these unequal training budgets and a single training seed do not establish an architectural ranking. The separate public-feature MLP at 500 updates scored 36.125%, CI 32.125–40%, against raw RL-WP on 200 protocol-2 pairs.

### Search evidence

The initial all-action search (16 particles, horizon 16, stochastic rollouts despite deterministic deployment) degraded the 2,000-update MLP by 20.5 percentage points, CI -29–-12, over 50 shared pairs. This implementation was replaced: rollouts now follow deployment action selection, use paired worlds/common policy randomness, and change the base action only when improvement exceeds a standard-error margin.

The corrected experiment (32 particles, horizon 32, margin 2.5, CUDA, all decisions, same 50 pairs) scored 44.0%; difference from no search on those same seeds was +0.5 percentage points, CI -3.5–+4.5. It changed 26 of 5,379 searched decisions and simulated 26,606,592 transitions. Mean decision latency was 32.63 ms versus about 0.18 ms for CPU MLP without search (concurrent workloads, not an isolated device benchmark). The intervals do not establish improvement; search remains off by default. Reports: `runs/ppo8_v2/eval_2000_search_guarded.json`, `guarded_search_difference.json`; naive results are retained separately as `eval_2000_search16.json` / `naive_search_difference.json`. Search transitions do not count as training matches.

Reproduce evaluation and paired comparison:

```powershell
./.venv/Scripts/python.exe -m koikoi.evaluate --agent runs/ppo8_v3/update_00003000.pt --opponent rl-wp-assisted --pairs 300 --endgame --deterministic --output runs/ppo8_v3/eval_3000_guarded.json
./.venv/Scripts/python.exe -m koikoi.compare runs/ppo8_v2/eval_2000_search_guarded.json runs/ppo8_v2/eval_2000_guarded.json --output runs/ppo8_v2/guarded_search_difference.json
```

### Active work

The new batched checkpoint evaluator compared MLP update 3,000 against update 2,000 on 10,000 development pairs / 20,000 matches: 50.815% (95% CI 50.243–51.375%), mean point difference +0.7107. Report: `runs/ppo8_v3/batch_3000_vs_2000.json`. Both policies used deterministic actions and endgame assistance. The measured 2.825 seconds covers batched simulation/inference and native startup, excluding model loading and report bootstrap, with training running concurrently; it is not a controlled throughput benchmark. This is evidence of a small gain against an earlier snapshot, not superiority to RL-WP.

`runs/ppo8_v2`, `runs/features8_v1`, and `runs/relational8_v1` have completed their pilot targets. `runs/ppo8_v3` continues MLP PPO from update 2,000 to 5,000, snapshots every 500 updates:

```powershell
./.venv/Scripts/python.exe -m koikoi.train --resume runs/ppo8_v2/update_00002000.pt --allow-code-change --run runs/ppo8_v3 --updates 5000 --checkpoint-every 500
```

Source changes were explicitly allowed and parent lineage is recorded. These development pilots began during code changes and are not publication-quality fixed-revision experiments. Do not treat their concurrent throughput as isolated measurements. Check `metrics.jsonl` / immutable snapshots for actual counters. A run-local STOP file requests a clean completed-update checkpoint. No 100M-match job has started.

## Next work

Evaluate the 5,000-update PPO checkpoint against assisted RL-WP with fixed development seeds. Current evidence favors continuing the cheap MLP baseline; neither Q-boosting, explicit features, relational scoring nor search has demonstrated superiority. Once a policy exceeds the baseline, validate across independent training seeds and test hidden-card prediction/search at matched budgets. Weight-only initialization now supports a separately counted research12 experiment. Centralized-V controls for Q-boosting, archived-opponent training, exact reduced-game exploitability, multi-seed strength experiments, 100M-match selected twelve-round training, and manuscript remain pending. Do not describe the rollout heuristic as equilibrium search.

No superiority result, 100M-match run, or publication is complete.
