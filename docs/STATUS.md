# Current status

Research is active. The published RL-WP strength gate and 100M-match scale target remain unmet; no strongest-AI or publication claim is established. Historical details, hashes and comparisons are in [EXPERIMENTS.md](EXPERIMENTS.md), to consult only when relevant.

## Verified implementation

- Independent Python oracle and shared C++/CUDA core; versioned paper8 and research12 rules. See RULES.md for game logic, DESIGN.md for architecture.
- GPU self-play PPO, native signed returns, MLP/features/relational/Transformer policies, optional belief loss and experimental Q-boosting.
- Exact checkpoint continuation, source hashes, separate weight-initialization ancestry and budget audit without duplicate resume counts.
- Published-model adapter, paired confidence intervals, GPU checkpoint duels, guarded observation-only rollout search, on-policy endgame masks and diagnostic koi-policy substitution.
- CUDA Graph minibatches and optional captured collection/returns. Warmup restores state/RNG; current weights are read each replay. FP32/BF16 eager/captured trajectories, complete updates and restart agree exactly, including real scale and relational checkpoint migrations.

Validation: full 74-test suite passed with no skips, including CUDA/upstream. Subsequent changes passed four resume cases and one additional Transformer full-update case; 77 cases now collected. Ruff passed. Two expected upstream Transformer performance warnings. No native rules changed in this milestone.

Hardware: RTX 5070 Ti 16 GB, Ryzen 9 9900X, about 62 GiB RAM; local Python 3.12 / PyTorch 2.12.0+cu130 / NumPy 2.5.4. Use .venv/Scripts/python.exe.

## Live jobs (recheck process command lines and advancing metrics)

| Run | Target | Last observed updates / matches | Source |
| --- | --- | ---: | --- |
| ppo12_scale_seed30001_captured | 100M cumulative new research12 matches | 7,481 / 10,795,279 | 142c265 |
| ppo8_seed10001 | 5,000 updates | 4,748 / 10,579,658 | 2f3a366 |
| relational8_fullgraph_seed50001 | 3,000 updates after initialization | 932 / 1,056,638 | 5dc3025 |

Counts above are cumulative within each run's resume lineage. Scale counters exclude initialization (1,000,464 research12 and 11,141,556 paper8 matches); relational counters exclude 2,245,436 paper8 pretraining matches. Final-test seeds (100,000,000+) remain unused.

Scale migrated cleanly at update 5,800 / 8,367,810 matches; relational at update 647 / 733,156 new matches. Source/model/optimizer/game/RNG/counter checks passed. Old directories ppo12_scale_seed30001 and relational8_graph_seed50001 are terminal and retain STOP; do not restart them. New process/advancing metrics verified. Their active commands are:

```powershell
./.venv/Scripts/python.exe -m koikoi.train --resume runs/ppo12_scale_seed30001/update_00005800.pt --allow-code-change --enable-cuda-rollout --run runs/ppo12_scale_seed30001_captured --target-matches 100000000 --checkpoint-every 1000
./.venv/Scripts/python.exe -m koikoi.train --resume runs/relational8_graph_seed50001/update_00000647.pt --allow-code-change --enable-cuda-rollout --run runs/relational8_fullgraph_seed50001 --updates 3000 --checkpoint-every 500
```

Runs/checkpoints/reports are ignored under runs/. Use immutable numbered snapshots. Never relaunch a live job. A run-local STOP requests completed-update save/exit. Source 5dc3025 has the same package code as 142c265.

## Decision evidence (development only)

- Small MLP seed 1 update 5,000, 11,141,556 matches: 47.825% vs assisted RL-WP, CI 46.05-49.65%, 1,000 pairs. Still below baseline. Independent seed 10,001 update 3,000: 46.4167%, CI 43.0833-49.5833%, 300 pairs; final replication pending.
- Relational update 2,000, 2,245,436 matches: 49.25% vs assisted RL-WP, CI 45.9167-52.6687%, 300 pairs. Improvement over update 500 on the same deals: +5.0 percentage points, CI +0.9167 to +8.9167. At matched 262,144,000 training decisions it beats small MLP at 55.035%, CI 54.4699-55.625%, 10,000 pairs. Prioritize relational learning.
- Relational next-stage update 500, 566,585 new matches: 48.3333% vs assisted RL-WP, CI 44.9167-51.8333%. Difference from parent -0.9167 percentage points, CI -5 to +3.1667. Sampled deployment scored 46.8333%; difference from deterministic -1.5 points, CI -5.5833 to +2.5854. No demonstrated improvement; continue the planned learning target.
- Width-512 MLP screening completed 500 updates / 1,108,105 matches: 41.75% vs assisted RL-WP. At matched decisions it beats small MLP (53.0025%) but loses to relational (46.7625%), both 10,000 pairs with CIs excluding 50%. Defer width-only extension. These are candidate screens, not isolated architecture effects.
- Endgame-training A/B completed: 49.125% assisted vs 49.175% control, 1,000 shared pairs; difference -0.05 points, CI -2.35 to +2.225. Keep training assistance off. Diagnostic MLP-card/published-koi hybrid scored 52.0833%, CI 49-55.1667%, 300 pairs; no established gain, no default change.
- Research12 transfer pilot beat initialization at 52.073%, CI 51.515-52.633%, 10,000 pairs. Scale update 7,000 beats that pilot at 51.2175%, CI 50.6275-51.7825%, +1.2146 points, 10,000 pairs. This is a small development gain against the parent, not superiority to the paper8 published baseline.
- Search, Q-boosting and explicit features have not demonstrated improvement. Keep search off. Positive point difference alone does not meet the win-rate objective.

## Performance and budget evidence

Twenty scale-config updates after warmup: 15.211 seconds ordinary collection versus 5.707 captured; identical 27,254 matches / 328,568 rounds / 5,242,880 decisions. Relational five-update comparison: 13.971 versus 6.570 seconds with identical counts. Concurrent jobs were active; these are not isolated hardware claims. Reports: rollout_*_benchmark.json and relational_rollout_*_benchmark.json under runs/.

Real checkpoint next-update equivalence: runs/rollout_scale_migration_validation.json and rollout_relational_migration_validation.json. Three fresh full-size updates also matched all weights, Adam tensors, simulator, RNG and metrics (rollout_production_validation.json). Diagnostic updates do not count toward scale learning.

Budget audit at resumed scale update 6,000: 8,656,699 current research12 + 1,000,464 research12 initialization = 9,657,163; paper8 initialization 11,141,556 separately. Resume parent counts are not added again. runs/ppo12_scale_seed30001_captured/budget_6000.json. Audits cover one ancestry, not a sum across models with shared ancestors.

## Next actions and completion gates

Finish seed-10,001 update-5,000 evaluation against assisted RL-WP and independent checkpoints. Evaluate relational saved checkpoints as training progresses; maintain the committed 3,000-update target unless evidence warrants a change. Keep the captured twelve-round scale baseline running. Further hyperparameter/curriculum/population/belief experiments require measured benefit; no default advantage is assumed.

Success still requires: exceed assisted RL-WP on paper8, replicate training gains, evaluate chosen policies on untouched final-test deals, verify at least 100M complete twelve-round training matches, and prepare the manuscript. Submission is a later decision. Do not mark the goal complete based on implementation, throughput or parent-model wins alone.
