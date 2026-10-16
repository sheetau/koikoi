# Historical record

Earlier observations and plans are retained below. Consult STATUS.md for live jobs and current decisions.

# Experiment history

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

`./.venv/Scripts/python.exe -m pytest -q`: 60 passed on this host, including CUDA and pinned upstream (no skips). Includes 2,000 scoring comparisons, 12 full upstream matches, replayed feature equality, CPU/CUDA/reference trajectory equality, hidden-state invariance (including original model inputs and search actions), sampler constraints, signed GAE, exact CPU/CUDA restart for both estimators, matching-pennies Q-boosting, pair replay, BF16 training, transfer budget/lineage, paired report compatibility and assisted baseline equality with upstream AgentForTest. Native batched duels match sequential reference results on both devices/profiles, including asymmetric endgame assistance and different chunk sizes; terminal rows remain frozen and evaluation split ranges are enforced. Native returns and complete PPO updates are bit-identical to their PyTorch counterparts on CPU/CUDA. Assisted training records the pruned action distribution correctly, retains behavior in checkpoint inference/search, and resumes exactly on CPU/CUDA. CUDA Graph tests cover no extra warmup updates, exact eager/captured agreement with identical capturable Adam settings across MLP/relational/Transformer, and exact FP32/BF16 checkpoint continuation. Ruff also passed. The upstream Transformer performance warning is expected.

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

All rows below use paper8 and deterministic action selection, with the public endgame guard applied to both candidate and RL-WP. The assisted opponent matches upstream's AgentForTest endgame rules; earlier raw-network rows above are a different comparison. Each row uses 300 paired seeds / 600 matches, seeds 1,000,000–1,000,299, except PPO update 5,000 (1,000 pairs / 2,000 matches, seeds 1,000,000–1,000,999) and the explicitly noted 100-pair early relational check.

| Model / update | Cumulative training matches | Score vs assisted RL-WP | 95% pair-bootstrap CI | Mean point difference |
| --- | ---: | ---: | --- | ---: |
| MLP PPO / 2,000 | 4,443,788 | 45.833% | 42.665–49.167% | -1.220 |
| MLP PPO / 3,000 | 6,675,832 | 49.250% | 45.917–52.583% | +0.750 |
| MLP PPO / 5,000 | 11,141,556 | 47.825% | 46.050–49.650% | +0.975 |
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

The same trained checkpoints were additionally replayed through the CPU sequential reference on the first ten pairs (`reference_3000_vs_2000.json`): both seat scores and point differences agree with the GPU batched report. Independent replication is running from scratch in `runs/ppo8_seed10001`, with `configs/ppo8_seed10001.json`, target 5,000 updates and snapshots every 1,000. Seed 10,001 also gives a disjoint block of initial simulator seeds from the first run's 1–2,048, while remaining separate from development and final-test seeds. It started from committed source 2f3a366, unlike the earlier development continuations.

The relational policy is also continuing in `runs/relational8_v2` from `runs/relational8_v1/update_00000500.pt` to update 2,000, snapshots every 500. Command: `python -m koikoi.train --resume runs/relational8_v1/update_00000500.pt --allow-code-change --run runs/relational8_v2 --updates 2000 --checkpoint-every 500`. This starts from committed b9393cb and uses the verified native return scan. Both live jobs were confirmed by their process command lines and advancing metrics; recheck those before resuming or launching replacements.

At update 1,000, the relational policy scored 47.648% (CI 47.080–48.220%), -2.5879 points, against the MLP 5,000-update checkpoint on 10,000 pairs (`runs/relational8_v2/eval_1000_vs_mlp5000.json`). It remains behind that baseline at the measured budgets; its 2,000-update target continues as planned.

Update 5,000 beat update 3,000 directly on 10,000 pairs: 51.068%, CI 50.525–51.608%, +0.8763 points (`batch_5000_vs_3000.json`). Against assisted RL-WP, its larger 1,000-pair test remains below 50% (`eval_5000_guarded.json`). On the 300 seeds shared with the earlier 3,000-update evaluation, the score difference is +0.75 percentage points, CI -3.417–+5.0; do not compare the unpaired headline rates as if they used identical seeds. Replica seed 10,001 at update 1,000 scored 50.675%, CI 50.103–51.218%, against seed 1 at the same update on 10,000 pairs; this does not yet establish multi-seed superiority to RL-WP.

`runs/ppo8_v2`, `runs/features8_v1`, `runs/relational8_v1`, and `runs/ppo8_v3` have completed their pilot targets. The latter continued MLP PPO from update 2,000 to 5,000, snapshots every 500 updates:

```powershell
./.venv/Scripts/python.exe -m koikoi.train --resume runs/ppo8_v2/update_00002000.pt --allow-code-change --run runs/ppo8_v3 --updates 5000 --checkpoint-every 500
```

Source changes were explicitly allowed and parent lineage is recorded. These development pilots began during code changes and are not publication-quality fixed-revision experiments. Do not treat their concurrent throughput as isolated measurements. Check `metrics.jsonl` / immutable snapshots for actual counters. A run-local STOP file requests a clean completed-update checkpoint.

### Twelve-round transfer

`runs/ppo12_transfer_v1` initialized actor weights from the 5,000-update paper8 model using `configs/ppo12.json`, then trained from fresh optimizer/deals/counters to 1,000,464 completed research12 matches (693 updates / 181,665,792 decisions / 12,016,762 completed rounds including rounds in still-running matches). The 11,141,556 paper8 pretraining matches are recorded separately and do not count toward the twelve-round target. Initial weights are in `initial.pt`, completed pilot weights in `update_00000693.pt`; source revision 42e7ade.

Against the initialization checkpoint on 10,000 paired seeds / 20,000 complete twelve-round matches, the pilot scored 52.073%, CI 51.515–52.633%, point difference +4.083. Both agents use deterministic actions and endgame assistance; report `eval_1m_vs_initial.json`. This supports adaptation to research12, not a comparison against the eight-round published model.

### Return-estimation optimization

Signed GAE now has a C++/CUDA reverse scan, one thread per environment, with PyTorch retained as the reference/optional backend (`returns_backend= torch`). CUDA compilation disables multiply-add contraction to preserve the reference's float32 operation order. Randomized returns for lambda 0/0.95/1 and entire PPO updates agree bit-for-bit on CPU and CUDA.

`python -m koikoi.benchmark --returns --envs 2048 --horizon 128 --steps 50` measured 40.510 ms for the PyTorch recurrence and 0.0336 ms for the native scan on this host (`runs/gae_benchmark_verified.json`). This is the return calculation alone, with another training job running, not overall training acceleration. The prior profiled loop spent about 10% of end-to-end time in the recurrence; broader throughput must be measured separately.

## Earlier planning snapshot (superseded by STATUS.md)

CUDA Graph minibatch updates passed correctness and restart checks. Under four concurrent training jobs, 20 full research12 learning updates took 32.223 seconds eagerly versus 10.757 seconds with capture, after five warmup updates (reports `runs/train_benchmark_eager.json` and `train_benchmark_graph.json`). This is about 3x in that concurrent setting, not an isolated hardware claim. The optimizers use different capturable settings and complete slightly different trajectories; only eager/captured runs with the same optimizer setting are bit-identical.

The scale baseline is prepared in `configs/ppo12_graph.json`: MLP PPO, twelve full rounds, seed 30,001, CUDA Graph updates, initialization from `runs/ppo12_transfer_v1/update_00000693.pt`. Its fresh counters will target 100,000,000 additional complete twelve-round matches. Prior twelve-round and paper8 learning stay in initialization lineage. Running this verified baseline alongside method selection advances the explicit compute requirement; it does not establish superiority or preselect the final strongest policy.

Controlled endgame-training experiments are running in `runs/ppo8_endgame_v1` and `runs/ppo8_control_v1`: initialize `configs/ppo8_endgame.json` and `configs/ppo8_control.json` from `runs/ppo8_v3/update_00005000.pt`, seed 20,001, 1,000 additional updates, snapshots every 250, committed source 0499968. Both reset optimizer and counters identically; only policy-mask assistance differs. Evaluate both with assistance against the same RL-WP and snapshot opponents. Unlike earlier inference-only guards, this tests whether training the deployed behavior improves the earlier decisions. No strength result exists yet for this option.

Complete the independent seed-10,001 replication and improve beyond the still-unbeaten RL-WP baseline. The cheap MLP improves against past snapshots but has not met the public-baseline gate; larger relational training and aligning training with deployed endgame assistance are reasonable next controlled experiments. Neither Q-boosting, explicit features, relational scoring nor search has demonstrated superiority. Centralized-V controls for Q-boosting, archived-opponent training, exact reduced-game exploitability, multi-seed strength experiments, 100M-match twelve-round training, and manuscript remain pending. Do not describe the rollout heuristic as equilibrium search.

No superiority result, 100M-match run, or publication is complete.

## Completed endgame-training comparison

Both arms initialized the same paper8 update-5,000 weights, reset optimizer/counters and used seed 20,001 for 1,000 updates / 262,144,000 decisions (source 0499968). Assisted training completed 2,236,221 new matches; control completed 2,233,075. Evaluation assists both policies and the published opponent.

| Training | Score vs assisted RL-WP | 95% pair-bootstrap CI | Mean point difference |
| --- | ---: | --- | ---: |
| Endgame mask | 49.125% | 47.275-51.000% | +1.657 |
| Control | 49.175% | 47.400-50.926% | +1.033 |

Each uses 1,000 development pairs / 2,000 matches. Paired assisted-minus-control difference: -0.05 percentage points, CI -2.35 to +2.225. Direct 10,000-pair duel: assisted 49.6925%, CI 49.1225-50.2325%. Neither comparison supports enabling assisted training by default. Control minus its initialization against RL-WP: +1.35 percentage points, CI -0.95 to +3.65; superiority to the published baseline remains unestablished. Reports: runs/ppo8_endgame_v1/eval_1000_wp.json, difference_vs_control.json, ab_1000.json; runs/ppo8_control_v1/eval_1000_wp.json and difference_vs_parent.json.

## Scale baseline started

The additional 100M-match research12 run started from committed source 5ddd893 in runs/ppo12_scale_seed30001, config ppo12_graph.json, seed 30,001, initialized from runs/ppo12_transfer_v1/update_00000693.pt. Command is recorded in STATUS.md. At update 1,000 it had 1,443,851 new matches; lineage adds 1,000,464 research12 and 11,141,556 paper8 pretraining matches separately. Against its twelve-round initialization on 10,000 pairs: 50.02%, CI 49.4674-50.5626%, +0.5784 points (eval_1000_vs_parent.json). No match-score improvement is established at this checkpoint.

Independent seed 10,001 at update 3,000 scored 46.4167%, CI 43.0833-49.5833%, -1.5567 points against assisted RL-WP on 300 development pairs (runs/ppo8_seed10001/eval_3000_guarded.json).

## Matched-decision relational comparison and later scale checkpoint

Relational update 1,000 (131,072,000 decisions) versus MLP update 500 (same decisions), both with inference endgame guard: 56.1075%, CI 55.5374-56.6575%, +4.1103 points over 10,000 development pairs. Report: runs/relational8_v2/eval_1000_matched_budget.json. This is a positive candidate screen at matched decisions, unlike the earlier comparison against MLP update 5,000. Architectures also differ in precision, rollout environment count and minibatch size; a single seed does not isolate an architectural effect.

Scale research12 update 3,000 versus its initialization: 50.6225%, CI 50.0725-51.1901%, +0.8892 points over 10,000 development pairs, both guarded. Report: runs/ppo12_scale_seed30001/eval_3000_vs_parent.json. Repeated development checkpoint evaluation is selection evidence, not a held-out superiority claim.

Wider MLP screening started from committed a6b9bb6 in runs/ppo8_wide_seed40001, target 500 updates (131,072,000 decisions). Config ppo8_wide_bf16.json changes width to 512 and uses BF16/CUDA Graph with independent seed 40,001. No playing-strength evidence exists yet for this candidate.

## Completed capacity and phase diagnostics

Wider MLP (configs/ppo8_wide_bf16.json, committed a6b9bb6) completed 500 updates / 131,072,000 decisions / 1,108,105 matches in 401.883 seconds under concurrent workloads. At matched decisions, it beats MLP update 500 at 53.0025% (CI 52.4375-53.5775%) but loses to relational update 1,000 at 46.7625% (CI 46.1625-47.35%), each 10,000 pairs. Against assisted RL-WP over 300 pairs it scores 41.75% (CI 38.25-45.25%), -4.3233 points. Reports: runs/ppo8_wide_seed40001/eval_500_matched_mlp.json, eval_500_matched_relational.json, eval_500_wp.json. Defer extension of width-only candidate and prioritize relational learning.

Relational continuation completed update 2,000 / 262,144,000 decisions / 2,245,436 cumulative matches. Versus MLP update 1,000 at matched decisions: 55.035% (CI 54.4699-55.625%), +4.0791 points, 10,000 pairs. Report: runs/relational8_v2/eval_2000_matched_mlp.json. Precision/environment/minibatch differences mean this is a candidate screen, not an isolated architecture estimate.

Koi substitution (committed a95391f), 300 paired development seeds against assisted RL-WP: MLP update 5,000 cards + RL-WP koi scores 52.0833% (CI 49-55.1667%), +2.2867 points. Its difference from the full MLP on the same 300 seeds is +2.0833 percentage points (CI -0.4167 to +4.5833). Reverse hybrid scores 47.75% (CI 45.3333-50.25%), -0.76 points. Reports: runs/ppo8_v3/eval_5000_wp_koi.json, eval_wp_5000_koi.json, difference_wp_koi.json. Both directions use --endgame --deterministic --pairs 300; component identities are recorded. Evidence is suggestive, not sufficient to promote the hybrid.

Next relational candidate uses configs/relational8_graph.json (BF16, width 64, seed 50,001, CUDA Graph) initialized from runs/relational8_v2/update_00002000.pt. Fresh optimizer/RNG/counters; initialization ancestry retains prior learning. Target 3,000 new updates, snapshots every 500. Graph feasibility benchmark completed five measured updates after five warmups in 6.541 seconds / 5,225 matches with concurrent work (runs/relational_graph_benchmark.json); not a controlled speedup or strength claim.

Relational update 2,000 public-baseline evaluation finished: 49.25%, CI 45.9167-52.6687%, +0.1567 points over 300 pairs (eval_2000_wp.json). Compared with update 500 on the same deals: +5.0 percentage points, CI +0.9167 to +8.9167, +2.9733 points (difference_2000_vs_500.json). This confirms development improvement, not superiority to RL-WP. The new graph run started from committed b9651ff; its process and advancing metrics were verified.

Timing diagnosis with configs/ppo12_graph.json: fresh trainer, five warmup updates, ten measured updates, CUDA synchronization around collect and whole update. Total 7.178 seconds; collect (including native returns) 5.178 seconds; remaining optimizer/minibatch work 2.000 seconds. Three other training jobs were active. Source b9651ff, report runs/scale_update_timing.json. This identifies collection as the next optimization target without attributing its cost to a particular CPU/GPU operation yet.

## Captured self-play collection

Optional cuda_rollout captures fixed-horizon observation, inference, action sampling, native transition/reset and return estimation. Warmup restores simulator state and CPU/CUDA RNG; actual match/round/decision bookkeeping remains outside graph replay. Model weights keep fixed addresses and BF16 casts are refreshed; the result dictionary is copied so advantage normalization cannot replace graph-output references. Public endgame masks and observation-only actor boundaries are unchanged.

Correctness: full suite 74 passed, no skips, two expected upstream warnings. Subsequent resume-metadata change passed four targeted resume cases (two were newly added). Eager and captured 256-step trajectories across complete matches, FP32/BF16 full updates and exact checkpoint continuation are tested. Real production configuration (2,048 environments / horizon 128 / MLP128) matched metrics, weights, Adam tensors, RNG and simulator state after three updates. An immutable scale checkpoint at update 5,000 also produced an identical next update with capture enabled versus ordinary collection: 7,215,134 cumulative matches at update 5,001. Reports runs/rollout_production_validation.json and rollout_scale_migration_validation.json; diagnostic updates are not added to the live scale budget.

Using configs/ppo12_graph.json, twenty complete updates after five warmups took 15.211 seconds ordinarily and 5.707 seconds with benchmark --cuda-rollout. Both completed exactly 27,254 matches / 328,568 rounds / 5,242,880 decisions. Reports runs/rollout_eager_benchmark.json, rollout_captured_benchmark.json. Three other trainers were active; approximately 2.66x here is not an isolated hardware speedup claim. Explicit train --resume --enable-cuda-rollout changes only execution mode and preserves optimizer, games, RNG and counters; changed source still requires --allow-code-change. Execution-change metadata survives later resumes.
