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

Scale migration completed: old process exited cleanly at update 5,800, 8,367,810 matches / 100,424,868 rounds / 1,520,435,200 decisions. Immutable runs/ppo12_scale_seed30001/update_00005800.pt SHA ae2c5358eaee93ed98f9d0e00b264e9f9e2f758a1def7a32682a76cd66de18ad matched final metrics. Resume in runs/ppo12_scale_seed30001_captured from committed 142c265 uses --allow-code-change --enable-cuda-rollout, --target-matches 100000000 and --checkpoint-every 1000. Cumulative counters, optimizer, simulator, RNG and initialization lineage were retained; live process and advancing counters were verified. Source 5ddd893 before, 142c265 after. Do not add resumed parent counters a second time. The old run retains its STOP marker and is terminal.

Relational rollout capture: five full updates after five warmups measured 13.971 seconds ordinarily versus 6.570 seconds captured, identical 5,225 matches / 41,583 rounds / 655,360 decisions; concurrent work differs from earlier timing trials. Reports runs/relational_rollout_eager_benchmark.json and relational_rollout_captured_benchmark.json. A real BF16 relational update-500 checkpoint gave identical next-update metrics/weights/Adam/environment/RNG with either execution mode (runs/rollout_relational_migration_validation.json). Extra Transformer BF16 full-update equivalence test passed as well.

Relational migration: old graph process exited cleanly at update 647 / 733,156 new matches / 5,747,999 rounds / 84,803,584 decisions. Immutable update_00000647.pt SHA 17676e08d07a256fefb47143dd429783e733800a32186db50ea1445a805f03d7 matched final metrics. New runs/relational8_fullgraph_seed50001 resumes that state with --allow-code-change --enable-cuda-rollout, target --updates 3000, snapshots every 500. Source 5dc3025, package identical to 142c265; original 2,245,436-match weight-initialization ancestry remains separate. Verified new process and advancing metrics. Old directory retains STOP.

Early relational continuation update 500 (566,585 new matches) versus assisted RL-WP: deterministic 48.3333%, CI 44.9167-51.8333%, -0.5433 points; sampled 46.8333%, CI 43.5-50.1667%, -4.1067 points. Each used 300 identical development pairs. Sampled-minus-deterministic difference -1.5 percentage points, CI -5.5833 to +2.5854 (point difference -3.5633, CI -6.06 to -1.0333). Deterministic-minus-parent update-2,000 difference -0.9167 percentage points, CI -5.0 to +3.1667. No evidence for sampled deployment or early strength improvement; continue the prerecorded training target. Reports under runs/relational8_graph_seed50001: eval_500_wp.json, eval_500_wp_sampled.json, difference_sampled.json, difference_vs_parent.json.

Scale update-6,000 budget audit confirms 8,656,699 current research12 matches + 1,000,464 research12 initialization = 9,657,163; paper8 initialization 11,141,556 separately. Parent resume counters are not double-counted. runs/ppo12_scale_seed30001_captured/budget_6000.json.

Captured scale update 7,000 versus the research12 transfer-pilot initialization on 10,000 development pairs: 51.2175%, CI 50.6275-51.7825%, +1.2146 points. Both deterministic and endgame-assisted. Report runs/ppo12_scale_seed30001_captured/eval_7000_vs_parent.json. This supports a small development gain against the parent; it is not a comparison with the eight-round published baseline or a held-out result.

## Longer-return controlled screen

configs/relational8_lambda99.json changes gae_lambda from 0.95 to 0.99, otherwise exactly matching the resolved config of relational8_fullgraph_seed50001. Initialize from the same runs/relational8_v2/update_00002000.pt, use seed 50,001, reset optimizer/RNG/counters, 1,000 new updates / 131,072,000 decisions, snapshots every 250. Planned run runs/relational8_lambda99_seed50001. Control update 1,000 is already saved. Earlier control phases used ordinary collection; its capture migration was verified bit-identical, so execution history is recorded rather than treated as a learning-parameter change. This is one deliberately paired training seed, not a multi-seed result. Native 10,000-pair and published-opponent shared-deal comparisons determine further study; no final-test deals used.

Initialized functionality check in runs/relational_lambda99_smoke completed three updates / 393,216 decisions / 3,064 matches with finite metrics. These duplicated pilot updates are not counted toward the controlled experiment's target or the research12 budget. No new algorithm code was needed.

The lambda-0.95 control at new update 1,000 beat its initialization on 10,000 development pairs: 50.9825%, CI 50.4299-51.535%, +0.6298 points (runs/relational8_fullgraph_seed50001/eval_1000_vs_parent.json). This is a small parent-relative improvement; public-baseline superiority remains unproven.

The lambda-0.99 experiment started from committed bedd9c3 in runs/relational8_lambda99_seed50001 with the planned initialization/seed/1,000-update target. Live process and advancing finite metrics were verified. The first three updates reproduce the functionality pilot; duplicated pilot learning is excluded from experiment counters.

Control new update 1,000 against assisted RL-WP: 49.5%, CI 46.25-52.75%, +1.3567 points, 300 shared development pairs. Report runs/relational8_fullgraph_seed50001/eval_1000_wp.json; paired comparison with its initialization in difference_1000_vs_parent.json. Public-baseline superiority remains unestablished.

## Old-source independent-replica continuation check

Concurrent captured trainers slowed the old eager replica, so its continuation was validated against the actual historical code rather than assumed equivalent. A detached ignored worktree build/compat2f3a366 at 2f3a366 restored runs/ppo8_seed10001/update_00004000.pt without allowing source changes: source SHA 635bcb228b1674fe3414c9c0e519647ccab7ed5244339becdd550ef5cfbf6ea3 matched the checkpoint. Two updates produced runs/replica_migration_old.pt and replica_migration_old_metrics.json. Current code with captured collection, native returns and the original non-capturable Adam produced runs/replica_migration_new.pt. All model/optimizer tensors, environment, counters, CPU/CUDA/Python/NumPy RNG and update metrics matched exactly. Report runs/replica_old_revision_migration.json. These duplicated diagnostic updates are not added to live training counts.

Original replica then exited cleanly at update 4,822 / 10,744,859 matches / 84,263,764 rounds / 1,264,058,368 decisions. Immutable snapshot update_00004822.pt SHA a381af634fd4c07f3457998abbbb2aa7c8a853a46dd093fc31b247575b5455e8 matched final metrics. Resume from committed f3eeba1 in runs/ppo8_seed10001_captured with --allow-code-change --enable-cuda-rollout --updates 5000 --checkpoint-every 1000. Optimizer arithmetic remains unchanged. Old directory retains STOP and is terminal. Independent-seed evidence is preserved, but do not describe the final run as one fixed source revision.

Final-replica evaluation watchers were started for the immutable update-5,000 checkpoint: 1,000 paired deals against assisted RL-WP plus shared-pair difference versus seed 1; 10,000 direct pairs versus runs/ppo8_v3/update_00005000.pt. They check the specific training process while waiting and fail on missing-checkpoint termination. Relational update-1,500 public evaluation also started on 1,000 development pairs. These are pending evaluations, not results.

## Larger relational development evaluation and return-trace midpoint

Relational new update 1,500 against assisted RL-WP on 1,000 pairs / 2,000 matches: 48.55%, CI 46.75-50.325%, +0.807 points. Shared-pair difference from small MLP update 5,000: +0.725 percentage points, CI -1.75 to +3.125. Difference from the relational initialization uses only their 300 shared pairs: +0.6667 percentage points, CI -3.6667 to +4.8354; do not compare unequal-sample headline scores as a decline. Reports under runs/relational8_fullgraph_seed50001: eval_1500_wp.json, difference_1500_vs_mlp5000.json, difference_1500_vs_parent.json. Public-baseline superiority remains unmet.

Exploratory midpoint comparison of lambda 0.99 versus 0.95 after 500 new updates / 65,536,000 decisions each: 48.9125%, CI 48.3799-49.4426%, -1.2088 points, 10,000 development pairs with deterministic endgame-assisted policies. Same initialization and training seed; report runs/relational8_lambda99_seed50001/ab_500.json. The midpoint is negative. The predeclared primary 1,000-update target is unchanged; no default change or early stopping. Watchers are already waiting for the final snapshot to run the 10,000-pair direct comparison and 300-pair assisted-RL-WP evaluation with shared-pair control difference.

## Completed independent replication and longer-return endpoint

Independent MLP seed 10,001 reached update 5,000 / 1,310,720,000 decisions / 87,366,796 rounds / 11,141,428 matches. Final checkpoint runs/ppo8_seed10001_captured/update_00005000.pt SHA edc44213bc5c5c840297afc33ee64a57b91e9dee48c202535906fc916cde109d. Against assisted RL-WP on 1,000 development pairs: 49.375%, CI 47.6-51.1256%, +0.49 points. Shared-pair difference from seed 1: +1.55 percentage points, CI -0.75 to +3.9. Direct duel with seed 1 on 10,000 pairs: 49.975%, CI 49.4025-50.5225%, -0.0007 points. No established difference or baseline superiority. Reports eval_5000_wp.json, difference_vs_seed1.json and eval_5000_vs_seed1.json under the completed run. Training retains the verified source migration documented above.

Lambda 0.99 reached its prespecified 1,000-update endpoint: 131,072,000 decisions / 8,847,828 rounds / 1,127,084 new paper8 matches, excluding 2,245,436 initialization matches. Against matched lambda-0.95 control: 48.12%, CI 47.5699-48.6725%, -1.9771 points, 10,000 pairs. Assisted RL-WP: 48.75%, CI 45.3333-52.0833%, -1.27 points, 300 pairs. Shared-pair difference from control: -0.75 percentage points, CI -5.1667 to +3.5; point difference -2.6267, CI -5.2835 to -0.0766. Reports runs/relational8_lambda99_seed50001/ab_1000.json, eval_1000_wp.json and difference_vs_control.json. Reject lambda 0.99 for further extension; retain 0.95. This inference is limited to the matched configuration/initialization/seed studied.

## Lower-entropy controlled screen

configs/relational8_entropy005.json changes only entropy weight from 0.02 to 0.005 versus the resolved relational control config. Same initialization runs/relational8_v2/update_00002000.pt, seed 50,001, fresh optimizer/RNG/counters, lambda 0.95 and captured rollout/minibatches. Target 1,000 new updates / 131,072,000 decisions, snapshots every 250, runs/relational8_entropy005_seed50001. Hypothesis: reduced exploration regularization allows further refinement; no measured advantage yet. Compare against control update 1,000 on native 10,000 paired deals plus assisted RL-WP on 300 shared development pairs. Do not adopt or extend based on midpoint results. This deliberately matched training seed is a controlled screen, not independent replication. Pretraining and duplicated smoke updates are excluded from new experiment counters and the twelve-round budget. Final-test deals remain unused.

Initialized three-update smoke in runs/relational_entropy005_smoke completed 393,216 decisions / 3,065 matches with finite losses and gradients. This validates execution only; duplicated smoke learning is excluded from the controlled run.

Entropy screen started from committed e626ac1 with the planned initialization/config/seed/1,000-update target. Advancing finite metrics verified. Final entropy and relational-update-3,000 evaluation watchers are queued; STATUS.md records their sessions, process guards and output paths. These are pending experiments, not strength results.

## Larger koi-substitution comparisons across two training seeds

Using deterministic endgame-assisted inference and the same 1,000 pairs against assisted RL-WP, seed-1 MLP cards + published RL-WP koi scores 50.825%, CI 49.075-52.65%, +1.481 points. Paired improvement over the original seed-1 MLP is +3.0 percentage points, CI +1.5 to +4.55; point improvement +0.506, CI -0.3980 to +1.3991. Independent seed-10,001 MLP cards + published koi scores 51.175%, CI 49.375-53.025%, +0.794 points. Paired improvement over its original MLP is +1.8 percentage points, CI +0.3 to +3.375; point improvement +0.304, CI -0.56 to +1.1830. Both win-rate differences exclude zero, making koi substitution a supported improvement for these tested MLPs. Neither public-baseline confidence interval excludes 50%. Do not claim that either standalone MLP learned this hybrid strength or that this establishes the globally best policy.

Reports eval_5000_wp_koi_1000.json and difference_wp_koi_1000.json under runs/ppo8_v3 and runs/ppo8_seed10001_captured. Source 903a0fe; package unchanged since 142c265. First 300 seed-1 paired records match the previous 300-pair hybrid pilot exactly; this is an extension, not an independent resample. Both reports' pair counts, component identities, development split and assistance settings checked. The independent training seeds share evaluation deals and the same published koi component; their 2,000 total pair records must not be pooled as 2,000 independent deals.

Next endpoint: 3,000 additional development pairs for each hybrid with --seed 1001000 --pairs 3000 --koi-agent rl-wp --opponent rl-wp-assisted --endgame --deterministic, outputs eval_5000_wp_koi_new3000.json under each run. These deals do not overlap the earlier public comparison but have appeared in checkpoint-versus-checkpoint development duels; final-test data remains untouched. Both evaluations started and advancing pair counts verified. Keep the prespecified endpoint and report separately from the first 1,000 pairs. This larger evaluation determines whether the composite merits final held-out validation and motivates further koi-specific learning work; no default deployment change yet.

## Completed relational, entropy and larger hybrid endpoints

Relational continuation ended at 3,000 new updates / 393,216,000 decisions / 26,636,332 rounds / 3,401,171 matches, excluding 2,245,436 pretraining matches. Checkpoint SHA 2d4d50a489230ec0080515baf3338957acc6708f839bd92fb4f96df41d971eae matches eval metadata and final counters. Parent duel: 51.27%, CI 50.72-51.825%, +0.8686 points, 10,000 pairs. Assisted RL-WP: 47.875%, CI 46.075-49.625%, -0.731 points, 1,000 pairs. Difference versus update 1,500 on shared public pairs: -0.675 percentage points, CI -2.825 to +1.475; point difference -1.538, CI -2.8551 to -0.2640. Reports eval_3000_vs_parent.json, eval_3000_wp.json and difference_3000_vs_1500.json in runs/relational8_fullgraph_seed50001. Do not extend unchanged; parent gains have not established public-baseline superiority.

Entropy-0.005 screen completed its prespecified 1,000 updates / 131,072,000 decisions / 8,916,621 rounds / 1,136,109 new matches. SHA 03490c82b1ffe5cc56a44f4fc573fabb927bbb0a03ff87c04cc4ca7930d7f51f and final counters verified against actual checkpoint/metrics. Versus entropy-0.02 control at matched decisions: 49.2925%, CI 48.745-49.8375%, +0.1617 points (point CI -0.16741 to +0.4738), 10,000 pairs. Against assisted RL-WP: 48.0833%, CI 44.75-51.3333%, -0.14 points, 300 pairs. Shared-pair difference from control: -1.4167 percentage points, CI -5.5833 to +2.75; point difference -1.4967, CI -3.8668 to +0.8002. Reports ab_1000.json, eval_1000_wp.json and difference_vs_control.json under runs/relational8_entropy005_seed50001. Reject extension; retain entropy 0.02. A point estimate above zero in direct point difference does not override the win-rate objective.

Additional hybrid evaluation completed 3,000 pairs per seed on development range 1,001,000-1,003,999. Seed-1 cards + RL-WP koi: 49.9417%, CI 48.9083-50.9669%, +0.2433 points (CI -0.3647 to +0.8841). Seed-10,001 cards + RL-WP koi: 50.6833%, CI 49.6667-51.6919%, +1.4667 points (CI +0.8207 to +2.1041). Neither win-rate interval excludes 50%; do not promote either as superior. Reports eval_5000_wp_koi_new3000.json under the two MLP runs. These deals are disjoint from earlier public comparisons but previously used in native development duels. Do not pool shared deals from the two training seeds as independent samples. This supplements the replicated positive substitution differences on the first 1,000 pairs, without establishing the overall strength gate.

All relational/entropy/hybrid trainer and evaluation processes have ended; expected reports are present. No final-test seeds were used. Next method work should address opponent diversity with optional frozen-history matchups, keeping off-policy opponent actions out of learner PPO updates and preserving checkpoint/budget invariants. This implementation is pending, not evidence of a successful method.

## Scale update-12,000 audit and comparison

Immutable runs/ppo12_scale_seed30001_captured/update_00012000.pt SHA 8fb7bcaf7c64717c451a16878e008693506aab64468eaf3f1916bc3bf8bae14e contains 17,324,780 current research12 matches / 207,908,627 rounds / 3,145,728,000 decisions. Adding only initialization ancestry gives 18,325,244 research12 matches including 1,000,464 transfer matches; paper8 pretraining 11,141,556 is separate. Report budget_12000.json. Resume ancestry is not counted twice and duplicated diagnostic updates are excluded.

Update 12,000 versus transfer initialization on 10,000 paired research12 deals: 51.42%, CI 50.87-51.97%, +1.3174 points, exactly 20,000 matches / 240,000 rounds. Shared-pair difference versus update-7,000 results against that same parent: +0.2025 percentage points, CI -0.5226 to +0.985; point difference +0.1028, CI -0.4404 to +0.6339. This supports superiority to initialization, but does not establish additional improvement since update 7,000 or superiority to paper8 RL-WP. Reports eval_12000_vs_parent.json and difference_12000_vs_7000.json. Source 121e2fd, unchanged package SHA. Scale training remains live toward 100M new complete matches.

## Fixed historical-opponent PPO implementation and planned screen

Optional opponent_fraction leaves current-policy self-play as the default. A fixed subset of environments uses a frozen checkpoint, with learner seats alternating only on completed-match resets. Opponent actions are excluded from learner PPO policy/entropy/belief/KL terms and advantage-normalization statistics. Its current joint-policy value is learned with a separate observation-only scalar critic rather than taken from the historical self-play value head. Signed GAE spans all simulator decisions and match boundaries remain the only return termination. The role critic, learner and Adam all participate in CUDA-graph warmup rollback. Frozen inference evaluates only the assigned environment prefix. Checkpoints embed frozen weights/config/provenance and restore without the original file. The budget auditor reports simulator and learner decisions separately; frozen-opponent pretraining is a disclosed dependency, not learner initialization ancestry.

Full suite: 89 passed, no skips, two expected upstream performance warnings. Dedicated checks cover match-fixed seats and round boundaries, legal opponent actions, complete-match outcome signs and unfinished-tail bootstrap, zero actor gradients from opponent-only rows even with extreme stored log probabilities, learner-only advantage normalization, immutable frozen weights, CPU/GPU exact restart after removing the original opponent file, all trainable optimizer tensors/RNG, and eager/captured full updates. The actual prior source 499990c in ignored build/compat499990c produced the same next two default-self-play updates as current code for MLP/FP32 and relational/BF16. Model/Adam/environment/counters/all RNG/metrics agree exactly; runs/frozen_default_equivalence.json and frozen_default_* artifacts. These diagnostic updates are not added to training budgets.

Planned controlled screen: configs/relational8_history20.json, seed 50,001, learner initialization and fixed opponent both runs/relational8_v2/update_00002000.pt. 204/1,024 environments use the frozen opponent. Resolved config changes only opponent_fraction 0 -> 0.2 from the previous control. However the method also introduces an auxiliary role critic and reduces learner-action count per simulator step; report this as a candidate screen, not an isolated causal estimate of diversity. Target 1,000 new updates / 131,072,000 simulator decisions with actual learner_decisions recorded, snapshots every 250, runs/relational8_history20_seed50001. Primary comparisons: control update 1,000 on 10,000 native pairs and 300 assisted-RL-WP shared development pairs. No midpoint adoption; any success requires an independent training seed and untouched final test. No strength or equilibrium guarantee is presumed.

Initialized three-update production-shape smoke runs/relational_history20_smoke completed 3,062 matches / 26,283 rounds / 393,216 simulator decisions / 353,918 learner decisions with finite metrics. It validates execution only. Duplicated smoke learning is excluded from the controlled run's budget; 2,245,436 paper8 initialization matches remain separate from new counters and the research12 requirement.

Additional coverage passed two targeted full-update comparisons including relational/BF16 learner and frozen opponent. Actual production-shape initialized smoke checkpoint update 3 then produced exactly identical next-update metrics, learner/role-critic/frozen weights, Adam, environment, counters and all RNG states with ordinary versus captured collection. Report runs/frozen_production_equivalence.json; diagnostic counters reach 4,117 matches / 524,288 simulator decisions / 471,883 learner decisions but are not added to the planned experiment. Verified package SHA 3f1107914d1c6c5019f1f684c6b7149deaf4982e9f57288533c256747243f72e.

The historical-opponent screen started from committed 14d62c0 with the planned initialization/frozen checkpoint/config/seed/1,000-update target. Process and advancing finite metrics verified; source SHA 3f1107914d1c6c5019f1f684c6b7149deaf4982e9f57288533c256747243f72e. A final-evaluation watcher is queued for the immutable endpoint; STATUS.md records handles and output paths. The captured scale baseline remains live on its existing source and was not restarted.

## Frozen-opponent identity comparison

A second history-mixture screen will keep configs/relational8_history20.json, learner initialization runs/relational8_v2/update_00002000.pt, seed 50,001 and the 1,000-update target. It replaces only the fixed opponent with the independently trained MLP checkpoint runs/ppo8_seed10001_captured/update_00005000.pt (SHA edc44213bc5c5c840297afc33ee64a57b91e9dee48c202535906fc916cde109d). This compares opponent identity, including its architecture/training history/behavior, not diversity alone. The first historical-parent screen keeps its original endpoint unchanged. No interim strength result was used to define this arm.

Planned run runs/relational8_history_mlp_seed50001; snapshots every 250. Target 131,072,000 simulator decisions with actual learner_decisions separately recorded. The frozen MLP's 11,141,428 paper8 training matches are a disclosed dependency and are not added to learner training ancestry; learner pretraining remains 2,245,436 paper8 matches. Primary evaluations: 10,000 direct pairs against current-self-play control update 1,000 and 300 shared public pairs against assisted RL-WP. Compare final history-mixture arms on 10,000 direct pairs to study fixed-opponent choice. No midpoint adoption, no untouched final-test deals, no presumption that this opponent is better. Independent replication is required for a selected improvement.

Mixed-architecture fixed-opponent smoke completed three initialized production-shape updates with finite losses/gradients: 3,057 matches / 26,308 rounds / 393,216 simulator decisions / 353,863 learner decisions, runs/relational_history_mlp_smoke. This validates execution only and its duplicated learning is excluded from the planned experiment.

## Scale update-22,000 improvement and budget audit

Immutable checkpoint SHA db58b75c2aca40f5a08bc9f9cc57c6149e5c3b9671388ece76c301f491476279 contains 31,771,972 research12 matches / 381,274,860 rounds / 5,767,168,000 simulator and learner decisions. Initialization ancestry adds only 1,000,464 research12 matches, giving 32,772,436; the 11,141,556 paper8 matches remain separate. Report runs/ppo12_scale_seed30001_captured/budget_22000.json. No duplicate resume counting or attribution of frozen-opponent pretraining.

On 10,000 paired development deals against transfer initialization, update 22,000 scores 52.275%, CI 51.7325-52.8375%, +1.6925 points (CI +1.2849 to +2.0916). Shared-pair score difference from update 12,000 against that same opponent: +0.855 percentage points, CI +0.1375 to +1.5925. The corresponding point-difference change is +0.3751, CI -0.1569 to +0.9131. This establishes development improvement against the fixed parent over that interval, not general superiority or a paper8 RL-WP result. Reports eval_22000_vs_parent.json and difference_22000_vs_12000.json; evaluator source bbb58ec, package SHA 3f1107914d1c6c5019f1f684c6b7149deaf4982e9f57288533c256747243f72e. Training remains on the unchanged original captured source.

The MLP fixed-opponent arm started from committed 642e9b2 with unchanged package source and the prespecified initialization/seed/1,000-update target. Process/advancing finite metrics verified. Its first three updates exactly reproduce the smoke's losses/gradients/counters, excluding timings; duplicated smoke learning remains excluded. Final control/public/opponent-identity evaluations are queued, with process guards and artifact paths in STATUS.md. No interim result is a strength claim.

## Fixed-opponent throughput entry point

The full-training benchmark now accepts --frozen-opponent together with a positive-opponent_fraction --training-config. It includes opponent inference and optimizer work, excludes five warmup updates, and reports both total simulator decisions and learner decisions/rates. The fixed opponent's metadata is recorded using the same loader/provenance as training; the benchmark starts a fresh learner. Existing ordinary benchmarks report equal learner/simulator decision counts.

CLI checks rejected all three invalid pairings (opponent without training config, positive fraction without opponent, zero fraction with opponent). A CPU ordinary benchmark reported exactly two measured updates / 128 decisions after warmup. Production-shape command `python -m koikoi.benchmark --training-config configs/relational8_history20.json --frozen-opponent runs/ppo8_seed10001_captured/update_00005000.pt --steps 3` measured 7.7419 seconds / 3,207 matches / 25,296 rounds / 393,216 simulator decisions / 351,369 learner decisions (50,790.69 simulator and 45,385.42 learner decisions/s). Actual checkpoint hash was checked. Three trainers were concurrently active; this is a functional throughput check, not an isolated comparison or strength evidence. Reports runs/history_mlp_training_benchmark.json and runs/frozen_benchmark_validation.json. These benchmark updates are excluded from all experiment budgets. Learning code and ongoing training settings were unchanged.

## Completed fixed-opponent endpoints

Both prespecified 1,000-update arms completed, and both evaluation watchers exited successfully. Learner initialization, continuation seed 50,001, opponent fraction 0.2, architecture and optimizer settings were held fixed. Each used 131,072,000 simulator decisions. The historical-parent arm completed 1,133,486 matches / 8,888,031 rounds / 118,021,781 learner decisions. The independent-MLP-opponent arm completed 1,127,533 matches / 8,829,434 rounds / 117,967,392 learner decisions. This is matched simulator budget, not identical learner-action budget; both share a learner initialization/seed and are not independent training replications.

Historical-parent arm: versus current-self-play control update 1,000, 50.0375% (95% paired-bootstrap CI 49.4899-50.5675%), +0.226 points (CI -0.0933 to +0.5442), 10,000 pairs. Versus assisted RL-WP, 47.6667% (44.4146-50.9167%), -0.5933 points (CI -2.7401 to +1.4667), 300 pairs. Shared public-pair score difference from control is -1.8333 percentage points (CI -6 to +2.3333), point difference -1.95 (CI -4.5301 to +0.5871).

Independent-MLP-opponent arm: versus control, 50.4475% (CI 49.8949-50.99%), -0.0092 points (CI -0.3346 to +0.3085), 10,000 pairs. Versus assisted RL-WP, 46.9167% (43.5833-50.1667%), -1.0067 points (CI -2.9468 to +0.9002), 300 pairs. Shared public score difference from control is -2.5833 percentage points (CI -6.5833 to +1.5); point difference -2.3633 (CI -4.6635 to -0.0599). Directly versus the historical-parent arm, 50.2575% (CI 49.7324-50.7951%), -0.0495 points (CI -0.3626 to +0.2682), 10,000 pairs.

Decision: neither mixture establishes a win-rate gain over ordinary self-play or public-baseline superiority. Do not extend these conditions or select one opponent based on point estimates. Keep the feature optional and default off. These results do not establish that all historical mixtures fail. The public intervals are wide, and the negative point-difference interval in the MLP arm does not establish a win-rate decline. Final-test seeds remain unused.

Immutable endpoint hashes: runs/relational8_history20_seed50001/update_00001000.pt SHA 199602a1f19a9c3acfbdd7f3d8791d0118445561d6e87894e7fe75aeefd879ca; runs/relational8_history_mlp_seed50001/update_00001000.pt SHA 61ba8eac83bedcb59abf859cb9900764c4561703dd23bce3a87cd3a8e97c5266. Source learning package SHA 3f1107914d1c6c5019f1f684c6b7149deaf4982e9f57288533c256747243f72e. Evaluations ran at 2fa1fb3, whose only package change was benchmark reporting.

Each directory contains ab_1000.json, eval_1000_wp.json, difference_vs_control.json and budget_1000.json; the MLP arm additionally contains ab_vs_history20.json. Audit runs/frozen_endpoints_audit.json verifies actual snapshot hashes, final-metric counters, finite metrics, report checkpoint identity, deterministic/endgame settings, pair counts and development seeds. Initialization-lineage totals: parent arm 3,378,922 paper8 matches / 393,216,000 simulator / 380,165,781 learner decisions; MLP arm 3,372,969 paper8 matches / 393,216,000 simulator / 380,111,392 learner decisions. Each includes the same 2,245,436-match learner pretraining. Opponent-only dependencies are not added again. These separate model lineages cannot be added as disjoint project learning, and none count toward research12.

## Next investigation: value estimation under partial observation

A candidate next screen is a separate scalar PPO critic, first using observation features alone and then observation plus privileged training features. This can distinguish a change in shared-network optimization from an effect of hidden information. Existing CentralQ/Q-boosting results are a different algorithm and do not decide this comparison. Actor and search inference must continue to use only their legal observations; no state-derived action correction is proposed.

[Baisero and Amato, AAMAS 2022](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p44.pdf) show why substituting a state-only critic under partial observability can introduce bias, and analyze history-state critics as an alternative. This motivates examining the conditioning information; it does not establish an improvement for this game's finite, compact history representation or an unbiased clipped-PPO update. Treat any such implementation as an empirical hypothesis, with separate parameter paths, restart/capture validation, a prespecified matched-budget screen and independent confirmation if positive. No new critic implementation or training arm has started at this milestone.

## Separate PPO return critic implementation and prespecified comparison

Implemented ppo_value=observation/privileged, default shared. The scalar return critic uses identical 656-input, width-256 MLP parameters in both controls, with 96 zero-padding features versus actual opponent-hand membership and remaining-stock ranks. Its output starts at zero and initialization does not shift policy sampling RNG. It supplies all collection values and tail bootstrap for signed GAE. The deployed actor/value model remains observation-only; its value head is still trained on the detached return target. The auxiliary MSE receives the same value_weight as the actor value MSE. Both trainable networks share Adam and gradient clipping, so comparison with the shared baseline changes more than hidden information; observation versus privileged controls hold those changes fixed. The critic_loss metric reports auxiliary MSE. Pure-self-play PPO only; unsupported Q-boosting/frozen-opponent combinations are rejected. See DESIGN.md for details and limitations.

Validation: 101 tests passed without skips, including CUDA and pinned upstream, with only two expected upstream Transformer warnings. New coverage checks hidden-stock observation invariance, zero private features in the observation control, actual private hand encoding, critic-gradient parameter separation, deployment compatibility, complete-match signs, correct unfinished-tail bootstrap, both value heads learning, CPU/FP32/BF16 exact restart, Adam/environment/counters/all RNG states, and eager versus captured minibatches/full updates. A test initially retained a parameter clone's autograd graph, causing a CUDA capture stream mismatch; changing that diagnostic snapshot to detach().clone() resolved it. Production code did not need a capture workaround.

Default self-play still exactly reproduces two historical-source updates (499990c artifacts), for MLP/FP32 and relational/BF16: weights, Adam, games, counters, CPU/CUDA/Python/NumPy RNG and metrics. Report runs/ppo_value_default_equivalence.json. Initialized production-shape smoke (1024 environments, horizon 128, relational64/BF16, critic256, both captures) completed three updates / 393,216 decisions in each arm: observation 3,075 matches / 26,426 rounds, privileged 3,074 matches / 26,412 rounds. First collection counts coincide at 894 matches / 8,455 rounds because both critics start at zero and actor/RNG are identical. Losses diverge during optimization as expected from their different critic inputs.

From each three-update checkpoint, captured versus ordinary collection yields an exactly identical fourth update, including actor/critic weights, Adam, environment, counters, all RNG and metrics; report runs/ppo_value_production_equivalence.json. Diagnostic counters 4,133 and 4,127 matches are not experimental training budgets. Smoke directories runs/relational_value_observation_smoke and runs/relational_value_privileged_smoke. Verified learning package SHA 2cddf9bbe6d644872310f54383d715950da565725c81cb4523b6aadf09e2ff73.

Prespecified screen: configs/relational8_value_observation.json and configs/relational8_value_privileged.json differ only in ppo_value. Both initialize actor weights from runs/relational8_v2/update_00002000.pt, seed 50,001, fresh Adam/RNG/simulator/counters, lambda 0.95, entropy 0.02 and 1,000 new updates / 131,072,000 decisions. New critics start fresh; historical critic weights are not transferred. Expected run directories runs/relational8_value_observation_seed50001 and runs/relational8_value_privileged_seed50001, immutable snapshots every 250. Each retains 2,245,436 paper8 pretraining matches separately. Duplicated smoke/equivalence updates are excluded; paper8 contributes nothing toward the research12 target.

Endpoint evaluations: each arm versus original shared-critic control update 1,000 on 10,000 paired development deals; each versus assisted RL-WP on the same 300 development pairs and paired differences from control; privileged versus observation on 10,000 direct pairs. Use deterministic endgame-assisted inference, development seeds beginning at 1,000,000 and no search. Do not select or extend based on midpoint results. Interpret direct and public results jointly; any selected improvement requires larger development evaluation, independent training replication and untouched final-test validation. Neither a critic loss reduction nor a direct control win alone establishes the public-baseline gate. Training starts only after committing this verified implementation.

Both separate-critic screens started from committed 16080dd with the prespecified configuration, initialization and 1,000-update targets. Process identity and advancing finite metrics verified. Each arm's first three updates exactly reproduce its isolated smoke excluding timings; duplicate smoke updates remain excluded. An endpoint watcher is live with process guards, queued budgets and direct/public/shared-pair comparisons, including privileged-versus-observation. Operational handles and paths are in STATUS.md. The research12 scale trainer remains live on source 142c265 without restart.

## Longer koi-rollout screen and captured search execution

After the replicated positive koi-substitution diagnostic, a separate inference screen was specified for the independent-seed MLP (runs/ppo8_seed10001_captured/update_00005000.pt): uniform hidden proposals, 128 particles, horizon 192, koi decisions only, paired standard-error margin 2.5, deterministic endgame-assisted deployment versus assisted RL-WP. The endpoint remains 100 development pairs, seeds 1,000,000-1,000,099. This is a small screen, not a held-out result or an adoption decision. Horizon 192 is a finite cutoff with observation-only value bootstrap where needed; it is not guaranteed full-match search. Both seats' rollout policies remain the same observation-only MLP, not an oracle or a learned public-opponent model.

Matching CUDA no-search control completed 100 pairs / 200 matches: 43.25% (CI 37-49.75%), -4.05 points; all 100 pair records exactly match the first 100 records of the existing 1,000-pair CPU evaluation. Report runs/ppo8_seed10001_captured/eval_5000_cuda_control100.json. This prefix is deliberately reused development data and its low score does not contradict the full 1,000-pair result. GPU decision latency was 32.97 ms under concurrent trainers/evaluation, so this is not isolated hardware latency. The PowerShell wrapper returned status 1 after rendering the upstream warning as a NativeCommandError, but the actual evaluator produced its complete report, whose counts/records/settings were independently checked.

The eager long-search process was too slow for the intended comparison. Its log runs/koi_search192.log reached 20 pairs before being explicitly stopped after validating a faster equivalent execution path; no endpoint report exists and no partial strength claim is made. The same 100-pair endpoint will be replayed from the beginning with capture. This execution change preserves particles/horizon/confidence/deals/model; it is not an interim-outcome-driven algorithm change. Partial aggregate scores (10 pairs 0.275, 20 pairs 0.3375) are retained in the log for later replay consistency, not for selection.

Implemented optional --search-cuda-graph for CUDA search. It captures only the tensor rollout loop, caching by particle-branch batch size and profile; root sampling, actions and public root sign are refreshed each decision. Three warmup rollouts restore simulator/RNG state before actual replay. Report search_preparation_decisions separately from logical search_simulated_decisions; neither counts toward training. First-use latency includes preparation. CUDA capture is off by default, and search remains off by default pending strength evidence.

Validation: targeted search/evaluation suite 15 passed, then expanded search capture tests 5 passed. Coverage includes deterministic/sampled policies, both actor seats, changing legal-action counts, changed model tensors, public endgame bypass, terminal outcomes retained across reset work, unfinished tails, actual particle values, final simulator states, all search counters, proposal/CPU/CUDA RNG and hidden-world invariance. CLI rejects capture without CUDA search; a one-pair smoke records exactly 10 searched decisions / 640 logical transitions / 192 warmup transitions. This smoke is functionality only. Full suite subsequently passed: 106 tests, no skips, only two expected upstream Transformer warnings. Ruff passed. No native rules or training algorithm changed in this execution milestone.

Actual 128-particle/horizon-192 MLP comparison against source 22b45d8: all four probe actions, final simulator states, counters and RNG agree exactly, across both root seats. Report runs/search_capture_production_equivalence.json; 196,608 logical transitions and 147,456 preparation transitions. Captured first-use time 10.6567 s versus eager 4.0712 s; subsequent probes captured 0.2310/0.2323/0.2387 s versus eager 3.4289/3.4275/3.4121 s. These concurrent-workload measurements demonstrate a local execution benefit, not isolated latency or strength. Package SHA 4ff30860b37e14ec4e769814c54e07db296cc3554c13bc4920b6042103168674.

## Critic midpoint budget checks and queued scale-halfway audit

Immutable critic update-500 budgets: observation SHA dbf06fbefe8d415a9f59bd36945cac358d583af60dafe23203ae8e23847cdf05, 567,128 new matches / 4,443,457 rounds / 65,536,000 decisions; privileged SHA c618a25fedaf020f499c1078d3bcf0a20c3bc83fff128d9cbed873c0fbd7e86f, 566,207 new matches / 4,436,417 rounds / 65,536,000 decisions. Adding the shared 2,245,436-match initialization yields separate lineages of 2,812,564 and 2,811,643 paper8 matches, 327,680,000 decisions each. Reports budget_500.json in each run. Counters match metrics, auxiliary tensors are finite and learned, and all Adam tensors have 48,000 optimizer steps. These are audit checks, not midpoint strength evaluations.

Queued research12 checkpoint-35,000 audit and evaluation, near the 50M-new-match milestone: budget_35000.json, 10,000 deterministic guarded development pairs against runs/ppo12_transfer_v1/update_00000693.pt into eval_35000_vs_parent.json, then shared-pair difference versus update 22,000 into difference_35000_vs_22000.json. Watcher guards the already-running scale process and does not restart it. A parent-relative result is still separate from paper8 public-baseline superiority. Operational handles are in STATUS.md.

## Completed separate-critic comparison

Both 1,000-update endpoints and all queued evaluations completed successfully. Observation-only auxiliary critic: 1,133,603 new matches / 8,873,596 rounds / 131,072,000 decisions; SHA b409de9136c4f3dca89c693f63233b86b12928fbca000eabadc76e620c0b8f81. Privileged critic: 1,130,954 matches / 8,848,700 rounds / 131,072,000 decisions; SHA 7ed8ff49fac387e9d08da72d54afaaa0f1eb4cde25ce5d4d93eaf8edf325fd1f. Actual hashes, final metrics, learned finite critic tensors, 96,000 Adam steps, report checkpoint identities, deterministic/endgame settings and pair ranges were verified; consolidated report runs/ppo_value_endpoints_audit.json. Including initialization gives separate paper8 lineages of 3,379,039 and 3,376,390 matches, 393,216,000 learner/simulator decisions each; budget_1000.json in each directory. No contribution to research12.

Observation versus shared-critic control: 50.3775% (95% paired-bootstrap CI 49.84-50.91%), +0.3733 points (CI +0.0550 to +0.6931), 10,000 pairs. Versus assisted RL-WP: 48.4167% (45.25-51.5833%), +0.2867 points (CI -1.5167 to +2.0867), 300 pairs. Shared-public difference from control: -1.0833 percentage points (CI -5.0833 to +3), -1.07 points (CI -3.4368 to +1.3535).

Privileged versus shared-critic control: 49.8675% (CI 49.325-50.41%), -0.2686 points (CI -0.6076 to +0.0603), 10,000 pairs. Versus assisted RL-WP: 46.9167% (43.5-50.3333%), -0.4 points (CI -2.3101 to +1.5267), 300 pairs. Shared-public difference from control: -2.5833 percentage points (CI -6.8333 to +1.6667), -1.7567 points (CI -4.1603 to +0.5635).

Privileged versus observation directly: 49.805% (CI 49.2624-50.3425%), -0.5607 points (CI -0.8844 to -0.2268), 10,000 pairs. Shared-public difference: -1.5 percentage points (CI -5.5833 to +2.5021), -0.6867 points (CI -2.9734 to +1.4635). Reports ab_1000.json, eval_1000_wp.json, difference_vs_control.json in each run; ab_vs_observation.json and difference_vs_observation.json in the privileged run. All development only. Both learners retain source 16080dd / package 2cddf9bb..., while evaluators used package 4ff30860... during the validated search-execution changes; the no-search policy and native transitions were unchanged.

Decision: neither separate critic establishes a win-rate gain over shared PPO, and privileged information does not improve the observation-only control. Positive observation-arm point difference and negative privileged-arm direct point difference do not replace the win-rate objective. Do not extend either condition unchanged; retain ppo_value=shared by default. No independent replication or held-out evaluation is warranted for these unselected conditions. This does not prove all privileged-critic methods ineffective.

## Scale update-35,000 halfway audit

Immutable scale snapshot SHA 31ca1e1af2c49afa5ddaac675834fe40a7ba75dff5126bfc7518724cfa5dd681: 50,546,968 new complete research12 matches / 606,574,785 rounds / 9,175,040,000 decisions. Initialization adds 1,000,464 research12 matches for a lineage total of 51,547,432; paper8 pretraining 11,141,556 remains separate. Report budget_35000.json. Actual file hash, checkpoint/evaluation counters, 10,000 paired development seeds and 20,000 evaluated matches / 240,000 rounds were verified.

Versus the research12 transfer initialization, update 35,000 scores 52.4425% (CI 51.88-53.0126%), +1.658 points (CI +1.2370 to +2.0830), 10,000 pairs. Shared-pair difference from update 22,000: +0.1675 percentage points (CI -0.565 to +0.8926), point difference -0.0345 (CI -0.5391 to +0.4877). Reports eval_35000_vs_parent.json and difference_35000_vs_22000.json. This retains a parent-relative gain but establishes no further improvement since update 22,000 or superiority to the published paper8 model. Watcher completed successfully. Continue the prespecified 100M-new-match target on its existing source; do not restart or count any diagnostic/search work as training.

## Captured 100-pair koi-search endpoint and fixed extension

The committed capture implementation (683f6a3, package 4ff30860...) completed the prespecified 100 pairs. First 10/20-pair aggregate scores reproduce the stopped eager log exactly. Against assisted RL-WP: 45.0% (CI 38.75-51.5%), -2.81 points (CI -6.9603 to +1.4). Difference from the matched CUDA no-search control: +1.75 percentage points (CI -0.5 to +4.25), +1.24 points (CI -0.04 to +2.7503). Report eval_5000_koi_search192_captured100.json and difference_koi_search192_100.json under runs/ppo8_seed10001_captured. Neither improvement nor public superiority is established.

924 searched decisions, 19 action changes, 45,416,448 logical search transitions plus 147,456 preparation transitions, all excluded from training. Mean candidate latency 12.60 ms, p95 100.48 ms including first-use preparation. This evaluation had fewer concurrent trainers than the 32.97-ms no-search control, so those means are not a valid speed comparison between search and no search. Endpoint source/settings/seed range/counter arithmetic verified; runner session 85467 completed successfully.

The favorable but inconclusive paired estimate warrants a fixed additional 900-pair development evaluation, seeds 1,000,100-1,000,999, preserving the same model, 128 uniform particles, horizon 192, margin 2.5, deterministic/endgame assistance and CUDA capture. These deals were used in earlier baseline development evaluations, but not this search screen. Evaluate matching CUDA no-search control first, then search; report their additional-900 results and the prespecified combined 1,000-pair endpoint (original 100 included). This is a nested development extension, not independent replication, and the low original-prefix score must remain in the full result. No further setting changes or selection from intermediate outcomes. Any selected improvement still needs independent training-seed validation and untouched final testing.

Expected outputs under the same MLP run: eval_5000_cuda_control_new900.json, eval_5000_koi_search192_new900.json and koi_search192_1000_summary.json. Runner runs/run_koi_search192_extension.py checks disjoint contiguous ranges, matching inference/search settings and checkpoint identities within segments, hashes source reports, compares both GPU controls with prior CPU records, and computes the full paired difference. It does not invent a pooled latency percentile. The aggregation script hash is stored in the summary. This extension will start from the committed protocol; it is not a strength result yet.

The additional-900 control/search pipeline started from committed protocol 53a4a45, unchanged package 4ff30860..., with its fixed settings and aggregation endpoint. Runner and control-evaluator process identities verified; control pair counts are advancing. Operational session/process/log paths are in STATUS.md. Do not infer a method decision from intermediate scores.

## Relational koi-substitution diagnostic

To check whether the earlier MLP koi-substitution finding also appears with the relational representation, an additional diagnostic fixed the card policy to runs/relational8_fullgraph_seed50001/update_00001000.pt and replaced only koi decisions with pinned published RL-WP. This 300-pair diagnostic was declared in the work log before execution, while the fixed 900-pair search extension was running. It did not change that extension's model, settings, deal range or aggregation. This documentation records the completed diagnostic; it is not a retrospective claim of a separately committed preregistration.

Commands (CPU default, development seeds 1,000,000-1,000,299):

```powershell
./.venv/Scripts/python.exe -m koikoi.evaluate --agent runs/relational8_fullgraph_seed50001/update_00001000.pt --koi-agent rl-wp --opponent rl-wp-assisted --profile paper8 --pairs 300 --seed 1000000 --endgame --deterministic --output runs/relational8_fullgraph_seed50001/eval_1000_wp_koi300.json
./.venv/Scripts/python.exe -m koikoi.compare runs/relational8_fullgraph_seed50001/eval_1000_wp_koi300.json runs/relational8_fullgraph_seed50001/eval_1000_wp.json --output runs/relational8_fullgraph_seed50001/difference_wp_koi300.json
```

Composite draw-adjusted win rate against assisted RL-WP: 51.4167% (95% paired-bootstrap CI 48.0833-54.75%), +1.66 points (CI -0.2967 to +3.5803), 300 pairs / 600 matches / 67,953 decisions. Full relational control on the same deals: 49.5% (46.25-52.75%), +1.3567 points. Shared-pair difference: +1.9167 percentage points (CI -0.9167 to +4.6667), +0.3033 points (CI -1.3268 to +1.86). Neither improvement nor public-baseline superiority is established. No default change or training method is selected from this result.

Actual checkpoint SHA 0cf7f198a04e5cb217650da2838cc4e509be6e83157e04241f833dfdac112296 agrees with both reports. Deterministic/endgame settings, CPU device, contiguous shared development seeds, means, bootstrap intervals and paired comparison were independently checked/recomputed. Audit runs/relational_koi300_audit.json stores report hashes. Diagnostic evaluator revision d1c0d1f, package 4ff30860b37e14ec4e769814c54e07db296cc3554c13bc4920b6042103168674; published revision 89d789565c7c36c84b74662e43001d960bf80901. The historical control evaluator used package 2bfd85f8..., with no-search actor/game behavior unchanged by the intervening optional training/search implementations. Runner runs/run_relational_koi_diagnostic.py completed successfully, session 10917 terminal.

This is an inference component diagnostic using the same development deals as earlier comparisons, not an independent training replication, held-out result or additional training budget. Published koi is an explicit dependency; composite strength cannot be attributed solely to our learned relational network. Mean candidate latency 0.5508 ms and p95 2.2138 ms were recorded under concurrent workloads, not an isolated deployment benchmark. Continue the already-fixed full-1,000 search endpoint before selecting a further method.

## Next scale checkpoint audit

Prespecified next scale endpoint: immutable update 52,000, expected near 75M new research12 matches based on current completed-match density. Audit actual counts rather than assume an exact 75M total. Run koikoi.budget, then koikoi.batch_evaluate versus runs/ppo12_transfer_v1/update_00000693.pt on 10,000 deterministic endgame-assisted development pairs (research12, seeds 1,000,000-1,009,999, default CUDA batch size 1,024). Compare the resulting report with eval_35000_vs_parent.json using koikoi.compare. Expected artifacts under runs/ppo12_scale_seed30001_captured: budget_52000.json, eval_52000_vs_parent.json, difference_52000_vs_35000.json and audit_52000.json.

Watcher runs/watch_scale52000.py waits for the atomic numbered snapshot while retaining a handle to the verified trainer process, and never restarts training. It rejects existing endpoint outputs, checks subprocess exits, then verifies actual checkpoint hash, metrics/counters, initialization budgets separated by profile, evaluation pair range and comparison pair range. This uses existing evaluation code with no training/search method change. The 100M-new-match target continues regardless of this intermediate result; parent-relative strength does not satisfy the public-baseline or final-test gates. Launch after committing this protocol.

Update-52,000 watcher launched from committed protocol 97366ec and confirmed live waiting on trainer 57432. Operational handles in STATUS.md. Meanwhile the search-extension additional-900 CUDA control completed (50.0556%). Its 900 records plus the original 100 exactly reproduce all 1,000 prior CPU records, combined score 49.375%. Actual MLP hash, complete contiguous seed coverage and inference settings verified. Runner advanced automatically to captured search without changing the prespecified protocol; no search-strength conclusion yet.

## Completed scale update-52,000 audit

The prespecified watcher completed all commands and its artifact audit. Immutable snapshot SHA 313f2299aba0b3b66252790ac718d614f6d2745d8404940e836b466995777768: 75,085,544 new research12 matches / 901,037,570 rounds / 13,631,488,000 decisions. Adding initialization gives 76,086,008 research12 matches; paper8 pretraining 11,141,556 remains separate. Actual hash, snapshot/metrics/report counters, lineage totals, deterministic/endgame settings and 10,000 contiguous development pair identities passed the recorded audit. Reports budget_52000.json, eval_52000_vs_parent.json, difference_52000_vs_35000.json and audit_52000.json under the scale run.

Versus transfer initialization: 52.1925% (95% paired-bootstrap CI 51.64-52.7676%), +1.9193 points (CI +1.4809 to +2.3449). Difference on the same deals versus update 35,000: -0.25 percentage points (CI -0.9675 to +0.4951), +0.2613 points (CI -0.2906 to +0.7973). No further strength improvement is established. The parent-relative advantage remains separate from public paper8 superiority. Session 51551 terminal; no duplicate audit or training restart. Continue the existing 100M-new-match target; next scale endpoint is the final target.

## Completed full-1,000 koi-search development endpoint

The fixed 900-pair extension and prespecified aggregation of original 100 plus new 900 completed. Full control: 49.375% (CI 47.6-51.1256%), +0.49 points (CI -0.6600 to +1.632). Full search: 50.025% (CI 48.2244-51.825%), +0.736 points (CI -0.43 to +1.8583). Paired search-minus-control difference: +0.65 percentage points (CI +0.125 to +1.2), +0.246 points (CI -0.057 to +0.558). Additional 900 alone: search 50.5833% (CI 48.7493-52.4444%), control 50.0556%; paired difference +0.5278 pp (CI 0 to +1.0556). The decision uses the prespecified full endpoint, not only the extension.

The paired score interval supports a small development improvement for this model, but is not adjusted for the preceding method search or nested sample expansion. This warrants fixed independent-training-model validation, not a confirmatory strength claim. Public-baseline superiority is not established: the full search interval includes 50%. Keep search off by default and leave final-test seeds unused.

Across both search segments: 230,709 total decisions, 9,096 searched decisions, 173 changed actions, 447,086,592 logical simulation transitions and 294,912 preparation transitions (separate graph preparation in each segment). None counts as training. Actual model/segment hashes, fixed search and guard settings, complete pair range, transition arithmetic, all control records versus prior CPU, recomputed score/point intervals and paired comparison checked in koi_search192_1000_audit.json. Summary koi_search192_1000_summary.json SHA d1393a850b94f36943aeee7e5ee69980fabe40fe0e2f30f8c2c692e86aeec38c under runs/ppo8_seed10001_captured. Runner session 76409 terminal, successful. Package unchanged at 4ff30860...; aggregation revision d0c5912. No pooled latency percentile is reported.

Prespecified replication: independently trained seed-1 MLP runs/ppo8_v3/update_00005000.pt, SHA d104ae00824aca114e6cd414c74d02670b9a8de5a009ad4fad557005f66233e5. Run the same 1,000 development pairs (seeds 1,000,000-1,000,999) against assisted RL-WP, deterministic/endgame-assisted on CUDA. Complete matching no-search control first, then uniform 128 particles / horizon 192 / margin 2.5 / koi-only captured search; no parameter tuning or intermediate selection. Compute paired difference and compare control records against the historical CPU result. This replicates across training models, not across independent evaluation deals; do not pool both seeds as 2,000 independent pairs. A positive result still requires larger development assessment and untouched final validation before a public-superiority claim.

Runner runs/run_koi_search192_seed1.py checks the actual checkpoint hash, rejects existing endpoint outputs, checks subprocess exits and validates final metadata/counters. Outputs under runs/ppo8_v3: eval_5000_cuda_control1000.json, eval_5000_koi_search192_captured1000.json, difference_koi_search192_1000.json and koi_search192_1000_audit.json. CPU near-tie differences are counted rather than assumed absent. Logs runs/koi_search192_seed1_control.log, runs/koi_search192_seed1_search.log and runs/koi_search192_seed1_compare.log. Launch once after committing this protocol.

## Final 100M scale endpoint protocol and independent count checks

The existing source-142c265 trainer stops after the first completed update whose new research12 match counter reaches 100,000,000, saving latest.pt atomically; its final update need not be a numbered checkpoint interval. Prespecified watcher runs/watch_scale100m.py retains a handle to that live process, waits for natural exit, requires exit code zero and at least 100M new matches, and verifies that the preceding metrics row was below target. It never starts or resumes training. A below-target or failed exit is an error for inspection, not success.

Before evaluation, verify actual simulator accounting: every environment has twelve rounds with bankruptcy disabled and has already reset any terminal state; sum of EPISODE equals the new complete-match count; 12 times that sum plus current ROUND indices equals all completed rounds, including partial ongoing matches; updates times environments times rollout horizon equals decisions. Model and Adam tensors must be finite and final checkpoint counters must equal final metrics. These stronger checks passed actual update-35,000 and update-52,000 snapshots, including 11,169 and 11,042 completed rounds within still-running matches respectively. Deliberately inconsistent match/round/decision counters and nonfinite model tensors were rejected. Report runs/scale_episode_accounting_audit.json. This validates the accounting routine, not an as-yet-unreached 100M endpoint.

After the trainer exits and checks pass, preserve the exact latest.pt bytes with exclusive creation as runs/ppo12_scale_seed30001_captured/final_100m.pt. Hash and audit that immutable copy; never evaluate a moving latest.pt. Run koikoi.budget to budget_100m.json. Keep initialization separate: exactly 1,000,464 prior research12 matches and 11,141,556 paper8 matches; neither contributes to the new-100M threshold.

Then run koikoi.batch_evaluate with final_100m.pt versus runs/ppo12_transfer_v1/update_00000693.pt, research12, 10,000 development pairs, seed 1,000,000, deterministic guarded CUDA inference with default batch-pairs 1,024, into eval_100m_vs_parent.json. Run koikoi.compare against eval_52000_vs_parent.json into difference_100m_vs_52000.json. Verify actual checkpoint/report hashes, training counters, guards, 20,000 matches / 240,000 rounds and complete paired seed range; write audit_100m.json. Logs runs/scale100m_budget.log, runs/scale100m_batch_evaluate.log and runs/scale100m_compare.log.

This is the final endpoint of the fixed scale-training run, still a development evaluation. It does not consume final-test seeds or establish paper8 public-baseline superiority, independent validation of selected methods, or project completion. Launch once from the committed protocol using python -m runs.watch_scale100m. No new algorithm or simulator change accompanies this audit.

## Koi-only adaptation implementation and controlled screen

The completed koi-substitution diagnostics and full-1,000 search endpoint motivate testing whether focused adaptation can improve learned koi decisions while retaining the existing card policy. This is a separate training experiment, not a change to the fixed independent-model search replication. No causal claim about shared representations follows from the earlier diagnostics.

Implemented optional policy_scope=koi and koi_width. Freeze card/shared/belief parameters; optimize the original koi head, optional width-128 observation-only residual MLP, actor value head and separate observation-only PPO return critic. The residual starts at zero and preserves initialization RNG. Card actions remain sampled from the frozen policy. Policy/entropy/KL losses and advantage normalization include only koi rows; both value losses and signed complete-match GAE still cover every phase. Restrictions, initialization migration, empty-phase handling and budget semantics are documented in DESIGN.md. Default all/zero behavior is unchanged.

Validation so far: 20 dedicated tests passed, covering initialization, fixed-observation card-logit invariance despite learning, both value heads and koi learning, residual export/frozen-opponent loading, hidden-world invariance, loss/normalization phase filtering, empty-phase normalization, counter arithmetic, CPU/FP32/BF16 exact restart and CUDA minibatch/full-update equivalence. Against actual pre-change package 4ff30860..., two captured default updates remain bit-identical for MLP/FP32 and relational/BF16: metrics, model, Adam, games, counters and all RNG; runs/koi_default_equivalence.json. Full suite subsequently passed: 126 tests, no skips, only two expected upstream Transformer warnings. Ruff passed.

Production-shape checks use the actual independent MLP update-5,000 checkpoint, SHA edc44213bc5c5c840297afc33ee64a57b91e9dee48c202535906fc916cde109d, with 1,024 environments, horizon 128, FP32, minibatch 4,096, both captures, seed 60,001 and fresh observation critic256. First collection hashes (observations, masks, sampled actions, log probabilities, values and returns) exactly match across all three arms before their first optimization. Three updates / 393,216 simulator decisions: control 3,011 matches / 25,695 rounds; head-only 3,014 / 25,667, 33,727 policy decisions; residual 3,015 / 25,690, 33,717 policy decisions. Frozen card tensors match the actual parent in both specialized arms. From each snapshot the fourth ordinary-versus-captured update is identical for both models, Adam, simulator/counters, all RNG and metrics, keeping the same capturable Adam settings. Report runs/koi_production_equivalence.json; smoke directories runs/ppo8_koi_{control,head,residual}_smoke. Every smoke and duplicated diagnostic decision is excluded from experimental budgets. Learning package SHA 3eb0e894aa14757912a157dd5fa44ffc1e84f83528c677a9a198c93ae8172bcc.

Prespecified screen: configs/ppo8_koi_control.json (all phases, no residual), ppo8_koi_head.json (koi-only, no residual) and ppo8_koi_residual.json (koi-only, residual128). All initialize from runs/ppo8_seed10001_captured/update_00005000.pt, retain its 11,141,428 paper8 pretraining matches separately, and start fresh optimizer, observation critic, simulator, RNG and counters. Same seed 60,001, 1,000 new updates / 131,072,000 simulator decisions per arm, checkpoints every 250. Run directories runs/ppo8_koi_{control,head,residual}_seed60001. No training endgame guard, no belief loss, entropy 0.02, lambda 0.95. Do not select from intermediate checkpoints. This is a matched-simulator-budget comparison, not equal policy-gradient decisions or independent training replication. All three share the same fresh observation critic; it does not isolate phase restriction from freezing relative to the full control.

At update 1,000, compare each specialized arm with the full-policy control on 10,000 deterministic guarded native development pairs; compare residual versus head on 10,000 direct pairs. Evaluate all three against assisted RL-WP on the same 300 development pairs (seed 1,000,000, deterministic/endgame-assisted, no search), and compute shared-pair differences from control plus residual versus head. Audit actual snapshot hashes, counters, simulator phase counts, preserved card tensors and profile-separated budgets. Positive candidates require larger development validation and independent training replication; untouched final seeds stay unused. Paper8 contributes nothing toward the research12 target. Launch only after the complete implementation validation and committed protocol.

The independent-model search replication completed its matched CUDA control at 47.825%, with all 1,000 records identical to historical CPU eval_5000_guarded.json. The live runner expected the newer eval_5000_wp.json naming convention, so a verified exact-byte alias was created; mapping and SHA 34d8b04b06d7e8f39082ceb76a5d58be0ed54c1fee2c56af4eb022c46ca58f97 in runs/ppo8_v3/control_cpu_alias.json. No data or evaluator settings changed and the search process was not restarted. Control loaded package 4ff30860..., search loaded 3eb0e894... during the optional koi implementation; the default actor path is unchanged and verified by old-source equivalence. Search evaluator child 57536 (parent 56492) is live.

The controlled koi screen runner runs/run_koi_screen.py will start all three learners in separate processes from the committed implementation, refuse existing run directories, retain their process handles and wait for all fixed endpoints. It then runs the prespecified budgets, three direct duels, three public evaluations and three paired differences. At completion it checks actual endpoint hashes, metrics, episode sums, finite model/critic tensors, specialized-arm fixed card weights, initialization budgets and evaluation identities. Expected consolidated audit runs/koi_screen_endpoints_audit.json; process identifiers runs/koi_screen_processes.json. Training logs runs/koi_screen_{arm}_training.log. No mid-run strength evaluation or data-dependent budget extension.

All three koi adaptation learners started from committed 5d4e05d with the fixed configs/parent and 1,000-update targets. Actual child process identities and advancing finite metrics verified; each arm reproduces its production-smoke first three update metrics exactly despite concurrent execution. Runner session 57508 owns the training handles and the prescribed endpoint evaluations. Operational details in STATUS.md. No midpoint strength selection; the independent-model search replication and research12 scale/final watcher continue separately.

## Koi adaptation training endpoints completed

All three learners completed exactly 1,000 new updates / 131,072,000 simulator decisions and exited successfully; the existing runner proceeded to the fixed evaluations. Control: 1,115,289 new paper8 matches / 8,743,052 rounds, checkpoint SHA a225b172f70ae968e02217a0eba2e4ea1bae7ba68cd52f817e72fb3aa41f3b80. Head-only: 1,116,221 matches / 8,758,951 rounds / 11,347,406 policy decisions, SHA 824e8f5392e099aaf0f49c311a2d9acd9f252c18230fac480c5b90baf3714905. Residual: 1,115,299 matches / 8,743,751 rounds / 11,364,073 policy decisions, SHA 996443b42fd1f4b91fd096e7292440668ca50c2ef8b67b3cb4682dac424f8f3c.

Actual file hashes, final metrics and counters, simulator episode sums, config/source identity, finite actor/critic/Adam tensors and 96,000 optimizer steps checked. In both specialized arms, every card/shared/belief tensor equals the initialization checkpoint exactly, and frozen parameters have no Adam state. Report runs/koi_training_endpoints_audit.json. Separate lineage totals including 11,141,428 pretraining matches are 12,256,717 / 12,257,649 / 12,256,727 paper8 matches; budgets in each run. These overlap in their initialization and must not be added as independent training lineages. None contributes to research12.

All three native direct duels have completed, and the runner is executing the three public evaluations before paired comparisons and its consolidated audit. No condition is selected from the partial set of endpoint results. Runner session 57508 remains live; trainer children 50444, 36772 and 57124 are terminal and must not be resumed. Operational state is consolidated in STATUS.md; full historical experiment evidence remains here.

## Completed koi-adaptation screen and fixed larger development endpoint

Runner session 57508 completed every prespecified evaluation, comparison and artifact audit successfully. Rechecked stored report hashes, recomputed all score/point means and bootstrap intervals, and recomputed the three public shared-pair differences. Reports and training budgets remain in the run directories described above; consolidated audit runs/koi_screen_endpoints_audit.json. All learners/evaluators in that pipeline are terminal; do not resume them unchanged.

| Comparison | Draw-adjusted win rate (95% paired-bootstrap CI) | Mean point difference (95% CI) |
| --- | --- | --- |
| Head versus full-policy control, 10,000 pairs | 50.545% (49.9999375-51.0975625%) | +0.3401 (+0.0102 to +0.6645) |
| Residual versus control, 10,000 pairs | 50.5475% (49.9975-51.125%) | +0.7006 (+0.3717 to +1.0317) |
| Residual versus head, 10,000 pairs | 49.9125% (49.6525-50.175%) | +0.3068 (+0.1589 to +0.4533) |
| Full-policy control versus assisted RL-WP, 300 pairs | 46% (42.75-49.25%) | -1.4033 (-3.3033 to +0.49) |
| Head versus assisted RL-WP, 300 pairs | 48.75% (45.5833-52.1667%) | -0.8967 (-2.9768 to +1.1935) |
| Residual versus assisted RL-WP, 300 pairs | 49% (45.75-52.5%) | +0.4533 (-1.6734 to +2.6034) |

Use the unrounded direct win-rate bounds: both head/control and residual/control intervals include 50%, despite their very close lower endpoints. No win-rate improvement is established by these duels. Public head-minus-control difference +2.75 pp (CI -1.25 to +6.9167), points +0.5067 (CI -1.9535 to +2.9467); residual-minus-control +3 pp (CI -1 to +7.0833), points +1.8567 (CI -0.5368 to +4.2703). Residual-minus-head public difference +0.25 pp (CI -1.3333 to +1.8333), points +1.35 (CI +0.5333 to +2.2067). Positive point differences do not replace the win-rate objective.

Supplementary post-screen comparison with the original initialization's already-recorded first 300 public pairs (48.25%): control -2.25 pp (CI -6.5833 to +1.9167); head +0.5 pp (CI -1.25 to +2.25); residual +0.75 pp (CI -1.25 to +2.75). Point differences: -0.3833 (CI -2.9303 to +2.0667), +0.1233 (CI -0.7467 to +0.96), +1.4733 (CI +0.5166 to +2.46). Per-run difference_vs_initialization.json explicitly identifies this supplementary analysis. It does not establish that full-policy continuation declined, nor that either specialized policy improved win rate.

The consistently positive but small specialized-policy estimates justify a fixed larger development assessment, not more training or default adoption. Extend all three public evaluations by exactly 700 pairs (seeds 1,000,300-1,000,999), same immutable models/CPU/deterministic/endgame assistance, no search or configuration change. Prespecified combined endpoint is original 300 plus additional 700 for each arm, 1,000 total; retain the original data in all conclusions. These are reused development deals, already present in baseline/native comparisons, not untouched test data. Report all arms and paired differences, including each versus initialization, and residual versus head; no intermediate selection or extra training updates.

Also run supplementary direct parent duels for all three endpoints against runs/ppo8_seed10001_captured/update_00005000.pt on 10,000 deterministic guarded native development pairs (seed 1,000,000). This separates comparison with the changing full-policy control from comparison with initialization, but does not substitute parent-relative wins for public superiority. Mark these as post-screen diagnostics, not part of the original prespecified screen.

Runner runs/run_koi_extension.py launches three independent CPU additional-700 evaluators and runs the three native parent duels, then waits and aggregates only after all finish. It rejects existing outputs, checks actual checkpoint hashes, contiguous/disjoint ranges, fixed inference settings, report identities and all paired comparison sample sizes; records source-report and runner hashes. Outputs per arm eval_1000_wp_new700.json and ab_vs_initialization_1000.json; combined runs/koi_screen_1000_summary.json. Process manifest runs/koi_extension_processes.json; logs runs/koi_extension_{arm}_{public,parent}.log. No pooled latency percentile. Launch once after committing this extension protocol. Any selected improvement still requires independent training replication and untouched final testing; default settings remain unchanged.

The fixed larger koi endpoint started from protocol commit 0d2fcf0. Runner session 63774, actual PID 35576 (parent 31968), owns three public evaluator handles and sequential native parent duels. Actual CPU evaluators 28252 / 31336 / 62264 (control/head/residual; launcher PIDs 2936 / 41756 / 49460) and advancing public pair counts verified. Process manifest runs/koi_extension_processes.json records launcher handles. No new training or configuration changes accompany this evaluation. Wait for the complete combined endpoint and audit before selecting a condition.

## Completed full-1,000 koi-adaptation development endpoint

Runner session 63774 completed successfully. All six segment/report hashes, combined seed ranges, means/paired-bootstrap intervals and six public paired comparisons independently recomputed; audits runs/koi_screen_1000_audit.json and runs/koi_parent_duels_audit.json. Combined summary runs/koi_screen_1000_summary.json retains original 300 plus additional 700 for every arm.

| Arm | Public win rate, 1,000 pairs (95% CI) | Public mean point difference (95% CI) | Direct win rate versus initialization, 10,000 pairs (95% CI) |
| --- | --- | --- | --- |
| Full-policy control | 47.125% (45.275-48.875%) | -0.699 (-1.83305 to +0.410025) | 49.5475% (48.9825-50.1150625%) |
| Koi head | 49.125% (47.399375-50.9%) | -0.023 (-1.137 to +1.072025) | 49.8075% (49.56-50.0375%) |
| Koi residual 128 | 49.25% (47.5-51%) | +0.602 (-0.548125 to +1.73805) | 49.835% (49.5674375-50.1075%) |

Public paired head/control +2 pp (CI -0.35 to +4.35), residual/control +2.125 pp (-0.25 to +4.5), residual/head +0.125 pp (-0.8 to +1.05). Versus initialization's 49.375% on the same 1,000 pairs: control -2.25 pp (-4.550625 to +0.1), head -0.25 pp (-1.075 to +0.65), residual -0.125 pp (-1.100625 to +0.925). Thus neither specialization nor full continuation establishes a win-rate improvement over initialization. No sampled-card adaptation is selected or extended unchanged.

Public point differences versus initialization: control -1.189 (CI -2.598025 to +0.2022), head -0.513 (-0.97605 to -0.046), residual +0.112 (-0.409 to +0.6291). Residual/head +0.625 (+0.122975 to +1.12). Native parent-duel point differences: control -0.7323 (-1.074125 to -0.4004775), head -0.4987 (-0.6423025 to -0.3597), residual -0.158 (-0.3104125 to -0.0060925). Positive residual-versus-head points do not meet the win-rate objective; these are reused development comparisons, not final tests.

## Completed independent-model koi-search replication

Runner session 95279 completed the fixed seed-1 MLP 1,000-pair endpoint and audit successfully. Search win rate 48.375% (CI 46.599375-50.2%), points +1.389 (+0.283 to +2.502025). Matching CUDA no-search control 47.825% (46.05-49.65%), points +0.975 (-0.132 to +2.116). Paired search gain +0.55 pp (CI 0 to +1.125), points +0.414 (+0.119975 to +0.728). Direction agrees with seed-10,001's +0.65 pp, but this replication's win-rate interval includes zero and neither model establishes public superiority. Search remains optional; no default promotion or additional unchanged search run is selected.

There were 230,191 candidate decisions, 9,011 searches, 163 changed actions, 442,908,672 logical search transitions and 147,456 preparation transitions; none are training. All 1,000 CUDA-control records equal prior CPU records. Model SHA d104ae00824aca114e6cd414c74d02670b9a8de5a009ad4fad557005f66233e5; search report SHA 52e79ab87491e87022ba1040e95332da3198bb9df331ea46a772fa2bda6cc813. Checked actual report hashes and independently recomputed both means/intervals and the paired comparison; koi_search192_1000_audit.json and koi_search192_1000_statistics_audit.json under runs/ppo8_v3. Same development deals across the two training models are not independent extra 2,000 pairs. Fixed uniform128/horizon192/margin2.5/koi-only/captured/deterministic/guarded settings were unchanged.

## Frozen-card action-selection diagnostic and controlled next experiment

Exploratory diagnostic at the original seed-10,001 MLP initialization: four captured collections, 1,024 environments x horizon128, seed 70,001, frozen-card koi configuration, no optimizer updates. Of 478,989 card decisions, 457,803 had multiple legal actions; 79,032 (17.2633%) sampled a non-argmax action. Mean argmax probability 0.827382, mean entropy 0.421013 nats on those choices. Of 45,299 koi decisions, 6.83017% differed from argmax, mean argmax probability 0.931835. Report runs/koi_card_sampling_diagnostic.json. All 524,288 decisions / 34,277 rounds / 4,070 complete matches are diagnostic simulation excluded from training budgets. This quantifies a training/deployment difference; it does not prove that this difference caused the adaptation result.

Add optional deterministic_cards=true, restricted to policy_scope=koi. Both players execute their frozen card policy's legal argmax, while koi remains sampled and optimized. Card behavior log-probabilities are zero, and those rows remain excluded from all PPO policy terms; full-match signed GAE and both value losses follow actual trajectories. This is an on-policy koi-learning environment with fixed deterministic card behavior, not deterministic policy-gradient learning on trainable card actions. Default false and inference settings are unchanged. Endgame training remains disabled to isolate this one factor. Existing sampled-card learners are complete and must not be resumed unchanged.

Validation: 33 dedicated koi tests passed (including CPU/CUDA collection behavior, sampled koi actions, frozen tensors, checkpoint resume, FP32/BF16 and capture/eager equality). Against immutable actual old-source default results (package 4ff30860...), two complete MLP/FP32 and relational/BF16 captured updates match all metrics, model/Adam/environment/counters and CPU/CUDA/Python/NumPy RNG; runs/koi_detcards_default_equivalence.json. Production-size validation from the actual parent: head three updates / 393,216 decisions / 3,046 matches / 25,798 rounds / 33,978 koi decisions; residual 3,041 matches / 25,771 rounds / 33,969 koi decisions. Initial collection hashes match across arms, every recorded card action equals legal argmax, and frozen card weights equal the parent. Fourth eager/captured updates match complete trainable/Adam/environment/RNG states and metrics; runs/koi_detcards_production_equivalence.json. All smoke/duplicate diagnostics excluded from budgets. Full regression suite passed: 139 tests, no skips, two expected upstream Transformer warnings, 278.75 seconds (session 88273 terminal). Ruff passed. Package SHA be4b1cd035affa520c8716620f14a0d62d95ac094a468c2c01cba870d0a26bd0.

Fixed next protocol: configs/ppo8_koi_head_detcards.json and ppo8_koi_residual_detcards.json, each differing from its completed sampled-card counterpart only in deterministic_cards. Initialize from the original seed-10,001 MLP update 5,000 (SHA edc44213bc5c5c840297afc33ee64a57b91e9dee48c202535906fc916cde109d), same fresh seed 60,001/observation critic 256/Adam/environment and 1,000 updates / 131,072,000 simulator decisions, checkpoints every 250, no midpoint selection. Directories runs/ppo8_koi_{head,residual}_detcards_seed60001. Their 11,141,428 paper8 initialization matches stay separate; none count toward research12.

After both fixed endpoints, run 10,000 deterministic guarded native development pairs for each versus its corresponding sampled-card endpoint and versus initialization, plus residual versus head. Public comparisons: exactly 1,000 CPU deterministic guarded no-search pairs per arm versus assisted RL-WP, seed 1,000,000, with paired differences versus the corresponding original300+additional700 sampled-card records, initialization and residual/head. These are development deals reused for method selection; final seeds remain untouched. The screen tests card selection under two fixed capacities, not an independent training replication. No added training or sample-size expansion from intermediate outcomes. Any selected gain still requires independent-model/training validation and untouched final evaluation.

Runner runs/run_koi_detcards.py rejects existing run directories, verifies initialization/control hashes and config differences, retains trainer handles, checks successful exits, all endpoint counters/metrics, episode sums, finite model/critic/Adam, 96,000 optimizer steps, preserved card weights and profile-separated budgets. It then checks full evaluation identities/seed ranges and computes paired comparisons. Artifacts: budget_1000.json, ab_vs_sampled_1000.json, ab_vs_initialization_1000.json, eval_1000_wp.json, difference_vs_sampled.json and difference_vs_initialization.json per arm; residual also ab_vs_head_1000.json and difference_vs_head.json. Consolidated runs/koi_detcards_training_audit.json and koi_detcards_endpoints_audit.json; process manifests koi_detcards_processes.json and koi_detcards_public_processes.json; logs runs/koi_detcards_*.log. Launch once only after validation and committed protocol.

## Completed 100M-new-research12 scale endpoint

The unchanged scale trainer exited successfully after update 69,267, the first update crossing the target: previous99,999,773 -> final 100,001,204 new completed research12 matches, 1,200,025,665 completed rounds including 11,217 rounds in partial matches, 18,157,928,448 simulator decisions. Trainer session 57088 and final watcher 51102 are terminal success; do not resume this completed scale run unchanged. Original training source 142c265/package 2bfd85f8... was retained throughout the captured continuation. The 1M research12 and paper8 initialization budgets are excluded from the new100M threshold.

Immutable final_100m.pt SHA 613ea0d992a54a0a18ab1848f106abe074c5d74640d76c3a86a131448aa29cce equals final stable latest.pt bytes. Audits inspect actual twelve-round/no-bankruptcy settings, nonterminal/reset rows, sum of EPISODE =100,001,204, 12*episodes+sum(ROUND)=1,200,025,665, update*envs*horizon decisions, finite model/Adam, final metrics and first target crossing. Total initialization lineage: 101,001,668 research12 matches; 11,141,556 paper8 matches separately. Reports budget_100m.json and audit_100m.json; independent recomputation audit_100m_independent.json under runs/ppo12_scale_seed30001_captured.

Final snapshot versus transfer initialization on 10,000 deterministic guarded research12 development pairs: 51.8125% (95% paired-bootstrap CI 51.225-52.3825), points+1.3454 (CI+0.93206 to +1.7790025). There are 20,000 complete evaluated matches/240,000 rounds. Shared-pair difference versus update 52,000: -0.38 pp (CI -1.1200625 to +0.3675), points -0.5739 (CI -1.10863 to -0.0258975). Parent-relative win advantage persists; additional training beyond52,000 does not establish further win-rate gain. No unchanged continuation is justified by this endpoint.

Actual checkpoint and report hashes, complete seed range, all score/point means/intervals and paired comparison independently verified. Files eval_100m_vs_parent.json and difference_100m_vs_52000.json. Evaluation source hash be4b1cd0... includes the validated optional deterministic-card training flag; inference behavior is unchanged. Snapshot training source remains142c265. This satisfies the requested100M-new-research12 computation gate only. Public paper8 baseline superiority, independent validation of a selected improvement, untouched final testing and manuscript remain unfinished; the goal stays active.

The deterministic-card comparison started from committed implementation/protocol 080f789 (package be4b1cd0...). Runner session59159, actual PID26416 (parent22964); head trainer23128 (parent47316), residual trainer2892 (parent50164). Both actual command lines and advancing finite metrics checked. The first three updates and counters for both learners exactly reproduce the prior production validation despite concurrent execution; runs/koi_detcards_start_audit.json. Fixed targets and all endpoint comparisons remain unchanged. Scale and both preceding evaluation pipelines are complete; these two learners are the only live training jobs. Operational handles are tracked in STATUS.md.

## Completed deterministic-card adaptation endpoints

Runner session 59159 completed both fixed learners, all five native duels, both public 1,000-pair evaluations and all five public paired comparisons. No intermediate selection or budget extension occurred. Head: 1,000 updates / 131,072,000 simulator decisions / 1,122,373 new paper8 matches / 8,803,468 rounds / 11,387,045 koi policy decisions; checkpoint SHA 0faf75b49c4709245e86734bae0a9f3dd5c2af9a6b2c6a4d7741fb2a3c8a9798. Residual: 1,120,934 matches / 8,786,985 rounds / 11,401,936 koi decisions, same update/decision budget; SHA 20847e7608b394474fefe1591388ecae6e6af87b158ba0fc44d284959500f871. Separate paper8 lineages including initialization: 12,263,801 and 12,262,362 matches, not additive across models and not research12.

Actual endpoint hashes, source 080f789/package be4b1cd0..., exact config difference versus sampled controls, final metrics/episode sums, finite model/critic/Adam and 96,000 optimizer steps verified. Every frozen card/shared/belief tensor equals the original initialization and has no optimizer state. Independently recomputed every native/public mean, bootstrap interval and five public paired comparisons. Reports: runs/koi_detcards_training_audit.json, koi_detcards_training_and_duels_audit.json, koi_detcards_endpoints_audit.json, koi_detcards_final_statistics_audit.json.

| Comparison | Draw-adjusted win rate (95% paired-bootstrap CI) | Mean point difference (95% CI) |
| --- | --- | --- |
| Head versus sampled-card head, 10,000 pairs | 49.925% (49.735-50.1175625%) | +0.3182 (+0.2061 to +0.4307025) |
| Head versus initialization, 10,000 pairs | 49.785% (49.56-50.0000625%) | -0.1215 (-0.24671 to +0.0029) |
| Residual versus sampled-card residual, 10,000 pairs | 49.86% (49.5925-50.1275%) | +0.0077 (-0.1364075 to +0.150115) |
| Residual versus initialization, 10,000 pairs | 49.7025% (49.435-49.9675%) | -0.1539 (-0.3045025 to -0.0069) |
| Residual versus head, 10,000 pairs | 49.9375% (49.6775-50.19%) | -0.011 (-0.152215 to +0.1308) |
| Head versus assisted RL-WP, 1,000 pairs | 49.25% (47.525-51.025%) | +0.398 (-0.74405 to +1.506075) |
| Residual versus assisted RL-WP, 1,000 pairs | 49.175% (47.399375-50.975%) | +0.642 (-0.5 to +1.76105) |

Public paired head-minus-sampled-head: +0.125 pp (CI -0.55 to +0.775), points +0.421 (+0.038 to +0.795). Residual-minus-sampled-residual: -0.075 pp (-1 to +0.825), points +0.04 (-0.421 to +0.500025). Versus initialization: head -0.125 pp (-0.875 to +0.625), residual -0.2 pp (-1.175 to +0.75). Residual-minus-head: -0.075 pp (-0.95 to +0.825). No win-rate gain is established from deterministic frozen-card behavior. The native residual/initialization interval is negative; the head/initialization upper bound remains slightly above 50%, so do not round it into a negative result. All are development comparisons with the previously disclosed selection history, not untouched confirmation. Do not extend either deterministic-card run unchanged or adopt the option by default.

## Search truncation diagnostic and full-action execution validation

A separate diagnostic examined the first unsolved koi roots at zero-based rounds 0, 3 and 6 for each original MLP (seeds 1 and 10,001), found in deterministic guarded self-play from seeds 5,000,000 onward. For each of the six roots, generated 128 uniform worlds and both root actions, then compared otherwise identical horizon192 and horizon512 rollouts. All 256 branches at every root had terminated by step192. Means, confidence margins and selected actions were identical across horizons. A wrapped terminal counter was verified against the original uninstrumented rollout for exact values/final states. Report runs/search_truncation_diagnostic.json, package be4b1cd0.... All 1,212,416 executed diagnostic transitions are excluded from training. This deliberately stratified six-root sample is not a population estimate or a full-match guarantee for horizon192; it gives no support for increasing horizon unchanged.

Earlier full-action corrected search only tested 32 particles/horizon32 on the older update2,000 model. The completed horizon192 experiments restricted search to koi. The next isolated change is to apply the already-supported full-action flag at the current 128-particle/horizon192 settings on the original seed-10,001 model; no new algorithm or simulator code is required.

Before strength screening, checked the first card-decision roots with 2, 3, 4, 5, 6, 7 and 8 legal actions from deterministic guarded self-play seed5,000,100. Current eager and CUDA-captured full-action search use the actual immutable original model, uniform128/horizon192/margin2.5. Every root matches in selected action, full simulator state, proposal RNG and CPU/CUDA RNG; repeated captured replay matches after restoring the proposal RNG. Report runs/full_action_search_production_equivalence.json. Eager measured 0.106-0.148 seconds, first capture 0.374-0.501 seconds and replay 0.0540-0.0613 seconds under concurrent CPU public evaluations. These are functional timing observations, not isolated deployment benchmarks or strength evidence. Peak allocated/reserved CUDA memory 315,927,040/459,276,288 bytes. Eager logical 860,160, captured logical 1,720,320 and capture preparation 2,580,480 transitions, all excluded from training.

## Fixed full-action horizon192 search screen

Model remains runs/ppo8_seed10001_captured/update_00005000.pt, SHA edc44213bc5c5c840297afc33ee64a57b91e9dee48c202535906fc916cde109d. Do not substitute an adapted endpoint. Evaluate exactly 100 development pairs, seeds 1,000,000-1,000,099, versus assisted RL-WP, deterministic guarded CUDA inference, uniform128 particles, horizon192, confidence margin2.5, --search-all --search-cuda-graph. This is an exploratory screen on the same existing prefix; preserve the original no-search43.25% and koi-only45% controls rather than changing the deal range. No model training, final seeds, interim tuning or default adoption.

Compare paired results with existing matching eval_5000_cuda_control100.json and eval_5000_koi_search192_captured100.json, both under the original MLP run. Runner runs/run_all_search192.py validates those controls, actual model hashes and fixed settings before starting; checks child success, contiguous endpoint seeds, checkpoint/settings identities and simulation-count bounds; writes both paired comparisons and report hashes. Output eval_5000_all_search192_captured100.json, difference_all_search192_vs_control100.json, difference_all_search192_vs_koi100.json and all_search192_100_audit.json. Log runs/all_search192_evaluate.log; process manifest runs/all_search192_process.json. Launch once from committed protocol. A positive screen requires larger assessment and independent-model/final validation; no superiority claim follows from 100 reused development pairs.

The full-action search screen started from committed protocol b66cb12, unchanged package be4b1cd0.... Runner session 41603, actual PID 61488 (parent 61620), evaluator PID 11808 (parent 11772). Actual command lines and the first 10 completed pairs verified. Keep the fixed 100-pair endpoint and both comparisons; no intermediate selection. This evaluator is the only current experiment process; all training and preceding evaluation pipelines are complete.

## Proposal preprocessing optimization during the fixed search screen

The fixed 100-pair full-action evaluation continues unchanged in its existing process, package be4b1cd0.... No restart, settings change or interim selection accompanies the following execution optimization for future runs.

CPU profiling of eight 128-world proposal groups showed 2,048 repeated ReferenceEnv.observe calls, consuming 0.037 of 0.071 profiled seconds. The old per-particle path recomputed the same observation twice, unknown set, hand count, public history and belief weights. It also rebuilt the sampled-hand set for every candidate stock card. Archived exact pre-change search source in runs/search_before_batched_proposals.py, SHA befa244d00900a88cd020a6570e0907dd2df6b45139947f3f78be0c8da1b748c; profile runs/search_proposal_profile_before.txt.

Implemented sample_hidden_batch: compute public root context once, allocate independent world rows, and preserve the exact sequential choice/rejection/shuffle/future-seed RNG calls per particle. Hoist the hand membership set within each particle. Scalar sample_hidden uses the one-particle path. Search uses the batch directly. No cross-root cache, proposal distribution change, new hidden inputs or rollout/value/selection change. API/default search settings and training are unchanged.

Compared with the actual archived old-source sampler over 96 evolving root states across paper8/research12, both uniform and weighted proposals: 6,144 worlds and final RNG states are identical, including 36 matching invalid-initial-hand rejection draws. Additional regression tests cover scalar/batch stream equality, hidden-world independence, original-root preservation and valid decks on both profiles. Targeted 19 search/capture/substitution tests passed. Full regression suite passed: 144 tests without skips, two expected upstream Transformer warnings, 163.74 seconds; session64198 terminal. Ruff passed.

Production comparison uses the actual original seed-10,001 MLP,128 particles/horizon192/margin2.5, deterministic guarded full-action CUDA capture. Seven card roots with 2-8 legal actions plus one koi root: initial and three further paired calls each match chosen action, full simulator state and proposal RNG against actual old-source SearchAgent. Logical/preparation counts are equal: 3,637,248/2,580,480 per implementation, all diagnostic simulation excluded from training. Report runs/batched_proposal_equivalence.json, package 20e9d7c361d17f37a1fccca269ea01dcc34ad3a49b60c057f60a0ff1d2bc76d1.

Median 128-world proposal times over three paired repetitions at each root: old 7.29-13.40ms, new 1.40-2.50ms, approximately fivefold component speedup. Whole-search observations vary under the concurrent original evaluation: old 92.4-111.1ms, new 92.4-108.3ms, including one slower new measurement. Do not extrapolate the sampler ratio to overall search or describe these as isolated deployment benchmarks. This optimization changes execution cost, not demonstrated playing strength.

## Completed full-action 100-pair search screen and fixed extension

Runner session 41603 completed successfully using the original pre-optimization package be4b1cd0.... Full-action search scores 49.25% (95% paired-bootstrap CI 43.25-55.25), points -1.63 (CI -5.57 to +2.2),100 pairs/200 matches. Same-prefix no-search 43.25%, koi-only 45%. Paired all-minus-control +6 pp (CI +2.25 to +10), points +2.42 (+0.72 to +4.25). All-minus-koi +4.25 pp (CI +0.24375 to +8.75), points +1.18 (-0.94025 to +3.27). This is a promising exploratory development screen after earlier method selection, not proof of public superiority; the absolute win-rate interval includes 50%.

Actual report/controls hashes and checkpoint/settings identities checked; all three reports' score/point means and bootstrap intervals plus both paired comparisons independently recomputed in all_search192_100_statistics_audit.json, alongside all_search192_100_audit.json. There were 22,936 total decisions,10,867 searched decisions,103 changed actions,1,286,922,240 logical simulation transitions and 2,580,480 preparation transitions. These are inference simulations, not training. Reported candidate mean 58.8836ms/p95 69.8845ms includes concurrent diagnostic/test workloads, not isolated deployment timing. Root changes are not automatically improvements; measured paired match outcomes support the screen result.

Fixed next endpoint: exactly 900 additional development pairs, seeds 1,000,100-1,000,999, same immutable original seed-10,001 model, opponent/guards/determinism/CUDA/uniform 128/horizon 192/margin 2.5/full-action/capture settings. Do not change the small-prefix data or adapt the model. Use the validated root-preprocessing implementation (package 20e9d7c3...) whose actual old-source world/RNG and search-state/action equivalence is recorded above. This is an execution optimization, not a new search policy or a claim that old evaluation used the new package.

The primary development endpoint is original 100 plus new 900,1,000 total. Compare with already-completed corresponding 100+900 CUDA no-search and koi-only controls, retaining all original records and reporting the additional900 separately in the segment metadata. The controls have different slash conventions in their agent paths; validate the actual checkpoint SHA rather than require textual path equality. No interim selection, new training or final-test data. This is a nested exploratory extension; unadjusted development intervals do not remove earlier method/sample-size selection. A positive full result still needs independent-model confirmation and untouched final validation.

Runner runs/run_all_search192_extension.py starts the fixed 900 evaluator and aggregates only after success. It rejects existing outputs, verifies actual model/control hashes, contiguous disjoint pair ranges, matching settings, full control records versus prior CPU, both source segments, and all paired comparisons. Summary retains segment hashes, runner hash, counters and separate graph preparation work; no pooled latency percentile. Outputs under runs/ppo8_seed10001_captured: eval_5000_all_search192_new900.json and all_search192_1000_summary.json. Log runs/all_search192_new900.log; process manifest runs/all_search192_extension_process.json. Launch once after committing this protocol.

The fixed additional900 full-action extension started from committed protocol b1f7012, verified package20e9d7c3.... Runner session17341, actual PID47200 (parent65592), evaluator PID45000 (parent28136). Actual command lines and first10 completed additional pairs verified; process manifest runs/all_search192_extension_process.json, log runs/all_search192_new900.log. Preserve the fixed combined1,000 endpoint and all original records; no interim tuning or additional training.

## Terminal-aware deterministic search execution diagnostic

The fixed additional900 evaluator remains live and unchanged (PID45000, package20e9d7c3...). Its intermediate scores are not used for selection. This separate execution diagnostic follows the earlier truncation evidence: some particle groups finish their original match long before horizon192, while the current rollout resets completed rows and keeps evaluating until the fixed horizon.

Prototype runs/diagnose_chunked_search.py uses32-step CUDA graph blocks, checking whether any original-match particle remains alive after each block. It retains the first terminal outcome and the same tail value at the original horizon if needed. Uniform128/horizon192/deterministic guarded policies only. It is not a shorter search horizon. Completed scratch environments may differ from the original192-step path because post-terminal reset work is omitted; do not claim full-state equivalence. Public-root state, future proposal stream and candidate values are the relevant deployment invariants, and production integration still requires explicit tests.

Selected the first card roots with2-8 legal actions plus first unsolved koi roots at rounds0/3/6 from deterministic guarded self-play seeds5,000,100 onward, separately for the two original MLPs. Twenty deliberately selected roots, not a deployment distribution. Each compares four alternating-order full-graph/block-graph runs on identical sampled worlds. Every particle return and CPU/CUDA RNG state is exactly equal. All original-match branches finish at a block boundary32,96,128 or160 in this sample. The no-terminal-before-horizon bootstrap case and stochastic policies are not covered; production must test the former and reject the latter unless RNG equivalence is separately established.

Report runs/chunked_search_diagnostic.json, SHA e9d5df8cd0d38b6e063a3afe30e40e2496eeb64300671c4c01c6acaff2474223; actual script/model hashes checked. Full logical work8,060,928 plus6,045,696 preparation transitions; prototype executed6,062,080 plus2,015,232 preparation. All excluded from training. Per-root median chunk/full time ratios span0.186-0.969, median0.821, under the concurrent live evaluator. This supports further implementation testing, not an isolated deployment speedup or strength claim. Full timing excludes initial state copy while prototype timing includes it; neither includes hidden proposals. Do not extrapolate these ratios to full-game latency.

Next execution work: optional deterministic-only terminal-aware captured search, retaining the default full-horizon path and the running protocol. Before any use in a future evaluation, verify both rule profiles, every block/tail boundary, nonterminal bootstrap, action/confidence equality, proposal and torch RNG, current model tensors, and logical versus actually executed/preparation counts. Search scratch state is overwritten from each new root; verify no omitted work affects later calls. Existing full-action1,000-pair strength endpoint and final-test reservation remain unchanged.

## Optional terminal-aware captured search implemented and verified

Added --search-chunk-size N (default0) to deterministic CUDA-captured search. Nonzero requires --deterministic --search-cuda-graph and an evaluation-mode model; stochastic and unsupported execution modes are rejected. Graph blocks retain each original particle's first terminal result, stop only when all have finished, and preserve the configured horizon including a final partial block and unchanged bootstrap for any unfinished particles. Root proposals, endgame guards, candidate comparison and confidence margin are unchanged. Existing eager/full-graph paths remain the default.

Each new block graph restores its persistent alive/value accumulators as well as the existing simulator/RNG rollback. Per-root actions/sign buffers and model parameter addresses remain live inputs. Scratch states after completed matches may differ because discarded subsequent episodes are omitted; every next root overwrites the complete state and initializes accumulators. Reports retain logical search_simulated_decisions, add actual search_executed_decisions, identify search_chunk_size, and separately count three executed warmups per unique block graph. None counts toward training.

Archived actual pre-change source runs/before_chunked_search.py (SHA0b9d4968d51c16fc5e65b455482f4d2fbd42a7acfdb9559093cb1d25be603da7) and before_chunked_evaluate.py. Production validation compares that archived SearchAgent with the current default and32-step option at uniform128/h192/margin2.5/full-action/guarded/deterministic settings. Both original paper8 MLPs plus the immutable research12 final100M model: ten card/koi roots per model, then reverse-order cache reuse,60 total searches. Every particle value, selected action, proposal RNG and CPU/CUDA RNG agrees. Current default scratch states and counters also equal old source. Actual checkpoint and runner hashes verified.

Report runs/chunked_search_production_equivalence.json, SHAe9db82d5a724f45919cd9f23432b62c32f425256c4f379d61be91901f78640a5. Per implementation combined logical6,045,696 transitions; old/current-default each execute6,045,696 plus7,741,440 preparation; chunked executes5,005,312 plus2,580,480 preparation. This is functional evidence, not a deployment benchmark or strength result. All work excluded from training.

Closed-loop CLI replay of existing development seed1,000,000 against assisted RL-WP, same original seed-10,001 model/settings and32-step execution: both seat results and point differences exactly equal the previously completed full-action100 report's first record. Counts13,910,016 logical/6,848,512 executed/860,160 preparation transitions. Report runs/chunked_search_replay_pair.json and independently checked chunked_search_replay_audit.json. This single reused pair is a regression diagnostic, not additional strength evidence; equality of final records alone is not a full action-trace proof.

Targeted44 tests passed; full suite173 passed without skips, two expected upstream Transformer warnings,318.94 seconds under concurrent evaluation/diagnostics. A subsequent test-only extension explicitly exercises opponent-seat roots and reuse after terminated roots; all29 chunk tests passed again in23.97 seconds. Includes both profiles, guarded/unguarded paths, horizons1/7/8/15/65/193, block sizes7/16/32, short/tail/early-terminal cases, hidden-world invariance, current weight reads, exact return/action/RNG and work accounting. Ruff and diff checks passed. Current package SHAe9c9f0867a18add428ab62bfd4ccef18e7f389924424b498c00e42b3c6812e94.

The ongoing fixed additional900 evaluator still uses its originally imported package20e9d7c3... and has not restarted or changed settings. No chunk flag is added to that run. Segment evaluation provenance was captured before execution; do not confuse a later aggregation's on-disk source hash with the live evaluator's imported implementation. Use this optional path only for a subsequently specified evaluation, preserving model/proposal/settings identity and disclosing its execution counters. No new training or algorithm change is selected.

## Full-action1,000 endpoint audited; independent-model replication protocol

The fixed original100+additional900 endpoint completed successfully (session17341 exit0; runner47200/evaluator45000 absent). Additional900 report SHA d5fd748bf16c3a966137d8d4a1e28ad7d3eaa9c3c920c13dcedf6fc1076d35fa; combined summary SHA50cb5fe828084fca92397d50ec5f9e63370b541dc7a7463e5e99d621790331cf. Additional900 scores51.361111% (CI49.472222-53.277778); preserve it as a segment of the specified combined endpoint, not a newly selected primary sample.

Combined full-action51.15% (CI49.325-52.95), points+1.262 (+0.126925 to+2.394025); no-search49.375%, koi-only50.025%. Paired full-minus-control+1.775pp (+0.774375 to+2.75), points+0.772 (+0.221 to+1.327). Full-minus-koi+1.125pp (+0.075 to+2.2), points+0.526 (-0.061025 to+1.093025). Thus the full-action development improvement persists, but its absolute win-rate interval still includes50%; no public superiority claim. Original100 was reused after method selection and unadjusted bootstrap intervals do not remove that selection bias.

Independent runs/audit_all_search192_extension.py imports no koikoi statistics code. Recomputes all six segment and three combined score/point means and intervals plus all three paired differences directly from records, with independent vectorized5,000-replicate seed99 bootstrap. Checks actual checkpoint/report hashes, contiguous disjoint seed ranges, seat score versus point sign, opponent/profile/guards/determinism/settings identities, CPU control record equality, and search accounting. All pass. Audit SHA2e18f29dee800c1d3190a8bf4546373b75d1d7b753c884135a0dc234a57ef8ac. Actual evaluation provenance is20e9d7c3... at b1f7012; the summary records later on-disk e9c9f086... at485fcbd. These are explicitly distinguished, not attributed interchangeably.

Combined full-action counts:231,002 decisions,109,446 searched,1,025 changed,12,944,375,808 logical simulation transitions plus5,160,960 preparation transitions. Old execution ran the full horizon. These are excluded from training. Additional900 mean51.4856ms/p9561.1813ms includes concurrent verification earlier in the run; neither isolated deployment timing nor a pooled1,000-pair latency percentile.

Fix independent-training-seed replication before launch: original seed1 MLP runs/ppo8_v3/update_00005000.pt, SHA d104ae00824aca114e6cd414c74d02670b9a8de5a009ad4fad557005f66233e5; exactly1,000 development pairs seeds1,000,000-1,000,999; assisted RL-WP/paper8/deterministic/endgame/CUDA/uniform128/horizon192/confidence2.5/full-action/capture, with validated --search-chunk-size32. Package e9c9f0867a18add428ab62bfd4ccef18e7f389924424b498c00e42b3c6812e94. Logical/actual/preparation work remains separate. The terminal-aware optimization retains values/actions; it is not a horizon/policy change.

Reuse the previously verified seed1 CUDA no-search1,000 and koi-only1,000 controls, checking their bytes against the prior koi audit before launch and again afterward. Primary replication endpoint: full-action minus no-search paired draw-adjusted win-rate difference,95% bootstrap interval lower bound strictly above0. Full-minus-koi and both point differences are secondary; absolute public superiority is a separate requirement. No intermediate stopping/selection, additional training or final-test data. A failure/inconclusive result will not trigger automatic unchanged extension. Shared development deals across models do not constitute2,000 independent deal samples. Replication can support a later fixed final-test protocol, never replace it.

Runner runs/run_all_search192_seed1.py verifies model/package/control identities, launches once, then checks all settings/records and writes both paired comparisons after success. Log runs/all_search192_seed1.log; manifest runs/all_search192_seed1_process.json. Outputs in runs/ppo8_v3: eval_5000_all_search192_chunk32_1000.json and all_search192_1000_summary.json. Commit this protocol before launching. No other strength experiment runs concurrently.

Replication started after protocol commit7376e5b. Session88215; actual runner63652 (parent14620), evaluator39016 (parent28516). Manifest records the Windows launcher child28516; actual executing interpreter39016 verified by command line. Packagee9c9f086..., actual model/control hashes checked by runner; first20 pairs completed. No other experiment is active. Keep the fixed1,000 endpoint and independently recompute all results after successful termination.

## Full-action independent-model endpoint: replication not established

Fixed seed1 replication completed (session88215 exit0; runner63652/evaluator39016 absent). All1,000 pairs retained. Full-action48.00% (CI46.225-49.825), points+1.183 (+0.055975 to+2.32405). No-search47.825%, koi-only48.375%. Paired full-minus-control+0.175pp (-0.75 to+1.15), points+0.208 (-0.29305 to+0.719025). Full-minus-koi-0.375pp (-1.3 to+0.575), points-0.206 (-0.721025 to+0.310025). Prespecified primary positive-CI criterion fails. This does not reproduce seed10001's+1.775pp development gain. Positive mean points do not imply win-rate superiority; the full-action absolute win interval is below50% on these development deals.

Independent runs/audit_all_search192_seed1.py uses the earlier independent raw-record/statistics helpers, no koikoi statistics imports. All means/intervals/differences, hashes, settings, seat-score signs, seed range, previous CPU control records and search counters agree. Audit SHA51f2fdf02504c23fa55e92a40e796537246718dcbdcf5aef7f1e16222214ed5f; report SHAad7f55ff80633851657600cec13014189cc127abf496997de14abd0e15c61a07; summary SHA40ab30bfe4d7ea09583336768cffb4a8eae529ff5fe37ff77095b89985f710ae. Evaluator/manifest/summary all identify protocol7376e5b and packagee9c9f086....

Counts:230,180 decisions,108,947 searched,960 changes,12,913,139,712 logical transitions,6,426,251,264 executed,860,160 preparation. All excluded from training. Mean candidate25.7815ms/p9545.6191ms is observed timing on this workload, not a controlled paired speed comparison with the other model's run. Shared deal seeds across models remain1,000 unique pairs, not2,000 independent pairs.

Decision: do not adopt full-action search as a general default or automatically extend the failed replication unchanged. Preserve the promising original-model result with its exploratory-selection caveat. No final-test seeds are used. Next diagnose whether changes selected with128 proposals remain advantageous under independent hidden-state samples; a failed strength replication alone does not identify Monte Carlo noise, posterior mismatch, opponent mismatch or policy quality as its cause.

Fixed diagnostic protocol: each original seed1/seed10001 immutable model,25 duplicate pairs/50 baseline matches against assisted RL-WP, seeds5,100,000-5,100,024, paper8/guarded/deterministic. Every candidate root with multiple legal actions and no solved guard receives shadow full-action uniform128/h192/margin2.5/chunk32 search. Actual trajectory always follows the guarded baseline argmax, regardless of shadow choice. At every changed root, use a separate SeedSequence([73003,model_index,pair,seat,candidate_decision]) stream to generate1,024 new worlds and evaluate baseline/chosen actions with the same deterministic guarded self-play rollout and horizon. Validation values never influence action selection or subsequent diagnostic roots. No adaptive sample size or model selection.

Record selection and independent paired return means/standard errors, positive independent margins, unfinished particle counts, root phase/round/public-observation hash and logical/executed/preparation work. Report conditional root statistics without treating correlated roots as independent match outcomes. These uniform-model self-play estimates do not validate the true hidden-state posterior or actual published-opponent behavior; persistent independent gain can only narrow the sampling-noise explanation. A positive diagnostic is not strength evidence. Script runs/diagnose_search_resampling.py, output runs/search_resampling_diagnostic.json, progress log runs/search_resampling_diagnostic.log, process manifest runs/search_resampling_process.json. Commit before launch; all simulations excluded from training, no final-test data or production policy changes.

## Independent resampling diagnostic completed

The fixed diagnostic ran after protocol eef8c91, packagee9c9f086..., session74195/actual PID26184(parent65380), terminal exit0. Both models25 complete duplicate pairs, with baseline actions controlling every real trajectory. Report runs/search_resampling_diagnostic.json SHAc635bd21f27eb7aa7ddccfbd8b9182a4d0aff9b5e079091cd1160fe485d040ed; script SHAce6608f49b2506703a065bf308d961879761d5de00e9665f918261db6715fce1. Independent row-aggregate/model/script-hash checks in search_resampling_diagnostic_audit.json; this is not an independent particle-level replay.

Seed1:2,755 searched roots,22 proposed changes (21 discard/1 koi), average selected-particle gain0.1711648 versus independent-sample0.0359996. Six independent means are negative;16/22 remain positive, but only6/22 retain margin2.5. Seed10001:2,711 roots,18 changes (17 discard/1 koi), selected mean0.1362847 versus independent0.0313043. Six independent means negative;12/18 positive,3/18 retain margin2.5. Every selected and validation particle group completed its original match byh192, so these40-root discrepancies do not involve the value bootstrap. The large drop after independent resampling is selected-sample optimism under the same assumed uniform hidden-state/self-play model. It does not establish real-opponent calibration, identify the sole cause of failed strength replication, or prove confirmation will improve match outcomes. Roots within games are correlated and no independent-root confidence intervals are asserted.

Seed1 selection logical326,074,368/executed160,763,904/preparation860,160; validation logical8,650,752/executed5,046,272/preparation393,216. Seed10001 selection320,667,648/159,965,184/860,160; validation7,077,888/4,128,768/393,216. All excluded from training. No final-test seeds used.

Next change is optional two-stage proposal confirmation: retain current128-world candidate selection; only for a proposed deviation compare that chosen action with baseline on1,024 fresh worlds using a separate pair-resettable RNG and margin2.5; accept only a positive confirmation lower estimate, otherwise play baseline. Never rerank candidates on confirmation data. Default remains disabled. This reduces reuse of the same samples for selection and acceptance, without promising an unbiased posterior, optimal response or strength. Before a fixed strength screen, verify default equivalence against actual previous source, deterministic pair replay/order independence of both RNG streams, hidden-state invariance, graph/eager agreement and separated proposal/confirmation/logical/executed/preparation accounting. No new strength protocol is launched yet; all jobs are terminal.

## Independent proposal confirmation implemented and verified

Added --search-confirm-particles N, default0. Deterministic evaluation-mode policies only; invalid budgets and stochastic/training-mode combinations rejected. Search selects a candidate from the original paired worlds, then only for a proposed change draws fresh worlds from a separate SeedSequence([seed,1]) generator. Compare exactly chosen versus baseline with the same margin; strict positive confirmation is required, otherwise baseline. No reranking or confirmation-driven proposal-stream consumption. Paired evaluation resets both generators through wrappers. Total and per-stage logical/executed/preparation transitions plus proposed/confirmed/final-change counts are reported. Shared deterministic candidate execution can reuse a graph by branch batch size/profile; first graph creation is charged to the triggering stage.

Targeted57 search tests pass. Full186 tests pass without skips, two expected upstream warnings,88.82seconds; session87973 terminal. Ruff and diff checks pass. Tests cover accept/reject/tie/no-proposal paths, separate RNG consumption, both profiles, opponent-seat roots, hidden-world invariance, eager/full-capture/chunk agreement, cache reuse with equal batch size but different branch/particle layouts, work accounting, and replay after unrelated stream consumption. Only a test dictionary-style lint fix occurred after collection; no behavioral change.

Actual pre-change sources archived at runs/before_confirmation_search.py and before_confirmation_evaluate.py. Production validator reconstructs all40 previously selected diagnostic roots on the exact baseline trajectories, advancing proposal RNG even at skipped roots. Archived SearchAgent versus current disabled-confirmation default: every value/action/scratch state/proposal and torch RNG matches. Current confirmation compares baseline/chosen on1,024 independent worlds; every particle return matches a separate old-source full-horizon rollout, as do action decisions, confirmation RNG and hidden-world invariance.22 seed1 roots accept6;18 seed10001 roots accept5. The confirmation RNG differs from the earlier diagnostic stream, so acceptance counts need not equal that diagnostic's6/3.

Report runs/search_confirmation_production_equivalence.json SHA307a66e4dabb929a3e6937984a2b72218bc60da29b383c9e3c5b9cd0c5254a53; actual script/archive hashes verified. Seed1 old/default each logical3,170,304/executed1,863,680/preparation860,160; confirmed11,821,056/6,844,416/1,253,376; old-source gold executes8,650,752. Seed10001 old/default each2,260,992/1,306,624/860,160; confirmed9,338,880/5,435,392/1,253,376; gold7,077,888. All excluded from training.

CLI integration: two reused development pairs seeds1,000,000-1,000,001 with original seed10001 model and uniform128/h192/margin2.5/full-action/chunk32/confirm1024 versus assisted RL-WP. Replayed both complete pairs in reversed order after consuming123 proposal and456 confirmation draws before each seat match; records and all simulation counters exactly equal. Four proposals/confirmations,one accepted. Logical28,041,216/executed13,590,528/preparation1,253,376 per run, with exact stage sum checks. Report runs/search_confirmation_cli2.json SHA1916b88b807f9d0b0e322d21997f8ec6e0ea5b40bec092886b2a479d91f9c425; audit search_confirmation_cli_replay_audit.json. Regression only, no strength inference. Package SHA b5fb6911d7a0fee266f98eca186408fad9148e7d481e743e5fb71312ce6d307f.

## Fixed two-model confirmation strength screen

Commit protocol before launch. Evaluate original seed1 then original seed10001 immutable MLP checkpoints, each exactly1,000 development pairs seeds1,000,000-1,000,999, assisted RL-WP/paper8/deterministic/endgame/CUDA/uniform128/horizon192/confidence2.5/full-action/capture/chunk32/confirmation1,024. Same margin for proposal and confirmation. Retain audited same-model no-search, koi-only and unconfirmed-full-action controls; verify actual bytes before/after each stage. No adaptation after the first model, interim selection/stopping, new training or final-test data.

Primary endpoint in each model is paired confirmation-minus-no-search draw-adjusted win-rate gain,95% pair-bootstrap interval lower bound strictly above0. Replicated improvement requires both model endpoints positive; do not pool shared deals as2,000 independent pairs. Confirmation-minus-unconfirmed, confirmation-minus-koi, corresponding point changes and absolute public win intervals are secondary. This fixed development screen follows selection of the method from the resampling diagnostic; it cannot replace untouched final evaluation. A failed/inconclusive endpoint does not automatically trigger an unchanged extension.

Runner runs/run_confirm1024_search192.py pins packageb5fb6911..., validates control audit sources, actual checkpoints, settings/records/seed range, all stage work and both model summaries. Logs runs/confirm1024_search192_seed1.log and confirm1024_search192_seed10001.log; process manifest runs/confirm1024_search192_process.json. Outputs in each model directory: eval_5000_confirm1024_search192_1000.json and confirm1024_search192_1000_summary.json. Independent raw-record audit remains mandatory after both endpoints. Search simulations never count toward the100M-new training gate.

The fixed two-model screen started after implementation/protocol committed as ed37caf. Runner session18601, actual PID63236(parent66440); seed1 evaluator55908(parent26416), confirmed command/settings and first20 completed pairs. Manifest records Windows child launcher26416; actual interpreter55908 verified separately. Seed10001 remains queued inside the same runner and must not be launched separately. No other experiment is active.

## Fixed confirmation screen completed; replication criterion not met

Both committed ed37caf endpoints completed without adaptation: session18601 exit0, runner63236/final evaluator59684 absent. Each original model1,000 development pairs against assisted RL-WP on the specified common seeds and unchanged uniform128/h192/margin2.5/full-action/chunk32/confirmation1024 settings. Source packageb5fb6911... unchanged; no training or final-test data. Same deals across models are not2,000 independent pairs.

Independent auditor runs/audit_confirm1024_search192.py recomputed all individual/combined endpoint statistics and six paired comparisons from raw paired seat records without koikoi imports, using5,000 pair-bootstrap resamples/default_rng99. Verified actual immutable checkpoint hashes, all report/control hashes, runner/manifest provenance, game identities, contiguous pair range, score signs, historical CPU control equality, phase counts and proposal/confirmation/total work identities. Audit passed, SHAe4a28dc9792271e2e8dab760f72725f4f16821288d2e37e85cf09c3e51e96b9e. This validates recorded endpoint arithmetic and identities, not independent replay of all matches.

Seed1: confirmation48.275% (95%CI46.525-50.05), mean point difference+1.254 (+0.133975..+2.388125). Paired confirmation-minus-control+0.45pp (-0.25..+1.15), points+0.279 (-0.078..+0.632); minus koi-0.1pp (-0.825..+0.625), points-0.135 (-0.545..+0.271); minus unconfirmed full+0.275pp (-0.45..+0.975), points+0.071 (-0.313025..+0.441). Primary positive lower-bound condition fails.

Seed10001: confirmation51.15% (49.35-52.925), points+1.412 (+0.25095..+2.538125). Paired minus control+1.775pp (+1.075..+2.475), points+0.922 (+0.574..+1.295025); minus koi+1.125pp (+0.4..+1.9), points+0.676 (+0.288..+1.061025); minus unconfirmed full0pp (-0.775..+0.775), points+0.15 (-0.282025..+0.583). Primary condition passes on this model only. The equal aggregate51.15% for confirmed/unconfirmed policies does not imply identical records.

Replication requires both model primary lower bounds above0 and therefore failed. Neither absolute interval excludes50%; neither confirmation-minus-unconfirmed interval excludes0. Positive point means do not establish win superiority. Do not adopt confirmation as a default, extend this screen unchanged or launch final testing. These are selected-method development results on reused controls/deals and retain all prior exploratory-selection caveats.

Seed1 counters:230,138 decisions/109,024 searched/983 proposals and confirmations/411 accepted. Proposal logical12,915,818,496/executed6,429,151,232/preparation860,160; confirmation386,531,328/190,382,080/393,216; total13,302,349,824/6,619,533,312/1,253,376. Seed10001:230,552 decisions/109,260 searched/1,024 proposals and confirmations/392 accepted. Proposal12,924,542,976/6,457,806,848/860,160; confirmation402,653,184/197,459,968/393,216; total13,327,196,160/6,655,266,816/1,253,376. All search work excluded from training. Fewer changes alone do not prove better decisions.

Raw report/summary SHA256: seed1 cf3fed38cadd9dc285d58407d7b26976cefbdc021e71a85665f2a50cc71a2da8 / 551123dcfcd0269693fe62707646ab30dbaecc6b038531185a22d25f6d3980e8; seed10001 881ce9c2240a837a209b5f43bbdb0a034822cbac22d7fd8d389f427cca1443b4 / 5e1979e42b781426fb1287b361c5f1f0371d16abc9b6fab1412826f2b0c0ad4c. Paths remain each model directory's eval_5000_confirm1024_search192_1000.json and confirm1024_search192_1000_summary.json. Auditor SHA38f286df8c0f821e75314dc7e488dd4d5e7dc3fa7a484a2d0475547dc4fd9d18; helperSHAfe105ed0033e4346c6b01af9a4540fec16f1f848b4028042d50939c1e7b0b350; manifestSHAcde964a378e30af43a564bc6ef144a7a5736d1bcfc53e56f88a026c8fbbcf37c.

Next inspect rollout-policy mismatch: current search uses candidate-policy continuations for both seats, whereas actual evaluation uses assisted RL-WP as opponent. Confirmation only changes sample independence, leaving this model assumption unchanged. Inspect feasibility of a fixed-root policy-comparison diagnostic before specifying or launching it; this endpoint alone cannot identify mismatch as its cause. Any diagnostic must preserve own/public-only policy inputs and separate its budgets from training and strength evidence. No experiment remains live.

## Rollout opponent-policy diagnostic: verified feasibility and fixed protocol

Replayed the prior resampling diagnostic's25 duplicate baseline pairs per original model, seeds5,100,000-5,100,024; recovered all40 selected roots (22 seed1/18 seed10001) with identical own/public observation hashes and baseline actions.11,486 replay decisions are diagnostic work, not new strength samples or training. Cached simulator states in runs/rollout_opponent_roots.json, SHA f7d2023685927b884adc9a54e674493b2c6614cffa26c0190e1af63d36fa9dc3. These packed states serve only the simulator; policy inputs remain acting-player observations.

Feasibility pilot: first cached root of each model,2 fresh worlds each, separate stream92003. Four arms share worlds: self-play baseline/chosen and candidate-versus-assisted-RL-WP baseline/chosen. Candidate policy is the immutable deterministic guarded MLP; published actions use the existing pinned CPU singleton PublishedAgent with the same endgame guard. Pilot verifies every state transition and policy action of the first world in each arm against independent ReferenceEnv/agent calls, and self-play particle values exactly against production SearchAgent.rollout at the same batch size. At each encountered opponent phase, resampling hidden information leaves native observation, reconstructed published feature tensor and published action unchanged;8 phase checks passed. Both roots' all arms terminate by192 (130/52 steps). Measured1.834/0.406seconds includes checks and startup effects; not a controlled throughput estimate or strength evidence.

Pilot report runs/rollout_opponent_pilot.json SHA34107f2e47b464f3e5d341cf93c23beba52ea2f30f1f79aa31e10d037ce8cf0c. Main logical1,536 transitions each; executed1,040/416; active938/308. Separate reference verification438/106 transitions and gold full rollouts1,536 each. Published compared actions236/61, disagreement74/25. These action disagreements alone do not prove value error. The pilot predates subsequent reporting-only phase/hash additions and enabling its checks for every full-diagnostic root; retain its recorded original script hash.

Fixed diagnostic protocol: all40 cached roots,128 fresh uniform worlds each, SeedSequence([92009,model_index,seed,seat,candidate_decision]), independent of selection/previous confirmation/pilot streams. All four arms share the same initial worlds and future-deal seeds. Continue to the configured192-decision horizon, freezing terminal rows; preserve first terminal match utility from the root player's perspective. At the actual-opponent arms' opponent turns, substitute the existing assisted RL-WP action; candidate turns and all self-play-arm turns use the candidate MLP. Do not change root actions, rerank or select a new policy. Report per-particle values, paired chosen-minus-baseline means/SEs for both continuation models, and paired difference-of-gains. Unfinished branches retain the same candidate-critic bootstrap and are explicitly counted; they cannot establish calibrated opponent values. This measures sensitivity within the assumed uniform proposal model, not the true posterior or a causal explanation of the failed match screen.

Every root repeats the pilot's production self-play equivalence, first-world four-arm ReferenceEnv/policy trajectory checks and hidden-feature checks. Report transition work separately for main batches, active rows, gold reference rollouts and serial checks; record actual checkpoint/published-model/root/script/package hashes before and after. Report action disagreement by phase as a descriptive auxiliary metric. Aggregate root summaries remain descriptive: correlated selected roots are not independent match samples. No root-level aggregate confidence interval, strength inference, new training or final seeds. Complete all40 roots without outcome-based stopping or adjustment; failures require investigation rather than partial result selection.

Script runs/diagnose_rollout_opponent.py SHA baf2c400b572cf96556f9ffdc9bdfdd9e617a3ffdba47b353db818419f93cfd2, packageb5fb6911...; main command .venv/Scripts/python.exe -m runs.diagnose_rollout_opponent run. Output runs/rollout_opponent_diagnostic.json; process manifest runs/rollout_opponent_process.json, log runs/rollout_opponent_diagnostic.log. Commit this protocol before launch, then independently audit raw arrays, exact roots/streams/hashes and work totals before interpreting results. No deployment change is made.

## Opponent-policy diagnostic completed and independently audited

The fixed40-root/128-world diagnostic completed without adaptation: session90654 exit0, actual interpreter10084 absent. Report runs/rollout_opponent_diagnostic.json SHA91180a92268ca04a7622c09b6056891edd71fb4ea58b1c98f3946cef3830f9b0. Independent audit runs/rollout_opponent_independent_audit.json SHA84836d2a3b46c9b8af45725f0ad05ca788c68d59577e6a1e244751d4cd2e15ca passed. Auditor SHA741bb592d4e4022c36e47f037863816bc622e8072361222d619058fb5e8649cd independently reconstructed all40 uniform world streams byte-exactly from cached roots and specified SeedSequences, recomputed all raw-array means/SEs in float64, and checked identities, checkpoints, published-model/source hashes and budgets. Recorded float32 SEs agree within1e-7. This is not independent replay of every published-opponent particle.

At every root, self-play particle values exactly match the production rollout, and four first-world trajectories/policy actions match independent ReferenceEnv/agent execution.154 hidden-feature phase checks passed. Every branch in all40 roots terminates byh192; no candidate-critic bootstrap is used. Utilities below are signed match outcomes (-1/0/+1), not win-rate percentage points. These are descriptive averages over selected, correlated roots.

Seed1 (22 roots): average chosen-minus-baseline self-play utility0.00994318 versus actual published-opponent continuation0.01420455; mean shift+0.00426136. Positive means9/22 versus12/22;2 positive-to-negative and4 negative-to-positive reversals. Seed10001 (18 roots): self0.04470486 versus published0.05815972; shift+0.01345486. Positive means10/18 versus12/18;2 positive-to-negative and3 negative-to-positive reversals. Each comparison shares the same128 worlds; this isolates changing continuation policies within the sampled model, with Monte Carlo uncertainty still present. Opponent changes can affect action rankings in both directions; this diagnostic does not support attributing the failed match replication to systematically optimistic self-play continuations. No causal claim, aggregate independent-root CI or strength gain follows.

Seed1 main logical2,162,688/executed1,042,944/active808,106 transitions; separate production-gold2,162,688 and serial reference6,161. Seed10001 main1,769,472/886,784/669,226; gold1,769,472; serial5,478. Published compared actions196,684/160,977, disagreements59,895/48,651, respectively. Action disagreement alone is not a measure of value error. All work is excluded from training; prior root-replay and pilot budgets remain separate. No experiment remains live, no production policy change and no final-test seeds.

Decision: do not replace search continuations with expensive published-opponent evaluation on this evidence alone. Inspect the other unchanged assumption: uniform hidden-world proposals do not condition on observed opponent action likelihoods. The existing sampled worlds and paired rollout values permit a support diagnostic without new rollouts.

Next fixed diagnostic: replay within-round public action histories for all40 cached roots, first validating all40 actual historical worlds against the exact assisted RL-WP opponent. ReferenceEnv.deal accepts the saved initial deck, so historical steps can be replayed with each phase's recorded public action until the current root; require exact reconstructed root state. For each existing128-world sample from stream92009, check whether every past opponent decision agrees with assisted RL-WP under that hypothetical hand, retaining all observed actions even after a mismatch. Candidate actions are replayed, not scored as opponent evidence. The resulting0/1 consistency indicator is specific to this known deterministic opponent and available within-round history; it is not a general calibrated posterior or a deployable opponent model.

Record compatible-world counts, per-phase mismatch counts and first mismatch, hashes, replay/teacher-call work, and conditional means of the already-recorded four-arm outcomes only when at least one world survives (otherwise null, never fill from unconditioned data or resample until success). Report sampling scarcity explicitly, including roots with no or few survivors; conditional selected-root means remain exploratory and are not strength evidence. Keep all40 roots and128 existing worlds fixed, without adapting sample counts or choosing a deployment policy. Before the full diagnostic, verify the actual-world histories and hidden-feature invariance on replayed opponent decisions; preserve existing rollout artifacts. Commit before launch and independently audit the output. No training or final seeds.

## History-support implementation and verified preflight

Implemented scripts/diagnose_history_support.py and five focused regression tests. The script replays observed candidate/opponent actions to the exact root even after prediction mismatches, scores only opponent decisions, preserves all fixed128 worlds and reports null conditional estimates for empty support. Full predictions, first mismatch, phase counts, separate replay/teacher/guard/invariance budgets and source/artifact hashes are retained. No production package change.

All40 actual cached histories match assisted RL-WP and reconstruct the complete root state exactly. Counts: {"candidate_decisions": 124, "feature_checks": 52, "feature_teacher_calls": 52, "guarded_decisions": 0, "opponent_decisions": 136, "replay_decisions": 260, "teacher_calls": 136}. Hidden resampling preserves legal/native observations, pinned upstream features and assisted actions. The initial pre-format preflight is preserved separately as history_support_preflight_before_format.json; the final-source preflight SHA is 799827876fb40f537685b945ad521c611116c2e31c3e0acbf072dc2c3c4a8a95, script SHA f95eeefdce3dc4264804b943deea8616f49f9d64a00c6ceb4c45a3573d2246a7. Five focused tests pass; Ruff passes.

Command: .venv/Scripts/python.exe -m scripts.diagnose_history_support run. This launches exactly the previously fixed40 roots and128 existing worlds, without new rollouts. Commit this implementation/preflight before launch. Independent native historical replay, raw event arithmetic and conditional-array auditing remain mandatory.

## Fixed history-support diagnostic completed and independently replayed

Implementation/preflight commit6b5b76a; full session32683 exited0, actual interpreter10064. Independent audit session93150 exited0. All40 fixed roots and128 existing worlds completed without sample/root adaptation, new rollouts, training or final seeds. Every hypothetical history reconstructs its target root exactly, including all actions after prediction mismatches. Report runs/history_support_diagnostic.json SHA ec8d8656e855f413779f0337bd3e207f745ce324d983598d474e53f06f5c8259; independent audit runs/history_support_independent_audit.json SHA 4989482b53401fd93194d05d9eb8fd41caf567dbe8b58d775376b014207093f7.

Tracked auditor scripts/audit_history_support.py SHA 53aa149c1aa2598f39033dfa57407b3fd47ee4756c5d2c4328710fba11a9eb5a independently reconstructs all40 sampled world streams, initializes round states without the diagnostic replay helper, and replays all40 actual plus5,120 hypothetical histories through native CPU transitions. Every recorded prediction is rechecked with pinned PublishedAgent and the native solved-action guard; all states, event identities, first mismatches, counts, hashes and conditional means agree. This is independent world/transition/arithmetic validation, not an independent teacher implementation or a rerun of future rollouts. Five focused regression tests and Ruff pass; production packageb5fb6911... unchanged.

Seed1:22 roots,1,164/2,816 sampled worlds compatible; no zero-support roots,4 roots with1-4 survivors,4 roots have no observed opponent decisions (all128 survive there). Seed10001:18 roots,620/2,304 compatible; no zero-support roots,6 roots with1-4 survivors,2 roots have no opponent history. These totals are descriptive sample counts, not an accuracy measure. All40 true historical worlds pass;52 phase-specific hidden-feature checks pass. Known deterministic assisted RL-WP is assumed exactly; no general posterior or deployable opponent inference claim.

Descriptive all-root mean chosen-minus-baseline signed utility: seed1 self-play unfiltered0.00994318 versus filtered0.00250591; published continuation0.01420455 versus0.04526429. Seed10001 self-play0.04470486 versus0.03101003; published0.05815972 versus0.03701789. These are signed-outcome utility units, not win-rate percentage points. At roots with at least one past opponent decision (18/16), corresponding published means unfiltered0.02256944/0.05810547 and filtered0.06053135/0.03432090. Conditioning shifts values in different directions across models;10/40 roots have only1-4 surviving worlds. Selected correlated roots and scarce conditional samples cannot establish improved ranking or match strength. No default adoption or new strength screen follows from these means alone.

Full replay work33,280 decisions (15,872 candidate,17,408 opponent);17,408 teacher calls,zero guarded decisions. Preflight260 decisions (124 candidate,136 opponent),136 main teacher calls plus52 invariance teacher calls; the preserved pre-format preflight repeats these same budgets and is separate. Native independent audit33,540 transitions and17,544 teacher calls. All work excluded from training; previous rollout values reused unchanged. Per-phase full comparisons/mismatches: {"0": {"comparisons": 14976, "mismatches": 6593}, "1": {"comparisons": 1536, "mismatches": 28}, "2": {"comparisons": 384, "mismatches": 29}, "3": {"comparisons": 512, "mismatches": 5}}.

Read-only combinatorial inventory runs/history_support_hand_inventory.json SHA 036266d1ae216077fd9082c19d89129684be4bc7cca92d33399ceab1ffb26894. Distinct current opponent hands in the128 worlds range12-128 per root. Most early roots have nearly no duplicate hands, so memoization alone cannot fix sparse rejection. Nine structurally late roots have at most1,000 possible current hands before valid-initial-deal filtering,4,792 candidate hands in total. Root selection here uses only public unknown-card/hand counts, not observed support or utility results.

Next fixed feasibility protocol: enumerate every current opponent hand at all nine existing roots whose combinatorial upper bound is<=1,000. Reject only invalid initial hands using the existing deal rule, then replay the same observed opponent history with the exact pinned assisted RL-WP. Compute exact within-round deterministic-policy support counts, retaining zero support explicitly; do not search until success. All4792 pre-validity candidates must be accounted for. Use a canonical remaining-stock order only for this past-action support computation, never for future rollout values. Check every existing128-world indicator against its enumerated hand, and verify remaining-stock/RNG invariance of features/actions in the enumerated histories. Preserve actual-world inclusion and independently audit enumeration completeness, compatibility, hashes and teacher/replay budgets. This is a feasibility/support diagnostic, not a deployment method, general calibrated belief, strength test or new rollout experiment. No automatic sample-size extension or final seeds. If exact support is useful, specify a separate fixed validation protocol before drawing future rollout worlds; do not choose that protocol from partial enumeration results.

## Exact-support enumeration implementation/preflight

Implemented scripts/enumerate_history_support.py for the fixed9-root/4,792-candidate protocol. Public observation determines unknown cards and opponent hand count; observed opponent discards reconstruct the initial hand. Canonical stock order and seed0 are used only for past-action support. Every opponent prediction checks native/public features, upstream features and assisted action equality after reversing the remaining stock and changing RNG state. All mismatched histories still replay to the exact root. Every previous sampled-world event sequence will be cross-checked against its enumerated hand.

All9 actual-hand preflights pass with identical historical teacher actions and root reconstruction. Budgets {'replay_decisions': 111, 'candidate_decisions': 58, 'opponent_decisions': 53, 'teacher_calls': 53, 'guarded_decisions': 0, 'feature_checks': 0, 'feature_teacher_calls': 0, 'stock_rng_feature_checks': 53, 'extra_teacher_calls': 53}. Preflight runs/history_support_enumeration_preflight.json SHA c228882dc7e0b8f0a4ca3fec3e554ada50181a3f949a304535115e4e41ca7c88; scriptSHA e1aa447a8dfc23696b8b84bb5243b85408c034751b4c7814b19580ea7616f9c1. Seven focused tests pass, including hidden-state-independent canonical hypotheses and invalid-initial-month rejection; Ruff/diff checks pass. No production package change. Commit before full command .venv/Scripts/python.exe -m scripts.enumerate_history_support run; independent completeness/native replay audit after all9 roots.

## Exact-support enumeration completed and audited

Full enumeration session48072/actual interpreter9860 exited0; independent audit session9967 exited0. All9 prescribed roots and4,792 hand combinations accounted for:4,789 valid initial hands,3 invalid. Report runs/history_support_enumeration.json SHA 57f3c88691b183aa3dfcdf85b0ef761fe7a08b90c2472d80b0e8a03017b6c052; audit runs/history_support_enumeration_audit.json SHA ff79e702f4871f47a9136d17acde346652a830b4275b6ed72ae770d51926d7e4. Auditor scripts/audit_history_enumeration.py SHA eb169374f0f73f3e815ff11aecafa7ee154802aa33be0c13419c537fb440cea8 independently enumerates combinations recursively, derives unknown cards/counts from native public observations, checks all invalid-month exclusions, and replays every valid hand through native CPU transitions under two different remaining-stock/RNG settings. Every teacher prediction and all1,152 previous sampled-world event sequences agree. Actual hands are included and compatible in every root. The pinned teacher inference adapter is shared, not an independent teacher implementation.

In cached-root order [1,7,9,11,12,22,32,37,38], valid-hand counts [816,680,816,77,816,78,12,679,815]; exact compatible counts [44,1,20,3,27,1,8,26,33]. No empty support. In root7, prior128 uniform draws have expected support size128/680=0.188235, so merely filtering128 worlds cannot reliably provide a substantial conditional sample. Root22 also has one compatible hand (among78). These are known deterministic assisted-RL-WP, within-round support statements, not general human-opponent certainty or strength evidence.

Budgets: enumeration52,135 observed transitions (24,884 candidate/27,251 opponent),27,251 teacher calls plus27,251 stock/RNG invariant action calls;27,251 full native/upstream feature checks. Preflight111 transitions/53 teacher+53 invariant calls. Independent native audit104,270 transitions/54,502 teacher calls. All work excluded from training; no future rollouts in this enumeration.13 focused tests now pass (support replay, enumeration validity/invariance, fresh conditional sampler), Ruff/diff checks pass. Production packageb5fb6911... unchanged.

## Fixed conditional-world value validation protocol

Proceed with exactly the same9 structurally selected roots, retaining their prior baseline and selected actions. Compare two world distributions: uniform valid hidden hands (existing sampler) and uniform selection from the complete audited compatible-hand set. For both, uniformly shuffle remaining stock and draw independent future-deal seeds; never evaluate canonical stock. scripts/sample_history_support.py validates nonempty unique support and samples fresh worlds; tests verify public-observation preservation, legal full-deck permutations, input immutability, fresh stocks and identical worlds/RNG states after changing actual hidden information or support-list order.

Main:1,024 worlds per root per distribution, SeedSequence([94009,model_index,seed,seat,candidate_decision,distribution_index]), distribution0=uniform,1=history-supported. Within each distribution, common worlds across four arms: self baseline/chosen and actual assisted-RL-WP opponent baseline/chosen. Candidate remains the immutable same-model deterministic guarded MLP. Horizon192, first terminal signed match utility, existing candidate-critic tail only if unfinished (count explicitly). Preserve native/reference trajectory, production self-play equality and hidden-feature checks from the audited previous rollout routine. Promote that routine into a tracked helper and require byte-exact pilot values/counts against its archived source before main launch.

Fixed feasibility pilot: first enumerated root of each model,2 worlds per distribution, separate stream94003. All four root/distribution combinations, no selection by outcomes. Main follows only if implementation checks pass. Every main root/distribution completes without adjustment. Record per-world raw utilities, sampled-hand indices for supported worlds, actual model/source/input/world hashes, all main/verification transition and teacher budgets, and unfinished counts. Paired chosen-minus-baseline estimates are within each distribution; supported-minus-uniform differences use independent-stream standard errors, never pretend worlds are paired across distributions. Root aggregates remain descriptive; no aggregate independent-root CI, new policy selection, match-strength inference, final seeds or new training. Independently reconstruct both full world streams and all raw-array statistics after completion. Any later deployment/strength screen needs a separate fixed protocol.

## Conditional-world value runner verified before launch

Promoted archived four-arm rollout into tracked scripts/rollout_opponent.py. scripts/validate_supported_values.py implements the fixed nine-root/two-distribution/1,024-world protocol; no production package changes. The fixed pilot runs both distributions at the first root of each model,2 worlds each on stream94003. All four groups exactly match archived rollout values and every existing counter; each also passes full-horizon production self-play equality, four serial ReferenceEnv trajectories and hidden-feature checks. All pilot branches terminate, no bootstrap. Pilot values are correctness checks, not strength evidence.

Independent scripts/audit_supported_values.py reconstructs all pilot worlds without the production sampler, including identical support indices, and checks raw arrays, independent-stream difference SEs, complete metadata/artifact hashes and work totals. Audit passed. Pilot SHA 408fe647e35a04712bcaee653f9fc16ae26b1e1252a8382f20f7fe61a4216411; pilot audit SHA 7fef507cf715db449328b35655f17751bf98c8fdf6dbfa694ee5f055c8adbf5b. Pilot main budgets {'executed_transitions': 840, 'active_transitions': 749, 'published_calls': 170, 'compared_opponent_actions': 170, 'opponent_action_disagreements': 71, 'feature_invariance_checks': 8, 'serial_checks': 365, 'logical_transitions': 6144, 'steps': 105, 'gold_executed_transitions': 6144, 'total_teacher_calls': 261, 'verification_teacher_calls': 91}; separate archived equivalence replay {'executed_transitions': 840, 'active_transitions': 749, 'published_calls': 170, 'compared_opponent_actions': 170, 'opponent_action_disagreements': 71, 'feature_invariance_checks': 8, 'serial_checks': 365, 'logical_transitions': 6144, 'steps': 105, 'gold_executed_transitions': 6144, 'total_teacher_calls': 261}. No training/final seeds.13 focused tests and Ruff/diff checks pass.

Runner SHA eb54c9e8ad8e4c69ff1d49903891c346906e8abf92b482505ed92bcbf40a1a42; rollout helper SHA c9780f9c975ac77ed34afd5aead28093a9d905f8835f3952e99aa95ddfafc02f; sampler SHA 4410552d1b20c59274f1af8d1ebbd2b3b58127b68626dc9c48214c561d39c714; auditor SHA 69fc8d97d6cc0c67589d89288a9d30926b7787da445f287979644e282d9d8243. Commit before main command .venv/Scripts/python.exe -m scripts.validate_supported_values run. Manifest runs/supported_values_process.json; log runs/supported_values_diagnostic.log; progress snapshot supported_values_partial.json; final supported_values_diagnostic.json. After completion run .venv/Scripts/python.exe -m scripts.audit_supported_values run. No interim policy changes, resampling, root selection or duplicate launches.

The fixed main conditional-world comparison launched after commit653998b. Session56729; actual interpreter26240(parent18788), Windows launcher18788(parent54260), verified live command .venv/Scripts/python.exe -m scripts.validate_supported_values run. First root1/uniform group completed:1,024 worlds,4,096 arms,all branches terminal by7 steps; exact self-play/reference/hidden-feature checks pass. Main28,672 executed/18,356 active/786,432 logical transitions, separate gold786,432 and22 serial checks;3,396 main teacher calls plus5 verification calls. This is operational confirmation, not interim method selection. Run all18 fixed groups without adjustment, then the independent auditor.

## Fixed conditional-world value comparison completed and audited

The full fixed9-root x2-distribution x1,024-world comparison completed; session56729 exited0 and actual interpreter26240 is absent. Report runs/supported_values_diagnostic.json SHA50b1da8bd09936a7657179af1265e8ffc81707549916f3c24e59382e5b0a3d0f. Independent audit runs/supported_values_independent_audit.json SHAe1cca166e4ee9348624211be925f275db2c8ec4bdc483bf3bdf434649640775e passed. Auditor69fc8d97d6cc0c67589d89288a9d30926b7787da445f287979644e282d9d8243 independently reconstructed every uniform/supported world stream and support index, recomputed raw-array means/SEs and independent-stream differences, and checked source/checkpoint/input hashes and work counts. This is not independent replay of every future trajectory. Each group already passed production self-play equality, four serial reference trajectories and hidden-feature checks.

All branches in all18 groups terminate byh192, no critic bootstrap. Selected-action-minus-baseline signed-match-utility estimates below use1,024 worlds per distribution. These are NOT win-rate percentage points, independent match samples or deployment strength. The uniform/supported draws use independent streams; the four action/continuation arms share worlds only within each distribution.

| Cached root | Model | Uniform self | Supported self | Uniform published | Supported published |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | seed1 | 0.07421875 | 0.04296875 | 0.04492188 | -0.00390625 |
| 7 | seed1 | 0.12109375 | -0.06933594 | 0.08203125 | -0.07714844 |
| 9 | seed1 | 0.02050781 | 0.00292969 | -0.01367188 | 0.00781250 |
| 11 | seed1 | 0.30957031 | 0.60546875 | 0.30371094 | 0.57324219 |
| 12 | seed1 | 0.05957031 | 0.00585938 | 0.04589844 | -0.00976562 |
| 22 | seed10001 | 0.11621094 | 0.07324219 | 0.11718750 | 0.05371094 |
| 32 | seed10001 | 0.08593750 | 0.04589844 | 0.05566406 | 0.02636719 |
| 37 | seed10001 | 0.01757812 | 0.01953125 | 0.03613281 | 0.06250000 |
| 38 | seed10001 | 0.00976562 | 0.00585938 | 0.02636719 | 0.00292969 |

At root7, the known-opponent continuation estimate reverses from+0.08203125(SE0.02182781) under uniform hands to-0.07714844(SE0.02716285) under exact history support; independent-distribution difference-0.15917969(SE0.03484643). Self-play also reverses+0.12109375 to-0.06933594. At root11, published gain increases0.30371094 to0.57324219, difference+0.26953125(SE0.04643820); self0.30957031 to0.60546875. These demonstrate sensitivity of selected-root decisions to the assumed hidden-hand distribution. They do not establish the cause of all failed strength screens or a generally calibrated belief model.

Descriptive mean gains across the five seed1 roots: uniform self/published0.11699219/0.09257813, supported0.11757813/0.09804688. Across four seed10001 roots: uniform0.05737305/0.05883789, supported0.03613281/0.03637695. Seed1 published means are positive at4/5 uniform and2/5 supported roots; seed10001 at4/4 under both. Correlated selected roots, unequal root importance and a large root11 gain prevent a match-strength interpretation of these averages. No aggregate independent-root CI, default adoption, model selection or final evaluation follows. Exact-support confirmation is motivated as a candidate method; it has not been implemented in the deployment path or strength-tested.

Main logical14,155,776/executed6,307,840/active4,736,438 transitions; separate production gold14,155,776 and serial reference4,634. Main published calls1,149,282, action disagreements347,459;1,221 additional verification teacher calls (total1,150,503),66 hidden-feature checks. Main group elapsed sum2,492.391seconds includes verification/startup overhead and is not a controlled throughput benchmark. Pilot/archive-equivalence/enumeration work remains separate. No new training or final seeds. The full production source remainedb5fb6911... for this diagnostic. No experiment remains live.

Next implementation priority: a compatible native CPU/CUDA builder for the pinned published model's300 input features, followed by an explicitly identified GPU published-policy adapter. Current slow CPU singleton/reconstruction calls constrain both history-aware search studies and direct published-opponent training. Existing frozen-opponent training uses only our PolicyValue checkpoints; the reviewed experiment record contains no direct published-RL-WP training or distillation. Do not infer that direct training will improve strength, and do not replace the strict CPU comparison opponent.

Implementation contract: preserve original card-feature quirks, chronological public-history logs, phase masks, koikoi tokens, score transforms and paper8 restriction. Derive features only from acting-player own/public information; never depend on actual opposing hand assignment, hidden stock order or RNG state. Compare with the existing pinned reconstruct(...).feature_tensor on all phases, true and resampled worlds, and full transitions. Validate CPU/native/GPU features independently, quantify GPU logits/action differences and timing by batch size, and retain the original CPU singleton as benchmark ground truth. Floating-point action equality is not assumed. A GPU policy with any disagreements is a separately identified approximation, suitable only for an explicitly specified experiment; it cannot silently replace the exact deterministic policy used to certify support. No teacher-training or new strength run until compatibility/throughput evidence and a fixed experiment protocol are committed. Keep previous diagnostic artifacts immutable and record any new production source hash.


## Fixed published-batch compatibility and timing protocol

Before launch: validate all exposed decisions from100 complete random paper8 matches, seeds6,200,100-6,200,199; action RNG62021, uniform legal card choices and continue probability0.7. Add the fixed40 cached diagnostic roots (SHAf7d2023685927b884adc9a54e674493b2c6614cffa26c0190e1af63d36fa9dc3) and two resampled hidden worlds for16 systematically spaced roots per phase (RNG62029). Include100 terminal states as separate zero-feature/no-action checks. This is a correctness corpus, not strength data or training.

For both published RL-WP and RL-Point, retain strict original CPU singleton feature/model output as oracle. Compare native CPU/CUDA feature hashes exactly and masks/guards/state preservation; quantify legal-logit errors, raw and assisted action disagreements by phase, disagreement indices and oracle margins. Native inference uses batch256, float32, two CPU threads, matmul/cuDNN TF32 disabled, cuDNN benchmark disabled. Preserve raw arrays and checkpoint/source/runtime hashes. No numerical action-identity assumption and no replacement of the strict comparison opponent or exact-support teacher.

Measure warm steady-state WP unassisted feature and complete policy latency at batch1/32/256 for each phase and mixed states, plus mixed1024. Three warmups,20 feature and5 policy repeats, GPU synchronized around every timed call; report raw times and medians. Compare serial strict original inference on32 states per phase, three repeats. Exclude model/environment construction and transfers; include feature creation, phase grouping, masks and action selection in batch policy timing. No concurrent training/testing during timings. No strength conclusion from throughput. Commands: python -m scripts.validate_published_batch and the subsequent standalone raw-array auditor. Artifacts runs/published_batch_validation.{json,npz}; preserve completed outputs.


## Native published features and explicit batch adapter: verified milestone

Implemented shared C++/CUDA300-feature extraction and BatchPublishedAgent for WP/Point, paper8 only. Historical upstream feature quirks and chronological logs are preserved; inputs use acting-player own/public information only. Original CPU PublishedAgent, transitions, search defaults and training remain unchanged. Optional native guard, frozen float32 models, terminal handling, torch RNG preservation and ambient-autocast exclusion are tested. TF32 remains an explicit caller setting; the dynamic phase-grouping path is not graph-capturable. New production source SHA3c3f6ab6abe426db61ae79b80016f95a3ec8f7645a8f6bc2496b00c503ac85d9.

Fixed diagnostic completed session22260 exit0, actual PID64236 absent. Runner scripts/validate_published_batch.py SHAfd1af63064964bcbd0ed6173d1d93c9c7be7ce3fc5538243d000ecc53ad7aed6.100 complete paper8 random matches produce781 rounds and12,861 decisions;40 cached roots plus128 hidden variants give13,029 active states. Phase counts discard11,014/pick-hand511/pick-draw506/koi998.100 terminal states separately checked per device. Every native CPU/CUDA feature digest exactly equals the original input, including hidden variants; native/reference masks and public guard agree, states remain unchanged. Neither simulator decisions nor these diagnostics count toward training/strength budgets.

Report runs/published_batch_validation.json SHA0b040cd31c1d42e945bd233ea94d4f57e06d8d2ffae4a63ffded5c68c8aed544; rawNPZ SHAe49004ce6795da3c23fbcb036c4d23712b6676a8d96cde1b43f6fca28d17d322. Standalone auditor scripts/audit_published_batch.py SHAb00a8bf4a6b02dc1771b55295ef2f2e81f898e14b6d2dcefdcd046f4bdd94304; audit runs/published_batch_independent_audit.json SHAe2a9567be3f05bf6a72ff6cbf3c56383463e9cd07d535f8bd90cddcb7a9599d4. Audit independently regenerates every random trajectory and checks terminal-derived rounds, cached identities, reference masks/guards/public observations, source/checkpoint hashes, stored feature-digest equality, all raw-logit summaries and68 timing-row calculations. It does not independently rerun every model inference or measure timings again.

Both unassisted and assisted actions agree with strict CPU singleton on all13,029 states for all four variant/device combinations at batch256. Maximum absolute legal-logit errors:

| Variant | Native CPU batch | CUDA batch |
| --- | ---: | ---: |
| RL-WP | 0.00012540817260742188 | 0.00015354156494140625 |
| RL-Point | 0.0006157755851745605 | 0.0013437271118164062 |

These finite numerical checks do not prove universal action identity, especially near ties. Do not use the GPU adapter as an exact CPU-teacher support certificate. Runtime Torch2.12.0+cu130/NumPy2.5.4, RTX5070Ti, two CPU threads, float32, matmul/cuDNN TF32 disabled and cuDNN benchmark disabled. Upstream89d7895 and all six actual model hashes are embedded in the report. WP hashes match prior diagnostics. No hidden information is used to select teacher actions.

Corrected-run mixed-state WP policy timings (median of five synchronized calls after three warmups):

| Batch | CPU ms | GPU ms | GPU states/sec |
| --- | ---: | ---: | ---: |
| 1 | 2.1062 | 1.1676 | 856.46 |
| 32 | 34.8256 | 2.9379 | 10,892.13 |
| 256 | 248.4141 | 4.6879 | 54,608.67 |
| 1024 | 888.1026 | 13.6661 | 74,929.94 |

Policy timing includes native features, phase grouping, models, masks and argmax, but excludes construction and transfers. Mixed rows use the corpus prefix, phase rows the phase-filtered prefix; smaller phase corpora cycle if needed. Original serial CPU inference on32 states per phase measured601.65/463.98/592.26/430.57 states/sec for discard/pick-hand/pick-draw/koi; these are different input mixtures, not a matched overall speedup baseline. Feature-only timing and raw samples are retained. GPU feature latency is not monotonic in batch size. Session variability is substantial: the initial pilot's mixed256/1024 policy timings were19.0430/18.1373ms, versus4.6879/13.6661ms in the final run. No universal speedup or complete training throughput claim follows. No concurrent tests/training ran during timed measurements.

The initial run incorrectly counted positive terminal outcomes as rounds (43). Inspection found the wrong result-column index; the initial auditor mirrored that mistake. Preserve both reports/source files under runs/published_batch_counter_pilot. The corrected runner uses the round-end flag; the auditor additionally derives rounds from each terminal state's round index. Rerunning the unchanged corpus yields781 rounds and all raw arrays exactly equal the pilot. The first PowerShell wrapper also returned1 after redirecting an expected upstream warning; the corrected subprocess wrapper records actual Python exit0. Pilot timings are retained separately, not selected as the final result.

Validation: python -m pytest -q =>210 passed, no skips,7 expected upstream Transformer warnings,76.88s. A final explicit float32 model-construction fix then passed all12 related feature/adapter tests (including one new global-float64 construction case),6 expected warnings,4.29s. Ruff lint across koikoi/scripts/tests passes; all seven changed Python files pass formatter check; git diff --check passes. Whole-tree formatter check found four pre-existing unrelated files (agents.py, opponents.py, train.py, test_koi_learning.py); left untouched. Added coverage includes full-history upstream/CPU/CUDA feature equality, hidden resampling, all phases and terminal rows, singleton model actions, variant/profile rejection, float32 buffer safety under a changed global dtype, ambient autocast, state/RNG preservation and frozen parameters. Tests and timing did not overlap.

Next implementation candidate: optional direct published-opponent paper8 training with the verified explicit adapter. Existing frozen-opponent collection supports our PolicyValue checkpoints only. Preserve masking of opponent actions from learner losses, the distinct opponent-role critic, player-relative returns and per-match seat alternation, and record exact teacher hashes/runtime in resumable checkpoints. Dynamic teacher grouping requires an explicit eager path or separately validated capture support. Fix a bounded multi-seed protocol only after complete training throughput/correctness checks; assess strength with unchanged strict CPU assisted-WP on development pairs, leaving final seeds untouched. This milestone supplies a faster experimental opponent API, not evidence of a stronger learned agent.


Final dtype hardening explicitly casts loaded models to float32 even when the caller's global default is float64; this prevents a mixed-dtype inference failure. The pre-hardening corrected report/audit and adapter source are preserved in runs/published_batch_pre_explicit_dtype. The final source above reran the identical fixed diagnostic; all stored raw arrays (states, masks, guards, features and logits) equal both previous executions. Final Python exit0 and all raw audit checks passed. Production behavior under the originally tested float32 default is unchanged; timing values above are from the final run, not selected across runs.


## Fixed direct-published-training throughput diagnostic

Before launch: use configs/ppo8_published_wp50.json, a fresh MLP128x2 learner, seed6,300,001, paper8/endgame/match PPO,2048 environments, horizon128, three epochs, minibatch8192, lr0.0001, entropy0.02, lambda0.95, CPU threads2, float32. Assign50% of environments to deterministic assisted batch RL-WP; the rest self-play. Eager collection and captured optimizer minibatches; TF32/cuDNN benchmarking disabled by the explicitly identified teacher mode. Five warmup updates then exactly20 timed complete updates, no initialization checkpoint, no model selection or evaluation. Record actual match/round/simulator/learner/published counters, warmup counts, teacher hashes/runtime and full source provenance. Timing includes teacher inference and training, excludes setup/warmup. Do not run other tests or GPU jobs during this measurement. All work is a throughput/correctness diagnostic and excluded from strength/scale claims.

Implementation validation before timing: real published-opponent CPU/CUDA trajectories against reference transitions/strict CPU teacher on the tested states, terminal and bootstrap signed returns, separate teacher exposure counters, opponent-only loss exclusion, embedded teacher checkpoint restart on CPU/CUDA and captured minibatches, missing/tampered weights/identity/runtime rejection, and WP/Point RNG preservation. Also compare current defaults against actual archived Trainer source at3a395248493f4037ea1fe43afb0a56d6c48ce3cb in five modes: CPU self-play/historical mixture and CUDA eager self-play/captured self-play/captured historical mixture, two updates each. Preserve source hashes, all exact-state checks and limitations. No strength run until a separate bounded protocol is fixed after this diagnostic.


Direct published training implementation checks passed before benchmark launch: full224 pytest tests, no skips,21 expected original Transformer warnings,80.59s. New13 tests cover actual CPU/CUDA teacher trajectories and full terminal/bootstrap returns, exact exposure counts, actor-loss exclusion, embedded-teacher exact restart (CPU, CUDA eager optimizer, CUDA captured optimizer), tampered parameter/identity/runtime/missing checkpoint rejection, unsupported modes, runtime drift and WP/Point RNG preservation. Tracked production source1c36895f2b5164c4c0a86862e2270364a74c2ff0ee4d950f4b0c4cd970575914. Source-default equivalence report runs/published_training_default_equivalence.json SHA894b3a02b8c68bceeb28a8661d38c2d7cea5f59b150c0ef910f838d06843292e; validator scripts/validate_training_defaults.py SHA88abc4462e8b5159cc7f591e8441905e7e15f339b54e2bf113f414c695096513. Actual archived trainer3a39524 equals current defaults for five modes/two updates each in metrics, trainable/Adam tensors, environments, counters and CPU/CUDA RNG. Shared unchanged model/simulator modules are not independent reimplementations. The earlier validator invocation before a lint-only comment is superseded by the recorded invocation; no extra strength samples.


## Direct published training throughput completed

Implementation/protocol commit747b26c. The fixed benchmark completed session31751 exit0, actual PID30096 absent. Report runs/published_training_benchmark.json SHA41df452c66111480e7c685c5cf0f4e6cbba60bfc1473d0cf64c1a72681924cfe. Source1c36895f... unchanged.20 timed updates take25.8062283seconds:5,242,880 simulator decisions,3,786,964 learner and1,455,916 published decisions;349,132 rounds/48,444 complete matches, including180,670 rounds/26,850 complete matches in the published-opponent environment subset. Rates203,163.36 simulator decisions/s,146,746.12 learner decisions/s,1,877.22 complete matches/s. These are this fresh-learner mixture's throughput, not a deployment or strength result.

Five excluded warmup updates:1,310,720 decisions (945,394 learner/365,326 published),86,966 rounds/11,320 matches (published44,306 rounds/6,223 matches). All benchmark work is separate from both strength endpoints and the completed research12 scale budget. Counter partition and rate arithmetic were checked; trajectory/count semantics are independently tested, but this entire timed trajectory was not independently replayed. WP teacher parameter SHA7d684d9063b8c4bbe406c7b50f9b43ac90bb774f5917b13522d28be1a81925f4, original three weight hashes and all runtime flags are recorded. This establishes practical throughput for a bounded direct-opponent training experiment.

## Fixed two-parent WP50 adaptation screen

Freeze before launch. Run scripts/run_published_screen.py, source SHA c5b7c63a5881222ce3b896b1c3dd7e585d3d9970cb9deb311c882ad9f13422d2, production1c36895f2b5164c4c0a86862e2270364a74c2ff0ee4d950f4b0c4cd970575914. Preflight runs/published_wp50_screen_preflight.json SHA7c13c7651a59c0c2ae5f8ee8be5f95d08b4af19b9589f99032446cf1da7a6f43 validates source, actual parent/teacher bytes, configuration and complete baseline pair identities. Runner creates an exclusive runs/published_wp50_screen directory, atomically records worker PIDs/job states and preserves all logs/configs/checkpoints. Four training jobs precede all four evaluations; no interim checkpoint evaluation or selection.

Parents are immutable original seed1 runs/ppo8_v3/update_00005000.pt SHAd104ae00824aca114e6cd414c74d02670b9a8de5a009ad4fad557005f66233e5 and seed10001 runs/ppo8_seed10001_captured/update_00005000.pt SHAedc44213bc5c5c840297afc33ee64a57b91e9dee48c202535906fc916cde109d. Weight-only initialize two fresh arms from each parent: WP50 uses the verified deterministic assisted published GPU teacher in50% of environments, while control uses ordinary self-play. Both use the same parent-specific new deal seed (6,300,001 or6,310,001), paper8/endgame, MLP128x2,2048 environments, horizon128, lr0.0001, entropy0.02, lambda0.95, three PPO epochs, minibatch8192, CPU threads2, float32, eager collection and captured optimizer. The runner disables matmul/cuDNN TF32 and cuDNN benchmarking for BOTH arms. The published arm has its separate role critic; control has no opponent-role critic. This measures the complete direct-opponent adaptation intervention, not an isolated causal effect of opponent identity or the role critic.

Exactly500 updates per arm,131,072,000 simulator decisions each;Total four-arm simulator budget is524,288,000 decisions. Learner decision counts differ because teacher actions are excluded; report actual learner/published and complete-match/round counts rather than claiming equal learner exposure. Save every100 updates solely for recovery and evaluate only update500. Fresh optimizers/deals/RNG/counters, no inherited optimizer or added opponent-pretraining budget. The benchmark uses no inherited actor and supplies no selected model to this screen. No unchanged extension, learning-rate adjustment or checkpoint reranking. Any process/non-finite failure stops the runner and preserves its last completed checkpoint.

Evaluate each final actor with the unchanged strict CPU assisted RL-WP comparator, deterministic candidate/endgame/no-search, paper8,1,000 duplicate pairs seeds1,000,000-1,000,999, protocol2, CPU threads2. Use existing full parent controls on the same pairs: seed1 eval_5000_wp.json SHA34d8b04b06d7e8f39082ceb76a5d58be0ed54c1fee2c56af4eb022c46ca58f97; seed10001 eval_5000_wp.json SHA1cd58138616254de2dc658307aaa5f7ebe7915aa4145bc0fe6b0ea6a0ffb4ca6. Record actual final checkpoint hashes, provenance, training budgets and every paired outcome. Bootstrap pairs5,000 times with seed99; no pooling model runs as extra independent deals.

Prespecified eligibility criterion: for BOTH parents, WP50-minus-control and WP50-minus-parent paired draw-adjusted win-score95% CI lower bounds must be positive. Separately require BOTH WP50 absolute win-rate lower bounds above50% for a replicated public-superiority development signal. Point difference is secondary and cannot replace either win criterion. Control-minus-parent is descriptive. Shared, repeatedly explored development deals and these multiple endpoints do not supply independent final confirmation or a universal strongest-AI claim. Independently audit raw results/checkpoints before any method decision; if promising, fix a separate independent confirmation protocol. Final seeds100,000,000+ and RL-Point final comparisons remain untouched.


The fixed screen launched after commit3c6ee04 was pushed. Protocol artifact runs/published_wp50_screen/protocol.json SHA9409200f1b6990b1c0d31087ee1967f40c586c11a806566fd2773d381f59ce5f; source remains1c36895f..., runnerc5b7c63a.... Session95032, runner44560; first train:seed1_wp50 worker shim57512/actual interpreter61808 verified live. Initial monitored25 updates:6,553,600 decisions (4,902,196 learner/1,651,404 teacher),439,054 rounds/55,303 matches, including published220,689 rounds/27,794 matches; finite losses/gradients. These are initial budget counters, not a selected checkpoint or win-rate result. Atomic process.json records later worker transitions. Retain the active pipeline, fixed stop condition and strict CPU comparator; audit raw outcomes and immutable final checkpoints after completion.


## WP50 screen: first parent completed, checkpoint audit prepared

The existing session95032/runner44560 completed both seed1 jobs with exit0 and advanced automatically to train:seed10001_wp50 (shim63960/actual interpreter61672 verified live). Production source1c36895f... and runnerc5b7c63a... are unchanged. No endpoint strength evaluation or intermediate checkpoint selection has occurred.

Seed1 WP50 finished exactly500 updates/131,072,000 simulator decisions,1,123,348 complete matches/8,788,643 rounds.98,076,843 learner and32,995,157 teacher decisions;564,658 matches/4,419,681 rounds in the published-opponent subset. Final artifact seed1_wp50/update_00000500.pt SHA1c6ab2b474bf4ef54cfe4932b595c1dfa3c5b075a295ec52bd1d8b9e4908d341; complete500-row metrics SHA16c4775b989a47a4ee4fcb550b98e8ca9dbd08db040b011aeb96f0e4301f30fd. Recorded training elapsed610.4091seconds includes checkpoint work.

Seed1 control also finished exactly500 updates/131,072,000 decisions,1,119,106 matches/8,751,686 rounds; every decision is a learner decision. Final artifact seed1_control/update_00000500.pt SHA4b1b5e94d428f4dea5c2c2ed810c3318258c84324edeebd9e526249cbb6d150e; metrics SHA47d04d66cf40e61c0d27a2733d534632973caa7792ee01a4a6c42bdfc8807280. These counters are exposure, not win-rate evidence.

Added standalone scripts/audit_published_screen.py SHA277e421d661e1da9d8ff4ee2d55f61d251edf51c7d57dc70ccdc7e9d4a4ee101 and six CPU-only fixtures. The auditor does not import runner/training/evaluation/comparison statistics helpers. It checks fixed protocol/source/parent/model identities, source config versus checkpoint config,500 monotonic metric records, parameter and Adam finiteness/shapes/48,000 steps, actual parent initialization ancestry, teacher tensor equality to the original public files, runtime identity, budget partitions, exact matches from saved episode IDs and round-count bounds. Round counts and every training/evaluation transition are not independently replayed. Only legacy-pickle compatibility loading is shared to read the original teacher tensors.

For completed endpoints, full mode additionally checks exact pair/seat identities, score consistency with point differences, decision-phase accounting, reported means/CIs, paired differences and all stored summary values. Bootstrap draws use an independent scalar-replicate implementation (5,000, seed99) with explicit percentile interpolation. Existing parent endpoint results reproduce exactly: seed1 win0.47825(CI0.4605..0.4965), points0.975(CI-0.132..2.116); seed10001 win0.49375(CI0.476..0.51125625), points0.49(CI-0.660025..1.632). Pinned legacy no-search parents legitimately omit newer executed/preparation counters; new reports must supply them. Six fixtures cover known two-pair/constant/zero-difference results and malformed records; Ruff passes. No GPU test/training workload was added alongside the active experiment.

Partial checkpoint audit passed for BOTH completed seed1 arms: runs/published_wp50_screen/checkpoint_audit_2.json SHA340a99fdb2287f5f3c139baf9b9515a7fb9b88dc336249c957e2d921573dc792. Report explicitly sets paired_outcomes_checked=false and lists remaining cases. Earlier first-case audit artifacts remain separate; they are superseded by this two-case snapshot. Full result audit remains pending until all four evaluations finish. Do not infer superiority, change the prespecified criteria or consume final-test seeds from checkpoint integrity or training losses.


## WP50 screen: all training complete and checkpoint integrity verified

All four fixed training jobs exited0 without changing source1c36895f... or runnerc5b7c63a.... Existing session95032/runner44560 automatically entered evaluate:seed1_wp50, shim20608/actual Python52876 verified live. The four1,000-pair strict CPU assisted-WP evaluations remain in the original sequence; no interim checkpoint selection, extension or final-test seed use.

Each arm completes exactly500 updates/131,072,000 simulator decisions and48,000 Adam steps. Final training budgets:

| Parent/arm | Complete paper8 matches | Rounds | Learner decisions | Teacher decisions |
| --- | ---: | ---: | ---: | ---: |
| seed1 WP50 | 1,123,348 | 8,788,643 | 98,076,843 | 32,995,157 |
| seed1 control | 1,119,106 | 8,751,686 | 131,072,000 | 0 |
| seed10001 WP50 | 1,122,802 | 8,778,789 | 98,077,103 | 32,994,897 |
| seed10001 control | 1,117,673 | 8,732,113 | 131,072,000 | 0 |

Aggregate4,482,929 NEW complete paper8 matches,35,051,231 rounds,524,288,000 simulator decisions:458,297,946 learner/65,990,054 teacher. The two published-opponent subsets account for1,129,196 complete matches/8,836,407 rounds. These are training-work totals, not independent model-result samples, opponent pretraining or additions to the research12 scale gate. Repeated parent initialization ancestry is not added to this new-work total.

The second parent WP50 final update_00000500.pt SHA094501f8e2fb6129b6ccce0bff42833d683ef52b6df4350ccb49a6ae5fec9f1a; metrics SHAe2d7b87f1013eebfe51ad046bd72aa8fd0a810c5d19418ae7e7f87614be7bc27. Its published subset completes564,538 matches/4,416,726 rounds; recorded training elapsed610.4407seconds. Second parent control final SHA45ade2c287b2dbe0d42ac066e5193ea983691521cf47d5952da111aeeb660878; metrics SHA6ea6ac99763fbd7df8acd2caec0b7741f9307cd68cf7367d410932405403c252. First parent checkpoint/metric hashes remain as recorded above.

All-four checkpoint audit passed: runs/published_wp50_screen/checkpoint_audit_4.json SHA46b4e9aa4416029ba68f101173b717d5882cfedbd2ebd1e7b024f4b59cab56d2, auditor277e421d.... Exact complete-match counters agree with saved episode IDs, metric logs/counters/configurations/ancestry agree, round bounds hold, actors changed from parents, model/Adam tensors are finite and optimizer counts agree. Both embedded published teachers equal the original public parameter tensors and recorded runtime identities. This explicitly remains checkpoint_checks_only with paired_outcomes_checked=false; earlier snapshots are preserved. Full trajectory replay and strength superiority are not claimed. Run the prepared full raw-result audit after all endpoint evaluations and runner summary finish.


## WP50 screen: all endpoints complete, independent audit and decision

All eight jobs completed with returncode0 under the fixed protocol. process.json records complete_awaiting_audit; runner44560 and last evaluator shim19376 are absent, and session95032 is no longer available. The manifest retains the runner's pre-audit terminal status; the separate audit supplies the final verification. No source changes, intermediate selection, protocol extension or new final seeds occurred during the screen.

Command: .venv/Scripts/python.exe -m scripts.audit_published_screen. Full audit passed for all four cases. Artifact runs/published_wp50_screen/independent_audit.json SHA5dfcd58cdce3f832e50d28cafaaae96a70146efd331b540dec74e7223bd6d8e3; summary SHA9c58ff27770a040adca3ae3bcb3ff9fc3cd835e62f47973949d46d16c1441214. Auditor277e421d..., production1c36895f..., runnerc5b7c63a... and protocol9409200f... remain as fixed above. It rechecks actual checkpoint/configuration/ancestry/teacher tensors/48,000-step Adam/budgets and independently recomputes complete raw paired statistics, score consistency, identities, work counters and summary criteria. It does not independently replay all training or evaluation trajectories.

All endpoints are1,000 duplicate pairs, seeds1,000,000-1,000,999, strict CPU assisted RL-WP, deterministic/endgame/no-search, paper8. Win score counts draws as one half. Confidence intervals use the prespecified5,000 pair bootstrap replicates with seed99.

| Parent/arm | Win score % (95% CI) | Mean point difference (95% CI) |
| --- | ---: | ---: |
| seed1 WP50 | 50.700 (48.875,52.575) | 2.023 (0.904975,3.146025) |
| seed1 control | 50.975 (49.175,52.800) | 2.434 (1.265875,3.628050) |
| seed10001 WP50 | 48.475 (46.750,50.300) | 0.789 (-0.317000,1.938025) |
| seed10001 control | 49.525 (47.675,51.325) | 1.747 (0.628900,2.856050) |

Paired differences on the same deals, win score in percentage points:

| Parent/comparison | Win-score difference (95% CI) | Point difference (95% CI) |
| --- | ---: | ---: |
| seed1 WP50-control | -0.275 (-2.450,1.950) | -0.411 (-1.646025,0.858000) |
| seed1 WP50-parent | 2.875 (0.700,5.050) | 1.048 (-0.191025,2.305050) |
| seed1 control-parent | 3.150 (0.825,5.325625) | 1.459 (0.195825,2.725025) |
| seed10001 WP50-control | -1.050 (-3.075,1.050625) | -0.958 (-2.158100,0.300050) |
| seed10001 WP50-parent | -0.900 (-2.925625,1.300) | 0.299 (-0.924025,1.606000) |
| seed10001 control-parent | 0.150 (-2.025,2.300) | 1.257 (-0.029075,2.541025) |

Endpoint report SHA256 values (each case/evaluation.json under runs/published_wp50_screen):

- seed1_wp50:6b923e24fcfba7c48b182ef047c99dedc414779e6c106c3cc790cd3c5b6d70f8.
- seed1_control:b1afcb186e48e9700bb0084912a50b5b8a7675b96fd1365e5ba48b3c52c3f903.
- seed10001_wp50:c9afa2e31fdc05bf355d5ec733f587c9566f60d4cd90a4ebc9ab8e508bbe80ca.
- seed10001_control:0dd01673cd2136982d2aa5b13ed3dcb071751ee8d485633fcb2fbc09383e78d0.

Decision: paired_eligibility=false and replicated_absolute_development_signal=false. Neither parent establishes WP50 superiority to its matched self-play control; seed10001 also fails improvement over its original parent. The negative point estimates do not establish harm. Seed1 improvement over its parent is shared by the control and does not replicate on seed10001. Do not reinterpret the descriptive control result as success of a newly selected primary endpoint. Positive point means cannot substitute for the win criteria. Repeatedly explored common development deals and multiple comparisons preclude independent confirmation; two parents are not2,000 independent deals.

Keep the verified direct-published opponent capability optional; do not adopt this WP50 method, continue unchanged, evaluate intermediate checkpoints retrospectively or consume final-test seeds. No next experiment is selected by this milestone. The next research decision should reassess learning objectives and optimization evidence before a new bounded protocol. All work remains separate from the completed research12 scale gate. Public superiority and the overall strongest-AI objective remain unfulfilled.


## Fixed update-free objective-gradient diagnostic

Before execution: scripts/diagnose_objective_gradients.py uses the two immutable original MLP parents from the WP50 screen, new diagnostic seeds6,400,001/6,410,001, pure sampled self-play paper8/endgame,1,024 environments x128 steps x4 consecutive collections per parent. Exactly1,048,576 simulator decisions across both parents; zero optimizer updates, no public teacher, checkpoint selection, strength evaluation or final-test deals. All original parent/source hashes are checked. Defaults entropy0.02/value0.5/lambda0.95 and full-rollout advantage normalization are retained. Collect complete-match/round counters separately and exclude diagnostic work from learned-model budgets.

Measure full-batch raw gradients for PPO policy, weighted negative entropy, weighted value MSE and card/koi policy contributions. Phase contributions use the full-batch denominator. Store raw vectors, parameter layout, full-model/shared-body norms/dots/cosines, forced-action fraction and advantage statistics. Chunk8,192 for memory; compare the first chunk of every rollout against actual production backward/clipping with only Adam.step suppressed. Verify unchanged actor tensors and empty optimizer state. Source/policy remain frozen throughout. This is local Euclidean first-order geometry, not Adam update geometry or evidence of causal interference/strength. Earlier separate-critic experiments retained the actor's shared-body value gradient, so they do not isolate its removal. Do not change architecture on presumed interference before this diagnostic is read.


The fixed diagnostic completed with exit0:1,048,576 simulator decisions,68,911 completed rounds and8,174 completed paper8 matches, zero optimizer updates. Both original model tensors remained byte-equivalent by torch.equal and Adam states remained empty. The first8,192-row chunk of each of eight rollouts reproduced actual production combined backward/clipping; maximum gradient-coordinate discrepancy3.25963e-9. No production source or trained checkpoint was changed. All work is diagnostic exposure, not additional learned-model training or strength evidence.

Report runs/objective_gradient_diagnostic/report.json SHAde07a5388c27a79bf358039a8bb2d62fde19667bfee00b50fe96007b3b114604; producer scripts/diagnose_objective_gradients.py SHA0833060c9c4c0c781abc55b0e41b54172eab0ff923a43322ff2207339bd49f14. Independent stored-vector audit passed, runs/objective_gradient_diagnostic/audit.json SHAfb933dddf96c6ad495a42098067a7171ba8884496f9e3eb932d8601b5eae9fbc, auditor scripts/audit_objective_gradients.py SHA7e43f2c840f22e7097d1057e8098e2218d6fd1f52f86c416085ef9f32bf38d51. It uses scalar math.fsum to recalculate vector norms/dots/cosines, checks full-vector/shared-body slicing and card+koi additivity, source/checkpoint/raw hashes, fixed protocol and counter arithmetic. It does not regenerate the trajectories or gradients; unchanged model/optimizer evidence comes from the producer's direct tensor checks. Ruff lint/format and diff checks pass; production remains1c36895f....

Shared-body policy-versus-weighted-value gradient cosines, in consecutive rollout order:

| Parent | Rollout0 | Rollout1 | Rollout2 | Rollout3 | Forced-action fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed1 | 0.163241 | 0.187230 | -0.065225 | 0.125280 | 4.8023% |
| seed10001 | 0.050950 | -0.072747 | -0.085323 | 0.027446 | 4.8847% |

The value gradient is not consistently opposed to the policy gradient across both parents. Norms: policy0.03369-0.03707, weighted value0.03570-0.03780 on seed1; policy0.03378-0.03425, weighted value0.00873-0.01299 on seed10001. Card-versus-koi policy cosines range-0.00687..+0.04301 across both parents, approximately orthogonal at this scale. Weighted entropy gradient norms0.00202-0.00266 are smaller than policy norms; entropy-policy cosines are all negative (-0.19208..-0.02344), which alone is not evidence that reducing entropy improves generalization or deterministic strength. Forced actions account for under5% of these trajectories, so large dilution from forced rows is not supported here.

Decision: do not introduce shared-body value-gradient removal, forced-row normalization or another entropy reduction based on this diagnostic. These measurements do not prove absence of interference during later training or under Adam, but do not establish it as a common explanation for the failed strength screens. Next investigate full-match return credit versus the current lambda0.95/truncated bootstrap estimator, preserving player-relative signs and full-match termination. This requires a separately fixed diagnostic/protocol; no training alteration or next experiment has been launched. Earlier lambda0.99 relational continuation failed and must remain disclosed; a complete-return study must distinguish its estimator/horizon hypothesis from merely repeating that hyperparameter sweep.


## Fixed complete-match return diagnostic

Before execution: scripts/diagnose_match_returns.py freezes both original seed1/seed10001 MLP parents and collects four independent1,024-match batches per parent, paper8/endgame/sampled self-play, no optimizer steps or resets within each batch. Initial seeds6,420,001-6,424,096 and6,430,001-6,434,096; maximum1,024 decisions per environment, requiring every match to finish. Terminal rows are frozen and excluded; report active decisions, executed padded rows, rounds and exactly8,192 completed diagnostic matches separately. No final-test seeds or new learned checkpoints.

On identical complete trajectories compare production native signed GAE lambda0.95 over full matches, lambda0.95 with128-step windows/recorded critic tail, lambda1 with the same windows, and direct realized terminal outcome in the acting player's perspective. Full-match lambda1 must equal the direct outcome. Store every action/result/value/player/valid/phase/round field, initial/final state and per-step full-state SHA for independent native CPU replay and scalar return recomputation. Summaries cover all rows, rounds1-2/3-6/7-8 and terminal distances>64/<=16; report realized-MC target RMSE, advantage spread/sign/correlation and h128-versus-full95 discrepancy. No parameter or horizon selection from this diagnostic.

Realized outcomes are noisy samples, not conditional true values. Lower error to a sampled outcome does not prove a less biased or lower-variance policy gradient. Match-start-aligned windows also differ from steady-state training windows. The earlier failed relational lambda0.99 experiment remains relevant; this diagnostic separates finite-horizon bootstrapping from trace decay without launching another lambda sweep. No architecture/training change is authorized by a favorable descriptive number alone.


Complete-match diagnostic finished and passed CPU replay/float64 arithmetic audit. Exactly8,192 matches/63,974 rounds/960,241 active decisions,1,189,888 executed rows including frozen terminal padding, zero updates. Every active action/raw result and every pre-step full-state SHA matches independent native CPU replay; every final state matches. Direct terminal outcomes agree with full lambda1, and seat-zero float64 recurrence differs from saved native estimates by at most1.311303e-6. Shared native transition core and non-replayed actor inference/RNG are disclosed limitations.

Report runs/match_return_diagnostic/report.json SHAa107da18790ed4821a1e407295f2f2aac35c22618c26fcedcd783b84e93cc3bd; audit audit.json SHA250c9eed61d51e961037118418f23fb3117b3909172e83331684326ab9ebba69. Producer59257d4a4168828bcdaeebefa828bec19848944dad66832ecd4151bef88ae0a3; auditorbe72797358950c5c9a009779a7be56485ab82d87e59461fab6a72ba5b8a0c522. Production remains1c36895f.... Ruff checks pass.

Across four batches, seed1 full-lambda0.95 versus realized-MC return RMSE0.67338-0.68364, whereas h128-versus-full95 RMSE0.04737-0.05924. Seed10001 corresponding ranges0.67656-0.68919 and0.04916-0.06255. In rounds1-2, terminal trace weight averages0.00938-0.01431 across both parents; full95-versus-MC RMSE0.86739-0.88151, but h128-versus-full95 RMSE0.000707-0.000975. Later rounds7-8 average terminal weights0.52544-0.53246. In these match-start-aligned trajectories, trace decay changes sampled targets more than truncation does. These are descriptive discrepancies, not measured gradient bias or strength gains; realized MC has substantially larger advantage spread.

Next implementation: optional complete-match collection for pure self-play PPO, collecting each environment to match termination, excluding terminal padding and retaining configured lambda. This enables a controlled full-lambda0.95 versus full-lambda1 study under the same collection/reset/minibatch semantics, unlike the earlier truncated relational lambda0.99 screen. Variable-size batches require eager collection and optimization initially; count actual active decisions and full matches separately from padded executed rows. Preserve all defaults and resumable state. No strength run until correctness, restart and complete-update throughput are verified and a bounded protocol is fixed.


## Complete-match PPO implementation and fixed throughput check

Added optional complete_matches=true with unchanged default false. New config configs/ppo8_complete_mc.json uses paper8/endgame, MLP128x2,2,048 environments, safety horizon512, lambda1, lr0.0001, entropy0.02, three eager PPO epochs/minibatch8192, float32 and two CPU threads. One full match per environment; terminal padding excluded; native or Torch full signed returns; actual decision counters and separate collection_rows. Existing model/export/checkpoint format retains the mode and exact restart. Unsupported capture/opponent/separate-critic/koi-only modes reject explicitly. Details in DESIGN.md.

Validation:15 new tests pass, including independent Python-reference trajectories and forward lambda-return mixtures at lambda0.95/1 on CPU/CUDA, all-state/result/observation/mask/count checks, exact model/Adam/environment/counter/RNG next-update restart, exhausted-bound rejection without silent bootstrap or updates, unsupported configurations, and twelve-round/Torch returns. Full suite245 passed without skips,21 expected upstream warnings,82.52s. Actual archived Trainer3a39524 versus current defaults matches all metrics/model/critic/Adam/environment/counter/CPU-CUDA RNG values for five configurations x2 updates; report runs/complete_matches_default_equivalence.json. No production rule/observation/actor changes.

Before timing: use the new config, fresh random learner with seed6,460,001, five warmup updates then exactly20 timed updates via koikoi.benchmark.benchmark_training. Disable matmul/cuDNN TF32 and cuDNN benchmarking in the invocation. No checkpoint initialization or strength endpoint; warmup and measurement counts separately recorded. Timing includes complete collection, return computation and all optimizer work. Require40,960 measured completed matches and10,240 warmup matches, report true active and padded row counts, actual rounds and rates. No concurrent GPU tests or training during measurement. No selected model or added research12 scale claim.


Fixed complete-learning benchmark completed:20 timed updates in6.5445146001seconds,4,935,865 active learner/simulator decisions,6,002,688 executed collection rows including frozen padding,322,211 rounds and40,960 complete paper8 matches.754,198.79 active decisions/s and6,258.68 matches/s. Separate five warmups:1,236,687 active decisions/1,509,376 collection rows/80,759 rounds/10,240 matches. Report runs/complete_matches_benchmark.json SHA0adb1f500636e8cee2aa97cd5df0313efb8622f387f8f498ee3588daa8cd3381. Actual-source default equivalence report SHA103f07abd6afac1189ae3dbf3e922d6e5e7e3beb2bdd1d4d511699dbc970111b. Source005484abeab7451d383d77ad13f26077d9650564fe5e69e9d0c6188b532f1e91. The benchmark is a fresh-policy throughput result, not learned strength; no concurrent GPU tests ran during timing.

## Fixed two-parent complete-MC screen

Freeze before launch. Runner scripts/run_complete_screen.py SHA6085505ed2b323445009f7e20e8860a8874e180fff9a2d6e678da37bd834d36e, production005484abeab7451d383d77ad13f26077d9650564fe5e69e9d0c6188b532f1e91. Preflight runs/complete_mc_screen_preflight.json SHA1378b071f83743dfe0d8251439b9d265348d880bd04b08f504aceda39a11d79b verifies both immutable original parents, config, original strict-comparator model files and complete parent baseline identities. Config configs/ppo8_complete_mc.json SHA209ecc190cc7dcc49ed72b737f20d4df686ef4207c00d581f869838e19ba5336. Runner creates exclusive runs/complete_mc_screen, atomically records worker PIDs and statuses, preserves logs, and aborts on failure without automatic restart.

Original seed1 runs/ppo8_v3/update_00005000.pt SHAd104ae00824aca114e6cd414c74d02670b9a8de5a009ad4fad557005f66233e5 and original seed10001 runs/ppo8_seed10001_captured/update_00005000.pt SHAedc44213bc5c5c840297afc33ee64a57b91e9dee48c202535906fc916cde109d each initialize fresh MC(lambda1) and control(lambda0.95) arms. Both arms use complete-match collection, otherwise identical configs and parent-specific new seed6,440,001/6,450,001. Fresh Adam/deals/RNG/counters; no inherited optimizer, teacher or intermediate selection. All arms disable matmul/cuDNN TF32 and cuDNN benchmark in their workers. All share paper8/endgame/MLP128x2/envs2048/safety horizon512/lr0.0001/entropy0.02/value0.5/three epochs/minibatch8192/two CPU threads/float32/eager collection and optimization. This compares return estimators with matched complete-match collection, not an isolated effect of terminal credit on an otherwise fixed target/value loss.

Exactly500 updates and1,024,000 new complete matches per arm,4,096,000 total paper8 matches. Report actual active decisions, rounds, executed padded collection rows and Adam steps; decision counts, final partial minibatches and optimizer steps can differ with trajectory length. Checkpoint every100 solely for recovery, evaluate only update500. Four training jobs then four evaluations. No unchanged extension, retrospective checkpoint selection or final seeds.

Each endpoint evaluates exactly1,000 duplicate pairs seeds1,000,000-1,000,999 with the unchanged strict CPU assisted RL-WP, paper8/deterministic/endgame/no-search/protocol2. Parent baselines remain eval_5000_wp.json SHA34d8b04b06d7e8f39082ceb76a5d58be0ed54c1fee2c56af4eb022c46ca58f97 and1cd58138616254de2dc658307aaa5f7ebe7915aa4145bc0fe6b0ea6a0ffb4ca6. Pair bootstrap5,000/default_rng99. Eligibility requires BOTH parents' MC-minus-control AND MC-minus-parent win-score95% CI lower bounds>0. Replicated absolute development signal separately requires BOTH MC win-rate lower bounds>50%. Points secondary; control-minus-parent descriptive. No pooling repeated shared deals as2,000 independent pairs. Raw/checkpoint independent audit required before any method decision; repeated development data never substitutes for independent confirmation or final evaluation. Earlier negative relational lambda0.99 and WP50 results remain disclosed.


Fixed complete-MC screen launched after implementation/protocol commita09e5f9 was pushed. Protocol runs/complete_mc_screen/protocol.json SHAb397ad2258f189b39265e83f88957d5a4ab66a44cddcf48d0adc78dd883b664b. Tool session32904, runner47352; first train:seed1_mc shim29252/actual Python5396 verified live. Initial observed95 updates/194,560 complete matches/1,517,769 rounds/22,945,115 active decisions/28,907,520 executed collection rows; finite policy/value/entropy/KL/gradient metrics. These are progress counters only. Source005484ab... and runner6085505e... fixed. Keep the existing pipeline; monitor atomic process.json and worker logs, without restarting on observation timeouts. Independent endpoint/checkpoint audit remains required. Adam-step expectation for each arm is three times sum over500 updates of ceil(delta active decisions/8192), because the final minibatch varies; verify exact complete matches from saved episode counters, active/padded partitions, final fresh-match states and monotonic logs before interpreting endpoints.


## Complete-MC screen: first parent finished, independent audit prepared

Existing session32904/runner47352 completed train:seed1_mc and train:seed1_control with returncode0 and automatically advanced to train:seed10001_mc, shim47864/actual Python20988 verified live. Last observed109 updates/223,232 matches with finite metrics. No endpoint strength evaluation, intermediate model selection or source/protocol change. Production005484ab... and runner6085505e... remain fixed.

Added scripts/audit_complete_screen.py SHAc40aa7b872a01c53d41dc92703e9c7afe1d5ed22a2af41079543eaf1eb1440f6. It imports no runner/trainer/evaluator/comparison helpers. Checks immutable protocol and actual production/runner/config/parent/comparator hashes, final checkpoint identity/configuration/ancestry,500 monotonic metric records, valid/padded row bounds and exact match counts from saved episode IDs. Reconstructs all2,048 final reset states per completed arm using the independent Python ReferenceEnv. Derives Adam steps as epochs times the sum of per-update ceil(valid decisions/minibatch); verifies every parameter/moment shape, finite tensor and step counter. Raw full mode independently recomputes paired means/bootstrap intervals and all summary differences and prespecified criteria. Does not replay every training or evaluation trajectory; final-state reconstruction is not full training replay.

14 CPU-only tests pass in1.67s: known pair-bootstrap fixtures, malformed pair scores/seeds, per-update partial minibatch rounding (8,9,2 rows require2+3+1 minibatches rather than ceil(19/4)), missing updates/match counts/padding quantum/padding lower bounds/horizon limits/nonfinite data/final-count mismatch. Both existing full parent baseline means and CIs reproduce exactly. Ruff lint/format and diff checks pass. No extra GPU workload or full-suite rerun was introduced during training.

Completed first-parent budgets and hashes:

| Arm | Updates | Complete matches | Active decisions | Rounds | Executed collection rows | Adam steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MC lambda1 | 500 | 1,024,000 | 120,818,216 | 8,007,768 | 151,640,064 | 45,000 |
| Control lambda0.95 | 500 | 1,024,000 | 120,029,550 | 7,997,797 | 151,255,040 | 45,000 |

MC final update_00000500.pt SHAa564589ac034894ac90fca8bf6942f11d383f209a03b10d3752e9e7602a33e9a; metrics SHAa9880104651d6b48285b792d1e22fb932d46dd529302fe90f5f55e13ef7ce63f. Control final SHA2294639540030ce4740a48e5db2868055f1e5f472580550fabe0224c2448044d; metrics SHAbc62cece6ffd22a901c775bbb28b0bc408afb06be024b1f8430cd47d69445941. Active decisions differ despite identical matches; both happen to require45,000 Adam steps, verified from their actual batch sizes rather than assumed.

Both checkpoint audits pass: runs/complete_mc_screen/checkpoint_audit_2.json SHA39686d93b0940fe2c7873c05d29daba4306bd02c04d6f8e1fccbed8e7afb5d0e. Earlier first-case snapshot checkpoint_audit_1.json SHAc2296e437b329917c6b7c56de74b994012e56a3633f9cd0ac119aa5dcc9a5b1d is preserved. Reports explicitly mark paired_outcomes_checked=false. No win-rate/strength inference, adoption or final-seed use; finish the original four training/four evaluation jobs, then run full independent audit before any method decision.


## Complete-MC screen: all training complete and audited

All four training workers exited0 at the prespecified500-update endpoint. Existing session32904/runner47352 automatically entered evaluate:seed1_mc, shim56188/actual Python2712 verified live. All four1,000-pair strict CPU assisted-WP evaluations remain in their fixed sequence. No production/runner/protocol change or interim selection. Source005484ab... and runner6085505e... unchanged.

| Parent/arm | Complete matches | Rounds | Active decisions | Executed collection rows | Adam steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed1 MC | 1,024,000 | 8,007,768 | 120,818,216 | 151,640,064 | 45,000 |
| seed1 control | 1,024,000 | 7,997,797 | 120,029,550 | 151,255,040 | 45,000 |
| seed10001 MC | 1,024,000 | 8,020,239 | 121,033,833 | 151,504,896 | 45,006 |
| seed10001 control | 1,024,000 | 7,995,888 | 119,915,431 | 151,296,000 | 44,988 |

Totals4,096,000 NEW complete paper8 matches,32,021,692 rounds,481,797,030 actual simulator/learner decisions and605,696,000 collection rows including frozen padding. Matched completed-match budgets are not identical active-decision or optimizer-step budgets. All excluded from the separate research12 scale gate; inherited parent ancestry and benchmark/diagnostic work are not added to new training totals.

Second-parent MC final update_00000500.pt SHA76ffcc078b31fbbbbb07299205357a0835f4c638d451c4d83dae1a9a342f0938; complete metrics SHA f8dd1e7c5ebbd94f3e10babd9fa8ef3819b107ab7d1ab4dbbe63abecf2eb66cd. Second-parent control final SHAb43d7f06be6a8214bc90a1eeea93811257f8bd81e56fc13c71072f98baa28955; metrics SHAd6de00d561280acfb26d9c34f27b40c49bd84caac95add09edfcd1b08b9885d7. First-parent immutable hashes remain as recorded above.

All-four checkpoint audit passes: runs/complete_mc_screen/checkpoint_audit_4.json SHA8d76df23b2f8b497d911f0c8416c52ff27e2c1fb1994de446770daabf5396c06, auditorc40aa7b8.... All8,192 final reset states exactly reconstruct in the Python reference at episode500; saved episode counts, configurations, parent ancestry, metric bounds/partitions, finite tensors and actual variable-batch Adam steps agree. Earlier two-case snapshot remains preserved. This is checkpoint_checks_only with paired_outcomes_checked=false, not independent replay of every training trajectory or strength evidence. Complete all four original evaluations and full raw-outcome audit before any method decision. Final seeds remain unused.


## Complete-MC screen: final outcomes and rejection

All four1,000-pair fixed endpoint evaluations completed with exit0; the entire eight-job session32904 exited0, runner47352 and last shim62224 absent. No process was restarted or protocol changed. Manifest remains complete_awaiting_audit, denoting the runner's terminal stage. Full command .venv/Scripts/python.exe -m scripts.audit_complete_screen passed. Audit runs/complete_mc_screen/independent_audit.json SHA2fea1ce2c6ffabaec75c9a90b7546d6c214844cedfddeba420e10e46c3822c07; summary SHAc69ebdb5051c689bfc2d94b78680d19ca8a76be9eab491e48d59d9284fa40912. Auditorc40aa7b8..., production005484ab..., runner6085505e..., protocolb397ad22... remain fixed. Scope includes all stored checkpoint/configuration/ancestry/Adam/budget/identity checks, Python reconstruction of final reset states, and independent raw paired arithmetic/criteria; not independent replay of every training/evaluation trajectory.

All endpoints: same1,000 development duplicate pairs seeds1,000,000-1,000,999, paper8/deterministic/endgame/no-search against strict CPU assisted RL-WP, protocol2. Win scores count draws as one half;5,000 pair bootstrap replicates with default_rng99. Multiple previously explored endpoints and shared deals limit these to development evidence.

| Parent/arm | Win score % (95% CI) | Mean point difference (95% CI) |
| --- | ---: | ---: |
| seed1 MC | 46.475 (44.650,48.275) | -0.430 (-1.529050,0.670025) |
| seed1 control | 51.075 (49.275,52.900) | 2.175 (1.037975,3.329000) |
| seed10001 MC | 46.725 (44.875,48.475) | -0.741 (-1.928025,0.422150) |
| seed10001 control | 49.775 (48.050,51.525) | 1.984 (0.884000,3.095125) |

Paired differences, win score in percentage points:

| Parent/comparison | Win-score difference (95% CI) | Point difference (95% CI) |
| --- | ---: | ---: |
| seed1 MC-control | -4.600 (-6.950,-2.425) | -2.605 (-4.032075,-1.248800) |
| seed1 MC-parent | -1.350 (-3.550,0.825625) | -1.405 (-2.770025,-0.121900) |
| seed1 control-parent | 3.250 (1.125,5.400) | 1.200 (0.021950,2.420075) |
| seed10001 MC-control | -3.050 (-5.325625,-0.799375) | -2.725 (-4.137000,-1.337975) |
| seed10001 MC-parent | -2.650 (-4.900,-0.350) | -1.231 (-2.681025,0.181025) |
| seed10001 control-parent | 0.400 (-1.675,2.475) | 1.494 (0.331000,2.675000) |

Report evaluation.json SHA256 values under runs/complete_mc_screen:

- seed1_mc:ac55fbf8a0a025a247fa809117dc07b60d980cee0a23819f2f411bbea3953605.
- seed1_control:fecef17d273b873b68441bd5b7f18b1ac3129f4a95a12ac88fea8ecaa805a57a.
- seed10001_mc:df19413587f757b11c3e57b1705c06a12aae2718c2bc9a9eebe8e8f4dd6851b5.
- seed10001_control:fe5543e2a806c1204cd8ecccb6483db03cd92707dd1ea0b650a1bf83d33e6d68.

Decision: paired_eligibility=false and replicated_absolute_development_signal=false. Both MC-minus-control score intervals are wholly negative; both MC absolute win-rate upper bounds are below50%. Reject this complete-MC intervention and unchanged extension. Retain optional complete collection for explicitly identified research; default fixed-horizon/lambda0.95 remains. No intermediate checkpoint selection, automatic promotion of the descriptive control, new final-test seeds or public-superiority claim. The control's seed1 gain does not replicate on seed10001. Higher-variance full returns, shared value targets, multi-epoch PPO and variable decisions/Adam counts are simultaneous features of the intervention; these outcomes do not isolate a causal mechanism. No experiment remains live.

## Source inspection informing the next research decision

After completing the fixed screen, inspected the pinned upstream89d789565c7c36c84b74662e43001d960bf80901 training sources. The WP branch in rl_train.py uses a round-ending target: next round index, resulting player points and whether that player will be dealer index win_prob_mat; terminal or bankrupt outcomes use exact draw/win/loss probabilities. It multiplies this probability by10 and assigns it to sampled actions in that round (TraceSimulator discount defaults1), then regresses selected-action Q values. This is a different estimator from our full-match PPO lambda1 target. The script's checked-in task_name defaults to point; inspecting the WP branch is not a reproduction of the published weight-training run.

rl_wp_reward_simulate.py describes per-score/per-round continuation simulations; rl_wp_reward_regress.py fits a logistic point-to-win model per round and obtains the opposite dealer role by symmetry. Their table is policy-dependent and not automatically valid for our sampled actors. No script was executed or external weights/data imported. The checked-in simulation call/signature also needs verification before claiming reproducibility. get_master_net loads SL agents for the comparison master; this does not by itself establish SL initialization of the RL actors, whose main path creates new action networks. Preserve this distinction.

Next research priority: inspect/calibrate a public round-boundary continuation estimator under our current policies, using only round/points/dealer context and separately held-out complete trajectories before choosing a new bounded training intervention. The current diagnostic corpus may supply calibration evidence, but its reused trajectories are not independent strength confirmation. Do not substitute round termination for true match termination or assume upstream-style targets preserve optimality without specifying the estimator and validating its use. No next training protocol has yet been selected.


## Public round-boundary calibration: fixed protocol (2026-10-17)

Before execution: reuse runs/match_return_diagnostic (report SHA a107da18790ed4821a1e407295f2f2aac35c22618c26fcedcd783b84e93cc3bd). Both immutable original actors, four existing batches of 1,024 complete matches each. Fit batches0/1; hold out batches2/3, disjoint by complete match. This corpus was already used in descriptive return diagnostics; it is not untouched final or independent strength evidence. No new simulation or policy updates.

Fit eight separate, seat-symmetric, two-coefficient logistic regressions per actor on round-start public information: (points0-30)/30 and 2*dealer0-1, no intercept. Target is final player-zero win credit (0/0.5/1). Fixed ridge1e-4, float64 Newton, maximum100 iterations, gradient infinity norm below1e-10, Armijo backtracking. No tuning on holdout outcomes. Labels, network values, hands, stock and RNG are excluded from features. Round0 score slope is unidentified because every match starts at30 points. Report holdout unseen (round,points,dealer) cells and cells with fewer than20 fit observations.

Holdout comparisons: constant0.5 and original actor value converted to seat-zero probability (clipped to[0,1], clipping count reported). Network observes new private/public cards, so this is not an information-matched comparison. Report Brier, cross entropy, ten calibration bins and per-round Brier. Paired public-minus-baseline Brier intervals use whole-match cluster bootstrap,5,000 resamples with default_rng99; rounds are not independent observations. Sixteen supervised calibration fits do not count as policy training. No strength claim, reward/default change or final-test launch follows from calibration alone. Verify extracted public boundary states with native CPU replay and independently recompute fits/statistics before adopting conclusions.

Implementation/preflight: scripts/calibrate_round_boundaries.py; five CPU fixtures passed for boundary point/sign extraction, draws, feature exclusions/seat symmetry, known regularized optimum, unidentified round-zero slope and whole-match resampling. Execution: .venv/Scripts/python.exe scripts/calibrate_round_boundaries.py. Results pending.


### Round-boundary calibration result and decision

Completed; all five CPU fixtures and independent native-replay/statistics audit pass. Producer scripts/calibrate_round_boundaries.py SHA8fa50fc346abd3f940e8e87758ae95cea23ff73e1bffff04947d8b414cfe9a5d; auditor scripts/audit_round_boundary_calibration.py SHAe5be0a55b7b4866c8f15638dabca1273613a75637b4d575858569cf1d50f359c. Report runs/round_boundary_calibration/report.json SHAb622e04bd92d8ffb3b89aeb10f4cd129627b02a8c52a0f08e2fe0bee0680fb55; protocol SHAaae0921405046c446e12a323f8d869441ce384d5b2eb5260dd7a24a8e136b41c; data SHA1606fbc09909fbe4c4939da13cb205668c7677c53a088f5198fc21048c0b0dc2; audit.json SHA8c89beff2555ca7e5c6b79daf2f0cc05b8dac9c692ef4134eef13039fb629ed5.

Independent audit command: .venv/Scripts/python.exe -m scripts.audit_round_boundary_calibration. All8,192 original matches/63,974 public boundaries/960,241 active decisions/1,189,888 executed padded rows replayed through native CPU against every saved CUDA-state digest, transition result and final state. Boundary points/dealer/round come directly from replayed state, independently of producer cumulative score deltas. All16 fit-only regularized objective gradients satisfy the fixed tolerance (maximum9.86671282e-11); positive ridge ensures a unique minimizer. Predictions, fit/holdout partition, support counts, losses, bins, per-round errors and cluster intervals checked without producer helpers. Production005484ab... replays the original1c36895f... corpus exactly. Native rules core is shared; actor inference and action RNG are not regenerated. Replay contributes zero new independent matches or training.

|Original actor|Fit boundaries|Holdout boundaries|Public Brier|Network Brier|Constant0.5 Brier|
|---|---:|---:|---:|---:|---:|
|seed1|16,048|15,927|0.17893244|0.17678520|0.24472594|
|seed10001|16,001|15,998|0.17771710|0.17672837|0.24399925|

Public-minus-network Brier: seed1 +0.00214725 [0.00158876,0.00271243]; seed10001 +0.00098874 [0.00048799,0.00150832]. Lower is better: both public fits are less accurate than the card-informed network, an expected information difference that does not itself establish model misspecification. Public-minus-constant: -0.06579349 [-0.07097123,-0.06063324] and -0.06628215 [-0.07131907,-0.06104786]. Public/network/constant cross entropy: seed1 0.53920354/0.53669948/0.69314718; seed10001 0.53874258/0.53649855/0.69314718. Each actor has2,048 fit and2,048 held-out complete matches; intervals resample these matches, not the much larger boundary count.

Unseen holdout contexts:63/15,927 and50/15,998; fewer than20 fit observations:3,058 and3,037 boundaries. Maximum absolute predicted-versus-observed difference across the ten nonempty pooled bins:0.02912487/0.02422933, descriptive only, not a formal calibration guarantee or validation of each sparse cell. Round0 score slopes remain unidentified and zero. These held-out outcomes were previously used in another diagnostic; not untouched final evidence.

Decision: preserve this public-only estimator as a diagnostic candidate; do not replace terminal outcomes, adopt a new reward or claim playing-strength gains. The existing actor value remains more accurate on the held-out fresh-deal states, and the fixed public model is policy-dependent. A defensible next learning experiment must separately specify how a round-end continuation target is formed, distinguish a true match terminal from a learning trace boundary, and compare against a matched baseline on both original parents. Recalibrating on outcomes after observing that experiment would constitute a new protocol. No further unchanged complete-MC or WP50 extension; final seeds remain unused. This diagnostic is terminal, with zero new simulations/policy updates and16 small supervised fits.


## Frozen round-continuation screen: implementation and fixed protocol (2026-10-17)

Previous goal turn made progress: public boundary calibration and independent replay completed/pushed18e4322. It showed useful public-state signal but worse prediction than a network observing fresh cards. This motivates a bounded denoised round-target experiment, not automatic adoption. It differs from the failed full-terminal MC intervention: finish every actual match but propagate each round-ending continuation target only within that round. At true termination retain the real win/draw/loss, including bankruptcy and draws. Fixed table fitted only to original calibration batches0/1; do not refit on held-out or later policy outcomes. Public target drift/misspecification remains a limitation. This changes policy and value targets together and does not isolate a causal mechanism.

Production SHA28cb54aa6589f373c3a0541e89226c4a0ebe93d065b05ba2fdb9f18ff96a1987. Optional Config.round_continuation embeds coefficients and source identities; requires complete paper8/lambda1/eager pure self-play/shared-value/all-action mode and exact calibrated-parent initialization. Separate learning_reward/learning_done keep actual match outcomes/counters intact. No actor/rules/observation changes. Checkpoint embeds the table and checks its identity/ancestry. Defaults remain disabled.

Full282 tests passed without skips (21 expected upstream warnings,84.52s). Eighteen new production tests include CPU/CUDA Python-reference round targets, bankruptcy/draw/sign fixture, seat symmetry, exact restart and malformed/drifting identities. Existing complete-match tests also pass. Actual archived18e4322 Trainer plus collector match current default complete collection after2 updates for CPU/CUDA x lambda0.95/1. Five standard CPU/CUDA eager/captured/historical-opponent modes also match archived3a395248 source exactly. These checks compare model/Adam/environment/counters/metrics/RNG, not merely source resemblance. Ruff/diff checks required before commit.

Validation report runs/round_continuation_validation.json SHAee19d969be9f410f78f393205fa2a1f7f892ae05cfdc77cab0e53667d52e8770; standard-default report runs/round_continuation_default_equivalence.json SHAe57e0111201e89075286ce42e523ba012baf2dfe12ef788fed9b24e51f116de3. Twenty timed full-learning updates after5 separate warmups on the original seed1 actor and diagnostic seed6,620,001: 7.3432130001 seconds, 40,960 complete matches/319,909 rounds/4,788,776 active decisions/6,047,744 padded collection rows. 5577.94 matches/sec and652136.33 active decisions/sec. Warmup counters {"collection_rows": 1521664, "decisions": 1197020, "matches": 10240, "rounds": 80062, "updates": 5}. Benchmark policy updates are excluded from selected training/strength and the earlier research12 scale gate.

Predeclared screen: two original parents (seed1 d104ae00..., seed10001 edc44213...), each round-target versus a fresh complete-match lambda0.95 control,500 updates/1,024,000 complete matches each;4,096,000 total new matches. Fresh seeds6,600,001/6,610,001 are shared within each parent comparison. Both arms env2048/horizon512/MLP128x2/lr1e-4/entropy0.02/value0.5/epochs3/minibatch8192/endgame/float32/eager/CPUthreads2; TF32/cuDNN benchmarking disabled. Treatment uses original fit-only public coefficients with lambda1, control uses the unchanged match GAE lambda0.95. Actual decisions and Adam steps vary with match length and must be audited. All parents/checkpoints/tables/comparator weights/config/source/runner hashes are fixed in the exclusive protocol. No intermediate reranking, table updates or budget extension.

Evaluate only final update500 on exactly1,000 duplicate development pairs seeds1,000,000-1,000,999 per endpoint versus strict CPU assisted RL-WP, deterministic/paper8/endgame/no search/protocol2. Primary advance criterion: BOTH parents have positive95% paired win-score lower bounds versus their control AND original parent. Separately require BOTH absolute win-rate lower bounds above50% for a development superiority signal. Preserve paired outcomes and whole-pair5,000-bootstrap/default_rng99 calculations; points secondary. Shared repeatedly used deals are not2,000 independent pairs and passing does not supply independent method confirmation or final evidence. No final seeds100,000,000+.

Commands: .venv/Scripts/python.exe -m scripts.run_round_continuation_screen --preflight (passed all4 cases); .venv/Scripts/python.exe -m scripts.validate_round_continuation; .venv/Scripts/python.exe -m scripts.validate_training_defaults --output runs/round_continuation_default_equivalence.json. Launch after verified implementation commit: .venv/Scripts/python.exe -u -m scripts.run_round_continuation_screen. Exclusive run directory runs/round_continuation_screen; four training then four evaluation jobs, authoritative worker PIDs recorded in process.json; fail on job error, no restart from an observation timeout. Independent checkpoint/budget/raw-outcome audit required before interpreting strength. Implementation/preflight complete; strength launch pending at this entry.


### Round-continuation screen launched

Implementation/preflight committed and pushed736bb57; .venv/Scripts/python.exe -u -m scripts.run_round_continuation_screen launched session72335. Runner actual PID6008 (venv launcher45628), initial train:seed1_round shim55060/actual worker7300 were confirmed live via Win32_Process; initial health check update154/315,392 completed matches with finite logged metrics. This is not a strength result. Protocol SHA1c335d678b0b94a15b7a1e2b40391381826140aa2fbad771344d8f884d119f21; runner SHAe3f3e7555d2d4dc91dc52edc7c8a9bc8f615f1d1beee19613294b148ad46b196; production28cb54aa... frozen for all jobs. Do not restart from observation timeout; inspect the session, process.json, logs and actual worker process.

Independent screen auditor scripts/audit_round_continuation_screen.py SHA3fb4e53cd19d208283ff97cad259acd2f6ba70bdf163500fad6159050b73e145 is prepared; protocol-only verification passed, including exact original calibration/table/parent/control identities. Nineteen CPU fixtures pass for paired bootstrap, invalid raw outcomes, variable-minibatch Adam accounting and embedded table/parent/fingerprint drift. These fixtures do not mean the live screen has passed final audit. Commands: .venv/Scripts/python.exe -m scripts.audit_round_continuation_screen --protocol-only; --checkpoints-only after final update500 files exist; omit flags only after all jobs finish. Audit must verify actual parent/checkpoint/metrics/table identities, all final reset states, budgets/Adam steps, final raw paired results and both criteria. No interim strength inspection or selection.


### Round-continuation learning completed; strict evaluation in progress

All four training jobs exited0 and completed exactly500 updates/1,024,000 complete paper8 matches each. Source28cb54aa... and runner e3f3e755... remain frozen under protocol1c335d67.... New totals4,096,000 complete matches/31,992,099 rounds/481,431,397 active decisions/606,203,904 executed collection rows. No intermediate selection, refit or additional budget. All real matches terminated normally before reset.

Independent checkpoint audit passed for all four endpoints: runs/round_continuation_screen/checkpoint_audit_4.json SHA298372fd9e71f15b309691f9b2c343f45539cca6d0e5374803f90c63f353eecc. It checks exact config/parent/calibration-table identities, immutable model artifacts, per-update metrics, variable-batch Adam step accounting and all8,192 final reset states against Python ReferenceEnv(seed,paper8,500). This is a stored-state/budget audit, not replay of every training trajectory and not a strength result. Earlier1/2-case snapshots remain preserved.

|Case|Active decisions|Rounds|Executed collection rows|Adam steps|Final checkpoint SHA256|
|---|---:|---:|---:|---:|---|
|seed1_round|120,077,096|8,008,851|151,410,688|44,997|905582ea5c909a52cc816df262c5d9c11bb9331bafb952f82476bb8351d2d445|
|seed1_control|119,823,410|8,001,833|151,220,224|44,982|736e246bcf9ffe76bdb9f17e5a1efd5a654ae45c2e25f6f23015aa6e39797812|
|seed10001_round|121,483,822|7,987,213|152,133,632|45,000|b41d7f709a5ffe7958d2633b5660105e040435f62d110af4d6cd7eb07098bcbb|
|seed10001_control|120,047,069|7,994,202|151,439,360|45,000|3ff4af1078bd5f39e5bd344d7677525d9559141f32fa6085fc1e9120e6d1e34c|

Session72335 / actual runner6008 remains live. Initial evaluation evaluate:seed1_round shim65820/actual worker41940 confirmed via Win32_Process; progress650/1,000 duplicate pairs at this checkpoint. Only progress counts were inspected, not interim win rates. The next three fixed evaluations follow automatically. All four final reports and raw paired-outcome audit remain pending; do not promote a model or change the protocol from these training/budget checks. Production/source/runner stay unchanged through the live screen. No final-test seeds used. Previous goal turn was progress (implementation/launch); this turn adds completed-training/budget evidence while verifying the live evaluation wait.


### Round-continuation screen: final independently audited result

All four training and four evaluation jobs exited0. Session72335 exited0; actual runner6008 and last evaluator27404/shim64632 absent in Win32_Process. process.json retains the runner terminal state complete_awaiting_audit; the separate full audit supplies final verification. Production28cb54aa... and runner e3f3e755... stayed fixed through all jobs; no refit, intermediate selection, final seeds or budget changes.

Full independent audit PASSED: runs/round_continuation_screen/independent_audit.json SHA5a2f51f7c959c57d2fed374ba1c70101293eb4fcb34b50e2d135040b98cc67fb; auditor SHA3fb4e53cd19d208283ff97cad259acd2f6ba70bdf163500fad6159050b73e145; protocol SHA1c335d678b0b94a15b7a1e2b40391381826140aa2fbad771344d8f884d119f21; summary SHA69cddadaa056f706f12d350e6d23f6028a7e673f9f1dd43cc7109d1e6c749134. All model/table/config/parent/comparator identities, counts/variable-batch Adam, final reset states, raw pair outcomes, endpoint/paired means and bootstrap intervals, summary entries and both prespecified criteria checked independently. Scope is stored-state/budget and raw-outcome arithmetic; not replay of all training/evaluation trajectories. Earlier checkpoint-only snapshots remain preserved. No additional code/tests were needed after the already verified auditor; the full audit executed successfully on all final artifacts.

Each endpoint:1,000 shared development duplicate pairs, strict CPU assisted RL-WP, paper8/deterministic/endgame/no search/protocol2. Win score is draw-adjusted; points use final candidate-minus-opponent difference.

|Case|Win %|95% CI %|Mean point difference|95% point CI|
|---|---:|---|---:|---|
|seed1_round|50.925|49.100000 to 52.775000|2.445|1.346950 to 3.555075|
|seed1_control|50.575|48.750000 to 52.400000|1.526|0.379000 to 2.714025|
|seed10001_round|49.700|47.825000 to 51.500000|1.510|0.320000 to 2.665000|
|seed10001_control|49.725|47.974375 to 51.450000|1.441|0.326950 to 2.551000|

|Parent|Comparison|Win difference, pp|95% CI, pp|Point difference|95% point CI|
|---|---|---:|---|---:|---|
|seed1|round_vs_control|+0.350|-1.800000 to +2.450000|+0.919|-0.376150 to +2.188000|
|seed1|round_vs_parent|+3.100|+0.875000 to +5.275000|+1.470|+0.182975 to +2.753025|
|seed1|control_vs_parent|+2.750|+0.600000 to +4.925000|+0.551|-0.699000 to +1.786125|
|seed10001|round_vs_control|-0.025|-2.125000 to +2.050000|+0.069|-1.224025 to +1.334125|
|seed10001|round_vs_parent|+0.325|-1.825000 to +2.450000|+1.020|-0.204050 to +2.257075|
|seed10001|control_vs_parent|+0.350|-1.750000 to +2.425000|+0.951|-0.289100 to +2.195025|

Evaluation report SHA256 identities:

- seed1_round: 6aee366a4be969b2fd7eab189fcb1bf691b749dd9de0f5d1f0051b61646c1b73.
- seed1_control: 31ff46b33de702d4a243136e4c5613498cb8606f071e34de5aab673779f86306.
- seed10001_round: 0f97b5a608c338fbfca079b338b9a1c97f38534ff65194b4d25f4f7ae09190b9.
- seed10001_control: 9d7c1b78b4666dddeeac87606626b62361eaccecf169ed00186f77770b46f5bf.

Decision: BOTH-parent paired eligibility FAILED and replicated absolute public-superiority criterion FAILED. Round-target versus control differences are+0.35pp [-1.8,+2.45] and-0.025pp [-2.125,+2.05]; neither proves an improvement or harm. Both treatment absolute win intervals include50%. Seed1 improves over its original parent, but its matched control does too; seed10001 does not reproduce a positive lower-bound improvement. Positive point means do not replace the win criterion. No default adoption, unchanged extension, table refit, intermediate checkpoint reranking or final-test launch. The shared repeatedly explored deals are not2,000 independent parent samples. This result does not isolate table bias/drift, variance, value interference or representation as a cause.

The full experiment contributed4,096,000 new complete paper8 matches/31,992,099 rounds/481,431,397 active decisions/606,203,904 collection rows; immutable checkpoint identities and per-arm counts are above. Do not add this work to the completed research12 scale gate. The screen is terminal and no experiment is live. Public superiority, independent selected-method confirmation, final evaluation and manuscript remain outstanding; goal active.

Next direction: assess initialization from the published trained policy, rather than repeat another unsupported adjustment of these MLP return targets. This is a proposed capability diagnostic, not yet an adopted method or strength claim. The source review confirms BatchPublishedAgent already embeds three frozen original models and consumes native public/own-only300-feature tensors. Pinned koikoinet2L.py uses separate discard/pick/koi Transformer encoders (256 embedding,512 feedforward,4 heads,2 layers) with ordinary differentiable modules. Current logits() is explicitly no_grad and construction freezes parameters; it is not already a learner. Outputs are action-value scores, not calibrated action probabilities. Before any new strength protocol, verify gradient-enabled/eval-mode forward compatibility, exact parameter initialization, masks/phase routing, dropout behavior and memory/throughput on saved states; then explicitly define any stochastic policy distribution. Keep strict CPU comparator unchanged and disclose published-weight dependency. No assumption that its RL actors were initialized from supervised models: inspected upstream main creates initial action networks afresh. This direction builds on an existing validated native feature/adapter capability; no public model modification or policy training was performed in this review.


## Pretrained published actor: fixed zero-update feasibility protocol (2026-10-17)

After the audited round-target screen failed, assess direct initialization from original published trained WP scores. Add a standalone differentiable PublishedActor that uses the unmodified original three phase networks and native public/own-only300-feature input; keep dropout disabled even when train() is called. No implicit probability mapping or PPO/critic integration. Keep frozen teacher/comparison adapters unchanged. Actor snapshots distinguish current parameter digest from original initialization identity; restore current tensors with the pinned upstream source. The ordinary MLP learner/defaults are untouched.

Before corpus execution: reuse runs/published_batch_validation.npz SHAe49004ce6795da3c23fbcb036c4d23712b6676a8d96cde1b43f6fca28d17d322 and report SHA0b040cd31c1d42e945bd233ea94d4f57e06d8d2ffae4a63ffded5c68c8aed544. All13,029 active states,100 terminal states,128 existing hidden-world pairs; zero new simulation or policy updates. CPU/CUDA float32,2 CPU threads, TF32/cuDNN benchmarking/autocast disabled. Compare gradient-enabled actor versus frozen same-batch scores and stored strict CPU singleton oracle; batch128. Verify every native feature digest and legal mask, unchanged state and initial/final parameter identity. Finite-corpus agreement does not establish universal numerical identity.

Gradient probe: first32 nonforced states per public phase0/1/2/3, original oracle argmax labels, cross entropy at temperature1. This is a feasibility loss only, not a selected training objective. Compare every parameter gradient against the original independent phase-module forwards with separately indexed, weighted log-softmax losses. Require finite nonzero gradients per model and close agreement; no optimizer constructed or stepped. Describe entropy/greedy probability for fixed temperatures[1,.25,.1,.05] on nonforced states per phase; do not select a deployment/training temperature from strength outcomes.

Component timing on CUDA: balanced batches32/64/128,3 warmups then5 forward/loss/backward repetitions each, resident features. Record all times/median and baseline/peak allocated/reserved memory. Excludes environment collection, critic, Adam and transfers; not a full training throughput estimate. Preserve raw scores/gradients and all hashes. Six CPU/CUDA WP/Point tests passed, including original score/action comparison, exact repeated eval-mode scores/RNG preservation, finite gradients, hidden-world invariance, terminal rows and actor-weight snapshot after one test-only SGD update; that isolated fixture is not research policy training or strength evidence. Corpus execution: .venv/Scripts/python.exe -m scripts.validate_published_actor; report under runs/published_actor_validation. Results pending.


### Pretrained actor feasibility result

Implemented standalone koikoi/published_actor.py. Source 8ffbdd08a18a58f1b270df33b0f9fa2762ba6834e88044ab157da93edc030691; default Trainer/model, simulator rules and frozen comparator/teacher adapters unchanged. The actor keeps original dropout disabled while enabling gradients, exposes raw masked scores with explicit public phase routing, and embeds current weights/digest separately from initial source identities. Tests also cover actor snapshot after one isolated test-only SGD step; no optimizer/resume integration is claimed. The pinned external source is still needed to reconstruct upstream modules from embedded weights.

The fixed zero-update corpus diagnostic completed session88717 exit0. CPU/CUDA each13,029 active states,100 terminal states and128 existing hidden-world pairs; every native feature digest/legality/state check passed. Zero unassisted greedy-action disagreements versus both frozen same-batch and stored original singleton CPU WP policy on this corpus. Maximum legal-score errors versus frozen same-batch: CPU4.768371582e-6/CUDA8.106231689e-6; versus original singleton CPU:0.0001254081726/0.0001535415649. This is finite-corpus compatibility, not a universal action-identity certificate. Public endgame assistance remains the caller's mask responsibility.

All4,019,715 parameters initialize exactly from the original WP weights and remain unchanged after all diagnostic forwards/backwards: parameter digest7d684d9063b8c4bbe406c7b50f9b43ac90bb774f5917b13522d28be1a81925f4. The128-row nonforced gradient probe agrees with independent original phase-module losses: CPU maximum error0; CUDA maximum3.725290298e-9 for each phase network. All parameters have finite gradients; per-network gradient norms are positive. These checks show differentiation works; the probe's cross entropy is not a chosen reinforcement-learning objective.

Report runs/published_actor_validation/report.json SHAb30227db83ee71d0e9144c0f5c2f3a2884455b44db61a4acc1d8badadcd0650a; protocol SHAa5f6f6e112356ea25086017fbb15fa3a756e631515233ca6ad982ded541ca842; raw data SHA5e07670b9b0ed3c1dce3b9ebfb835badf3f36ca9531ef57ab9d2ef799f91e2e5; independent audit.json SHAff9816c1dc387cf21be030ca41680ecc8cbd8b2c8b65f19c302496ef36780ba1. Auditor scripts/audit_published_actor.py SHAe17e92cc960613cb450df1831efa3d9b255fee42ec08bececb744ee6ac26dc5d independently recomputes the original-weight digest and stored score/gradient-vector/temperature/timing arithmetic. It does not rerun inference/backward/timing; producer separately checks original-module backward and every native feature digest. No new matches, optimizer steps or policy updates in the corpus study. Full307 tests passed without skips,27 expected upstream warnings,95.39s; six new actor tests exercise WP/Point on CPU/CUDA, RNG, dropout, gradients, hidden-world invariance and snapshots. Ruff/diff checks required before commit.

|Balanced CUDA batch|Forward/loss/backward median ms|Baseline allocated bytes|Peak allocated bytes|Peak reserved bytes|
|---:|---:|---:|---:|---:|
|32|8.418400|115102208|162326528|601882624|
|64|8.733000|117050368|209187840|601882624|
|128|8.632400|120947200|303974400|601882624|

Resident-feature component timing only:3 warmups/5 timed repetitions at each batch size. Excludes simulator collection, value critic, Adam, transfers and rollout storage. Peak allocations include other resident diagnostic objects; they are not the eventual full learner footprint. Actor weights are unchanged and no benchmark model is selected. Full-learning throughput must be measured after integration.

|Temperature|Mean greedy probability, discard|Pick-hand|Pick-draw|Koi|
|---:|---:|---:|---:|---:|
|1.0|0.294628|0.531894|0.538285|0.681823|
|0.25|0.450325|0.600557|0.609954|0.839003|
|0.1|0.621814|0.675268|0.687548|0.903392|
|0.05|0.737699|0.735223|0.751467|0.935674|

These are descriptive probabilities on nonforced reused states (discard10,287/pick-hand511/pick-draw506/koi998), not strength estimates. Temperature1 would introduce substantial action variation relative to the original deterministic policy, especially discard. Even temperature0.05 averages only0.738 greedy probability on discards. Raw Q scores must not be silently treated as calibrated policy logits; explicitly label the policy transformation and compare initial deployment behavior before learning.

Decision/next work: technical feasibility supports integrating an explicitly defined pretrained-actor PPO path, with a separately parameterized public-observation value estimator, exact state/optimizer/RNG restart, initial greedy-policy compatibility and a full-learning benchmark. Keep the score temperature/exploration choice explicit, and specify critic initialization/warmup before a bounded strength experiment. This is a different initialization/architecture intervention from the rejected MLP return experiments. No published-weight policy has yet been optimized or evaluated for new strength. No final seeds used; superiority, independent method confirmation, final tests and manuscript remain incomplete.


## Pretrained published-policy PPO integration and feasibility (2026-10-17)

Implemented koikoi/published_train.py with explicit score temperature0.05 and independent public560-input critic, actorLR3e-6/valueLR1e-4, dropout off/deterministic cuDNN, fixed-horizon signed-GAE0.95, original endgame assistance and embedded published_ppo_v1 restart. Original pretrained weights remain an explicit dependency. The default20-update critic-only warmup differs intentionally from the benchmark's three updates. No claim about calibrated probabilities, upstream SL initialization or improved strength. Base train.py/default policies retain their path. See DESIGN/README for input layout, restrictions and CLI.

Production SHA4b4360ed783c1d12cb5258e255f4b85f2e30dfa2e111fd8a078cf3439ce612cb. New20 CPU/CUDA tests cover independent Python transitions/observations/nonzero signed GAE, hidden-world invariance, independent value gradients, warmup actor stationarity, actual actor updates, exact saved-next-update metrics/tensors/Adam/environment/RNG and restore without original weight files. Initial GPU differences of about1e-7 in metrics were resolved by deterministic cuDNN, not tolerance relaxation. Full suite327 passed without skips/39 expected original Transformer warnings in101.66s. Ruff/diff checks pass.

Fixed benchmark scripts/benchmark_published_ppo.py SHA6ebbde7ccb3336981588e5b9e79fe3b18900643a04f0aa600012bb207d473c54, seed6,680,001/env128/h128/minibatch512/epochs3/critic256. Three critic-only updates:382 terminal matches/3,266 rounds/49,152 decisions,288 value Adam steps/no actor steps. Five timed actor/value updates:681 terminal matches/5,529 rounds/81,920 decisions,480 Adam steps for value and each phase. Full-learning elapsed16.401248699985445s,41.52122880745 matches/s and4,994.74165038 decisions/s. Includes collection/GAE/forward/backward/Adam; excludes initialization1.148877s, warmup and checkpoint IO. CUDA allocator baseline86,652,416 bytes/peak2,279,617,536/reserved3,613,392,896; this is not total device memory. Timing is a single bounded feasibility run, not an uncertainty estimate or a comparison with prior hardware/settings.

Totals1,063 completed matches/8,795 rounds/131,072 decisions/eight updates (three value-only/five policy updates); these do not add to the research12 scale gate. Matches can span warmup; period counts are terminal events. Initial parameter digest7d684d9063b8c4bbe406c7b50f9b43ac90bb774f5917b13522d28be1a81925f4 remains unchanged through warmup; final digest1c223cc0be73491ee78b5382221ad7e08cf52c5ab47e386a06454bd02daa440e differs after training. No selected strength model results from this benchmark.

Artifacts runs/published_ppo_benchmark/: report.json SHA5611d2f0d7d0fcc44ae727f2c0c45859b29db0342cfe029ade4e95667110be70; initial.pt78adb6eec79d21c0baa5a79ccb63e12f45c2fa4bb0b74074daae396029af3954; latest.pt91f297390694cd73a2634126df2049afafd16ceb9705ead1b0aa91a6f425c061; checkpoint_audit.json2f8290333b34874d2a08df0095fad386602c6223995a4b2754c9545e6d7b14ef. Separate audit recomputes tensor digests, actual optimizer states, all metric/count deltas and throughput arithmetic; it does not rerun timings or all trajectories.

Sequential CLI initial-checkpoint comparison against strict CPU assisted WP:two duplicate development pairs6,642,001-6,642,002, four matches/454 decisions,50%/point0 with identical paired seat outcomes and phase counts. initial_cli_check.json SHAbfe7d12f47bc61060cdcdc6d30792b088f90e90734f720011faa078c7f680722. This is a small compatibility check, not new strength evidence. Commands: python -m scripts.benchmark_published_ppo; python -m koikoi.evaluate --agent runs/published_ppo_benchmark/initial.pt --opponent rl-wp-assisted --profile paper8 --pairs 2 --seed 6642001 --device cpu --deterministic --endgame --output runs/published_ppo_benchmark/initial_cli_check.json. Next fix a two-seed screen before strength learning; no final seeds or unchanged extension of rejected methods.


Fixed pretrained-PPO screen is READY (not yet launched). Runner scripts/run_pretrained_ppo_screen.py SHA827ecd17d5728b4ac2e6f1b8a60f1634277a7e133ed4ed930c693014c9e23f7c pins production4b4360ed... and original WP weights. Two training seeds6,681,001/6,691,001, each20 critic-only +500 actor/value updates;128env/h128/minibatch512/epochs3/T0.05/actorLR3e-6/valueLR1e-4/lambda0.95/entropy0.02/value0.5/critic256/endgame. Exactly8,519,680 decisions per case,327,680 during warmup; terminal match counts measured. Checkpoint every20 for recovery only; only update520 evaluated. Baseline initial snapshot first runs1,000 duplicate development pairs1,000,000-1,000,999 versus strict CPU assisted WP, paper8/greedy/endgame/no search; every paired score must be0.5 and point difference0 or the pipeline aborts. Same evaluation deals for both final actors. BOTH absolute win CI lower bounds>50% AND paired improvement over initial lower bounds>0 required before advancing; points secondary, no pooling, no interim selection/adaptation/unchanged extension/final seeds. This tests the entire initialized PPO procedure, not one isolated component. Estimated full run roughly one hour of training plus CPU evaluations from measured throughput, not a completion guarantee. Preflight and compatibility-gate rejection fixture pass; production source remains unchanged since327-test suite. Command .venv/Scripts/python.exe -m scripts.run_pretrained_ppo_screen; independent endpoint/checkpoint audit required before any method decision.


Pretrained-PPO screen LAUNCHED after protocol commit/pushbc0decd. Session99853 remains running; actual runner25908 (launcher48408), initial evaluation actual61984 (shim51412) confirmed via Get-Process and same-session polling. Last observed initial evaluation400/1000 pairs, mean0.5; this is incomplete compatibility evidence, not a strength result. Pipeline runs initial evaluation, seed6681001 training, seed6691001 training, then both final evaluations sequentially. No training was live at this observation. Protocol runs/pretrained_ppo_screen/protocol.json SHA5823b638c7c0a17246e06b7a1ba6c6d46854147c479739a3e389b90c7c39a718; runner827ecd17d5728b4ac2e6f1b8a60f1634277a7e133ed4ed930c693014c9e23f7c and production4b4360ed... frozen. process.json records authoritative worker transitions; check actual process/session before interpreting stale log/status files.

Prepared independent auditor scripts/audit_pretrained_ppo_screen.py SHA620058f96d1414848eff0617cad8281977666ff15d86ad15ec7e9de31f5d6c45. Pinned protocol/source/runner/teacher checks pass. Actual initial checkpoint290e301702428a1762ffec2004ec9e9fafbb35a7d68a79e849c8e47a1126893e passes tensor/ancestry/runtime/zero-optimizer/state counters; existing real eight-update benchmark passes independent accounting.17 CPU fixtures pass including corrupt work counters/episode totals/terminal state/nonfinite rows and BOTH-seed strict criterion. Ruff/diff pass; no production change since327-test suite. Auditor imports independent statistical primitives from the earlier round-screen auditor, never production train/evaluate/compare helpers. Its scope is saved tensor/Adam/count/identity/raw-outcome arithmetic, not all-trajectory or timing replay. Full audit is pending final artifacts. Commands: python -m scripts.audit_pretrained_ppo_screen --protocol-only; after initial evaluation, --checkpoints-only audits available update20/update520 artifacts without ranking intermediate models; after all jobs, omit flags for independent_audit.json and fixed criterion. Do not claim success from a checkpoint-only pass.


Initial pretrained-screen compatibility gate is COMPLETE (worker61984 exit0). All1,000 duplicate development pairs against strict CPU assisted WP yield paired win0.5 and point0; report SHA f4d46806b35c0082f0fac2aa315c4f9979c89a61a91830a7a60f79c077d35269. Independent raw records/statistics/identity audit passes. This establishes finite outcome compatibility of the evaluation path, not improvement or universal action equality.

Seed6681001 critic-only warmup20 completed:2,778 terminal matches/22,035 rounds/327,680 decisions; value Adam1,920 steps, each actor0. Checkpoint update_00000020.pt SHAe3c9f5309707ce6b13575b7d0735f80e03a79ad1e3218f4ef9bf07d2efa1aaa4 retains the exact initial actor digest7d684d9063b8c4bbe406c7b50f9b43ac90bb774f5917b13522d28be1a81925f4. Independent checkpoint_audit_2.json SHA1c085072aa6dacbcbabfbfa773ef3d72179a1ac5a6fc10bc75024281380567f5 validates original tensors/ancestry, actual Adam counts, metric arithmetic and simulator episode totals. Its initial report is fully audited; only available warmup artifacts are covered for training, not final endpoints or all trajectories. Prior checkpoint_audit_1.json retained.

The same session99853 remains LIVE; runner25908 and actual first training worker21416 (shim13104) confirmed via process inspection and successful same-handle polling. Latest observed update35 includes15 actual policy updates,4,886 total terminal matches/38,511 rounds/573,440 decisions; finite health metrics, no strength evaluation. Second training and both final evaluations remain queued. Source/runner/protocol unchanged, no interim model selection, no final seeds. Next continue observing the existing job and audit final update520 when available; do not restart or rerun initial evaluation. Goal remains active.


Pretrained-screen first TRAINING endpoint completed normally: seed6681001 worker21416/shim13104 exit0,520 total updates (20 critic-only +500 actor/value),73,247 terminal matches/570,644 rounds/8,519,680 decisions. Includes warmup2,778 matches/22,035 rounds/327,680 decisions; terminal counters may span phase boundaries. Actual optimizer steps: value49,920, each actor48,000. Final checkpoint update_00000520.pt SHA9ae3911b436e67f219b016e25bf598cf1ad0734a0420332485b5b05d9ce166ff, actor digest953d74fb597d1c3b4b89737d481909f2e92562777619fe3c1fd554790cfc2b53. No intermediate model was evaluated or selected; strength evaluation remains queued after second training.

Independent available-artifact audit checkpoint_audit_4.json PASSED, SHAa38f558bda06ee3e899ae5a50e838cab6614f94a5d483d58f37c6de545bbec2c; prior audit_3 preserved. It covers initial evaluation plus both update20 warmups and the first update520 endpoint, checking actual tensors/Adam/metric arithmetic/environment episode totals/identities. Second warmup seed6691001:2,759 terminal matches/21,969 rounds/327,680 decisions, value Adam1,920/actor0, exact initial actor digest unchanged; checkpoint SHA7d388f86281f034478a9c06d9f91462b601b6ba00c7a79d822a846048fc44e78. This is correctness/accounting evidence, not strength. Full raw-outcome audit remains pending.

Session99853 and runner25908 remain LIVE; second training actual worker34692 (shim27168) confirmed by process inspection and same-session polling. Last observed second update26 (six policy updates),3,596 matches/28,571 rounds/425,984 decisions; finite metrics. Both final1,000-pair evaluations remain queued. Protocol/source/runner unchanged; no final seeds or method decision. Next monitor the same session, audit second update520 when available, then independently audit both full evaluations before applying the frozen replicated criterion.


BOTH pretrained-screen training jobs are COMPLETE, exit0. Each520 updates =20 critic-only +500 policy/value updates,8,519,680 decisions, value Adam49,920 and each actor48,000 actual steps. Seed6681001:73,247 terminal matches/570,644 rounds; seed6691001:73,321/570,725. Combined146,568 matches/1,141,369 rounds/17,039,360 decisions, including warmup5,537 matches/44,004 rounds/655,360 decisions. These paper8 training counts are separate from the completed100M research12 gate; matches may span warmup/update boundaries.

Available-artifact audit checkpoint_audit_5.json PASSED, SHA126c1fa4105cac3d2d10ed59616167280e7bc5cabc5925de61a361b5f46ceed6, covering initial and both update20/update520 snapshots. Checks actual parameter digests, original ancestry/config/runtime/source, Adam steps, all recorded update counters and simulator episode totals; initial1,000-pair compatibility stats also rechecked. No all-trajectory replay or final strength audit yet. Second final checkpoint SHA6d997a980848bb3c068f89a0c4219dba20958810a3a2a75e7033e22087b16d96, current actor digesta58d9ca50b6b8082b1b891f1ab142cc0d2aa709ceaa48a5d3974a8a2fbea8106; first endpoint remains9ae3911b436e67f219b016e25bf598cf1ad0734a0420332485b5b05d9ce166ff. No intermediate endpoint selection or source/condition change.

Same pipeline session99853 is LIVE. Runner25908 and first final-evaluation actual worker47864 (shim63268) confirmed by process inspection; both training workers exited0. First seed6681001 evaluation against strict CPU assisted WP is running; seed6691001 evaluation remains queued. Each fixed1,000 duplicate development pairs, same1,000,000-1,000,999 deals/greedy/paper8/assistance. No final seeds, no strength decision. Next wait on existing session, then run full scripts.audit_pretrained_ppo_screen without flags after all jobs finish and apply the frozen BOTH-seed criterion.


The fixed pretrained-PPO screen is COMPLETE and independently audited. All5 jobs (initial evaluation, two training, two final evaluations) exited0; session99853 exited0, runner25908 and final evaluator9416/shim39860 are absent. process.json retains complete_awaiting_audit; independent_audit.json PASSED, SHA643432440c451836559ab398d45a095f21043c4f04b60e26ec470dfa09d8c982; summaryd3dc54b0b5509f32d65e01754e8ce17e9cc66bae181810a4df7eb02feef450f1. Protocol5823b638.../runner827ecd17.../source4b4360ed... stayed fixed. Full audit checks actual checkpoint/tensor/ancestry/Adam/state/work identities plus all raw paired outcomes, endpoint and paired means/CIs, summaries and criterion. It does not replay every training trajectory or inference.

Final win scores versus strict CPU assisted WP: seed6681001 51.275% [49.725,52.85], seed6691001 51.075% [49.475,52.725]. Relative to the unchanged initial policy (exact50% on each duplicate pair): +1.275pp [-0.275,+2.85] and+1.075pp [-0.525,+2.725]. Both absolute and paired criteria FAILED; neither interval demonstrates win improvement or harm. Secondary final candidate-minus-opponent point differences:2.297 [1.359,3.225025] and1.956 [0.92985,2.959025]. Positive points do not replace the win criterion. All1,000 development pairs per endpoint use the same explored deals; two models are not2,000 independent deals. Initial and final evaluation budgets are separate from training.

Final report hashes: seed6681001 eab9128bf7c2d6f75ada9093f049e2b4dd40d9a9917442ac4f97578a4a164e39; seed6691001 3e884d93df56e7991595c30b9b89f5a964d65142c0e4c0ea649d59eb6dcd05b6. Training remains146,568 terminal matches/1,141,369 rounds/17,039,360 decisions, two500-policy-update endpoints after20 warmups each. Keep both final snapshots and the original-weight dependency. Do not describe this optional PPO mode as an established strength improvement, adopt it by default, rerank intermediate checkpoints, extend the same endpoints unchanged or use final seeds.

Next diagnostic question: how different are the explicitly sampled training policies (softmax original/current scores atT0.05) from deterministic greedy deployment? Original zero-update corpus already showed nonforced discard mean greedy probability0.7377 atT0.05; that is a documented distribution change, not evidence that it caused this inconclusive screen. Assess initial versus both final actors on a fixed already-audited public-state corpus, by phase and with declared lower temperature probes, without new strength games or intermediate-model selection. Use descriptive probabilities/action agreement only; do not infer winning strength from them. Then specify a genuinely changed, controlled learning experiment if justified. Both point gains and positive win means remain exploratory; public superiority, independent selected-method validation, untouched final testing and manuscript remain unfinished. All experiments now terminal; full goal active.


Saved-state pretrained-policy diagnostic is COMPLETE and independently audited (session36187 exit0). Scripts diagnose_pretrained_policy.py SHA47ccd61d562a2232c9bb51fd6edfb736d6137369c3bce0c9eb06d9ec70339ad9 and audit_pretrained_policy.py SHA7d02e1876a890125a8293c264f2a5b87cbefb031483306bda52d83943201f4a2. No production code/default change, new match, optimizer step or strength evaluation. Initial and both fixed final actors scored on all13,029 previously audited states, CPU/CUDA batch128; native feature hashes/masks/public guards all agree. No initial-vs-original-CPU or CPU-vs-CUDA greedy disagreement; all64 predefined per-phase CPU single-row probes per actor agree with batched actions. These are finite checks, not universal floating-point equivalence. Actor weights and inference RNG unchanged.

Primary descriptive summaries exclude selected search roots and synthetic hidden variants: only the original100 random-policy matches, with nonforced/post-guard rows discard10,219/pick-hand479/pick-draw474/koi853. They are correlated, off-policy states, not actual learner/deployment visitation frequencies; no confidence intervals or causal/strength claims. Temperatures0.05/0.025/0.01 fixed before scoring. Four-phase greedy changes from initial: seed6681001 1,067/33/30/65, seed6691001 1,223/32/35/53. Discard changes10.4413%/11.9679%; koi7.6202%/6.2134%. Koi stop-to-continue27/28 versus continue-to-stop38/25: no consistent blanket direction.

AtT0.05 single-argmax probability (discard/pick-hand/pick-draw/koi): initial0.737611/0.739719/0.750182/0.951623; seed6681001 0.781281/0.751413/0.767729/0.958209; seed6691001 0.773204/0.756016/0.768743/0.956248. AtT0.01: initial0.898671/0.841264/0.848659/0.988367; final1 0.926105/0.859904/0.867874/0.990424; final2 0.923998/0.858678/0.864585/0.990773. Temperature changes the hypothetical probability transformation on fixed weights, not those weights or greedy ranking. It does not predict that lower-temperature training improves wins.

Post-hoc tie clarification was added without rerunning inference or altering the original diagnostic artifacts. Exact-score ties occur in300/288/284 of10,219 discard rows (initial/final1/final2),91/479 pick-hand and85/474 pick-draw rows for each actor, no koi rows. Summing probabilities over ALL exactly maximal-score actions gives discard mass atT0.05:0.750326/0.793926/0.785698, atT0.025:0.837297/0.874266/0.869068, atT0.01:0.912723/0.939836/0.937620. Thus about20.6%/21.4% of sampled final-policy discard probability remains outside the maximal-score set atT0.05, versus6.0%/6.2% atT0.01 on these saved states. For capture phases, exact-tie correction raisesT0.01 final masses toabout95.4%-95.8%. Exact score ties do not prove strategic action equivalence. All reported changed greedy actions lie outside the initial maximum set, so their counts are not merely alternative tie breaking.

Artifact directory runs/pretrained_policy_diagnostic: protocol8e58e3ae31f33991fce636e3965fde20dcb679f3f207c1391f6c4ea8c1d8c85f; reportb10b32a59fbe9c8c0085ea428a7855a5e5a71354ffbe09f75512aec390930142; score archiveb34b56d132bb691eb1cc1dfe85d6f88e90b941548e4694a778ed0724ee2859c8; independent audit21aac38fb9d5c45715fbcccf8e15eafbafa78a15e9d8c4583c9cca153721fdc5. Audit verifies actual file/tensor identities and all72 statistical groups via scalar math/explicit quantiles; it does not independently rerun inference. Post-hoc tie_auditc3cba0c0b4e708f4e4c7852491b0c65ffa4d3449c52c3dd4337c7e022cfd9f89 verifies vector versus scalar partition-function arithmetic; script audit_pretrained_ties.py SHAebb6979f8446c412ad22c76fe530846a65f0d0e083285a4cafb3640f2ef4ba05.14 hand-calculated/malformed-input/tie fixtures pass, Ruff and diff checks pass. No production change since the327-test suite.

Decision: the sampled/greedy distinction is quantitatively substantial on this off-policy corpus, including after exact-tie correction. It remains an unproven explanation for the earlier inconclusive win screen. Next preregister a controlled fresh-seed study ofT0.01 with actorLR6e-7 versusT0.05/actorLR3e-6 controls, retaining other learner settings and original WP initialization. Reducing LR withT is a heuristic to avoid increasing logit-step sensitivity fivefold; Adam/global clipping/visitation/entropy changes mean this is a bundled intervention, not a guarantee of matched updates or an isolated exploration effect. Fix training/evaluation budgets and criteria before learning; use new development deals and a budget suitable for effects around1pp rather than extending old endpoints or selecting intermediate weights. No such training has begun. Original screen remains failed/inconclusive; no adoption or final-test launch. Public superiority, independent confirmation, untouched final testing and manuscript remain outstanding; goal active.


Controlled lower-temperature/LR study is READY, not yet launched. Runner scripts/run_cold_ppo_screen.py SHAab36a65f52b768f01ff068b357a1206c6f5cb1a61e68f870a635b555bfcbd202 pins production4b4360ed... and original WP weights. Fresh training seeds6,710,001/6,720,001 each have warmT0.05/actorLR3e-6 and coldT0.01/actorLR6e-7 arms. Same20 critic-only +500 policy updates/env128/h128/minibatch512/epochs3/valueLR1e-4/lambda0.95/entropy0.02/critic256/endgame,8,519,680 decisions per case. Temperature and actorLR change together; no isolated exploration or matched-Adam-update claim. Old screen/models remain terminal and inconclusive; no continuation or added evaluation of old trained endpoints.

Fixed new development evaluation:5,000 duplicate pairs7,200,000-7,204,999 per initial/four final endpoints against strict CPU assisted WP, greedy/paper8/endgame/no search. Existing463 JSON report ranges contained no overlap (scan limited to recorded seed_start/first_seed with pairs/matches, files<=64MB). Initial baseline reuses the zero-update pinned snapshot290e3017...; its new full outcome-compatibility gate must pass before final evaluations. Initial cold/warm float32 score division preserves all13,029 cached greedy actions on both devices, finite evidence only. Evaluation budget increased prospectively for approximately1pp effects on new model/deal data, not by extending old endpoints.

Two distinct questions are preregistered. Temperature-bundle benefit requires BOTH cold-minus-warm paired95% win-score CI lower bounds>0. Separately, each arm is publicly qualified only if BOTH of its training seeds have absolute95% win-score CI lower bounds>50%. Points secondary. Selection: cold if qualified and paired benefit replicates; otherwise warm if qualified; otherwise cold if qualified; otherwise none. Absolute qualification alone never establishes cold-versus-warm improvement. Canonical checkpoint always seed6710001 of the selected arm, never the better-scoring seed. Original failed screen is retained as evidence. All selections remain development signals pending independent audit and separate untouched final confirmation; shared deal sets are not extra independent samples.

Runner limits1 GPU training plus2 CPU evaluation workers, with explicit dependencies and recorded PIDs. Initial CPU evaluation overlaps training; every final evaluation waits for its training and initial compatibility. Latencies under concurrent CPU load are not isolated benchmarks. Worker failure aborts only owned live process trees and preserves artifacts. No intermediate ranking/adaptation/unchanged extension/final seeds. Preflight and11 gate/selection/resource/dependency/failure-cleanup tests pass; Ruff/diff pass. Production learner remains unchanged since327-test suite. Command python -m scripts.run_cold_ppo_screen; independent saved-state/outcome auditor must be ready before any selection.


Cold/warm study LAUNCHED after protocol commit/push2ce405b. Session85269 LIVE; actual runner57300 (launcher56396), initial CPU evaluator65092 (shim30576), first warm GPU trainer47200 (shim49172) verified through process inspection and same-handle polling. Last observed warm update106/86 policy updates,14,962 terminal matches/116,748 rounds/1,736,704 decisions; initial evaluation690/5000 pairs. These are progress/health observations, not strength results. Source/runner/protocol frozen. Protocol runs/cold_ppo_screen/protocol.json SHAae81104920a1b31f84c1ee45045c0e984f91540c0b279420c456ab4bcf279950. New runner manifest jobs is a DICTIONARY keyed by job ID; each record has resource/dependencies/PID/status. Check actual processes/session before restarting; observation timeout is not termination.

Independent auditor scripts/audit_cold_ppo_screen.py SHAfcc30b9fd56826101cb7fa52b8dfb58709f37e016970544f0a689d90a0ed75ae is ready. Protocol/source/runner/dependency checks pass; initial snapshot and first warm update20 accounting pass in checkpoint_audit_2_0.json SHAad7ff750eda75aea89563eb1a528d1a4f72ab55eb0b47e80b9a4dbe49c17156a. Warmup checkpoint SHAe1c7a82308ced65fafbc91d21823b1d3d9ac0c6f1ed9fc3f769644cbc3bced59, counters {"updates":20,"decisions":327680,"rounds":21996,"matches":2762,"policy_updates":0,"warmup_updates":20,"warmup_decisions":327680,"warmup_rounds":21996,"warmup_matches":2762,"optimizer_steps_value":1920,"optimizer_steps_discard":0,"optimizer_steps_pick":0,"optimizer_steps_koi":0}. Actor digest exactly initial; actual Adam steps and environment episode totals checked. Initial new5000-pair evaluation is NOT yet complete/audited; the suffix _0 records this explicitly. Partial auditing skips unfinished metric tails and covers only saved20/520 snapshots, never intermediate ranking.

25 focused tests pass (selection boundaries/replication/canonical seed, independent score and point records, metric tails, resource/dependency schedule overlap, and owned-worker failure cleanup), Ruff/diff pass. No production learner changes since327-test suite. Auditor reuses earlier independent scalar-bootstrap/accounting primitives, not production training/evaluation/selection helpers; verifies actual identities/tensors/Adam/work, full completed worker schedule, all raw paired summaries and independent selection decision. No all-trajectory/inference replay claim. Commands: python -m scripts.audit_cold_ppo_screen --protocol-only; --checkpoints-only for available accounting (initial evaluation optional until complete); no flags only after all9 jobs finish. Any model selection is provisional until the full audit and remains development evidence before untouched final testing. Full objective active.

First cold/warm-study training endpoint seed6710001_warm is COMPLETE: worker47200/shim49172 exited0 and are absent; session85269/runner57300 remains live. Exactly520 updates (20 critic-only +500 policy),73,473 terminal matches/572,602 rounds/8,519,680 decisions. Warmup included2,762 matches/21,996 rounds/327,680 decisions; match/round event counts are separate from decisions and can cross update-phase boundaries. Actual Adam steps value49,920 and each of discard/pick/koi48,000 agree with the saved history.

Independent checkpoint_audit_3_0.json PASSED (initial + first warm20/520), SHA 6beb145d425c00eaf47b485867461a2161d9f9161bfc9d674c436b4ab7528ab1. Final checkpoint runs/cold_ppo_screen/seed6710001_warm/update_00000520.pt SHA37d06849261e0c1f0c31a9d2f228388fa1f391b670ce6772093a1c276091cc36; current actor parameters SHA7275e898c1b1922c7f48993de72ecf1ebfd87c1589db6862df0743023d73110c. Protocol/source/runner identities, pinned initialization, actual tensors/config/counters/environment episode sum and Adam state checked. Stored-state/accounting audit only, no trajectory/inference replay or strength claim. Command: .venv/Scripts/python.exe -m scripts.audit_cold_ppo_screen --checkpoints-only. Earlier audit snapshots retained.

Runner automatically started seed6710001_cold (actual33288/shim54316), confirmed live with initial evaluator65092. Initial evaluation last observed3650/5000 pairs; it is not complete or independently audited, so final-model evaluations remain pending its compatibility gate. No final outcome/selection decision is available. Frozen source, protocol, budgets, dependencies and selection rules unchanged; no additional tests or learner changes were needed for this accounting milestone. Full objective remains active.

Initial cold/warm-study evaluation is COMPLETE: actual evaluator65092/shim30576 exited0 and are absent. All5,000 duplicate pairs use exactly development seeds7,200,000-7,204,999. Independent raw-record audit verifies each pair score0.5 and point difference0, identities/range/uniqueness/phase work and recomputed statistics. Thus this fixed corpus passes the initial outcome-compatibility gate; no all-action/trajectory equality or universal equivalence claim. Initial report runs/cold_ppo_screen/initial/evaluation.json SHAe4284aff62646ba26275ae9f62c7f2fb0a3359f92bb20108ba8b9456a7cd87b2.

Independent checkpoint_audit_4_1.json PASSED, SHA4245453ceee81440844233a71153abbe59e80f0f567785a054b52a7ac516b434: initial snapshot, first warm20/520, first cold20, plus the completed initial evaluation. Earlier pre-initial-evaluation checkpoint_audit_4_0.json SHA310fc7956b6f828d4d3d5e0ff62016057535e3adc6a0d4b30a76e42bef43ad17 is retained. First cold warmup checkpoint SHA289c7b3542c81431bbd8d6a1e25122456842756fe369dad093c7889e2e5a30a7;2,798 matches/22,226 rounds/327,680 decisions,20 critic-only updates/value Adam1,920/each actor Adam0. Actor parameters exactly retain initial digest7d684d9063b8c4bbe406c7b50f9b43ac90bb774f5917b13522d28be1a81925f4. Protocol/source/runner/config/tensor/accounting checks remain passed. Command: .venv/Scripts/python.exe -m scripts.audit_cold_ppo_screen --checkpoints-only.

After the initial gate, runner57300 automatically started first warm final-model evaluation39516 (shim33888); it runs alongside first cold trainer33288 (shim54316). Session85269 and both workers verified live. Cold last observed205/520 updates/28,982 matches/226,125 rounds/3,358,720 decisions. No trained-model endpoint evaluation is complete, so no selection or strength conclusion is available. Frozen conditions/criteria and final unused seeds unchanged. Full selected-method validation, untouched final testing and manuscript remain outstanding; goal active.

First cold endpoint seed6710001_cold is COMPLETE: actual worker33288/shim54316 exited0 and are absent. Exactly520 updates (20 critic-only +500 policy),73,552 terminal matches/572,736 rounds/8,519,680 decisions. Warmup included2,798 matches/22,226 rounds/327,680 decisions. Actual Adam steps value49,920 and each actor48,000. Combined completed first-seed warm/cold training:147,025 matches/1,145,338 rounds/17,039,360 decisions; these are training counts, not strength evidence or new independent evaluation samples.

Independent checkpoint_audit_5_1.json PASSED, SHAc830072c4b1af4c60c985bab3ffc54fcc8517e12480df3f73c274581985c4b4e: initial snapshot, first-seed warm/cold20/520, and all5,000 initial evaluation pairs. Cold final checkpoint runs/cold_ppo_screen/seed6710001_cold/update_00000520.pt SHAbc5133b9b3926185bd23b6ef9139d361789e116bee24fcccafa8fdf261ffd25d; actor parameters SHA90d9ced5db46b30174141a73a3f91c856afb1388b877a845aa587c414ef0fb4f. Source/protocol/runner/pinned initial identities, configurations including T0.01/LR6e-7, tensors/counters/Adam/environment episode totals checked. Stored-state/accounting/raw-initial-outcome scope, not all-trajectory/inference replay. Command: .venv/Scripts/python.exe -m scripts.audit_cold_ppo_screen --checkpoints-only. Earlier snapshots retained.

Runner automatically started second-seed warm training64888 (shim61692) and first-seed cold evaluation36548 (shim52356). First warm evaluation39516 (shim33888) continues, last observed2310/5000 pairs. Session85269/runner57300 and all three workers confirmed live: one GPU training and two CPU evaluations. No trained-model evaluation has completed, so no adoption, strength conclusion or temperature-benefit decision is available. Fixed conditions/dependencies/budgets remain unchanged. No learner change or extra testing for this accounting milestone; independent selected-method confirmation, untouched final tests and manuscript remain outstanding. Goal active.

First trained-model evaluation seed6710001_warm is COMPLETE: actual39516/shim33888 exited0 and are absent. Exactly5,000 development duplicate pairs7,200,000-7,204,999 against strict CPU assisted RL-WP, fixed update520/greedy/paper8/endgame/no search. Independently recomputed draw-adjusted win score51.805% [51.090,52.525], point difference+2.3802 [1.9608,2.802005]. Paired versus zero-update initial score+1.805pp [1.090,2.525], same point difference because every initial pair has zero point difference. This is a positive result for ONE prespecified development endpoint, not replicated arm qualification, a temperature comparison, final-test confirmation or general strongest-AI evidence. Other results do not change the frozen experiment/selection rules.

Completed-endpoint audit runs/cold_ppo_screen/endpoint_audit_seed6710001_warm.json PASSED, SHA6226c0f92bf709067f77d5df814e092dacd47bade3c7eccc86db9c7331aec419; evaluated report SHA3096ec3770bf8034316d8eb9b4a24be9db3487ac8968dff2fddf43e42bfb3532. Invoked existing audit_cold_ppo_screen protocol/checkpoint/metrics(final=True)/report/statistics helpers for this completed endpoint and the initial baseline; no auditor/learner/protocol changes. Actual identities/config/tensors/work/Adam, terminal job return codes, all unique raw seeds/seat-score point signs, phase counters and independently recomputed means/CIs agree. No trajectory/inference replay; complete study schedule/summary/paired cold-versus-warm/selection audit still pending all9 jobs. Artifact decision=null. Later full audit rechecks this same fixed endpoint; no new matches, endpoint extension or interim checkpoint ranking.

Third training endpoint seed6720001_warm is COMPLETE: actual64888/shim61692 exited0 and are absent.520 updates (20 critic-only +500 policy),73,390 matches/571,033 rounds/8,519,680 decisions; warmup2,759 matches/22,004 rounds/327,680 decisions. Value Adam49,920 and each actor48,000. Combined three completed training endpoints220,415 matches/1,716,371 rounds/25,559,040 decisions. Final checkpoint SHA5e8a123d5fafb6ebae8c1ac8d094c9de65bf364381cc10b78208337051e181cc; actor parameters SHAfbb7f2b203f8c7f228549c6122529a08932a9a014e93cc7b035b6633c6121925.

Independent checkpoint_audit_7_1.json PASSED, SHA943acd3626b6df2bd53e786cc568d213c5c90757c92decb1f76bd00082437eda: initial, first-seed warm/cold20/520, second warm20/520, plus initial5000-pair records. Earlier checkpoint_audit_6_1.json SHA211797436481ff2261ea96034f4b5acc3b7ff7d497d829567d394fd7b3bf2a32 retained; second warm20 checkpoint SHA10d244322e665f9173330f1e3663180a1ead45ddd181c9697edce6aa15440105 retains initial actor parameters and value Adam1,920/actor Adam0. Command: .venv/Scripts/python.exe -m scripts.audit_cold_ppo_screen --checkpoints-only. This mode does not audit completed trained-model evaluation records; the separate completed-endpoint artifact above does.

Session85269/runner57300 remains live. Runner automatically started final cold training19196 (shim32384) and second warm evaluation52776 (shim57224); first cold evaluation36548 (shim52356) continues. All three actual workers confirmed live. No model selection or new final seeds; both-seed absolute qualification and paired temperature-bundle benefit remain separate pending criteria. Independent selected-method validation, untouched final testing and manuscript remain outstanding. Goal active.

First cold-model evaluation seed6710001_cold is COMPLETE: actual36548/shim52356 exited0 and are absent. Exactly5,000 development duplicate pairs7,200,000-7,204,999, same frozen comparator/settings as warm. Independently recomputed draw-adjusted win score50.755% [50.134875,51.395], points+1.3046 [0.93479,1.676425]. Paired versus initial score+0.755pp [+0.134875,+1.395]. Both first-seed endpoints have positive absolute lower bounds, but neither arm has yet met the required BOTH-seed qualification.

On these same5,000 paired deals, cold-minus-warm score -1.05pp [-1.875,-0.200], points -1.0756 [-1.54141,-0.59215]. This first-seed comparison favors the warm bundle; it does not establish a replicated difference or isolate temperature from the jointly changed learning rate, entropy/visitation and optimizer effects. The fixed both-seed-positive paired-benefit requirement fails at this seed; nevertheless complete every prespecified endpoint and the full audit, with no early termination, resampling, extension, intermediate ranking or changed selection rule. Absolute qualification remains a separate question. Shared deals are not extra independent evaluation samples.

Completed-endpoint audit runs/cold_ppo_screen/endpoint_audit_seed6710001_cold.json PASSED, SHA4fbf9b67736ac6e884d07edbe1d24b597f8d3c50594817032999d34ae86898ac; cold report SHA6c3accccd580333d3b8a9188ab66cbf2adea36aaa559028cc0c65331462085a5. Invoked unchanged audit_cold_ppo_screen protocol/checkpoint/metrics(final=True)/report/statistics helpers on cold, warm and initial. Actual identities/config/tensors/counts/Adam, terminal return codes, all unique seeds/seat-score point signs, phase work, endpoint and paired means/CIs agree. No trajectory/inference replay; decision=null and final study schedule/summary/selection audit remains pending. Original warm endpoint audit retained. No new training/evaluation games were added by auditing.

Last cold warmup20 checkpoint also passed in checkpoint_audit_8_1.json SHA1a6d69163f5514ac1b8bd7ab30283e149473f829884726834a324ff873829eb4. Checkpoint SHA2700fca47a3d74918e6f227da8ed5e6c804a502924a6e95bf376eb4aa82ed6c9;2,775 matches/22,125 rounds/327,680 decisions,20 critic-only updates/value Adam1,920/each actor Adam0. Actor parameters retain initial digest7d684d9063b8c4bbe406c7b50f9b43ac90bb774f5917b13522d28be1a81925f4. Command: .venv/Scripts/python.exe -m scripts.audit_cold_ppo_screen --checkpoints-only. This snapshot covers initial and all available20/520 checkpoints plus initial outcomes, not trained-model outcome records; separate endpoint audits above cover those.

Session85269/runner57300, final cold trainer19196 (shim32384) and second warm evaluator52776 (shim57224) confirmed live. Last observed final cold241/520 updates/34,041 matches/265,052 rounds/3,948,544 decisions; second warm1590/5000 pairs. Remaining trained endpoints will determine replicated qualification under the fixed decision rule. No adopted model or final-test launch. Independent selected-method validation, untouched final testing and manuscript remain outstanding; goal active.

ALL FOUR cold/warm-study training endpoints are COMPLETE with returncode0. Final cold seed6720001_cold actual19196/shim32384 are absent; exactly520 updates (20 critic-only +500 policy),73,482 matches/571,424 rounds/8,519,680 decisions. Included warmup2,775 matches/22,125 rounds/327,680 decisions. Value Adam49,920 and each actor48,000 agree with actual saved history. Final checkpoint SHA8975a91193af42e21f82ea697f50c3f0220ff2bc2417d2d9ffbc8020a35b9bc5; actor parameters SHA193cbe26b53c07bdb866e7409b029fd9ceefb776158d3fd0b92976da80ae4826.

Independent checkpoint_audit_9_1.json PASSED, SHAb81eb3e5732482eaf610140dead3c4fad2a6a3fac24e185b47afbe764b58d1dd. Covers initial snapshot plus all eight20/520 snapshots, initial5000-pair outcome records, protocol/source/runner/dependency identities, configs/actual tensors/counts/Adam and environment episode totals. No full trajectory/inference replay. Trained-model evaluation records remain covered by the separately completed endpoint audits and later full-study audit. Command: .venv/Scripts/python.exe -m scripts.audit_cold_ppo_screen --checkpoints-only. No changes to production/runner/protocol/auditor or additional tests for this accounting milestone.

Total fixed study training budget verified from all four final checkpoints:2,080 updates (80 critic-only +2,000 policy),293,897 matches/2,287,795 rounds/34,078,720 decisions. Warmup included11,094 terminal matches/88,351 round endings/1,310,720 decisions. These event counts can cross phase boundaries and are not independent evaluation samples or evidence of strength. Existing research12 100M gate remains separate; no resumed or extra endpoint learning.

Runner automatically launched final cold evaluation48436 (shim29860), alongside second warm evaluator52776 (shim57224). Both workers and session85269/runner57300 confirmed live; no GPU trainer remains. Last observed second warm3640/5000 pairs. Two trained-model evaluation jobs remain unfinished; full schedule/summary/paired comparison and final selection audit must wait for all9 jobs. First-seed results and all frozen criteria remain unchanged; no model adoption/final seeds. Goal active, with independent selected-method validation, untouched final testing and manuscript still outstanding.

Second warm evaluation seed6720001_warm is COMPLETE: actual52776/shim57224 exited0 and are absent. Exactly5,000 development duplicate pairs7,200,000-7,204,999 against strict CPU assisted RL-WP, frozen update520/greedy/paper8/endgame/no search. Independently recomputed draw-adjusted win score51.500% [50.764875,52.245], point difference+2.1546 [1.719195,2.588615]. Paired versus initial score+1.500pp [+0.764875,+2.245], same point difference because initial paired points are zero.

Completed-endpoint audit runs/cold_ppo_screen/endpoint_audit_seed6720001_warm.json PASSED, SHAe71c8b280fa9ddc3d4d1d355667a4a097f30f4ed4b93cca9173bdacb2982462c; evaluated report SHA2b8527eb583f1240ea2c60620bc4b597bf2922b82d6a4c38300079295e6dccfd. Invoked the unchanged audit_cold_ppo_screen protocol/checkpoint/metrics(final=True)/report/statistics helpers on this completed endpoint and initial baseline, as for first warm. Actual identities/config/tensors/work/Adam, terminal job status, all unique raw seeds/seat-score point signs, phase counters and independent means/CIs agree. No trajectory/inference replay, complete-study selection or new matches; artifact decision=null. Full audit will recheck all fixed endpoints together.

Warm endpoint qualification now replicates across BOTH fresh training seeds: each absolute95% win-score lower bound exceeds50%. This satisfies the preregistered development qualification for that arm, not untouched final-test superiority. These models share the SAME5,000 development deals; never pool them as10,000 independent pairs. Prior inconclusive warm screen remains disclosed. First-seed cold-minus-warm benefit criterion failed, but final cold evaluation and full schedule/summary/decision audit must still finish. Canonical checkpoint rule remains seed6710001 if this arm is selected; no intermediate/model reranking, budget change or final-test launch.

Only final cold evaluation48436 (shim29860) remains live alongside runner57300/session85269, last observed1620/5000 pairs. No GPU trainer or other evaluator remains. After the fixed experiment and full audit, freeze the selected method/checkpoints and specify untouched final evaluation, preserving independent-model confirmation and assisted RL-Point as the secondary comparator in COMPARATORS.md. No final seeds have been used. Full goal remains active; final validation and manuscript remain unfinished.

Prospective final-test runner and independent auditor implemented in scripts/run_final_published_test.py and scripts/audit_final_published_test.py. Preparation refuses a running or unaudited development study before creating the final directory; actual missing-audit gate verified. Fixed plan: BOTH selected-arm training seeds6710001/6720001 against assisted RL-WP (primary) and assisted RL-Point (secondary), each10,000 duplicate pairs on untouched seeds100,000,000-100,009,999, paper8/strict CPU/deterministic/endgame/no search. Four CPU jobs at2 threads each,80,000 played matches but only10,000 unique deal streams shared across jobs. Both models must independently have primary win-score95% CI lower bounds>0.5; secondary criterion is separate. Canonical remains first seed; no test-outcome reranking, adaptation, extension or early strength stopping. Concurrent timings are not isolated latency evidence.

Runner freezes source, complete audited development evidence, selected model/metrics, comparator checkout/revision/all17 tracked Python files and6 weights, evaluation settings and auditor. It rechecks identities before/after each worker and refuses silent reruns. Independent auditor recomputes raw paired statistics, identities/accounting and all4 terminal jobs; it does not replay every trajectory/inference.43 focused tests passed in1.65s and Ruff passed, including incomplete evidence, drift/missing dependencies, wrong split/model/comparator, nested historical seed overlap, altered summaries and owned-worker failure cleanup (fake processes). Real recursive scan found144 stored JSON records with ranges or raw seeds, none overlapping the reserved range; historical UTF16/BOM JSON inspected without rewriting. Scope excludes unrecorded work. Readiness artifact runs/final_published_test_readiness.json SHAf1ad8a6e2642f4065b94334ad20bf157f2a6ce3ecd577940e7403dd442775372. Production source4b4360ed783c1d12cb5258e255f4b85f2e30dfa2e111fd8a078cf3439ce612cb unchanged. No final protocol/directory or final game yet; complete development audit remains a hard prerequisite.


Cold/warm development study COMPLETE and independently audited. Session85269 exited0; runner57300/final evaluator48436 absent. All9 jobs completed successfully, no extra training or outcome-driven extension. Full audit runs/cold_ppo_screen/independent_audit.json SHA6b8548605b50226738227ce83dba1a2e946efd282461b0fa8d48132a42cfe9ba; summary SHA4a59f27352c6139003f06ad1fb46e39dda9e02a1dbf36a48bd283307e3872c9e. Command: .venv/Scripts/python.exe -m scripts.audit_cold_ppo_screen. Rechecked all9 checkpoints, all5x5000 raw paired evaluations (50,000 played matches on5,000 shared deal streams), identities/accounting/Adam/phase work, independent means/CIs, schedule, summary and fixed selection rule. Scope excludes full trajectory/inference replay.

Final cold seed6720001 endpoint win-score50.565% [49.930,51.215], points+1.1602 [0.78336,1.54021]; report SHA39143275af0c8610bb7a7bd50d6f34168b174e9a947d1fd19bdf6b96da55222c. Cold-minus-warm on that seed -0.935pp [-1.810,-0.075], points-0.9944 [-1.49225,-0.48399]. Together with first-seed -1.050pp [-1.875,-0.200], both paired intervals favor the warm bundle. Do not isolate temperature causally because actor learning rate changed too. Cold fails both-model absolute qualification and the paired-benefit criterion; warm passes absolute qualification on both seeds:51.805% [51.090,52.525] and51.500% [50.764875,52.245]. These are development results; previous exploratory/inconclusive endpoints remain disclosed.

Formal fixed-rule decision: selected_arm=warm; canonical_case=seed6710001_warm. Selected final update520 checkpoints SHA37d06849261e0c1f0c31a9d2f228388fa1f391b670ce6772093a1c276091cc36 and5e8a123d5fafb6ebae8c1ac8d094c9de65bf364381cc10b78208337051e181cc. No better-seed ranking. Freeze T0.05/actor LR3e-6,20 critic-only+500 policy updates and existing public/endgame greedy evaluation. Development study training remains293,897 matches/2,287,795 rounds/34,078,720 decisions across all4 endpoints, separate from research12's100M computation milestone.

Final protocol PREPARED, independently checked BEFORE any final game: runs/final_published_test/protocol.json SHA4a69cff95079e2bd4cfcda84241f4096149c05cfe408bd11cbfde154eaa060e4. Commands: .venv/Scripts/python.exe -m scripts.run_final_published_test --prepare; .venv/Scripts/python.exe -m scripts.audit_final_published_test --protocol-only. Both succeeded. Uses the previously committed4x10,000 duplicate-pair plan, both frozen warm models and WP/Point comparators, seeds100,000,000-100,009,999,80,000 matches on10,000 shared deal streams. Models, dependencies, runner/auditor and criteria remain frozen through execution/audit. Primary requires BOTH WP lower bounds>50%; secondary Point is separate; point means cannot replace wins. Next authorized command: .venv/Scripts/python.exe -u -m scripts.run_final_published_test --run. No final outcomes yet. Goal active; final validation and manuscript unfinished.


Untouched final test is LIVE. Command .venv/Scripts/python.exe -u -m scripts.run_final_published_test --run; unified session17151, actual runner45028 (shim61876). Four actual CPU workers: seed6710001_warm__wp35564 (shim21716), seed6720001_warm__wp64780 (shim58100), seed6710001_warm__point45896 (shim40536), seed6720001_warm__point45888 (shim16824). All four confirmed running and advancing at30/10000 pairs on first observation. Protocol SHA4a69cff95079e2bd4cfcda84241f4096149c05cfe408bd11cbfde154eaa060e4 matches the manifest. Final reserved seeds are NOW being consumed; never describe them as unused or silently restart these jobs. No final strength conclusion until all4 fixed endpoints finish and independent audit passes. Do not alter frozen production, runner, auditor or bound study/comparator artifacts during the test. Observation timeout is not termination; inspect the original session/process. Isolated latency measurement and manuscript remain unfinished; goal active.

Research manuscript DRAFT created while the frozen final test runs. Tracked docs/paper/manuscript.tex, references.bib, README.rst and generated tables/evidence; scripts/export_paper_evidence.py binds17 audited/protocol/report artifacts and refuses to transcribe incomplete final results. It exports explicit draft_final_results_pending/final=null and an empty final table until a complete final independent audit exists. Development table values and combined293,897 matches/2,287,795 rounds/34,078,720 decisions independently checked against expected audited totals. The draft separates research12's100M computation from pretrained paper8 models, discloses earlier inconclusive warm screen and negative search/confirmation evidence, and includes exact selection/final rules, public-only observations, signed GAE, limitations and reproducibility. Method description checked against actual source/config. Original paper/pinned source plus official PPO/GAE bibliographic records cited. No final strength result or submission claim.

Compiled8-page working PDF output/pdf/manuscript.pdf, SHAadafc7993395263739e734c12ff67dd677126c2399d937150c114b98c425d5d7, from manuscript SHA10aa92d9b84bb3e9807cef6a6b5b3c298f54c327d66a1ffb70c22dc978956319; generated evidence SHAaab18bb0f0f36b889ae25e378fb8e63bc7721850fca9baf49d2f852a028359e8. QA artifact runs/manuscript_draft_qa.json SHA5593932840d7d1265b35655ae57f505cd0ef516a9687ac32ada9f894bcf10e6b: table/counter transcription, citation keys, no unresolved references/missing glyphs/TeX overfull or underfull boxes,14 embedded fonts, page-bound checks and visual inspection of all8 latest rendered pages passed. Tectonic0.17.0 official archive SHAverified; portable compiler and PyMuPDF1.28.2 renderer installed only under ignored external/. Engine emits a missing default Fontconfig configuration diagnostic, but every actual PDF font embeds and all pages render correctly. Build/PDF QA images are ignored in output/pdf/ and tmp/pdfs/. New text is UTF8-noBOM/LF; existing document endings preserved. Ruff and diff checks pass; no production changes or redundant learning tests.

Final test remains LIVE: session17151/runner45028 and all four actual workers35564,64780,45896,45888 confirmed alive and advancing. Last observations: first warm WP1180/10000, second warm WP1120/10000, first warm Point1110/10000, second warm Point1160/10000. validate_frozen() passed after manuscript work: production/model/development/comparator/protocol/runner/auditor identities unchanged. No final model selection or outcome conclusion. Still required: complete all4 final jobs, independent raw-record audit, isolated decision latency, final manuscript revision and a local reproducibility artifact package; submission remains a later decision. Do not restart or tune the running test. Goal active.

Post-final CPU latency benchmark IMPLEMENTED and PREPARED, not measured. scripts/benchmark_final_latency.py and scripts/audit_final_latency.py use the fixed previously audited development corpus runs/published_batch_validation.npz SHAe49004ce6795da3c23fbcb036c4d23712b6676a8d96cde1b43f6fca28d17d322. Only its12,861 random-trajectory states are eligible; selected search roots and hidden variants are excluded. Deterministic spaced samples:1,024 mixed,64 unguarded per four phases,32 guarded; group union1,287 cached states. Both frozen warm models plus assisted WP/Point run on CPU2 threads, exactly the final evaluator's agent.act path including candidate outer and internal endgame guard behavior. Two warmup and8 measured passes; agent order rotates, each position twice. Planned41,184 measured/10,296 warmup/5,148 validation calls; zero simulated matches and no additional final deals. Loading/state construction/transitions/correctness checks/process inventories excluded from call timings; load times recorded separately. Groups overlap and repeated cached-state statistics are descriptive, not independent samples or a deployment-policy distribution.

Protocol runs/final_latency/protocol.json SHAddec2df361cc2e206ab4759ac6f8a477fbf712003ff56dfac8f8887bfdc4083f, runnerSHAcedce89c464284689d7658c87b584de248c4dc580bcfb1d658c039c3a3d185ed, auditorSHA4cca3f05fc4c75796c0e687562fa3f1b4eb9659fecf0d9c61500a2569a44eeac. Actual live-study guard tested BEFORE preparation: run() raises before loading any agent or creating a timing directory. Then prepare()/validate() succeeded and independent sampling reconstruction agreed. Only protocol.json exists in the timing directory; no measured report/process/raw arrays. Readiness runs/final_latency_preflight.json SHA6408908455edec14142eefa15f0a41d4e8b0626f500fb4cc3ce09f8a43181d96. All10 current final runner/worker/shim PIDs detected by read-only inventory. Completed final study+independent audit and absence of project workloads are required before measurement. Workload checks also occur at measured-pass boundaries; unrelated OS activity remains uncontrolled. Never kill another process for a benchmark or silently overwrite/restart a timing run. Exceptions preserve failed status/artifacts.

12 focused latency tests and43 existing final-test tests all pass (55 total,2.45s); Ruff format/lint and diff checks pass. Checks cover independent group sampling and hand-calculated percentiles, corrupted timings/actions/guards/indices/repeat counts, live-study and concurrent-worker gates, and exact endgame wrapper arrangement. Auditor independently recomputes cached-state/action legality/stability, raw timing statistics, counts/ordered passes and timing-versus-wall accounting; verifies source/dependency/model identities. It does not rerun inference or independently measure the clock. No production/strength-runner/auditor change or new real inference benchmark during the live study. Commands AFTER all4 final jobs complete and audit passes: .venv/Scripts/python.exe -m scripts.benchmark_final_latency --run; .venv/Scripts/python.exe -m scripts.audit_final_latency. Prepared script/protocol identities must remain fixed through measurement/audit.

Final study remains LIVE: session17151/runner45028 and all4 actual workers35564,64780,45896,45888 confirmed alive. Last observed first warm WP2050/10000, second warm WP1970/10000, first warm Point1950/10000, second warm Point2020/10000. Do not infer strength from interim logs or change the frozen model/protocol. Final audit, actual isolated latency, final manuscript and local reproducibility package remain unfinished; goal active. The preceding goal turn made concrete progress with the verified8-page manuscript; this turn adds verified measurement infrastructure without interfering with final outcomes.

Local research artifact packaging IMPLEMENTED, not built. scripts/package_research_artifacts.py provides read-only --plan, gated --build and portable --verify; instructions in docs/ARTIFACTS.rst. Current plan371 files/~422MB existing uncompressed data, with11 required unfinished outputs absent (four final raw reports, final summary/audit, four timing outputs, final manuscript QA). Whitelist includes committed source, all cold/warm20/520 checkpoints plus initial, metrics/configs, selected models, bound/scanned historical JSON evidence, both comparator source/weights/licenses, timing corpus, separate100M checkpoint/budget evidence and final manuscript. It excludes Python/compiler caches, unselected intermediate/other exploratory checkpoints, Git internals and unrelated upstream assets. This is a selected local evidence archive, not a claim that every historical experiment is fully replayable from it, and no upload/publication occurred.

Build requires final and timing terminal status/full independent audits, refreshed final manuscript evidence, final PDF QA binding current source/evidence/PDF and a clean committed source snapshot. Actual live-final-study refusal verified before any archive/partial creation. Files stream into a temporary ZIP; source revision/readiness are rechecked, then bytes and final statistics verified before rename. Existing archives/partials are never overwritten. Default ignored output dist/koikoi_research_artifacts.zip; final archive receipt will be runs/research_artifact_bundle.json. Original absolute Windows protocol paths remain unchanged and map under recorded original_root during portable verification. Outside-root/absolute archive/traversal/case-collision paths are rejected; no extraction or model deserialization is performed. Full execution still needs recorded runtime/native compiler and clean complete pinned upstream Git checkout; its archived source subset has no Git metadata. No new license grant is invented.

Portable verifier uses Python+NumPy only. It checks exact members/byte counts/SHA256, original protocol bindings, production source identity, all9 cold/warm checkpoint identities, timing audit/report/raw bindings and separate100M checkpoint identity. It independently recalculates final raw pair means/5000-replicate seed99 bootstrap intervals and separate both-model WP/Point criteria. This is byte/statistical verification, not training/trajectory/inference/timing replay or an authenticity signature; retain the externally recorded archive digest. The same new arithmetic was exercised on all4 actual completed development reports and matched their previous independent audit to1e-12. No final outcomes were read and no additional matches played.

20 new artifact tests plus prior55 final/latency tests passed (75 total,1.89s); Ruff and diff checks pass. Synthetic ZIP fixtures cover hashes/sizes/exact members/case collisions and path mapping; known two-pair statistics and invalid seed/count/sign fixtures cover arithmetic; actual running-study gate blocks packaging. Full real archive build and final-data verification remain pending complete inputs, not claimed tested. Script SHAdabb66f3f7b9365340b7343b4b96491d0ea1a36bc275335d013e914563634e37. Readiness runs/research_artifacts_preflight.json SHAae42e6599351e810b913bf4c5263aa080bc5a7be93c63be3b6119bec4e925de1.

Final frozen identity validation passed again; no production/model/strength/latency protocol changes. Session17151/actual runner45028 and all4 actual workers35564,64780,45896,45888 confirmed alive. Last observed first warm WP3160/10000, second warm WP3030/10000, first warm Point3000/10000, second warm Point3110/10000. Do not restart on observation timeout or derive adoption from interim scores. Full final audit, actual isolated latency, revised/verified final manuscript and actual evidence archive remain unfinished; goal active.

Final frozen evaluation COMPLETED 2026-10-17. Original session17151 exited0; runner45028 and all4 actual workers35564/64780/45896/45888 absent. All4 fixed10000-pair jobs finished without extension, model reranking or retraining. Total80000 matches/9281432 decisions on10000 shared deal streams100000000-100009999. `.venv/Scripts/python.exe -m scripts.audit_final_published_test` exited0 and independently passed identities/accounting/raw-pair statistics; no trajectory/inference replay. Audit SHA891c765dec23edec33538174433d5105dabc443ca2da1a4f91d4273c965eade8; summary SHA49a2babbb4e84043ef7101eff683ec7071094261dc2914ee99cb1e51dd7c692f. Both-model primary WP criterion TRUE, secondary Point criterion FALSE. Canonical seed remains6710001_warm.

Final scores in percent (95% CI), candidate-minus-opponent points (95% CI): first WP51.4800 [50.9700,51.9750],2.4203 [2.122395,2.7133025]; second WP51.4450 [50.9449375,51.9525],2.0221 [1.715885,2.317115]; first Point50.4925 [49.9549375,51.0400625],-1.8918 [-2.2427125,-1.53859]; second Point51.2875 [50.7200,51.8525],-1.9603 [-2.3197125,-1.6088825]. First-seed Point interval includes50%; both Point expected-score intervals are negative. These findings do not support replicated wins against both opponents, all public implementations, other scoring profiles or humans. All original final deals are consumed and the original outcomes must remain unchanged.

Isolated latency COMPLETED after final audit and worker exit. `.venv/Scripts/python.exe -m scripts.benchmark_final_latency --run` session77792/actual29440 exited0; `.venv/Scripts/python.exe -m scripts.audit_final_latency` exited0. Measured41184 calls, warmup10296, validation5148 on1287 unique cached development states,0 simulated matches. Mixed mean/median/p95/p99 ms: seed6710001 1.584908984375/1.4789/2.035545/2.302272; seed6720001 1.5954383544921875/1.50395/2.033645/2.285609; WP1.9743964233398437/1.91075/2.7142/2.925063; Point1.92618193359375/1.8339/2.68788/2.901381. All actions legal/stable/guard-consistent and cached states unchanged. Exact deployment path including candidate value forward and wrappers; CPU2 threads. Timings are descriptive repeated-cache data, project workload absence sampled at pass boundaries, OS activity uncontrolled. Audit checks recorded timing/action arrays and statistics, not clock/inference remeasurement. Audit SHAe3b48a44568597f3cc528396c3a700407054137af4c540e2983a3486434f34cc; report SHA24ec5e4d5574e48a92d590f3f757ef701bfad5cd2daf7cf072dfa8a284f84531; raw SHAd0b4454a18135c3cbfddf6d4d385d2bf657983f57667253d739fbadaea17fc94. Original protocol/runner/auditor unchanged; never overwrite this run.

Manuscript revised with final primary success/secondary failure, negative Point means, exact workload/latency scope and archive instructions. Exporter now binds26 artifacts and produces final/latency tables from passed audits, checking final/protocol/report/raw timing digests. Final9-page PDF SHA506319f6faaa74bd669e80f0e173d1597eea6924e004893ec03640d6b4413177; manuscript SHA9cd591877efe88ffe1bf66bc3bbce606fedfc2af27d1c979d89f78cb1a6fa71f; evidence SHA104066c182779b3911fa511269b1dab0f7336e9d02a75e37f261c02c2b22fcb5; exporter SHA9b6c8d3c0cd46abd83998b344f49ec38dfc24daa7908e36f50c50fce7d871a6f. Final QA runs/manuscript_final_qa.json SHAa63cee06d64828856ab4cff265cb75b954bb78515ad4a6bef61bda82b2e3e0d5: passed/pending_final_results=false, all9 latest pages visually reviewed,14 fonts embedded, no TeX overfull/underfull/unresolved-reference/missing-glyph/page-bound defects; audited numbers and26input hashes/citation keys verified. After final layout edits, pages4/5/9 reinspected and other6 PNGs byte-identical to reviewed renders. Existing Fontconfig diagnostic remains nonfatal with actual embedded fonts verified. No production changes; Ruff checks and actual exporter/build/QA used instead of redundant learning tests. Local archive build and full original-plan completion audit still remain; author/venue/submission and external release are later decisions.

First real archive build from commit529dfa7 retained a failed partial ZIP (315286888 bytes, SHA9de95e11b7bfad390f305eb63baf407b22f807e4af59b38dab0f5be3fc3b9313). All372 members passed byte/hash checks, but production-source recomputation failed. Root cause: the verifier sorted raw strings while training provenance sorted pathlib components. koikoi/native/core.h precedes koikoi/native.py in original Path order, opposite raw string order. All29 production source payloads were checked equal to the live files; using originating Windows Path order exactly reproduces frozen SHA4b4360ed783c1d12cb5258e255f4b85f2e30dfa2e111fd8a078cf3439ce612cb. No model, training/evaluation artifact or production source changed.

Packaging verifier corrected to use original-root platform Path ordering (Windows case folding or POSIX case sensitivity). Two independent hand-ordered Windows/POSIX fixtures cover nested-versus-flat paths, case and excluded files;22 artifact tests pass, Ruff passes. Corrected verifier also passed the full retained actual ZIP:372files/428842562 uncompressed bytes, recomputed primaryWP=true/secondaryPoint=false. The old ZIP contains the earlier verifier source and is retained as dist/koikoi_research_artifacts_failed_path_order.zip, not promoted to the release artifact. Rebuild from a newly committed corrected snapshot is required; no silent overwrite or strength rerun.

Corrected actual archive build from ef554ba COMPLETED: session92587 exited0,372members/428846065 uncompressed bytes,315288189-byte ZIP. Archive SHA01b66f11f7bf0672859d12c39ccdb00ba9d859d48d8da53cd9acd4d29e5ad412. Full byte/member/source/model bindings and independently recomputed final raw statistics/conclusions pass. Original failed path-order ZIP remains separately retained. Final usage audit then found stale README statements claiming final results unproven and actor integration absent; README, COMPARATORS and DESIGN are updated to the measured canonical model, exact evaluation protocol/limitations and a small explicitly developmental CLI example. A new delivery archive must include this corrected documentation; preserve the ef554ba ZIP under a versioned name before rebuilding.

Full original-plan requirement audit: actual C++/CUDA kernels and BatchEnv GPU dispatch inspected; current tests cover reference/CPU/CUDA trajectories/observations/signs/noninterference. Final100M checkpoint loaded directly: CUDA/research12/2048env/captured PPO, counters updates69267/matches100001204/rounds1200025665/decisions18157928448 match the independent audit and the sum of all2048 environment episode counters. Its separate research12 parent evaluation contains10000pairs/20000matches,51.8125% [51.225,52.3825]. Pretrained two-model frozen primary WP superiority independently audited; secondary Point failure is not hidden. Development ablation, search negatives, timing scope, manuscript9-page QA and actual archive cover the research/deliverable requirements. Optional algorithm suggestions are not compulsory implementations or strength claims. Root/tests AGENTS, ignored plan/agents/external/runs/dist and intended Git remote verified. Venue/author/submission/external publication remain later decisions per DESIGN gate6. Final delivery synchronization is pending.

After the real archive-order failure was fixed, final integrated suite `.venv/Scripts/python.exe -m pytest -q --junitxml=runs/final_release_tests.xml` exited0:460passed,0skipped,0failures,0errors,39 expected upstream Transformer warnings,107.92s. JUnit XML SHA cf9925750fddcf9e5bd5da84282591d368e4de87b8d312bf0d6eeb0b11013a1f; counts independently parsed from XML. Includes real CUDA/upstream coverage and the2 new archive ordering fixtures. No production source change. Documentation/formatter/diff checks and actual source/model/PDF/evidence hashes remain consistent.

Final delivery archive COMPLETED from source snapshot f5cf955ac8f65c935a9f54c13bd0e8f25b6c39de; session82391 exited0. dist/koikoi_research_artifacts.zip is315291737bytes,372members/428853553 uncompressed bytes, SHA2ddce4d2033006e332f45d19ed7313f09eff9d11f7e9a625cbd4be4d7ff1f352. Actual complete verifier passed member bytes/sizes/hashes, original absolute-path mapping, production source identity with original Path order, all model bindings and independently recomputed final paired statistics:primaryWP=true/secondaryPoint=false. Receipt runs/research_artifact_bundle.json records the delivery. Independent PowerShell Get-FileHash matches; manifest revision checked, updated README present, archived PDF hash matches final QA. Retained earlier valid ef554ba bundle and original failed partial under distinct versioned names. No training, match replay, timing rerun, external upload or journal submission. Subsequent commits only record delivery metadata and do not rewrite this source snapshot.

Completion interpretation preserves original requirements and limitations: C++/CUDA GPU simulation, verified self-play learners,100001204 newly completed research12 matches, published-baseline improvement at both fixed model seeds on original paper8 rules, controlled ablation/negative research record, isolated latency, final manuscript, local reproducibility archive and repository practices are evidenced. Universal strongest play, human superiority, equilibrium and replicated RL-Point superiority are NOT established or claimed. The named minimum match-win comparator is RL-WP as prospectively fixed in DESIGN/COMPARATORS; the Point result remains separate and negative for expected points. Final460-test/no-skip suite and artifact/visual checks pass. Delivery metadata is being committed/pushed to the user-specified origin; remote tip and clean worktree must be verified before marking the active goal complete.


## Expert arbitration extension: implementation and first completed endpoint


The user requested stronger play, a clearer research contribution and a substantial manuscript revision. Initial dealer selection remains unchanged. The previous final evaluation and archive below are completed historical results; this extension has not yet established a new strength claim.

Implemented a public-observation controller over frozen adapted-win and original-point expert proposals, with duplicate actions collapsed. Full-context and phase-only controllers will be compared with ordinary continued PPO at both existing parent checkpoints, using identical additional decision budgets and fresh development deals. See [ARBITRATION.md](ARBITRATION.md). Source/parent-bound machine protocol is written before the study starts. The playable canonical model remains unchanged.

Validation:47 tests passed across arbitration, published training and published PPO, including CPU/CUDA trajectory equivalence, hidden-state invariance, duplicate-action zero gradients, frozen experts, critic warmup and exact resumed updates. Feasibility only: five updates/81,920 decisions at seed18990001,653 complete matches,7.35 seconds for updates; no strength evaluation. Ruff passes. Study command: `.venv/Scripts/python.exe -m scripts.run_arbitration_study` (fresh `runs/arbitration_study`). Manuscript restructuring and primary-source Related Work expansion are in progress.

The six-arm study is running in exec session75964, with one GPU trainer and up to three CPU evaluators. Production source SHA256 is `2b718198a4f7bc640f457c9fc12311dcb9a8689fba1ffcb09411234ba0920cdb`; do not modify production modules or the frozen runner while this study is active. Analysis is implemented separately in `scripts/analyze_arbitration_study.py` and requires all jobs complete. Four additional tests pass, including actual-parent initial greedy equivalence and corrupted-result rejection.

Manuscript revision:12 references, substantial Related Work, direct Abstract/Introduction, shorter interpretation and reproduction sections. Current11-page PDF SHA256 `4df2ef588e1b88375707d33a73c1b99df1ec686dea6137cc433d46cf8673539f`; current QA `runs/manuscript_arbitration_revision_qa.json` passes all11 visually reviewed pages,14 embedded fonts, citation/box/bounds checks and all26 retained evidence hashes. The new arbitration section explicitly says results are pending. The prior9-page PDF and its historical QA remain preserved in the earlier archive.

A second hypothesis has a tested training-target prototype: same-world paired complete-match outcomes at public expert disagreements. CPU/CUDA reference trajectories, signs, duplicate branches and grouping tests pass (3 tests). CPU feasibility generated64 roots/128 branch completions/15,018 branch decisions; it is not a policy-strength result. See [PAIRED_ROLLOUTS.md](PAIRED_ROLLOUTS.md). Direct outcome-difference regression and embedded deployment are now implemented in `scripts/train_paired_arbitration.py`. The8-test combined pipeline/branch/study suite passes, including a four-match evaluation/statistics round trip. The queued runner `scripts/run_paired_arbitration_study.py` freezes65,536 roots per parent,20 epochs, full/no_match ablation and24 fresh1,000-pair matchups. It waits for the earlier training jobs before GPU work, and for all earlier evaluations before CPU work. No new strength result is available yet.

Execution milestone: parent1_full finished520 updates/8,519,680 decisions/73,390 source matches. Its checkpoint SHA256 is `3eead4c639dc49f66247c51a31b8fb6e04441f32bf7b5ca01e70db8d258da46f`; independent training-record checks pass. Its fixed endpoint is under CPU evaluation while parent1_phase trains. No new endpoint win result has been selected or claimed.

The paired-regression study is queued in exec session28640 (`runs/paired_arbitration_study/process.json`: `waiting_for_previous_training`). Its bound protocol and runner are immutable while queued. Session75964 remains the preceding PPO study. Analysis entry points are `python -m scripts.analyze_arbitration_study` and `python -m scripts.analyze_paired_arbitration_study`; each requires its whole study complete. The paired analyzer checks source-match splits, targets/counters, frozen experts, CPU-recomputed validation MSE and all raw paired comparisons. Its tiny pipeline test passes. Keep both goals active: stronger verified play and the final evidence-based manuscript revision are still unfinished.

The first full-controller endpoint completed its two public comparisons on the fixed1,000 development pairs. Against WP:52.600% [50.975,54.275], parent52.500%, paired gain+0.100pp [-1.750,+1.950], point difference3.281 versus2.546 (gain0.735 [-0.328,1.783025]). Against Point:52.075% [50.450,53.700], parent51.575%, paired gain+0.500pp [-1.325,+2.275], point difference-0.090 versus-1.603 (gain1.513 [0.463,2.536025]). Raw records and statistics were checked using the analysis functions. These are completed endpoints for one parent, not an interim training checkpoint or a completed two-parent study. The two win-gain intervals include zero. No arm/model/deployment selection or budget change follows. The paired-regression protocol was already frozen and queued before these new-controller results were inspected. Machine-readable partial summary: `runs/arbitration_study/parent1_full/completed_public_comparisons.json`.

Rechecked upstream `rl_train.py` lines91-103 and related simulation/regression scripts: RL-WP's code uses a round-ending continuation-probability table, while the paired-target candidate uses two actual complete-match outcomes. This source distinction was already documented in the earlier round-boundary research record and should be made explicit in the revised manuscript. It is not a claim to reproduce the original released weights' training history.


## 2026-10-17: first-parent arbitration diagnostics and fixed hybrid controls

On2,048 evenly spaced states from the retained random-play corpus, the completed parent1_full controller selects Point in188 of523 disagreements (35.946%). Holding expert proposals, legality and other inputs fixed while changing only explicit scores/round changes choices in8 disagreements. Phase-only selects its frozen parent expert at every phase. Report `runs/arbitration_diagnostic_parent1/report.json`, NPZ outputs and `output/figures/arbitration_parent1_context.{png,svg}` retain group counts and the controller-only intervention grid. These states and interventions do not estimate match strength or a causal source of win improvement. Diagnostic tests:3 passed; Ruff and diff checks pass.

A raw-pair discrepancy in the phase control was traced at seed21000447, candidate seat1, decision82, round4, koi phase. Identical weights, features, masks and states produce stop/continue scores6.9657087326049805/6.965709686279297 when actor parameters are frozen, versus an exact6.9657087326049805 tie with requires-grad enabled. A2x2 intervention on both in-memory actors reproduces the flag effect, including repeated identical outputs; dividing by temperature does not explain it alone. Guards both decline to override. The one changed pair leaves wins unchanged and shifts mean points by0.002 on the1,000-pair Point comparison. Record `runs/arbitration_diagnostic_parent1/parent_inference_path_difference.json`; reproduce with `python -m scripts.diagnose_parent_inference_tie`. Checkpoints/workers are untouched. Initial greedy selection exactly selects the frozen parent proposal; it is not a universal bitwise equivalence guarantee to the older parent wrapper. Full and phase share the frozen path.

Added `scripts.run_static_expert_controls` and `scripts.analyze_static_expert_controls`: Point cards+parent koi and parent cards+Point koi for both parent seeds, versus both published comparators. All eight1,000-pair matchups use the paired-regression22,000,000--22,000,999 development streams, intentionally reused for paired contrasts, and wait for that study to finish. Both hybrids are fixed before paired-regression outcomes. No extra training or search. The current production hash, runner, parent/comparator weights and predecessor protocol are bound at launch. Actual CLI smoke runs cover both routing directions, four matches each at19830001--19830002; recomputation validates both reports and rejects altered points, seeds and decision totals (`runs/static_expert_controls_smoke/audit.json`). Smoke outcomes are not strength evidence.

The manuscript now distinguishes original RL-WP round-ending continuation-table targets from terminal-match self-play returns, and records the observed inference-path near-tie in the reproduction appendix. Latest11-page PDF passed changed-page visual inspection (2,8,9,10,11; other six render hashes unchanged),14 embedded fonts, box/reference/bounds checks and26 retained evidence hashes. QA `runs/manuscript_arbitration_diagnostic_qa.json`; PDF SHA256 c39c118aae3da31780f5495439319c881e03e35a647ec977acc4989af5daa470. This remains a provisional research extension; stronger verified play and its completed paper are outstanding.

Launch receipt: fixed controls started successfully in exec session48159 at source commit7142ffa; process state waiting_for_paired_study and all bound identities validate. First-parent continued PPO training has completed; its evaluations are running while parent2_full trains. Earlier study75964 and queued paired study28640 remain live. No strength endpoint selected; the active goal remains unfinished.


## 2026-10-17: held-out target diagnostics and direct Koi-Koi antecedents

Added `scripts.diagnose_paired_targets`: validation-only source matches, prediction MSE against zero and fixed training-mean baselines, fixed calibration bins, actual argmax expert choices including native action-index ties, and observed one-action win-score gains against both fixed experts. Source-match bootstrap uses5,000 replicates at seed19730001 and computes root-weighted ratios after cluster resampling. Correlated roots are not independent samples. Phase summaries are retained; no model/threshold/epoch is selected. These are frozen-continuation one-action diagnostics, not repeated-selector match performance. Four analytical tests and expanded actual32-root collection/fit/evaluation/diagnostic integration pass5 tests in13.97s, with2 expected upstream warnings; Ruff passes. Active production and bound scripts unchanged.

Primary-source literature check found a particularly relevant antecedent: Nagamine and Suzuki2021, IPSJ83rd National Convention2:317--318, construct Monte Carlo fast-finishing/blocking strategies and explicitly propose switching scoring/fast-finishing strategies across a match as future work. Their simplified game disables koi-koi continuation. Original PDF retrieved from https://ipsj.ixsq.nii.ac.jp/record/214960/files/IPSJ-Z83-6P-07.pdf and read locally after web-tool throttling. Sato and Ikeda2017, Game Programming Workshop:64--71, compare neural policy gradients and NFQ using268 features. Primary repository https://dspace.jaist.ac.jp/dspace/handle/10119/16089. The manuscript now places both before Guan2023; it does not claim inventing objective switching. Rollout policy iteration (Dimitrakakis/Lagoudakis2008) and common-random-number planning (Yadav et al.2026) are also cited directly. Their verified arXiv entries are0805.2027 and2605.04732.

Added the implemented paired complete-match regression method, grouped validation and fixed-hybrid controls to the manuscript, with results still pending. Latest12-page PDF SHA256 `5fd4e21fc6e32e8c80a4590b2f25f012bab46013f91fc89e201e36ae41016de6`; QA `runs/manuscript_paired_revision_qa.json`. All12 rendered pages inspected,15 embedded fonts,16 citation keys resolved, no box/glyph/bounds defects,26 historical evidence input hashes retained. Previous goal turn made progress; current turn adds tested diagnostic capability and substantively improves research positioning. All three existing experiment handles75964/28640/48159 were re-polled live; no restart or endpoint selection.


### Completed first-parent ordinary-PPO control

The continued actor passed checkpoint/config/counter accounting at8,519,680 additional decisions. All three completed1,000-pair outcomes were independently recomputed, including paired contrasts on shared21,000,000--21,000,999 streams. Record `runs/arbitration_study/parent1_ppo/completed_comparisons.json` binds the fixed protocol. This is the complete first-parent comparison, not a selected intermediate checkpoint; second-parent replication remains pending.

| Opponent | Continued PPO win score,95% CI | Point difference,95% CI | Full gate minus PPO win difference,pp,95% CI | Full gate minus PPO point difference,95% CI |
| --- | --- | --- | --- | --- |
| RL-WP | 53.375% [51.800,54.925] | 3.565 [2.592,4.556] | -0.775 [-2.725,1.300] | -0.284 [-1.459025,0.91315] |
| RL-Point | 51.425% [49.650,53.225] | -1.154 [-2.297025,0.017025] | 0.650 [-1.375,2.575] | 1.064 [-0.12305,2.25105] |
| Own parent | 50.850% [49.400,52.375] | 0.711 [-0.145,1.56805] | not applicable | not applicable |

Continued PPO minus unchanged parent: against WP,+0.875pp [-0.875,2.600],+1.019points [0.056925,1.991075]; against Point,-0.150pp [-1.925,1.625],+0.449points [-0.551025,1.448025]. Thus the first-parent full selector has no established advantage over ordinary additional PPO in either win score or points. Retain the complete uncertainty and all endpoints; do not infer superiority from its better Point point estimate. The queued paired-regression study and static controls remain fixed and live.


## 2026-10-17: second-parent full selector trained; independent audits queued

Second-parent full training completed520 updates /8,519,680 decisions /568,871 round endings /73,025 matches in765.127435s. Checkpoint SHA256 `182567a3f68f514d041ae7b0cc1861a8b88bf7fcce342a81d94897eccf624cdb`. Independent `training()` audit verifies initial/final frozen experts, source/config, history and counts. Evaluation is running; no strength result yet. Parent2 phase-only training follows.

The same2,048 cached public states give502 expert disagreements; the second full selector chooses Point in120 (23.904%), and25 disagreements respond to the fixed explicit score/round grid. The first model selected188/523 (35.946%), with8 grid-sensitive disagreements. On429 states where both models have exactly the same two distinct proposals, their chosen actions differ184 times. Report `runs/arbitration_diagnostic_parent2_full/report.json` and cross-model `comparison.json` bind cached data and endpoints. This shows considerable decision variation across the two trained controllers; neither observed feature sensitivity nor decision agreement establishes playing strength. Parent and controller training seeds both vary. It reinforces the need to retain both model outcomes and the same-architecture no_match control.

Started `python -m scripts.watch_extension_audits`, exec session98461, `runs/extension_audits`, commit e7dcfa4. Its first live state is waiting/arbitration. It runs all three existing independent analyzers after the respective study's complete status, then both paired-target held-out diagnostics. Analysis source hashes are recorded and checked before execution; failures are recorded and surfaced. It does not change training, opponent/deal schedules, model selection or deployment. Existing live experiment handles75964,28640,48159 were revalidated before launch. Ruff passes. Do not concurrently rerun the watcher-owned analyses into their target files.


## 2026-10-17: reusable paired-change figure

Added `scripts.plot_extension_results` for the completed study outcomes, plotting win-score changes (percentage points) and mean point-difference changes relative to each unchanged parent against both public opponents. Raw rows and stored summaries are independently recomputed before plotting. Full two-parent plots require a passed whole-study audit and exact agreement with its reports; partial plotting requires explicit `--parents 1` and labels replication pending. Source/result/PNG/SVG/PDF hashes are retained in a JSON receipt. Axes are symmetric around zero and shared across opponents within each metric. No independent sample claim is made for shared deals.

Executed the first-parent-only preview against actual completed endpoints: `output/figures/arbitration_parent1_changes.{png,svg,pdf,json}`. All layout/labels were visually inspected, and Ruff/diff checks pass. This prepares concise result presentation without replacing uncertainty with long prose; full-study plots and paper integration await outcomes. Main study75964, queued paired28640, static48159 and audit98461 were all re-polled live at turn start. Parent2 full evaluations reached600/1,000 pairs and phase training340/520 updates at the last progress checkpoint. No new strength conclusion or model selection.


## 2026-10-17: complete second full-controller outcomes and score-distribution prototype

Second full controller completed all three1,000-pair matchups; raw records and paired intervals were recomputed in `runs/arbitration_study/parent2_full/completed_public_comparisons.json`. Versus WP:51.800% [50.275,53.400], points+2.961 [1.991975,3.91405]; versus Point:51.575% [49.800,53.350], points-1.545 [-2.71005,-0.38185]; versus own parent:50.475% [49.150,51.875], points+0.329 [-0.417025,1.139]. Relative to unchanged parent, WP win gain-0.050pp [-1.475,1.425], point gain+0.881 [0.055725,1.74505]; Point win gain-0.550pp [-2.125,1.025], point gain+0.238 [-0.62205,1.122]. Neither parent establishes win gains against either public opponent. First-parent Point point gain does not clearly replicate. No deployment promotion.

Second phase control completed8,519,680 decisions /569,740 rounds /73,036 matches in837.539240s, checkpointSHA `a3fbcc33e1b1168867a373d53700015c0ab1149786d39410df9ec7b5514a513b`. Independent training audit passes. Parent-minus-Point logits in phases0--3 are[0.8084475994,1.4040405750,1.3095242977,1.8956142664], so greedy selection always chooses its parent proposal. `runs/arbitration_diagnostic_parent2` now contains both full and phase endpoints on the same2,048 cached states; the phase control chooses Point zero times across502 disagreements and its context grid is invariant. Ordinary parent2 PPO training and phase evaluations remain active.

Prepared an **unlaunched** score-distribution prototype in `scripts/train_score_distribution.py`, documented in DESIGN and PAIRED_ROLLOUTS. It reuses paired final margins to supervise a shared public/action scorer. Active paper8 scores sum to60 and both remain positive, hence current lead is even in[-58,58]. Clipping even remaining-margin increments to65 classes spanning[-64,64] preserves every win/draw/loss outcome, including extreme tails. A categorical head minimizes cross entropy and decodes expected signed wins at the current lead; scalar controls regress unbounded mean increments or directly fit wins with draw credit0.5. Same seeds give identical initial shared-encoder hashes and independent identical minibatch-order seeds; output head sizes differ and are reported. No Bellman update or privileged model input. This follows the standard distributional-value perspective (Bellemare/Dabney/Munos2017, https://proceedings.mlr.press/v70/bellemare17a.html), with a specific terminal-margin/threshold formulation for the present selection problem; no novelty or strength result is presumed.

Validation:7 passed,0 skips,1 expected upstream warning,11.74s. Cases cover all active lead values and extreme tails, a safe-versus-risky decision reversal under fixed return distributions, shared action-order/duplicate consistency for all three heads, real32-root generation plus two-epoch CPU fits/reloads and hidden-state invariance, equal encoder initialization/shuffle seeds, and CPU/CUDA loss/gradient agreement. A final metadata-only edit records the fitting device explicitly; Ruff/diff checks pass. The small fitting runs are correctness checks, not strength evidence. Dataset generation work is marked reused; fitting adds zero simulation decisions. No large-scale fit, evaluator runner or new study was launched. Existing paired/static protocols still validate against unchanged production SHA256 `2b718198a4f7bc640f457c9fc12311dcb9a8689fba1ffcb09411234ba0920cdb`. Decide whether to run the follow-up only after the current paired-regression and static-control evidence is available.


## 2026-10-17: second phase-control evaluation and traced numerical difference

All second-parent phase-control evaluations completed. Against WP, win score51.850% [50.300,53.400] and points2.105 [1.193,3.012025]; every win pair equals the parent, but one pair changes points (parent mean2.080). Against Point, every pair's wins and points equal the parent:52.125% [50.400,53.825625], points-1.783 [-2.89305,-0.6989]. Against own parent, every pair gives0.5 win score and zero points. Report `runs/arbitration_study/parent2_phase/completed_parent_equivalence.json` records finite-pair equality, not universal action identity.

The differing WP pair is seed21000109, seat0, where the phase wrapper finishes+72 versus the original parent's+22; seat1 remains-22. First divergence is decision74, round index4, discard phase, public scores53:7. Frozen actor gives exactly tied raw scores10.20788860321045 for actions13 and19, choosing13; requires-grad=true yields10.207887649536133 and10.20788860321045, choosing19. Both actor instances reproduce both paths when only their gradient flags are changed; repeated calls are exact, temperature scaling preserves this distinction, and all weights/public inputs/masks/states match. Neither guard forces an action. This single changed pair contributes+0.025 mean points and zero win gain. Diagnostic `runs/arbitration_diagnostic_parent2/parent_inference_path_difference.json`.

`scripts.diagnose_parent_inference_tie` now accepts explicit controller, parent, opponent and fresh output arguments. Without --output it prints without overwriting old records. Reproduction:

```powershell
.venv/Scripts/python.exe -m scripts.diagnose_parent_inference_tie --seed 21000109 --seat 0 --controller runs/arbitration_study/parent2_phase/final.pt --parent runs/cold_ppo_screen/seed6720001_warm/update_00000520.pt --opponent rl-wp-assisted
```

The generalized command was run successfully with a fresh explicit output; Ruff passes. Bound study/analysis scripts and production remain untouched. All four live handles75964/28640/48159/98461 were revalidated. Remaining ordinary PPO training is still running; no restart, selection or deployment change.


## 2026-10-17: all arbitration training finished; paired generation started

Parent2 ordinary PPO completed its fixed520 updates /8,519,680 decisions /568,049 round endings /73,045 matches in1701.6233375s. CheckpointSHA `e210cae056ee71254e2519bb1c602a5427a6349dc85a2738ed9e37e8ddbee03d`; independent checkpoint/config/optimizer/counter/metric audit passed (`runs/arbitration_study/parent2_ppo/training_audit_completed.json`). All six training arms are now complete and checked. Last three ordinary-PPO evaluations are running; no strength conclusion before their completion. The audit watcher retains ownership of the whole-study final analysis.

Queued paired study28640 transitioned from waiting_for_previous_training to training without a restart. Parent1 generation is live: first2,048 disagreement roots produced4,096 branch endpoints /318,917 branch decisions /23,156 branch round endings in9.201281s; source trajectory work9,728 decisions /7 completed matches /596 round endings. At16,384 roots,32,768 branches /1,961,151 branch decisions /147,125 round endings, source77,440 decisions /616 matches /5,176 rounds, elapsed63.118635s. These are rollout/data-generation work counts, not strength estimates or independent complete-match sample counts. All remaining fits, matchups and static controls retain their frozen protocols. Main and paired handles were re-polled live at the transition.


## 2026-10-17: arbitration study audited; paired fits and held-out diagnostics complete

Original arbitration session75964 exited0. All22 fixed matchups and six training endpoints pass `runs/arbitration_study/analysis.json`, SHA256 `66f125fe8a2b9245e6cf91e20cbd6121516d76c1f7186846cd1a694ea42aabb2`. Training totals51,118,080 decisions /3,425,571 round endings /439,406 completed matches.22,000 evaluated duplicate pairs reuse1,000 unique deal streams;44,000 matches are not44,000 independent deals. None of full selection, phase-only selection or continued PPO establishes replicated win gain over the parent against both published opponents. Parent2 ordinary PPO scores52.150% [50.525,53.725] against WP,51.275% [49.500,53.050] against Point,50.250% [48.850,51.650] against its parent. Paired win differences over unchanged parent are+0.300pp [-1.425,2.075] against WP and-0.850pp [-2.575,0.925] against Point; point differences are+0.650 [-0.348,1.704075] and-0.460 [-1.482,0.57105]. No checkpoint promotion. Full two-parent figure `output/figures/arbitration_changes.{png,svg,pdf,json}` uses independently recomputed raw paired outcomes, was visually inspected, and retains all input/output hashes.

Paired study28640 completed both datasets and four fixed fits; it is now evaluating24 fixed matchups. Separate preview directories `runs/paired_arbitration_preview_parent1` and `runs/paired_arbitration_preview_parent2` contain dataset/fit audits and held-out diagnostics. These do not overwrite watcher-owned final analyses. Parent1 datasetSHA `f7ac8557f77af28f6569ef6f4c655b8f8ef15e024c9a73dac592aff8d4d68415`; parent2 `7e48fac140a044e5187244f858056e3bbbadc9d3f7e1bb18117f6844cfa91401`. Each contains65,536 disagreement roots /131,072 completed branch continuations. Parent1 source311,552 decisions /2,654 completed matches /21,079 round endings; branches7,669,301 decisions /578,401 round endings. Parent2 source311,680 decisions /2,612 completed matches /20,800 round endings; branches7,736,826 decisions /573,858 round endings. Source matches may be partly observed; branch endings are not independent source matches.

Held-out groups: parent1 has13,953 roots in590 source matches; parent2 has12,939 roots in547. Full/no_match MSEs are0.31734357/0.31733093 versus zero0.31577439 for parent1 and0.31754295/0.31755431 versus zero0.31617590 for parent2. Full selected-branch win-score gains over always-parent are+0.232925pp [-0.093509,0.562255] and+0.459850pp [0.089560,0.843572]; over always-Point they are-0.139755pp [-0.468753,0.177674] and+0.208671pp [-0.110850,0.528091]. No_match gains over parent are+0.254426pp [-0.063954,0.574351] and+0.448257pp [0.083145,0.830433]. Intervals resample whole source matches and retain root weighting. Similar full/no_match results do not establish explicit-context benefit. These estimate one changed action followed by frozen continuation, not full-match repeated deployment. No endpoint/epoch/threshold selection.

Descriptive target inspection `runs/paired_target_information.json` verifies categorical tail compression preserves every branch outcome on both actual datasets, without new fitting or simulation. Parent1 has6,314 different-win pairs versus27,315 different-margin pairs; parent2 has6,324 versus27,317 (each out of65,536). Same-win/different-margin counts are21,001 and20,993. All65 categorical labels occur;1,571 and1,365 branch labels exceed the support and are tail-compressed exactly with respect to wins. The training-only subsets show the same sparsity pattern. This is a rationale for considering score-distribution supervision, not evidence that it generalizes or improves play. The documented decision to await current paired/static results before launching that follow-up remains unchanged. Static session48159 and watcher98461 remain live; watcher has completed arbitration audit and now waits for paired evaluation.


## 2026-10-17: completed arbitration results incorporated in manuscript

Replaced pending arbitration statements with the audited six-arm/22-matchup result and a full two-parent paired-change figure. Introduction, interpretation and conclusion now state the observed lack of replicated win improvement; the paired-supervision study remains explicitly under evaluation. Added generated/arbitration_evidence.json and its tracked figure separately from the unchanged historical evidence ledger. No positive strength or novelty claim is inferred from these negative outcomes.

Built13-page PDF SHA256 `16ac69ab47349fe83e2d809db921e064bc18151cc8e7a3a29244ac730ede90db` with Tectonic. All page content visually checked; latest pages10--13 rechecked after adjusting float layout, and page1 is pixel-identical to the preceding inspected version. All18 fonts embedded,16 citation keys resolved, no TeX box/glyph warnings or out-of-bounds content. All26 historical and34 extension input hashes verified. QA: runs/manuscript_arbitration_results_qa.json. Training/evaluation production files unchanged; paired28640, static48159 and watcher98461 re-polled live. This is an updated working manuscript, not completion of the stronger-model objective.


## 2026-10-17: pretrained-feature reconstruction constraint verified

Investigated whether frozen expert representations could be extracted exactly from existing paired datasets to strengthen low-signal supervised learning. Native ordinary560 observations use acting-player-relative dealer status, whereas pinned published300x50 rows50/51 encode absolute dealer identity. Added `scripts.diagnose_public_feature_information`; it exchanges player labels, corresponding ownership/captures/scores/koi counts and initial-hand deck segments, leaving public relative content unchanged. Involution, exact ordinary-observation/mask/guard equality and restriction of published differences to rows50/51 are asserted on2,048 fixed cached random-play states.

Actual CPU2/batch128 diagnostics completed for both parent/Point expert banks. Parent1 action changes47/2,048; parent2 changes40; Point changes35. Both expert proposal IDs remain unchanged on1,967/1,975 states for parent1/2 respectively. Among those identical562-input pairs, parent scores change above1e-5 on1,967/1,974 states, and Point scores change on all1,967/1,975. Maximum legal-score changes:0.5984306 and0.3946834 for the parents,1.6960869 for Point. Reports `runs/public_feature_information/parent1.json` and `parent2.json` bind script/corpus/checkpoint identities and sampling indices. An earlier first-parent diagnostic output is retained separately; the final script additionally swaps initial-hand deck segments and has clearer metric names. Final scriptSHA `7d8ba7ea6f3063b3399a5041496b28a3200d15dfab37e4c9e4960350fe268746`.

This provides counterexamples to exact reconstruction of released-model features/scores from the saved inputs alone. It does not establish a strength defect, causal game benefit from seat invariance, or a rule change. No model trained, no new match simulated, no production/bound script modified. Ruff/diff checks pass. Any future exact pretrained-feature caching must retain the public seat bit, or explicitly evaluate a different representation. Current score-distribution prototype needs no reconstruction. Existing paired/static experiments continue; no additional strength study was launched.


## 2026-10-17: exact public expert-feature replay helper verified

Added unlaunched `scripts.cache_paired_expert_features`. Instead of inferring missing absolute public seat identity from stored relative observations, it reconstructs the original source stream using its seed/device/frozen experts/dedicated behavior generator. Every root's562 public input and source-match identifier must match the existing dataset exactly, and aggregate source work must agree. No branch outcomes are regenerated. Exports are public dealer identity, two expert encoder vectors per proposal [N,2,2,256] and raw scores [N,2,2]; no simulator state/private labels. Newly repeated source work is separate from reused branch work. Source/script/dataset/device/batch identities are retained; score differences across batch shapes remain possible.

Four tests passed,0 skipped,5 expected upstream warnings,14.13s: actual32-root datasets with complete branch targets on each CPU/CUDA followed by exact root replay/cache reload; malformed public root and output-overwrite rejection; both expert score paths exactly match original heads at the same batch shape across all four phases; exported latent vectors reconstruct their corresponding linear-head scores. The tests create64 roots total for validation, not a new strength study. Ruff/diff checks pass. No full cache extraction or new feature-based learning experiment launched; active paired/static protocols and production unchanged.


## 2026-10-17: first paired-regression full selector matchups complete

All three first-parent full-controller evaluations completed on1,000 duplicate pairs22,000,000--22,000,999. Raw pair statistics, counts, checkpoint/raw hashes and bootstrap intervals independently recomputed with outcome(). WP:51.050% [49.375,52.700], point difference+2.886 [1.86795,3.891],230,392 decisions. Point:49.650% [48.100,51.300], points-1.825 [-2.83305,-0.803975],232,196 decisions. Own parent:49.450% [47.850,51.075], points+0.594 [-0.405025,1.596],230,299 decisions. No established match-win superiority on these endpoints. No parent-relative public-opponent gain inferred before unchanged-parent evaluations on these same deals complete. Remaining controls and second-parent replication continue without changing budgets or settings. Record runs/paired_arbitration_study/parent1_full/completed_matchups.json; original session28640 re-polled live. No checkpoint promotion or new strength-study launch.


## 2026-10-17: full public feature caches and controlled score-fitting preparation

Public-feature replay session33507 completed both datasets. Each65,536-root stream matched stored public562 inputs/proposals/source groups exactly. Parent1 cacheSHA4136ea7c465f2775bd296f36a4a3d8eb84349ad474fbf8a6323fd342667d8ba4; parent2 cacheSHA69bc44a17be183fe5db3bf2dfd55481e59a841962f8607631c1141745ca9fff6. Each269,552,737 bytes. Replay adds311,552/311,680 source decisions,2,654/2,612 source match endings,21,079/20,800 round endings, and zero branch decisions; the original branch targets are reused. GPU timings33.7604522/34.7620271 seconds. `runs/paired_public_features/audit.json` passes source/dataset/cache identities, finite shapes, expected metadata, and every expert head score reconstructed from its stored latent vector. CPU/GPU head reconstruction max errors3.43323e-5/2.67029e-5 satisfy rtol1e-5+atol1e-5. No second full-trajectory replay or strength claim.

Added FeatureScoreSelector/FeatureScoreDistributionAgent: a shared546-input action scorer over two256-dimensional pretrained public expert embeddings, two Q scores scaled by1/16 and32 public context features. It supports categorical remaining-margin, direct-win and mean-margin losses, using the same fixed20-epoch fitting pipeline as the raw-public scorer. One-pass deployment shares each frozen encoder computation between proposals and representations. Feature cache/checkpoint identities and replay accounting are explicit. No private state enters the learned model. Added held-out actual-choice win/point gains clustered by source match; only win-valued heads report win-delta MSE. Pure analytic fixture independently checks win/draw scaling and point-margin units.

Validation: initial combined14-test run caught an inherited forward slice that included appended features as proposal IDs; fixed it to columns560:562. Rerun14 passed,0 skipped,6 expected upstream warnings,23.93s. After adding diagnostics, expanded feature/replay suite8 passed,0 skipped,5 expected warnings,15.92s; raw seven tests had passed and their code did not subsequently change. Covers CPU/CUDA exact same-batch expert-head/cache agreement in every phase, actual hidden-world invariance, proposal-order/duplicate consistency for all heads, real small-data fits/reloads, malformed cache/root rejection, identical per-representation initial encoders/shuffle seeds, and selected-branch diagnostic arithmetic. Ruff/diff checks pass.

Prepared fixed12-case fitting-only screen in scripts.run_score_fit_screen. Actual plan construction checked12 unique cases, seeds19410001/19420001,20 epochs, CUDA and15 bound input/script artifacts; existing-directory rejection checked. It crosses raw/features x categorical/win/mean at both parents, retains every final epoch, and runs held-out development diagnostics without model selection or new matches. Sequence is revised explicitly: fixed supervised fitting can use the idle GPU now because both weak held-out regression diagnostics and first-parent full-match outcomes are available. Existing paired/static CPU protocols remain untouched; any new match study still waits for their completion. This supersedes the earlier scheduling decision to wait before full-scale fitting, not the current studies' frozen budgets or evidence rules.


## 2026-10-18: complete12-case score-supervision screen; no diagnostic qualification

Ran python -m scripts.run_score_fit_screen in session66203 after committing sourceee19fe0. It exited0 with12 complete final20-epoch fits and held-out diagnostics. PlanSHA174a6091df7f76c15040d3eff38c632433a8e48305df418aaa381ff3bae1a089; auditSHA35a5b2012b7ab15ce4bfaf10f80852cceac1e13fa5b6170deede3653ab72a30e. All15 bound input/script hashes were checked after completion. Independent stored-artifact checks verify embedded experts/parents, dataset/cache identities, original shared encoder initialization and identical shuffle seeds within each parent/representation, complete epoch/optimizer counts, reused-work accounting, and CPU recomputation of final validation losses within1e-5 of GPU. Total recorded fitting31.6151751s; diagnostic/load/hash time is separate. Fitting adds zero simulation. Each dataset/replay is shared across heads, not independent new training trajectories. Audit does not replay training.

No configuration establishes one-action win gain over both always-parent and always-Point at both parent datasets. Feature categorical selects Point78.112%/77.309%; feature mean91.099%/90.973%; feature win9.855%/9.939%. Feature categorical gains over always-parent are+0.379847pp [-0.014350,0.762122] and+0.266636pp [-0.121710,0.665695], while gains over always-Point are+0.007167pp [-0.229512,0.229582] and+0.015457pp [-0.204148,0.244151]. Raw categorical over-parent gains+0.301011pp [0.007300,0.592244] and+0.135250pp [-0.171929,0.448925] do not replicate. Full twelve-case results are retained in runs/score_fit_screen/summary.json and fit_audit.json. These are held-out source-match development diagnostics, not repeated-deployment strength or a selected-method confirmation. No candidate promoted.

The observed expert preferences and weak paired win-delta errors motivate a next hypothesis: train the difference between candidates directly using the now-available frozen features, so shared outcome variation is removed from the learning target. The original paired regression used raw public inputs; this is a controlled representation/target question, not a claim that another loss will necessarily improve strength. Existing paired/static full-match studies continue, and no new score-selector matchup has launched.

First-parent no_match regression also completed all three1,000-pair matchups. WP51.275% [49.600,52.925], mean points+3.073; Point49.175% [47.625,50.800], points-1.858; own parent49.375% [47.749375,50.975], points+0.571. Full-minus-no_match win changes are-0.225pp [-0.950,0.475] against WP,+0.475pp [-0.050,1.025] against Point,+0.075pp [-0.525,0.700] against own parent. None establishes explicit-context benefit. Raw statistics/identities and paired bootstrap recomputed in runs/paired_arbitration_study/parent1_no_match/completed_matchups.json. Remaining unchanged-parent/PPO controls and second-parent replication are still pending.


## 2026-10-18: paired-difference objectives implemented and verified

Added paired_win and paired_mean to the shared raw/feature scorer. MSE supervises f_Point-f_parent using signed final-win difference or final-margin difference/64; no absolute probability interpretation is made for these unbounded scalar outputs. This retains the same input dimensions, scalar head, seeds, initial encoder, independent minibatch-order seed, optimizer and20 epochs as the completed marginal-head comparison. Original12-case sourceee19fe0 and audit remain preserved. Only completed score-screen scripts changed; production and active paired/static/watcher-bound files remain untouched.

Tests17 passed,0 skipped,5 expected upstream warnings,25.08s. Extended actual small-data fit/reload/diagnostic loops to both new objectives; CPU/CUDA loss/gradient checks include them. Two independent fixtures check signed-win/point target scaling, analytic loss gradients and invariance to per-root common score offsets. Original raw/feature inference, hidden-world and head reconstruction checks remain passing. Ruff/diff checks pass. Actual plan construction verifies8 unique fixed follow-up cases with seeds19410001/19420001 and20 epochs; no full-match claim or deployment promotion.


## 2026-10-18: paired-difference screen complete; fourfold data follow-up prepared

Session77750 exited0 after all8 fixed20-epoch fits and held-out diagnostics. Audit runs/paired_difference_fit_screen/fit_audit.json SHA856a5e6c667285d65a149205f291c5e0828fa026a1edf45f49c559e2d572e957 passes all bound inputs, embedded experts/parents, counters, epochs, dataset/cache identities, exact initial encoder and shuffle agreement with preceding marginal heads, and CPU final-loss recomputation within1e-5 of GPU. Recorded fitting time22.9450649s; no new simulation. No configuration has a positive one-action win-gain interval against both fixed expert choices at both parents. Raw paired-win over-parent gains+0.387014pp [0.088168,0.691871] and+0.115929pp [-0.235190,0.478856]; feature paired-win+0.193507pp [-0.134581,0.516030] and+0.081150pp [-0.237892,0.406510]. Full results retained. No deployment promotion or new match evaluation. The win comparison bundles output-link/loss changes; only the point comparison retains the same unbounded scalar link while changing marginal to difference MSE.

Prepared scripts.run_scaled_paired_targets to test data coverage:262,144 disagreement roots per parent, exactly4x the prior count, fresh source seeds19610001/19620001. Same parents, Point weights,90/10 behavior,128 environments,256-root batches and frozen continuation. Original datasets remain untouched. The runner separately counts source/branch/feature-replay work, verifies dataset accounting and Point identity, and caches public features only after exact source replay. It performs no fitting or strength evaluation. Source/parent/Point/10 dependency identities are fixed in its protocol. Actual plan construction and existing-directory rejection passed; Ruff/diff checks pass. Core collection/replay helpers retain the previously tested CPU/CUDA paths unchanged. Disk free227.3GiB before launch; expected dataset/cache output is a fewGiB. Current paired/static CPU studies continue without changed settings.


Scaled collection launched after source commit7670d29 in session5720, actualPID36744/wrapper37060. The handle was re-polled live. Parent1 reached12,288 roots /24,576 branch endpoints /1,499,462 branch decisions /111,505 branch round endings in44.832623s; source57,600 decisions /422 completed matches /3,831 round endings. This is verified ongoing computation, not a completed dataset or strength result. Current paired evaluation28640, static queue48159 and audit watcher98461 retain their original roles. Do not restart into runs/scaled_paired_targets or edit its bound collection/cache/audit dependencies while it runs.


## 2026-10-18: scaled collector resource failure corrected; preference objective verified

Original scaled session5720 exited1 with WinError206 in os.add_dll_directory during construction of a new CUDA branch environment. Last logged complete batch:83,968 roots; source397,568 decisions /3,362 completed matches /26,766 round endings; branches9,885,498 decisions /167,936 endpoints /740,196 round endings,300.877731s. No targets.pt was saved. Later unlogged work is unknown, so these are lower bounds. Same-seed replacement computation must not be counted as additional independent data. runs/scaled_paired_targets/failure_receipt.json retains the failure and repetition accounting. A separate fresh-process reproduction failed after334 unclosed registrations of the same97-character DLL directory;1,500 explicitly closed registration cycles passed.

Added scripts.collect_reusable_paired_targets with one reusable branch BatchEnv, full state overwrite at every root batch and original source/continuation/label/accounting semantics. It avoids changing production or helpers bound to ongoing CPU evaluations. Original versus reusable32-root/four-batch datasets match exactly on CPU and CUDA, including all exported public inputs, targets, groups, embedded experts and work counters. Exact public replay and existing dataset audits pass. A terminal-action fixture independently checks outcomes and unchanged source states through512 workspace reuses per device, above the observed resource-failure threshold. Replacement scaled runner keeps original262,144-root budgets and seeds, binds its new collector, and uses a fresh directory.

Added preference supervision to raw/feature selectors: mean over all roots of abs(G_Point-G_parent)/2 times pairwise BCE. Ties have zero data gradient, draw transitions half weight. Its conditional optimum chooses the expert with greater expected win score; logits are not return differences, so diagnostics do not report win-delta MSE for this head. Independent analytic fixtures verify signs, draw weights, ties, common offsets and a weighted-majority reversal. Same original body initialization, scalar head, shuffle, optimizer and20 epochs are retained for the fixed4-case screen. Standard cost-sensitive classification is not claimed as a novel loss.

Command: python -m pytest tests/test_reusable_paired_targets.py tests/test_score_distribution.py tests/test_paired_expert_features.py -q. Result23 passed,0 skips,7 expected upstream warnings,52.61s. Includes actual six-kind raw/feature fits, reloads, diagnostics, hidden-world invariance and CPU/CUDA loss/gradient agreement. No new full-match evaluation or deployment promotion. Existing paired/static evaluation and watcher remain active with their bound files unchanged.


## 2026-10-18: preference screen complete; matched first-parent controls available

Preference screen session90509 exited0, source29b4447. All4 fixed20-epoch endpoints pass stored input/expert/count/initialization/shuffle checks and independent CPU softplus loss reconstruction; auditSHA47d169a62fc1b8f462638a956e1d2a69f2b1039acd9c075b0541a2c1580bfe87. Raw preference gains over always-parent are+0.404931pp [0.106745,0.710060] and+0.154571pp [-0.192219,0.517587]; over always-Point+0.032251pp [-0.312077,0.373215] and-0.096607pp [-0.438207,0.240370]. Feature preference gains over parent+0.129005pp [-0.100403,0.350987] and+0.027050pp [-0.292079,0.341640]. Neither representation qualifies against both experts at both parents.19.0478165s recorded fitting overlapped scaled CUDA collection, so it is not isolated throughput. No deployment promotion. Historical screen source/hash snapshots remain authoritative after later runner parameterization.

All12 first-parent paired-study matchups now have raw-statistic/identity checks, including same22M-deal unchanged-parent controls. Preview runs/paired_arbitration_preview_parent1/matched_controls.json SHAe66f33c3a6e3ee569778641a9981ed4d44253a83aa947b43bd8d79af2795cd38. Full selector minus unchanged parent: WP win+0.350pp [-1.550,2.300], point margin+1.260 [0.159875,2.43005]; Point win+0.125pp [-1.950625,2.175], point margin+0.444 [-0.798125,1.714025]. No win improvement established. Second parent/full-study audit pending. These are exploratory matched contrasts, not final confirmation.

Replacement scaled collection session88030 is live and passed the original65,536-root scale: source310,144 decisions /2,610 matches /20,870 round endings; branch7,723,398 decisions /131,072 endpoints /578,768 round endings at238.844703s. A follow-up inspection found the original feature-replay helper also constructs a native environment each batch. The live bound runner is left untouched; it can retain its complete first dataset before that replay hits the same resource limit. scripts.replay_reusable_paired_features reproduces the original features/scores/dealer bits/counts exactly on CPU/CUDA using one workspace. scripts.complete_scaled_paired_targets waits for terminal status, revalidates identities, copies a completed audited dataset without redoing branches, and finishes remaining collection/replay with bounded resources into a fresh directory. Failed predecessor replay work is additional unmeasured computation and cannot be assigned a fabricated exact count.

Score-fit runner now accepts explicit dataset/feature directories and epoch budget, and binds selected cache/data hashes to the passed feature audit. This prepares larger-data20-epoch and smaller-data80-epoch controls; exact optimizer counts will be reported because split sizes make budgets only approximately equal. No enlarged-data fit has launched. Extended reusable-replay and custom-plan tests passed5/5,0 skips,2 warnings,36.27s; recovery orchestration is tested separately before queueing.
Recovery orchestration test passed1/1 (4 unrelated deselected),20.89s: real32-root first dataset is reused byte-for-byte, only the missing second parent is collected, both public replays complete, reused/new work stays separate, and existing-directory/protocol mismatch rejection passes.

Completion runner launched from source0fa49f5 in session4899, actualPID62020/wrapper49088; process waiting_for_predecessor. Re-polled live along with generation88030, paired28640, static48159 and watcher98461. Generation actualPID35592/wrapper33916 reached122,880 roots, beyond the original resource-failure point: source582,144 decisions /4,966 matches /39,230 round endings; branches14,431,365 decisions /245,760 endpoints /1,081,044 round endings,444.778716s. All work remains ongoing. Completion runner and reusable replay helper are now bound and must remain unchanged.


## 2026-10-18: fixed data/update-budget controls and reusable feature audit

Prepared the bounded comparison in PAIRED_ROLLOUTS: paired_win/preference x raw/features x2 parents; existing65,536-root20-epoch endpoints, new80-epoch fits on identical small data, and later262,144-root20-epoch fits. Seeds/initialization/shuffle/optimizer stay fixed. Larger and smaller validation populations differ, so direct endpoint contrasts will use the same new held-out source groups, explicitly checked against each training split. Exact optimizer counts must accompany approximately matched budgets. No new score-selector matches or final confirmation data are used.

scripts.audit_paired_feature_cache checks dataset/producer/source identities, shapes, finite features, phase/dealer metadata, source/branch accounting, and every cached score using independent CPU linear reconstruction from the embedded expert's output weights. Its scope excludes a second full source replay and strength inference. Actual CPU/CUDA collection/replay fixtures pass and detect altered scores and dataset identities:2 passed,3 unrelated deselected,3 upstream warnings,33.70s. Live collection/recovery and original paired/static/watcher inputs are unchanged.


## 2026-10-18:80-epoch control complete; common-cohort comparator verified

Session39519 exited0 after8 fixed80-epoch fits, source7468c07. Audit runs/score_update_budget_control/fit_audit.json SHAcd240a440794ac221fb5804396f96831901bb90d32e01b60996d543e0a168792 passes bound inputs, stored experts/parents/counters, exact initialization/shuffle agreement with20-epoch endpoints,4x optimizer steps (4,080/4,160 versus1,020/1,040), and independent CPU MSE/softplus validation loss reconstruction.126.753215s total recorded fitting overlapped ongoing CUDA generation and is not isolated throughput. No configuration qualifies against both fixed experts at both parents on the existing validation set.

Raw paired-win80-epoch train/validation MSE: first parent0.27706791/0.35229497; second0.28811331/0.36016726. Prior20-epoch validation was0.31571254/0.31649479. Lower training error with worse validation error indicates overfitting for these raw regression configurations; this is not a claim that every objective is fully optimized. Raw preference80-epoch one-action gains over parent+0.394180pp [0.106784,0.692786] and+0.166164pp [-0.190895,0.533043], still no replicated superiority over both expert choices. All endpoints retained, none promoted.

Added scripts.compare_score_training_budgets for the prespecified three-budget comparison. It evaluates every endpoint on identical larger-data held-out source matches, checks training-data/expert/source/cache identities and train/evaluation group separation, verifies dataset sizes/epochs/representation/shared initialization/shuffle, and reports paired root-weighted changes with source-match bootstrap intervals. It refuses incomplete fitting conditions and existing outputs. Actual distinct-source32-root raw/feature fits and transfer evaluation, direct outcome-unit checks, identity/cache rejection and analytic group-overlap/sign fixtures passed2/2,20.46s,1 upstream warning. Large-data comparison is not yet run; these are correctness tests.

Second full paired-regression selector completed all three1,000-pair matchups: WP49.250% [47.450,51.000], points+2.269 [1.185,3.338125]; Point49.550% [47.975,51.025], points-1.012 [-1.971,-0.065]; own parent48.425% [46.800,50.050], points+0.487 [-0.493075,1.507]. Raw outcome/identity checks passed in runs/paired_arbitration_preview_parent2/full_completed_matchups.json SHA406fa1f199b2ae3ed69a2dd76c92b3824f6bf8af7b2d1ffb3923fa33ef44929c. Same-deal second-parent controls and full-study audit remain pending; no improvement is established by these endpoints.


## 2026-10-18: first scaled dataset/cache complete; fixed follow-up prepared

Generation session88030 exited1 in the known original replay DLL-registration path, after successfully saving and auditing the entire first262,144-root dataset. DatasetSHAdc4389c863af6699f5be30222ccfd96e8b63bb6acb3a36157465199bc521811a; source1,242,624 decisions /10,691 matches /84,009 round endings; branch30,668,584 decisions /524,288 endpoints /2,302,398 round endings. Collection949.513761s. Completed versus incomplete source-match groups are counted separately:8,657 training groups and2,160 validation groups; these include unfinished source episodes and therefore need not sum to completed matches. Paired delta counts(-2,-1,0,1,2)=(9,976,2,696,236,948,2,705,9,819), variance0.322649568. Original failed replay has no exact work receipt; it remains unmeasured overhead.

Completion session4899 revalidated/copied the same dataset byte-for-byte, then finished bounded replay in108.895090s with no repeated branch simulation. New source replay work equals the original source counts. CacheSHAcd6df331e90bdfe0a01478bb7b94334bf875464c3ec35bedf561dc4d48eafc66,1,078,201,569 bytes. Independent full cached-head reconstruction passed (max CPU error3.8146973e-5 within fixed relative/absolute tolerances) in features/parent1_audit_preview.json. Parent2 collection is now running under the same completion handle.

Added scripts.audit_win_selector_screen to automate the already-used endpoint checks and independent CPU MSE/softplus formulas. It passed all8 existing80-epoch endpoints; reusable_audit.json SHA2e1330c1306b0fc64bbc6d3b54dff74d091ad1216adfe039ceeaea294cccf98f agrees with the earlier manual audit. Prepared scripts.run_score_scale_followup with47 bound inputs: wait for recovered data completion, audit both feature caches, fit all8 fixed larger-data20-epoch cases, audit endpoints, and compare all24 budget endpoints on common held-out source matches. No model selection or strength matches. Actual plan construction and existing-output rejection pass; subordinate feature/fit/comparison paths have the recorded actual tests and audits. Queueing will freeze its additional input scripts. It owns recovered/features/audit.json, runs/score_scaled_fit_screen and runs/score_scale_followup/comparison.json; do not duplicate those outputs while queued/running.

Follow-up launched from0eef853 in session71402, actualPID22128/wrapper27812, status waiting_for_data. Its47 input identities are frozen and its future output paths reserved. Completion4899 is live at63,488 parent2 roots: source304,384 decisions /2,557 matches /20,351 round endings; branch7,476,483 decisions /126,976 endpoints /555,600 round endings,220.020442s. Both handles plus paired28640, static48159 and watcher98461 were directly re-polled live. Previous turn and this turn are progress, not a blocked wait; no goal completion claim.


## 2026-10-18: manuscript clarity and classification prior work

Revised Abstract/Introduction to lead with concrete measured gains, budget and preserved inference. The warm/cold development result is stated as a0.935--1.050pp advantage of the tested joint configuration, with identical initial greedy policy. Method prose describes the joint temperature/learning-rate intervention directly and removes repeated blanket qualifications. Historical final-test wording now applies explicitly to its frozen checkpoints/budget, without implying that later development never occurred. The standalone64-character digest was removed from the prose; evidence files retain full digests.

Expanded classification-based policy-improvement prior work using original ICML2003 Lagoudakis/Parr and ICML2005 Langford/Zadrozny papers; corrected Dimitrakakis/Lagoudakis to its verified Machine Learning72:157--171 journal record. Existing PPO/Transformer/rollout-classification ideas are attributed, while this experiment's Koi-Koi expert-choice question stays explicit. No new supervised/data-scale outcome is asserted in the manuscript.

Built with Tectonic and visually reviewed all13 pages; final pages11/12 reinspected after the digest paragraph edit. QA runs/manuscript_clarity_revision_qa.json passes18 embedded fonts,18 resolved citations, no TeX box/glyph/reference warnings, page bounds and all26 historical plus34 arbitration input hashes. PDF SHAa6538307993938f84291ebe757a12e8dc2cc7f99d74523ac467302fbe548e56a. Prior PDF preserved at output/pdf/manuscript_arbitration_16ac69ab.pdf. Original local research archive remains unchanged. This improves the manuscript deliverable while stronger-model experiments continue.


## 2026-10-18: phase population and fixed-hybrid target diagnostics

Retrospective read-only diagnostics on the existing65,536-root datasets, using the two hybrids already fixed in runs/static_expert_controls/protocol.json. Dataset/root/parent identities checked. Discard roots59,661/59,168; koi roots3,547/3,757. Thus about90--91% of roots are discard decisions and about5--6% are koi, while the other capture phases are sparse. This motivates examining phase-specific learning behavior, without changing any fixed fit/evaluation protocol.

Point-cards/parent-koi one-action win gains over always-parent: first+0.455099pp [0.021918,0.881474], second+0.405750pp [-0.031592,0.850268]; over always-Point+0.082420pp [-0.075084,0.238545] and+0.154571pp [-0.015235,0.325701]. Opposite-role hybrid changes are the corresponding negatives. Neither hybrid establishes replicated superiority over both experts on these source-match diagnostics. Frozen-parent continuation and held-out-root uncertainty apply; full repeated-policy matches remain in the original queued static study. Report runs/static_paired_target_diagnostic.json SHAeb9dd3c35b3d94bb228c4c1502cb681906746720b6c537efc985a078c1edfb50. No fitting or new simulation.

All five active handles were directly re-polled live after the manuscript revision: recovery4899, data-scale follow-up71402, paired28640, static48159 and watcher98461. Parent2 scaled generation reached253,952 roots at878.347770s; fitting still waits for completed data/replay. Goal remains active: manuscript clarity improved, but further verified strength and final integration remain open.


## Completed data-scale comparison and preference-collapse control

Recovery4899 and follow-up71402 exited0. Both262,144-root datasets and exact
feature caches passed the full stored-score audit (SHA256
1fa7a22cac0c17f403c0b5aa19b66e3d9e4e3234c584b13787ad9ca42ef52f48).
Parent2 dataset SHA637040e53c9150c45ecbc5b6322c517b128823c406122da3b6bd8875fbb9afaa;
its source work is1,251,456 decisions /83,824 round endings /10,708 matches;
branch work30,690,157 decisions /2,281,272 round endings /524,288 endpoints.
Parent1 data was reused byte-identically, not generated again.

All8 large-data20-epoch fits passed the independent endpoint audit
(SHA44cd498ec413820a85edba7a33dd26e86fae781c373cebec6d62c7d1422d26c8).
The24-endpoint common-cohort comparison completed with all47 bound follow-up
inputs verified before subsequent source edits. Report
runs/score_scale_followup/comparison.json has SHA256
b90db78726ae8c6a511f1f6462c685594d927b1164d65dce6cd7f249b6138653.
No condition establishes one-action gain over both fixed experts at both parents.
All8 large20-minus-small80 win-change intervals include zero. Validation uses
52,090 roots /2,160 source matches for parent1 and52,055 /2,142 for parent2.
These cohorts are now consumed development evidence. The previous small-cohort
static-hybrid pattern also does not replicate on this cohort; diagnostic SHA
 a0167471123a32e3e78b18091c6477ce34cd7520c8a338059db4aa93a013bea2.

Further inspection found all4 large-data preference endpoints produce exact ties
at every held-out root; choices are consequently native action-index tie-breaks.
The first body-layer weight maxima are about2e-40 to5e-40. Some parameters remain
nonzero, so describe exact score collapse, not literally zero parameters.
Diagnostic runs/scaled_preference_collapse_diagnostic.json SHA256
2300e85e9903ea88663c8ba9ff90002ee1b4b9a8d70d289e09f8f65a0c1ed2ea.
This finding warrants the fixed zero-decay control specified in PAIRED_ROLLOUTS.
The fitting CLI now accepts finite nonnegative weight decay; its default remains
numerically identical. Actual CPU fitting verifies default equivalence and that
zero decay reaches the optimizer. Combined score/cache/screen tests pass20,
with0 skips and5 expected upstream warnings. No deployment promotion.


## Completed paired full-match and regularization controls

Paired session28640 exited0. Automatic watcher98461 completed the whole24-matchup
analysis and both held-out diagnostics, then waited for static controls. Audit
runs/paired_arbitration_study/analysis.json SHA256
d5709d50e51101240d0d9df7cc47a569d55da8fa32f7f3dd5b3304b597c286ff.
Evaluation totals48,000 matches /5,551,259 decisions on1,000 shared unique deal
streams, not24,000 independent streams. Full selector minus parent win changes
are parent1 WP+0.350pp [-1.550,2.300], Point+0.125pp [-1.950625,2.175]; parent2
WP-1.500pp [-3.600,0.650], Point+0.225pp [-1.825,2.150]. Corresponding point gains
are+1.260 [0.159875,2.43005],+0.444 [-0.798125,1.714025],-0.168
[-1.4341,1.060025],+1.660 [0.411975,2.863025]. Full/no_match win-difference
intervals include zero against both public opponents at both parents. Neither
full nor no_match establishes replicated win gain. Static controls48159 started
following paired completion; no checkpoint promoted.

Zero-decay screen16265 exited0 with8/8 complete. Audit19572 and matched-comparison
9981 exited0. Fit audit SHA c69a4e5b4e8ab0f420e2342400a34e53ecd6e754d101ca903cc2d1fb778dee60;
comparison SHA ed9f5d7bbd8ab9dc7cb09499e41e339dc084a3995d26ece5b399678e8b6faddf.
Every endpoint now has zero exact proposal-score ties. Preference score-difference
RMS is2.691388/0.195728 for parent1 raw/features and0.962448/0.202594 for parent2.
Changing only weight decay prevents the observed collapse under this fixed setup,
but none establishes win gain over both expert choices at both parents. All8
zero-minus-default win-change intervals include zero. The comparator checks
identical dataset, cache, network, counters and training settings except decay;
actual-endpoint matching and learning-rate/decay tampering checks pass. No new
simulation is charged to these fits. Default training weight_decay remains1e-4.

Prepared a separate source/continuation helper for the next opponent-distribution
control. It collects only learner disagreement roots, alternates learner seats,
and crosses fixed WP/Point opponents across four-match cycles. Metadata never
enters scorer inputs. The continuation retains the original learner seat and
opponent throughout all remaining rounds. Four CPU/CUDA tests pass, including
independent reference outcomes, point signs, decisions/round accounting and real
expert collection/public reconstruction. Initial CUDA device-alias comparison
failure was fixed by comparing actual tensor devices; final run4 passed,0 skipped.
No new large collection or fitting has been launched for this hypothesis.


## Opponent-continuation study runner verified

Implemented scripts.collect_opponent_paired_targets and
scripts.run_opponent_target_study for the fixed two-parent/two-continuation design.
Collection exports no hidden worlds. Both conditions share exact public roots,
source-match IDs, opponent metadata and frozen feature tensors; source is counted
once and each condition's two branches separately. Public features are extracted
before branch outcomes from the original root states. Explicit at_collection cache
receipts report zero replay work; the original replay-cache path is preserved.

The runner binds all input identities, audits both caches and shared roots, fits
four final20-epoch feature/paired-win endpoints and cross-evaluates both models on
both label sets. Main target is opponent-preserving held-out outcomes. It checks
matching initialization/order, fixed budgets and independent CPU validation loss.
It performs no full-match evaluation or automatic model promotion.

Combined actual collection/fit/cache/old-scorer tests passed25,0 skipped, with10
expected upstream warnings (93.35s). A separate complete reduced-budget study,
including both label conditions, actual fits, cross-evaluation, fresh-directory
and bound-input rejection, passed1 with2 expected warnings (55.57s). The initial
runner smoke exposed tensor-valued opponent metadata in the JSON protocol; this
was corrected to the existing parameter/model hashes before the passing rerun.
Ruff and diff checks pass. Production and active static/watcher dependencies are
unchanged. Launch uses the fixed PAIRED_ROLLOUTS protocol, fresh source seeds
19710001/19720001 and65,536 roots each; these tests are correctness evidence only.


Launch receipt: The fixed study launched from9013980 in exec session59220 (actualPID5940, wrapper42868), collecting parent1. Protocol SHA54d0a365737022b66e7761d970523aa7d300f3f0dd176c9cd8690e0f5d575b76; all18 listed inputs and unchanged production SHA were verified after launch. Keep these dependencies frozen until terminal. First2,048 roots completed with19,200 source decisions /1,226 round endings /128 matches; parent-only and opponent-preserving branches completed4,096 endpoints each. Static48159 and watcher98461 were re-polled live. No strength result yet.


## Completed paired-results paper revision

The manuscript now includes the completed24-matchup paired-supervision study.
New scripts.export_paired_paper_evidence independently reloads raw pair outcomes,
checks them against the passed audit and exports the four-row full/no_match table.
The paired evidence ledger binds117 inputs, including all48 score-screen fitted
checkpoints and their fixed plans/audits. Its SHA256 is
99b1d46189c69932587779b94f66c49906c209dfc4f194cd389fa9facbeb556d;
table SHAa312eb5008c2d652b0db236f827c21022fe975ea3678ad78b29b9c1d2de4d3e0.

Abstract/Introduction retain the verified adaptation gain and now state the
completed negative paired-selection result. A compact results table preserves
both parents and the context ablation. Data-size, fitting-budget and collapse
controls move to a separate appendix rather than crowding the main argument.
The missing "no" in the historical unchanged-final-test statement was corrected.
Static/opponent-preserving results remain pending and are not reported as findings.

PDF output/pdf/manuscript.pdf has14 pages and SHA256
075411f59901a0418417dc8608d5d6c23aa729051ec324ea0eeb315e0bfefd1f.
QA runs/manuscript_completed_paired_qa.json SHA256
6e0d3d9df741a8925e5e20bb2f6bff6a5423655abe1136178b0495a946d376cf passes:
changed pages1,2,6,11--14 visually reviewed; remaining7 renders byte-identical
to earlier reviewed pages;18 embedded fonts and18 citations; no TeX box/glyph/
reference or page-bound defects; all26 historical/34 arbitration/117 paired input
entries verified. Independently checked48-endpoint qualification claims, all8
large20-minus-small80 and all8 zero-minus-default intervals, and source/branch/
evaluation accounting. First QA output name collided with an older preserved
receipt and was correctly rejected; a fresh completed_paired name was used.
Previous clarity PDF preserved as output/pdf/manuscript_clarity_a6538307.pdf.
Original evidence archive remains unchanged. Opponent-target59220 and static48159
continue with bound code untouched; no new strength improvement is established.


## Static controls complete; fixed switch-budget diagnostic prepared

Sessions48159 and98461 exited0. All8 static matchups and the automatic watcher
completed successfully. Static audit SHA256
ebd960ce36e97ab51a20f14be9360c5eab0313bf5d6d31faf374e99f39319ad8.
Supplemental all-hybrid versus unchanged-parent comparison, independently checked
from raw records, is runs/static_expert_controls/parent_comparisons.json SHA256
1bbfaad279a585cc82dfa0bc045c5086511a0be3daf3010c196adb0aadb9a4f5.
Actual static work:16,000 matches /1,855,405 decisions on the reused1,000 unique
22M development streams. No new independent deal population is implied.

Point-only koi substitution increases expected point margin over the parent in
all four comparisons: parent1 WP+1.723 [0.921,2.532025], Point+1.956
[1.152,2.758025]; parent2 WP+0.926 [0.128975,1.723], Point+2.086
[1.23595,2.902]. Corresponding win changes are-0.800pp [-2.125,0.550],+0.900pp
[-0.425625,2.225],-0.850pp [-2.225,0.500],+0.900pp [-0.500,2.275]. All8
hybrid-versus-parent win intervals include zero. Learned full/no_match selection
does not establish replicated win improvement over these static controls either.
No deployment promotion.

The opponent-target study59220 completed first-parent collection, cache audits,
both fits and cross-evaluation, then began parent2 collection. First-parent
source:601,728 decisions /301,419 learner /300,309 opponent decisions;
40,283 round endings /5,129 matches (2,571 WP /2,558 Point). Opponent-preserving
minus parent-only fitted model on opponent-preserving labels: win+0.046104pp
[-0.215089,0.302850]; points-0.052559 [-0.197653,0.086176]. This partial result
does not establish target-construction benefit; parent2 and the full audit remain
pending. Both first-parent scorers choose Point at roughly49--53% of held-out
source disagreements, versus10% in source behavior.

A separate fixed10% training-score-quantile diagnostic is implemented in
scripts.diagnose_switch_budget; its first application covers only the two existing
larger-data feature/paired_win endpoints. It retains ties with the parent and
reports unrestricted/expert/same-rate-randomized expected-outcome controls on the
same consumed held-out roots. Two tests pass,0 skipped, including actual
collection, fitting and diagnosis; one expected upstream warning. No neural
uncertainty guarantee or match-strength claim. POLICY_IMPROVEMENT.rst documents
the exact design and verified primary DAgger/AggreVaTe/SPIBB positioning. All
18 active opponent-target dependencies remain unchanged.


## Fixed switch-budget diagnostic and opponent targets: completed

The fixed10% training-score-quantile diagnostic completed on the two existing
large-data feature/paired_win endpoints (session7907 exit0). Parent1 threshold
0.00625252724, held-out Point fraction0.10101747; parent2 threshold0.01129694656,
fraction0.10233407. Restricted-minus-parent win changes are+0.090228pp
[0.005733,0.178805] and+0.057631pp [-0.028169,0.140142]. Restricted-minus-greedy
changes are-0.023997pp [-0.168435,0.134072] and+0.006724pp [-0.159928,0.179072].
No replicated win benefit. Expected-point gains over parent are+0.112882
[0.067354,0.159371] and+0.120565 [0.071495,0.170024], but restricted-minus-greedy
point intervals also include zero. No threshold sweep or full-match promotion.
Reports runs/switch_budget_diagnostic/parent1.json SHA90bdc85f69c92ad0dc3a0d874293bac11eb94759dab9f00a1f9e3735d13fad46;
parent2.json SHA70b4ccaf40b4ee96bc1d29f860c03386c5e3d855de3181d57b0d666a91c85f19.

Opponent-target session59220 exited0. All four endpoints, both source/cache audits
and cross-evaluations completed; all18 protocol input identities reverified after
completion. Results SHA4187328f8d77757bd51cebdb0cd6be747de43ca3427a678fecf8db6108fa5665.
Parent1 data audit SHAb0f2e17e7275c9e0b555374219c1fd6ae0533dd1c9b8d8c5130513ecce26f06e;
parent2 SHAf01acc4f4bc47ab819e914f1616cd5ac475319f9ee950f413a84cff5546537e1.
Opponent-preserving-minus-parent-only training on opponent-preserving labels:
parent1 win+0.046104pp [-0.215089,0.302850], points-0.052559
[-0.197653,0.086176]; parent2 win-0.023348pp [-0.282387,0.246278],
points+0.069577 [-0.073287,0.217382]. No target-construction benefit is established.
On parent-only labels parent2 has a point improvement+0.212001
[0.059409,0.364997], while its win interval includes zero; this is not evidence
of improvement against the preserved opponents. Retain all labels/endpoints.

Combined shared source:1,211,392 decisions (606,950 learner /604,442 opponent),
80,859 round endings /10,311 matches (5,165 WP /5,146 Point). Combined forced
branches:30,876,654 decisions /2,295,860 round endings /524,288 endpoints.
Four branches per learner disagreement across two parents. Features extracted
inline; zero additional replay simulation. The runner's per-parent collection,
CPU fit-loss, initialization and cross-label comparisons are retained in results.
All previous studies/watchers are now terminal; bound dependencies are released.

## Direct pretrained response capability

Source review confirms the published actor already sees score difference
(native published.h rows17--19) and round identity (rows26--33). Adding those
inputs is not a new information capability. The earlier koi residual studies
used the project's MLP actor; nevertheless, existing published-context features
and failed selector controls rule out claiming novelty from merely adding context.

scripts/published_response.py instead enables actual on-policy pretrained actor
updates against fixed WP/Point opponents. Its separate config validates all
ordinary published-PPO restrictions while explicitly allowing a fixed opponent
fraction and wp/point/wp_point routing. Across four episodes per lane, the
learner visits both seats against both opponents; assignment stays fixed across
rounds. Actor inputs remain unchanged; routing metadata is not given to the actor.
Existing signed GAE follows all decisions to match termination. Frozen decisions
are excluded from actor losses/advantage normalization; a public observation-only
opponent-role critic supplies their values. The joint clipping operation includes
both critics, and real learner/opponent counts accompany total simulator work.

Seven tests pass,0 skipped,8 expected upstream warnings (15.32s): CPU/CUDA
reference transitions, independently recomputed seat-zero returns, expert action
routing and accounting, unchanged frozen experts, exact optimizer/environment/RNG
restart, all-opponent actor-loss exclusion, partial fixed-opponent lanes, and
canonical-parent greedy action equality across all four phases. ResponseAgent
reuses the existing deployment action path and loads no opponent model. This is
implementation validation, not a strength result or a new PPO algorithm.

Production-size capability session6936 exited0:128 environments x128 horizon,
three updates (one critic-only),49,152 decisions /3,242 round endings /383 matches.
Actual learner24,419 /opponent24,733 decisions. Elapsed update time12.094623s;
peak allocated CUDA memory2316669440 bytes (allocator-reserved3531603968).
All tensors finite; every actor/value/opponent-value Adam step matches counters;
per-opponent sums match totals. Report SHAfa356e15be7d4c3f7e9e71889b02269f1ae8007b2bde12ef08dcba7a890fa427.
No strength inference, checkpoint promotion or production behavior change.


## Direct response study: fixed protocol and pipeline verified

Added scripts.run_response_study and RESPONSE_STUDY.rst. Two retained parents x
response/selfplay continuation, fixed520 updates (20 warmup+500 policy),
8,519,680 simulator decisions each. Fresh training seeds19910001/20010001.
Four endpoints plus unchanged parents face strict CPU2 assisted WP/Point on
5,000 common development pairs23,000,000--23,004,999 (12 matchups /120,000 matches,
5,000 unique deal streams). Existing protocol inventories have no overlapping
recorded evaluation range. All results retained; qualification requires positive
paired lower95 bounds over BOTH parent and selfplay for BOTH parents/opponents.
Point differences secondary; no final-test/deployment promotion.

Three tests pass,0 skipped,17.43s: actual reduced subprocess scheduler through
both training conditions, six evaluations and full analysis; protocol/receipt/
opponent identity tampering; matched fixed production settings; owned-worker
failure cleanup and resource/dependency scheduling. The reduced run performs
24 matches and has no strength qualification. Ruff/format/diff checks pass.
Bound production source, helper/runner sources, upstream Python/weights and
parents are rechecked. Independent raw outcome and checkpoint/accounting checks
run before results are accepted. Planned maximum concurrency is one GPU trainer
and three CPU2 evaluators. Equal simulator budgets differ in actual learner
samples and additional opponent-role critic/clipping; no isolated causal claim.

Launch from8bfbc05: session26076, actual runnerPID15700/wrapper46640.
First response trainer60324/wrapper36508; actual CPU evaluators63964,30804,41228
with wrappers62440,60200,61212. Protocol SHA2342e1e80ebc79fc917891ebbf30acd7ef5dd0b4a21413d9d78e297feed57ffb.
All bound inputs reverified after launch; first two updates completed32,768
decisions /2,150 round endings /250 matches, finite metrics. Live process command
lines and progressing records confirmed. No outcome/qualification claim.


## Completed follow-up results integrated into the manuscript

The15-page revision includes the static Point-continuation gain in Abstract/
Introduction and a four-row paired win/point table. Both assignments/all8
comparisons remain in its59-input evidence ledger; every table effect is
recomputed from raw results and checked against the completed comparison.
Completed opponent-target and fixed-switch diagnostics move to the appendix.
Related Work adds DAgger and AggreVaTe primary references (20 total citations),
positioning fixed-source fitting against iterative learner-distribution data.
No result from the active response study is reported as a finding.

PDF SHAc5d7c21532e318f11c66fb4932f454e17899f20c3a9970479c99c65eef39a88e.
QA runs/manuscript_followup_qa.json SHAe10ea1c7e7c26b294e9b421cdee298d842179429a95b5346c623b7c70efb1e74:
all15 pages visually reviewed,18 fonts embedded,20 citations resolved, no TeX
box/glyph/reference or page-bound defects. All26/34/117/59 evidence input ledgers
verified, including newly bound datasets/caches/checkpoints and completed audits.
New followup_evidence.json SHAa72b8ebd49b69d6bed3bb7c78b519e864c5cc36de68880377522b12511b6967d.
PriorPDF output/pdf/manuscript_paired_075411f5.pdf and original archive preserved.
Live response26076 remains bound to its unchanged29 inputs and production source.


## Manuscript structure and existing-target stage diagnostics

Editorial revision keeps15 total pages/20 cited references but moves the conclusion
from page12 to10. Main-source whitespace tokens decrease4,925 to3,775 (23.35%).
All prior numeric tokens, generated table inputs, citation keys and equations
are retained; implementation/timing and controller details move to appendices.
This is not a new strength result. QA runs/manuscript_structure_qa.json SHA
9b817ce74800e74fd1dfed42995038fabb47a33b8127d3d19fbd7e6510122e0f verifies all4
ledgers,18 embedded fonts, clean TeX/bounds and all changed pages1--14 visually;
page15 equals its prior reviewed render. PDF SHA90f9d9f812f7e5656ec981664f50abb8edcc14d5af554edfd09fb7321587f17c.
Prior follow-up PDF/archive preserved. Active response results remain excluded.

scripts.diagnose_target_stages inspects only the training source-match splits of
the completed opponent-preserving datasets. Fixed descriptive partitions: card
versus koi, rounds1--6 versus7--8. Counts/sign units and complete partition checked;
dataset identities match the completed study. No fitting or simulator work.
Report runs/opponent_target_stage_diagnostic.json SHA
b0168490793960b9ee4d1ceff64c864242fc633d07debd78f484630d6982a259.

Parent1: early cards3,691 differing win outcomes/37,636 roots (9.81%); late
cards793/12,128 (6.54%); early koi511/2,134 (23.95%); late koi98/624 (15.71%).
Parent2: early cards3,443/37,457 (9.19%); late cards739/12,184 (6.07%);
early koi485/2,458 (19.73%); late koi93/588 (15.82%). Each group retains its
Point-minus-parent signed mean and positive/negative realized-label counts.
These describe correlated source roots and noisy one-action branch outcomes,
not a public-information oracle, causal horizon effect or repeated-policy gain.
The result does not support assuming the last two rounds automatically yield
denser supervision. No stage-selected model or change to response-study conditions.

## Public turn-win certificate capability and coverage screen

Added scripts/public_win_certificate.py and an independent reference-tree test.
The public unknown-card set overapproximates next draws. Hand capture precedes
universal draw enumeration; draw capture follows the reveal. The wrapper changes
an action only when a continuation ends the match with a win in the same turn.
Continuous use at subsequent decisions supplies the first-deviation coupling
argument for weak match-win nonregression. This is an algorithm argument, not a
machine-checked implementation proof; it does not imply point-margin improvement.
The method note relates it to established observation-based synthesis and
shielding, without claiming these general principles as new. Production, human
play, dealer choice and active study sources are unchanged.

Validation command: .venv/Scripts/python.exe -m pytest
 tests/test_public_win_certificate.py -q
 --junitxml=runs/public_win_certificate_tests_final.xml
Result: 5 passed, 0 skipped, 4.41 seconds. Tests compare 120 phase/context fixtures
against independently expanded ReferenceEnv trees, check public legal-mask
reconstruction, run 256 complete test matches across both profiles and base/guard
arms, and verify hidden reassignment and public-only access. These are correctness
tests, not learned-policy strength estimates. Initial test failures were fixture
issues: KOI was incorrectly placed on an automatic-stop turn, and a random policy
against greedy had no override in the small research12 sample. Fixtures now keep
KOI on valid turns and use seeded random opponents for the coupling check.
Ruff passes after import formatting; git diff checks pass.

Command: .venv/Scripts/python.exe -m scripts.diagnose_win_certificates
 --output runs/win_certificate_diagnostic_final.json
The source-match TRAINING splits of the two completed opponent-preserving target
datasets contain 52,522 and 52,687 expert-disagreement roots. Certificates occur
at 157 and 149 roots: discard 138/138, hand capture 4/2, draw capture 15/9, koi 0/0.
The parent proposal is uncertified at 12 roots per parent. Point is uncertified
at 102/97 roots; neither proposal is certified at zero roots. Every stored parent
continuation at a certified root already wins in its sampled hidden world.
Thus this screen identifies no recovered parent loss or draw and does not justify
launching a new full-match study or promotion. The biased correlated root sample
is not an estimate of ordinary-game prevalence or an independent strength test.

The screen performs 574,272 symbolic draw cases and 621,402 leaf checks, with
zero new simulator matches, rounds, decisions or rollout decisions. Input hashes
were checked before and after. Final report SHA
49c56dcaf3644e92d03796a41875f5d7b044355784882d3d606479a3d00f077a.
An earlier pre-format report is retained separately as
runs/win_certificate_diagnostic.json; the final report binds the final script.

The live response update240 snapshot is independently preserved at
runs/response_snapshot_checks/parent1_response_update00000240.pt, SHA
f921f8e3234c8b60a4f6c355ed67074c636701fa5c9c17e5722c9a2d94b85649.
Audit report SHA475d7dcd50131c077e22a53182920927d51debe4c5b57b9c10e00f4f8c202fce.
Checks cover parent initialization, frozen opponents, actual episode totals,
source/protocol identities, parameter digests, finite model/Adam tensors and all
102 optimizer parameter step counts. It records 3,932,160 decisions /262,547 round
endings /33,869 matches, split into 1,958,692 learner and 1,973,468 opponent actions.
This snapshot is neither an evaluated endpoint nor a model-selection decision.
The live protocol and all 29 bound inputs were reverified after the new work.

## First direct-response endpoint and independent Adam-budget check

The parent1_response trainer exited0 after the fixed 520 updates (20 critic-only,
500 policy updates). Its scheduler recorded completion before starting the
parent1_selfplay control. Endpoint SHA
ae7afccd9d39e18bb102d264d2b0c4d4384cddfe83fe3d47ed85bee626c82db5.
The runner's training audit passes initialization, frozen opponents, provenance,
metrics, receipts, actual episode totals, tensor finiteness and accounting.

Actual work: 8,519,680 decisions /569,437 round endings /73,491 matches.
Learner decisions: 4,241,242; opponent decisions: 4,278,438.
WP lanes: 2,131,812 opponent decisions /287,159 round endings /36,751 matches.
Point lanes: 2,146,626 opponent decisions /282,278 round endings /36,740 matches.
The source proportions are measured lane activity, not exact equal-duration
samples. Warmup contributes 327,680 decisions /21,926 rounds /2,782 matches.

Added scripts/audit_response_optimizer.py as a separate post-training check.
It derives expected Adam steps directly from the frozen updates, warmup,
minibatches and epochs rather than trusting the saved actor step counters.
It does not call the runner's training auditor. Existing checked() is used only
for protocol/source identity validation before and after the read-only check.
All 102 actual Adam states have the expected shapes, finite values and steps:
48,000 for every actor tensor and 49,920 for every critic tensor.

The actor changes 29 of 30 parameter tensors in each of discard/pick/koi. The
unchanged tensor in each is out.bias, a common scalar shift of action scores.
Initial-to-final L2 movement is 0.926381 /0.660436 /0.590032 respectively; these
norms are descriptive implementation facts, not a behavioral or strength metric.
Both six-tensor critics change all their parameters. This check does not replay
training and supplies no evidence of a higher win rate.

Command: .venv/Scripts/python.exe -m scripts.audit_response_optimizer
 --case parent1_response --output runs/response_optimizer_audits/parent1_response.json
Result: passed. Report SHA
71cbcb1d77fc058cbc8b86f2d0ab8c27028281f4bfb3aaf2b687dc1e006d3c32.
Ruff passes. The live protocol remains unchanged and all bound inputs verify.
No endpoint ranking, additional training, promotion or final-test use occurred.

A negative check modified the loaded endpoint in memory only: discard's saved
step counter and all corresponding Adam step tensors were increased together.
The independently derived budget rejected this mutually consistent false claim;
no checkpoint file changed and no success report was created.

Three completed baseline jobs (parent1 versus WP/Point, parent2 versus WP) have
exit0 receipts. Recomputed raw means/intervals, exact 5,000-deal ranges, seat-score
consistency, model identities and decision sums all pass using the existing
outcome auditor. Total: 30,000 matches /3,475,900 decisions on 5,000 shared unique
deal streams. No round count is inferred from this evaluation format. Report:
runs/response_baseline_audits/first_three.json, SHA
bdcc01c61c8cd3e41e45b261673ecc7eac34e608cde9b18ac445395571c222ed.
The scheduler started parent2/Point and both parent1_response matchups. Actual
processes and completed pair rows in the open raw files were confirmed. Windows
file metadata temporarily reported zero length while direct reads showed rows;
this was not a stalled evaluator and no restart was attempted. No partial
candidate outcomes were ranked or used to change the study.

## Opponent-conditioned one-action diagnostic on training roots

Question: do coarse opponent/decision groups support preferring Point proposals
for match wins, motivating an opponent classifier or a conditional static switch?
The diagnostic uses only TRAINING source matches from the completed
opponent-preserving datasets; no model fitting or additional simulation occurs.
Fixed groups are all decisions, four phases, and the two directions of koi
expert disagreement. Opponent identity is analysis metadata, never a policy input.
The two opponents induce different source states, so the contrast between their
groups is not the causal effect of changing opponent at an identical state.

Command: .venv/Scripts/python.exe -m scripts.diagnose_opponent_conditioning
Report runs/opponent_conditioning_diagnostic.json, SHA
858fec4688afc7e21e2d7e1238d7cc682f580bdb9048fb6eca8bdb24dbf64082.
The report binds its script, split/bootstrap helpers, completed-study result and
both input datasets. Source-match bootstrap uses 5,000 replicates, seed19730001,
and retains a root-weighted mean. Intervals are marginal and exploratory, with
no multiple-comparison adjustment. The sample is biased toward expert disagreement.

Training roots: 52,522 across 4,222 source matches for parent1, and 52,687 across
4,258 source matches for parent2. The overall Point-minus-parent win changes are
-0.2087pp against WP and +0.2499pp against Point for parent1; -0.1735pp and
-0.2636pp for parent2. All four intervals include zero. None of the 28 subgroup
win intervals has a positive lower bound. In particular, the sign reversal
between opponents is not replicated across parents. This supplies no coarse
win-improvement signal for the proposed conditional switch; finer interactions
remain untested, and no classifier or new selector was launched from this screen.

At koi roots where Point stops and the parent continues, against Point opponents,
point-margin changes are +3.9450 [0.8788,6.9149] (291 roots) and +3.0837
[0.9319,5.3960] (478 roots). Their win-change intervals still include zero.
These are one changed action followed by frozen continuation in sampled worlds,
not repeated execution of a new policy or a fresh full-match strength estimate.
The previously completed static-hybrid match results remain the relevant evidence
for repeated-policy expected-point gains.

Validation: input hashes rechecked; direct NumPy aggregation from both datasets
matches reported weighted means; phase and koi-direction partitions sum exactly;
opponent identities are constant within every source match. Ruff and diff checks
pass. The live response study continues with all original bound inputs unchanged.

## First parent's matched training conditions complete

The parent1_selfplay trainer exited0 after all 520 updates, and the scheduler
started parent2_response. The live handle remained session26076 throughout;
progress was observed without restarting or selecting intermediate checkpoints.
Self-play endpoint SHA
e9c9bdb8936a0b01759c670dcef1e190fb2adc0536408d26df37a4054af8abfd.
Its full runner audit passes. Actual work is 8,519,680 decisions /570,032 round
endings /73,019 matches, including warmup 327,680 decisions /22,107 rounds /2,770
matches. Together with parent1_response, the two completed conditions total
17,039,360 decisions /1,139,469 round endings /146,510 matches.

Command: .venv/Scripts/python.exe -m scripts.audit_response_optimizer
 --case parent1_selfplay --output runs/response_optimizer_audits/parent1_selfplay.json
The companion check passes all 96 Adam parameter states: 48,000 steps for each
actor tensor and 49,920 for each critic tensor, derived from the fixed budget.
Each actor phase changes 29/30 tensors; all six value tensors change. Actor L2
movement is 0.926705 /0.686848 /0.594241 for discard/pick/koi. Parameter movement
is not a behavioral or strength estimate. Audit SHA
e73818f9350cd606e7ae21611cad673fdc9631acb2322cdb131f6ad831427b03.

An additional comparison directly loads both initial checkpoints. Actor and
learner-value tensors and complete initial environments are exactly equal.
Both optimizers have empty initial state. The only config differences are
opponent_fraction (0 to1) and published_opponent (empty towp_point).
Initial CUDA RNG states are equal; CPU torch RNG states differ. Matching seed
settings therefore must not be described as equality of all RNG states or fully
coupled training trajectories. The response arm also has its separate opponent
critic and excludes opponent turns from actor loss, as the original protocol
already discloses. The simulator budgets match; learner sample counts do not.
Report runs/response_optimizer_audits/parent1_matched_initialization.json SHA
74a9e33b051d0afcce28196f101af0371585e5a3d3856d8902bd9cb66074ab6e.

The second parent's response trainer is confirmed live (actual46240 under
wrapper58580). First-parent response evaluation continues on both fixed public
opponents; its self-play control is queued for evaluator capacity. No partial
candidate result is ranked or used to alter this study. The remaining conditions
and full raw-outcome/contrast audit are still required before a strength claim.

## Six response-study evaluation cells complete

The parent2/Point baseline and both parent1_response evaluations exited0, each
at exactly 5,000 paired deals. Recomputed raw means and bootstrap intervals,
seat-score consistency, seed order, decision sums, endpoint identity and job
receipts all pass. The response endpoints additionally match their passed
training-audit receipt. New report:
runs/response_completed_evaluation_audits/second_three.json, SHA
b22d5bbf241b04d8b2f1b21a91d9c1b2cdde36736242097adee7337443c4fa3a.
These three cells add 30,000 matches /3,497,403 decisions. Together with the first
three audited cells: 60,000 matches /6,973,303 decisions on the same 5,000 unique
deal streams. All four baselines and the first response model's two matchups are
complete; remaining self-play and second-parent conditions still preclude the
fixed replication decision. No inter-arm contrast or ranking was performed.
The scheduler started both parent1_selfplay evaluations and continues the second
response trainer. Source identities remain fixed.

## Exact observer-switching bootstrap counterexample

The next methodological investigation concerns the information used to bootstrap
an actor's return, rather than another action-selection threshold. A uniform
private signal X and public binary action A lead to a fixed exposed opponent move
and terminal Bernoulli signed payoff. Conditional payoff means by X/action are
[[0.20,0.21],[1.0,-1.0]], and the initial learner policy is uniform in both states.
Both critics are exact conditional expectations of their own observations.
The opponent observes A but not X. Full derivation and prior-work positioning
are in docs/OBSERVER_RETURNS.rst.

With lambda0.95, the signed recurrence gives lambda*R+(1-lambda)*E[R|A]-V(X).
For the policy parameter at X=0, the true unconditional gradient is +1/800;
the surrogate expectation is -161/32000. This reverses that coordinate, not the
entire gradient vector. Correct signs and exact conditional-value fitting alone
do not resolve this information mismatch. This is a finite counterexample to a
general unbiasedness assumption, not evidence that it caused the observed Koi-Koi
results. The design already disclaims unbiased finite-observation PPO.

scripts/diagnose_observer_bootstrap.py enumerates all eight signal/action/reward
combinations, including zero-probability outcomes, and checks analytic fractions,
a direct objective derivative, Python signed GAE and native returns. Four cases
are checked on both CPU and CUDA: the example, lambda1, a learner-to-terminal
transition, and an intermediate joint-information critic. The latter three
restore the true expectation in this example. The compressed transition has
higher single-sample gradient variance (0.1197453 versus0.1080973 for theta_0),
so this result alone does not predict stronger finite-budget learning.

Commands: python -m scripts.diagnose_observer_bootstrap
 --output runs/observer_bootstrap_verified_cpu.json
and the same with --device cuda
 --output runs/observer_bootstrap_verified_cuda.json.
Both pass. CPU report SHA
4e3e05b73b4e019bf9e68cf2b90afe0d61a2829b0e821dc303d1224b6a13aeab;
CUDA report SHA
50ecb0247856ed6ddd19b91fb1db841fc52f0783682ea6fc59780fbb4306de7b.
Both bind the diagnostic and full production source identity. Initial assertions
incorrectly demanded double-precision equality from a recurrence whose sign/live
multiplication uses float32; the observed 1.50e-9 discrepancy was numerical.
Final checks separate the analytic value from Python/native values and use
3e-8 gradient /3e-7 advantage tolerance. Earlier pre-identity reports are retained.
There is no new simulation, fitting or Koi-Koi evaluation in this calculation.

The concrete next capability is to accumulate fixed-opponent transitions until
the next learner observation or terminal outcome, retaining primitive-step
accounting and explicit unfinished-tail masking. It has not yet been implemented
or adopted. It is a decision-time reduction with known theoretical relatives,
not a new general policy-gradient theorem. The existing response study remains
unaltered and must finish under its original protocol.


## Learner-decision return implementation and CPU capability (2026-10-18)

Following the exact observer-bootstrap counterexample, two isolated scripts now
implement learner-time GAE and pretrained PPO against the fixed public WP/Point
mixture. No production or live-study input changes. Returns use next learner
observations, lambda**primitive_gap and terminal learner payoff; an unfinished
final learner action is omitted, while its initial value still bootstraps the
preceding complete transition. Match reset blocks future reward leakage.

Nine return tests pass in6.46s and six integration tests pass in22.41s, zero
skipped. The latter has six benign upstream Transformer nested-tensor warnings.
The independent forward TD expansion, real Python/native CPU/CUDA traces, role
reward conversion, frozen opponent routing, initial retained-parent inference,
CPU exact restart, optimizer momentum exclusion and source tamper rejection all
pass. Ruff and whitespace checks pass. No GPU optimizer update was tested while
the fixed response study occupies the training GPU.

CPU production-size capability session94836 exited0. Config128env x128horizon,
minibatch512,epochs3,width256,seed20295001,one warmup then two policy updates.
Actual49,152 decisions /3,219 round endings /375 match endings; learner24,521,
valid learner24,315,omitted tails206,opponent24,631. No empty minibatches. All96
parameter optimizer states finite and correctly counted:90 actor tensors192steps,
6 value tensors288steps. Warmup preserves actor and both later updates change it.
WP12,403 opponent actions /1,644 rounds /188 matches; Point12,228 /1,575 /187.
Training-only capability, no new win-rate evaluation or deployment promotion.

Audit runs/learner_response_capability_cpu/audit.json SHA
216cc2ef25fead9f58b22ccb0ed66d54169fa6c958ee6cf664b2fdb361b3a1de;
checkpoint SHA94c19d41adada8b421d11685086a3bf64bfffbc86c8a139f457b219a1eeedcde;
metrics SHAd5341d1a7e8f9e2e081a811b895b3c94e372bc3bb393bdac5f8642f832f3f910.
Full config, source/runtime/parent identities and actual parameter-step mapping
are embedded in the audit. See OBSERVER_RETURNS.rst for method, commands and
reproduction. The exported learner_response_v1 actor keeps the same public input
and greedy inference. The retained human-play model and dealer rule stay intact.

This intervention additionally removes the opponent critic, rescales learner
value loss, filters minibatches and skips empty Adam updates. A future comparison
must disclose these changes and actual valid sample counts; it cannot identify
critic information alone as the cause. Tail exclusion is also boundary-dependent.
General partial-observation critic theory is prior work. Neither this capability
nor the counterexample proves stronger Koi-Koi play. The existing response study
continues unaltered; all29 bound inputs rechecked before/after capability work.


## Third response endpoint and learner-return comparison preparation (2026-10-18)

The original parent2_response worker58580 (actual46240) exited0. It completes
520 updates /8,519,680 simulator decisions /569,090 round endings /73,432 matches.
Learner4,242,171;opponent4,277,509. WP2,130,999 opponent decisions /287,006 rounds
/36,710 matches; Point2,146,510 /282,084 /36,722. Warmup327,680 decisions /21,771
rounds /2,765 matches. Endpoint SHA
406800e63f3a740d557309edadf18a76a9e6c8abf33374fe38b071e95cf107bf.
Both original training_audit and the separate actual-Adam-budget check pass.
Optimizer audit runs/response_optimizer_audits/parent2_response.json SHA
43dfa40190b91554893fc2f5958f020f87bac97c744c85a2229685ed9c2b382d.
All102 parameter states checked:actor48,000;both critics49,920steps. Parameter
changes are30/30 discard,29/30 pick,29/30 koi,6/6 learner value,6/6 opponent value;
actor-block L2 changes0.887236/0.670185/0.578468. These are implementation checks,
not strength. The three completed conditions together record25,559,040 decisions,
1,708,559 rounds and219,942 matches. The final original self-play condition runs
as actual PID48292/wrapper50264; original evaluation and input identities remain
unchanged. No interim win contrasts or model ranking have been inspected.

LEARNER_RETURN_STUDY.rst fixes the next comparison before the original full result:
learner-time lambda0.95 and ordinary ResponseTrainer lambda1, two retained parents,
original paired training seeds and fixed520-update budget. The two completed
signed-lambda0.95 response endpoints and unchanged parents are retained controls.
All eight players face WP/Point on new24M development deals,5,000 duplicate pairs
per cell:160,000 matches on5,000 shared streams. Four fresh training conditions
add34,078,720 decisions; reused response training is separately17,039,360. Eight
positive paired win-change lower95 bounds are required (learner over parent and
over response for both parents/opponents). Lambda1 and points remain secondary.
No intermediate endpoint selection, extensions or automatic deployment.

scripts/run_learner_return_study.py implements frozen input checks, per-condition
training/accounting audits, all16 matchups, complete-pair statistics/contrasts,
owned-worker cleanup and cross-study scheduling. The lambda1 arm retains the
original opponent critic and finite-tail bootstrap. Learner-time loss scaling,
critic removal, clipping and omitted tails are explicit full-intervention changes.
The runner verifies the original training receipts before using the GPU and waits
for the original terminal analysis before launching new CPU evaluations.

Final verification command:
.venv/Scripts/python.exe -m pytest tests/test_learner_return_study.py tests/test_learner_response.py -m "not cuda" -q
Result9 passed,2 CUDA tests deselected,5 upstream Transformer warnings in59.81s.
The reduced real CPU pipeline has10 subprocess jobs,2 trained conditions with256
simulator decisions and8 evaluation cells totaling32 matches /3,640 decisions.
It reports no strength qualification. Tests cover receipts/protocols, evaluation
opponent identity, separate upstream GPU/CPU gates and launch-failure records.
Updated CPU restart test also checks all CUDA RNG states without training on GPU.
No production/current-study inputs changed; Ruff and whitespace checks pass.

The final-source smoke data were independently re-read through the frozen input,
training, scheduling and raw-statistic checks. Pipeline audit
runs/learner_return_pipeline_audit.json SHA
b25d92871a89397fe29b79698b029acc5cfa7845f03c8cdd5abfdee50d70d428.
The exact39-file smoke directory is retained as
runs/learner_return_pipeline_evidence.zip (237,143,656 bytes), SHA
07f2d0e5e2a5aefb42f9f7620f8bcf41efec137d187242cdf47643ac6eb25e26;
every member SHA and ZIP CRC verified. Historical absolute candidate paths inside
the reports are preserved; the archive prevents pytest temporary-directory cleanup
from discarding this evidence.

scripts/check_learner_gpu.py supports a single guarded transition: wait for the
existing GPU training jobs to exit successfully, check frozen verification files,
run the actual CUDA exact-restart test, then three production-size GPU updates.
Only a successful checkpoint-bound audit enables the prepared research runner.
Failure stops the sequence and retains logs; no retry or test waiver. This gate
has been prepared but its CUDA checks have not yet run.

Guarded launcher started from commit7c6c470 with
.venv/Scripts/python.exe -m scripts.check_learner_gpu --wait --launch-study
Session23277 is live; wrapper55528/actual5680 verified through process inventory
and handle polling. runs/learner_response_capability_cuda/process.json records
waiting_for_training_slot, seven frozen verification inputs and the original
protocol2342e1e80ebc79fc917891ebbf30acd7ef5dd0b4a21413d9d78e297feed57ffb.
At this observation the original parent2_selfplay is at261/520 updates. No second
GPU training job or new research evaluation has started. Keep the seven captured
files unchanged while waiting and during the follow-on study. Progress records
remain in STATUS/EXPERIMENTS, outside those frozen inputs.


## Critic-conditioning finding integrated into the manuscript (2026-10-18)

The manuscript now states both the completed match-win adaptation and the exact
observer-bootstrap diagnosis in its title/Abstract/Introduction. Section7 gives
the binary-private-signal game, payoff matrix, exact conditional critics and
reversed first expected gradient component. AppendixB derives the formula and
lambda>199/201 threshold, records all three controls, distinguishes one reversed
coordinate from the full vector, and describes learner-decision tail semantics.
The expected update is before PPO clipping/empirical advantage normalization.
Current Koi-Koi results still use the original estimator; running response and
learner-return studies are not reported as completed findings.

Related Work adds Baisero--Amato,AAMAS2022,44--52 and Lyu/Baisero/Xiao/Amato,
AAAI2022,9396--9404. Primary texts/records were checked at
https://arxiv.org/html/2105.11674v2 and
https://ojs.aaai.org/index.php/AAAI/article/view/21171;
the official AAMAS contents and paper confirm its venue/pages. The paper links
the specific example to known critic-conditioning theory without first-priority
or generally unbiased-PPO claims.

scripts/export_observer_paper_evidence.py independently derives the two gradient
coordinates and threshold with Fraction arithmetic, validates both pinned
CPU/CUDA reports and all bound source identities, and checks analytic/Python/
native numerical agreement. It produces generated/observer_evidence.json SHA
5b5419a725ff72e01678b65fbfc273fc813bcea3fec8730cc7d2cb783a780623.
No sampled games, training updates or new GPU workload were added.

Tectonic builds the17-page PDF with22 cited references and20 embedded fonts.
Abstract191 whitespace words; main conclusion page11. All previous numeric tokens
and generated result tables remain. The initial new inline gradient vector made
one overfull box; splitting its two coordinates removes the warning. All17 final
pages and enlarged equation pages8/12 were visually checked. Glyph, bounds,
font embedding, TeX warnings, citation keys and all5 evidence ledgers pass.
Prior15-page PDF retained as output/pdf/manuscript_structure_90f9d9f8.pdf.
New PDF SHAeeff7b6d9d98595692b32408d3ebae66dc49b6b06b6b8d92baf070dc5a740ef7.
QA runs/manuscript_observer_qa.json SHA
7ffb1430d37ce5acd0c1c1063f1e5d5a08af956654d8eda5a56533137595fea1.
The running original study and all seven GPU-gate verification inputs remain
unchanged. This is a manuscript/evidence milestone, not new strength evidence.


## Original training complete; CUDA gate passes and learner study starts (2026-10-18)

Parent2_selfplay finishes520 updates /8,519,680 decisions /571,259 round endings
/73,368 matches. Warmup327,680 decisions /21,851 rounds /2,752 matches. Endpoint
SHAe805e9840e16453df26e407242a4e331bd65bf93c8cd8f95e8592fe9a0c3f564.
Full training audit passes; separate optimizer audit
runs/response_optimizer_audits/parent2_selfplay.json SHA
48a052fe90085d9295931f67ce855fc7a0ff4007535d029197a48d1956897a97.
All96 parameter states agree:actor48,000,value49,920steps. Actor blocks each
change29/30 tensors, with L2 deltas0.903499/0.675680/0.586213;value6/6 changes.
All four original training jobs total34,078,720 decisions /2,279,818 rounds
/293,310 matches. Eight original evaluation jobs have exited0; the original full
analysis still waits for the remaining four, with no interim outcome ranking.

The live guarded launcher23277 proceeds only after all original training exits.
Actual CUDA test test_warmup_restart_and_frozen_weights[cuda] passes:1 passed,
0 skipped,2 upstream Transformer warnings in6.14s. The CUDA production-size
capability then finishes3 updates (1 warmup,2 policy),49,152 decisions /3,240
rounds /386 matches in11.616s of update-loop time. This time is not an isolated
throughput benchmark. Learner24,599,valid24,406,omitted193,opponent24,553;empty
minibatches0. All96 finite Adam states match actual phase budgets:actor192,
value288. WP12,176 opponent decisions /1,631 rounds /193 matches;
Point12,377 /1,609 /193. Initial actor stays fixed in warmup and changes in both
policy updates. CPU and CUDA capability runs do not claim identical samples.

CUDA audit runs/learner_response_capability_cuda/audit.json SHA
c6f9d428bba1d253ea5344994c7ffcab446dc38b99648e120b770ce91060fb17;
checkpoint SHA5f11e298ac807c65fa52d2dee2734774e0e32fdfb30ea71621dabf72f7e17750;
metrics SHA371b5b5ab371525203cc75fb51ce0166ed93a3035b8bc5d82f405aa7af0570b7.
The audit binds its exact-restart log and seven verification sources. Frozen
source checks pass before research launch; maximum old training exit <= CUDA
validation start <= capability end <= new training start was checked directly.

The prepared learner-return research runner is now live under session23277:
actual runner18512/wrapper65160,first trainer46584/wrapper55492 (parent1_learner).
Protocol SHA675638eb8561f52e2d023c7f8fedc3006eac59a051d32e3b54e8e7cb78e10674
binds51 inputs. Its first training condition has passed the20-update warmup and
reached37 updates /606,208 decisions at the recorded check. CPU research evaluation
waits for the original study's terminal audit. No new strength result is available.

Direct initial-state comparison with original parent1_response verifies exact
actor,value and environment tensors and fresh Adam states;CUDA RNG matches and
CPU RNG differs. Report runs/learner_return_initial_audits/parent1.json SHA
32682c77b9086674a7bcc7a7e3a3d3228855ea630fcb6ba09dbecb45223d9e43.
This validates initialization matching, not equal subsequent trajectories or
learner sample counts. Production, all old/new bound sources, public comparison
agents, dealer choice and the playable canonical model remain unchanged.


## Response evaluation: nine completed cells independently checked (2026-10-18)

Both parent1_selfplay cells and parent2_response versus WP have now exited0.
The independent script runs/response_completed_evaluation_audits/audit_third_three.py
checks all original protocol inputs, endpoint/receipt/raw identities, exact seed
coverage, seat-score/point consistency, match/decision accounting and recomputes
both mean outcomes and5,000-bootstrap percentile intervals directly from raw
records, without importing the production analysis helpers. All checks pass.
These three cells total30,000 matches /3,484,172 decisions on5,000 shared23M deal
streams. With the previous six audited cells, the total is90,000 matches
/10,457,475 decisions; shared streams remain5,000, not45,000 independent deals.
Round endings are not recorded in this evaluation format.

Report runs/response_completed_evaluation_audits/third_three.json SHA
3f37607c7f48b22a0152ef9f935f14f33ec343353310e02a53f222010ff1bc06 binds the
checking script and exact closed-cell artifacts. This verifies stored statistics,
not an independent replay of policy inference. No intermediate ranking, strength
qualification, model selection or protocol change was made. The original study
still has three active evaluation jobs; the new learner-return study continues
its first training condition. Both live launcher handles were polled successfully.


## Shared-parameter observer counterexample (2026-10-18)

The exact two-step game now has an additional scalar-policy calculation using
logits(theta,-theta/100). At theta=0 both original critics remain exact and
both private-signal policies depend on theta. The true gradient is3/800; the
expected signed-lambda0.95 gradient is-8301/3200000. The negative directional
derivative -24903/2560000000 establishes strict local descent for a sufficiently
small positive population update. This is a different, explicitly restricted
policy family; the original two-coordinate example remains a one-coordinate
reversal. Lambda1, learner-to-terminal and joint-information controls all recover
the exact scalar gradient. The positive-sign threshold is lambda>6567/6767.

Command: .venv/Scripts/python.exe -m scripts.diagnose_shared_observer_bootstrap
--output runs/shared_observer_bootstrap_cpu.json. The final run passes rational
outcome enumeration, direct scalar-objective autograd, fresh Python/native CPU
checks and projection of the already pinned CPU/CUDA records. It makes no new
CUDA calls, samples no games and fits no policies. Report SHA
f6e500b64ecd8505ad0a2ccf52438df9df9c7efed4dadd65fdb7b799d082536e.
An initial run caught a mistyped pinned CUDA-report hash before computation;
comparison with the unchanged file corrected the literal. No report was written
by that failed run. The source was formatted before the successful evidence run.

OBSERVER_RETURNS records the derivation and its scope. This adds a concrete
population-descent example to the established critic-conditioning issue, without
asserting new general theory or improved Koi-Koi strength. The current17-page
paper is unchanged pending incorporation with the completed strength study.


## First learner-return research endpoint complete (2026-10-18)

Parent1_learner exits0 after520 updates:20 critic warmup and500 policy updates.
It records8,519,680 simulator decisions /568,536 rounds /73,379 matches.
Warmup327,680 decisions /21,926 rounds /2,782 matches. Learner4,234,808;
valid learner4,201,750;omitted unfinished tails33,058;opponent4,284,872.
No attempted minibatch is empty. WP contributes2,137,156 opponent decisions
/286,788 whole-match round endings /36,693 matches;Point2,147,716 /281,748
/36,686. These are training counts, not evaluation wins.

Final checkpoint SHA7374420f849f134c30cc22347df907e2fd913849e4dacd74782a02d093384849.
The full runner audit passes, including exact parent initialization, frozen
opponents, complete logged budgets, environment episode counts and actual Adam
state. Initial actor/value/environment match the original response control;
CUDA RNG matches and CPU RNG differs, as previously recorded.

The existing read-only optimizer companion also runs successfully against this
learner endpoint, without modifying it or the runner:
.venv/Scripts/python.exe -m scripts.audit_response_optimizer --base
runs/learner_return_study --case parent1_learner --output
runs/learner_return_optimizer_audits/parent1_learner.json.
Report SHA1b0dbe1eda888a03924829a9cb62801a73990ce913525ea9cd8ee89b28fef75b.
This particular run attains the full fixed schedule:all90 actor parameter states
have48,000 steps and all6 value states have49,920. The independent check derives
these maxima from updates,warmup,epochs and primitive minibatch counts and checks
actual stored Adam state. Learner filtering can legitimately omit phases or
whole minibatches in other runs; attaining these maxima is an observed fact here,
not a general requirement for learner-time returns.

Changed tensor counts are30/30 discard,29/30 pick,29/30 koi and6/6 value.
L2 changes are0.932972/0.696705/0.599001/7.295458 respectively. These establish
parameter movement and finite optimizer state, not stronger play. The scheduler
proceeds to parent1_lambda1, actual trainer52576/wrapper30572 under existing
runner18512/wrapper65160 and launcher session23277. Live processes were checked.
Original CPU evaluation remains active, so new CPU evaluation remains gated.
The two study protocols, their29/51 bound inputs and production stay unchanged.


## Shared-policy descent result integrated into manuscript (2026-10-18)

The Abstract and Introduction now state the shared-parameter example's local
payoff decrease directly. Section7 uses logits(theta,-theta/100); AppendixB
retains the independent-logit coordinate reversal and derives the scalar update,
threshold6567/6767 and negative directional derivative-24903/2560000000. The
conclusion identifies learner-decision returns as the example's correction and
an empirical method to test. It does not claim stronger Koi-Koi play from the
mathematical calculation. Related Work retains the established conditioning
literature. All previous empirical numbers,tables and22 citation keys remain.

The updated evidence exporter independently projects the exact original rational
gradients and verifies the scalar diagnostic/report sources. Ledger SHA
f7472951c106e3a069e09ad89628fba75d85f82351738f92c64306a411338b6d. No new matches or CUDA
training/validation were run for this paper revision. The current studies keep
all29/51 bound inputs unchanged; both actual launcher handles remain live.
Original evaluation has10 finished cells; the new study is trainingparent1_lambda1.

Build: external/tectonic/tectonic.exe --keep-logs --outdir output/pdf
 docs/paper/manuscript.tex. An overlong inline rational product was reformatted
as a displayed directional derivative; final build has no TeX box warnings.
PDF SHA6d7406c923befcf78d5d6a89b3351f0e51ed63d1ca0540cdc037a2fa62f4ca6b.
The prior17-page PDF is retained as output/pdf/manuscript_observer_eeff7b6d.pdf.
QA runs/manuscript_shared_observer_qa.json SHA
3c97e47e27e60ff86a2e90a3610234938119591dc1d58502c60070eb0d200dc5. All18 pages were visually
reviewed through rendered contact sheets; pages1/8/12/13 were checked at full
size. All19 fonts embed correctly,22 references resolve,Abstract is199 whitespace
words,andConclusion starts on11. All prior numeric tokens,all5 evidence ledgers,
page bounds and glyph checks pass. Ruff and the actual evidence export pass.


## Fixed-opponent response study complete: no replicated win gain (2026-10-18)

Original launcher26076 exits0. All4 training jobs and12 evaluation jobs finish
with their original budgets,sources,seeds and conditions. Training totals
34,078,720 simulator decisions /2,279,818 round endings /293,310 matches.
Evaluation totals120,000 matches /13,947,308 decisions on5,000 shared unique
23M deal streams. Evaluation round endings are not recorded in this format.
No new training,checkpoint selection,sample extension or deployment promotion.

All training and actual-optimizer-budget audits pass. The new independent
checker scripts/audit_response_study.py reads the closed raw pair records and
recomputes every outcome and all12 paired contrasts using one joint set of5,000
whole-deal bootstrap resamples with seed99. It does not import the production
analysis helpers. Closed-job event sweeps verify peak1 GPU trainer /3 CPU2
evaluators and endpoint-before-evaluation ordering. All protocol inputs and
training/optimizer receipt identities are rechecked. This is record/statistic
verification,not an inference replay or a training-seed uncertainty estimate.

Command: .venv/Scripts/python.exe -m scripts.audit_response_study --output
runs/response_study/independent_audit.json. Actual run passes; Ruff passes.
The earlier preflight correctly rejected the still-running study without writing
a result. Independent audit SHAff6677bcb24ffac367b5003301be4a04236a913785d725cdd2672b2d09a3f79c.
Original final analysis SHA2c148cff866dad57baeb60302c86a7b213fb3337ad9502e2240772ab6cf0babe.

All outcomes (win score is draw-adjusted; intervals are95% paired-deal bootstrap):

| Candidate | Opponent | Win score %,95% CI | Point margin,95% CI |
| --- | --- | --- | --- |
| parent1 | WP | 51.7300 [51.00988,52.42500] | +2.4626 [+2.037520,+2.876405] |
| parent1 | POINT | 50.4900 [49.66500,51.29500] | -2.0162 [-2.510205,-1.515800] |
| parent2 | WP | 51.2550 [50.50500,51.96000] | +1.9636 [+1.538195,+2.368005] |
| parent2 | POINT | 50.0650 [49.27000,50.85000] | -2.4892 [-3.009410,-1.985385] |
| parent1_response | WP | 51.0050 [50.25000,51.73500] | +1.9938 [+1.552400,+2.417225] |
| parent1_response | POINT | 50.1300 [49.32000,50.95012] | -2.2440 [-2.734615,-1.748190] |
| parent1_selfplay | WP | 51.9800 [51.21488,52.71000] | +2.8984 [+2.441385,+3.336210] |
| parent1_selfplay | POINT | 50.2250 [49.41500,51.03000] | -1.8262 [-2.342615,-1.322395] |
| parent2_response | WP | 51.7600 [51.03487,52.49012] | +2.0454 [+1.620960,+2.462415] |
| parent2_response | POINT | 50.1250 [49.32500,50.92500] | -2.5868 [-3.096810,-2.076955] |
| parent2_selfplay | WP | 51.5800 [50.83500,52.34000] | +2.7396 [+2.298195,+3.174405] |
| parent2_selfplay | POINT | 50.8850 [50.05000,51.71500] | -1.4118 [-1.920205,-0.920395] |

All within-parent paired contrasts; win changes are percentage points:

| Candidate minus control | Opponent | Win change pp,95% CI | Point change,95% CI |
| --- | --- | --- | --- |
| parent1_response-parent1 | WP | -0.7250 [-1.57500,+0.16000] | -0.4688 [-0.956635,-0.000200] |
| parent1_selfplay-parent1 | WP | +0.2500 [-0.57000,+1.08000] | +0.4358 [-0.043620,+0.907000] |
| parent1_response-parent1_selfplay | WP | -0.9750 [-1.83000,-0.11000] | -0.9046 [-1.393015,-0.422590] |
| parent1_response-parent1 | POINT | -0.3600 [-1.20500,+0.50000] | -0.2278 [-0.699220,+0.238025] |
| parent1_selfplay-parent1 | POINT | -0.2650 [-1.10000,+0.59000] | +0.1900 [-0.279025,+0.664600] |
| parent1_response-parent1_selfplay | POINT | -0.0950 [-0.92512,+0.76500] | -0.4178 [-0.889830,+0.055425] |
| parent2_response-parent2 | WP | +0.5050 [-0.31000,+1.33000] | +0.0818 [-0.367020,+0.532685] |
| parent2_selfplay-parent2 | WP | +0.3250 [-0.51512,+1.16000] | +0.7760 [+0.324595,+1.244615] |
| parent2_response-parent2_selfplay | WP | +0.1800 [-0.69512,+1.06000] | -0.6942 [-1.168605,-0.222940] |
| parent2_response-parent2 | POINT | +0.0600 [-0.75500,+0.85500] | -0.0976 [-0.563410,+0.365605] |
| parent2_selfplay-parent2 | POINT | +0.8200 [-0.00500,+1.66000] | +1.0774 [+0.593980,+1.538615] |
| parent2_response-parent2_selfplay | POINT | -0.7600 [-1.59512,+0.10512] | -1.1750 [-1.675000,-0.661790] |

The fixed primary criterion fails:zero of its eight required lower win-change
bounds are positive. All four response-minus-parent win intervals include zero.
Parent1 response loses0.975pp [-1.830,-0.110] to its matched selfplay continuation
against WP. Response-minus-selfplay point margins have negative intervals for
both parents against WP and for parent2 against Point. These are comparisons
of the tested complete implementations,not a causal diagnosis of the return
estimator. Continued selfplay also has no positive parent-relative win lower
bound in any of its four matchups; the parent2/Point interval starts at-0.005pp.
This borderline result does not change the prespecified criterion or budget.

The already running learner-return study remains fixed. Its first fresh24M
CPU evaluation starts at1792269284.845334,after the last original evaluator
ended at1792269278.3249133 (6.520 seconds gap). Initial evaluation wrappers
63508/7976/61196 cover parent1 versusWP/Point and archived parent1_response
versusWP. Parent1_lambda1 GPU training continues under52576/wrapper30572.
All51 bound learner-study inputs remain unchanged. The complete response
results now need integration into the manuscript; the new learner-return
strength comparison remains pending.


## Completed response comparison added to manuscript (2026-10-18)

Section8 is now Further policy improvement,including a new Section8.4 for the
closed response study. Its four-row generated table includes all eight primary
paired win contrasts (response versus unchanged parent and continued selfplay
for two parents against two opponents). The text reports the failed primary gate,
the parent1/WP response-versus-selfplay win loss,and the three negative point
intervals. It distinguishes the completed response study from the ongoing
learner-return comparison and preserves the original final-study results.
Abstract,Introduction,conclusion and reproduction/accounting text are updated.

The new scripts/export_response_paper_evidence.py verifies the pinned independent
audit and all input identities,checks displayed contrasts against independent
raw-record recomputations,and writes the evidence/table. Actual export and Ruff
pass. Ledger SHAdc4a0d1e42ce10285b4d53b55ceac0b9660fb25af4b2711395dcc4b3d972d106;table SHA
7232486aa5e80203c1c044cfb7587dee15ea3f2a8e3827405c45b43f1f0932c3.

Terminology correction:reference.observe sets card feature0 from the acting
player's hand; native observe agrees,PublishedPolicyValue consumes the first560
features,and the controller receives the same observation. Manuscript references
to public inputs/features/critic now say acting-player observations,observation
features or observation critic as appropriate. Public state,endgame assistance
and released opponents keep their distinct meanings. No bound source changed.

Build: external/tectonic/tectonic.exe --keep-logs --outdir output/pdf
 docs/paper/manuscript.tex. Final PDF SHA
99db5ce08838111024016e8800bcdfced949cb3f1e752dd14a80b5449042787b. Prior18-page PDF retained as
output/pdf/manuscript_shared_6d7406c9.pdf. QA runs/manuscript_response_qa.json SHA
8b84da1e3e4b48017dd32e6e464f19a5406bc23f32e1eb28d101a4294690ee1b. All19 pages visually reviewed;
table/proof pages11/13 checked at full size. All19 fonts embed,22 references
resolve,Abstract is202 whitespace words,andConclusion begins on12. All prior
numeric tokens,six evidence ledgers and page/glyph/TeX checks pass. The51 live
learner-study inputs remain unchanged; its actual launcher handle is live.


## First lambda1 control complete; second learner arm starts (2026-10-18)

Parent1_lambda1 exits0 after520 updates (20 value warmup,500 policy). It records
8,519,680 decisions /569,058 round endings /73,731 matches. Warmup327,680
/21,926 /2,782. Learner4,214,681 andopponent4,304,999 decisions. WP contributes
2,149,053 opponent decisions /287,624 whole-match rounds /36,864 matches;
Point2,155,946 /281,434 /36,867. Endpoint SHA
91f5f7d03436674771f479b3153c2ffbf99421745a0d0177918002048fd04529.
These are training counts,not evaluation wins. The first two new endpoints total
17,039,360 decisions /1,137,594 rounds /147,110 matches.

Full runner audit passes,including exact initial actor/value/environment against
its archived original response control. Both CPU and CUDA RNG states match for
thislambda1 initialization. Subsequent trajectories are not asserted identical.
The fixed protocol and source identities remain unchanged.

A separate read-only optimizer companion now supports thelambda1 arm explicitly,
leaving the original bound optimizer checker unchanged. Command:
.venv/Scripts/python.exe -m scripts.audit_lambda1_optimizer --case parent1_lambda1
--output runs/learner_return_optimizer_audits/parent1_lambda1.json.
Actual run and Ruff pass. Audit SHAfd68f366579dc34d5f3cf4c45c3a5d1cd0ef5d99788568589f92b59d2b640611.
All102 parameter states are finite and match the budget derived from the frozen
update/minibatch/epoch settings:90 actor tensors48,000 steps,and each6-tensor
value block49,920 steps. Changed tensors29/30 in each actor block and6/6 in each
critic. L2 movements:discard1.065544,pick0.763670,koi0.599053,value7.421758,
opponent value10.808631. Movement and optimizer accounting do not establish
playing strength or an isolated return-information effect.

The scheduler starts parent2_learner only afterlambda1 exits. Actual trainer
57276/wrapper23300 is confirmed live; runner18512 and launcher23277 continue.
Direct comparison of the saved new/original parent2 initial checkpoints verifies
identical actor,value and environment tensors,fresh optimizers and zero initial
counters. CUDA RNG matches;CPU RNG differs,as inparent1_learner.
Initial checkpoint SHA784e860bea05c2be6690c1fe20f3cd8e962ee95bb488eb6b33a93100b19d62f3.
Audit runs/learner_return_initial_audits/parent2.json SHA
eac8ac2dbfccad0c862d6054fd746be8230c904e3cc32789edcb95a6f4a3aee4. The comparison is initial-state only.
Three fresh24M evaluation workers remain active; no partial outcome ranking or
strength qualification is performed. All51 live protocol inputs are unchanged.


## Both learner arms complete; final lambda1 control running (2026-10-18)

Parent2_learner exits0 after520 updates (20 value warmup,500 policy),with
8,519,680 decisions /569,574 round endings /73,584 matches. Warmup327,680
/21,771 /2,765. Learner4,236,521,valid4,203,388,omitted33,133;opponent4,283,159.
No empty minibatches. WP accounts for2,134,372 opponent decisions /287,430
whole-match round endings /36,796 matches;Point2,148,787 /282,144 /36,788.
Endpoint SHA236e1b5be66257892a389c7de7aeff89299414b0341e744a1a55fb57f0f10cc5.
The three completed new endpoints total25,559,040 decisions /1,707,168 rounds
/220,694 matches. These counts establish executed training,not stronger play.

The full runner audit passes,including initial actor/value/environment equality,
CUDA RNG equality and differing CPU RNG versus the archived original response
initialization. The existing independent optimizer helper also passes for this
learner endpoint,whose actual phase counters attain the full derived budgets:
.venv/Scripts/python.exe -m scripts.audit_response_optimizer
--base runs/learner_return_study --case parent2_learner
--output runs/learner_return_optimizer_audits/parent2_learner.json.
Audit SHAd0616b8825c48c5cacb4fc0527d728ce860abc543204c3d9074d898160ca2113.
All96 parameter states are finite:90 actor tensors48,000 steps and6 value tensors
49,920. Changed29/30 tensors in each actor block and6/6 value tensors;L2 movement
discard0.912137,pick0.681352,koi0.571533,value7.141733. No opponent critic exists
in this arm. The old helper remains unchanged as a bound source.

Parent2_lambda1 starts after the learner trainer exits;actual process52764,
wrapper50552 is confirmed. The first three completed evaluation cells cover
parent1 versusWP/Point and archived parent1_response versusWP. Their30,000
matches /3,467,481 decisions use5,000 shared24M deal streams. Read-only checks
verify exact seed coverage,raw/report/checkpoint identities,seat score encoding
and counts;no partial ranking or paired strength comparison. Receipt
runs/learner_return_study/first_three_integrity.json SHA
8876e8bbda52554a31a52fdcef26973ad18801f603f44a6fc861d3f5d4d9db07.

The new scripts/audit_learner_return_study.py prepares the final independent
raw-record recomputation:16 outcomes,24 paired contrasts,all eight primary
bounds,and joint5,000-replicate seed99 resampling across80 outcome/contrast
columns. It verifies complete old/new job schedules,global resource limits,
fixed candidate/input identities and separate stored-Adam receipts. It requires
a terminal audited study;the actual running-study rejection check passes and
creates no output. Ruff passes. Full execution awaits all16 cells and the final
training endpoint;the routine has not yet established any strength result.
All51 live protocol inputs and production remain unchanged. No deployment
promotion,additional experiment or manuscript change at this milestone.


## All four learner-return study training endpoints complete (2026-10-18)

The final parent2_lambda1 job exits0 after520 updates (20 critic warmup,500
policy),8,519,680 decisions /567,462 round endings /73,775 matches. Warmup
327,680 /21,771 /2,765. Learner4,205,471 andopponent4,314,209 decisions. WP
contributes2,153,194 opponent decisions /286,965 whole-match round endings
/36,885 matches;Point2,161,015 /280,497 /36,890. Final checkpoint SHA
9e09ae504832c19194b58bf85c0604ad8e703d646efa78f6fedd770a82cd7370.
Full runner audit passes,including initial actor/value/environment equality
and matching CPU/CUDA RNG against the archived original response initialization.

Independent actual optimizer command:
.venv/Scripts/python.exe -m scripts.audit_lambda1_optimizer
--case parent2_lambda1
--output runs/learner_return_optimizer_audits/parent2_lambda1.json.
Audit SHA324571f25cf1724fee15098245024acd4cc2fe591c681f796904bc6e16dea39f.
All102 parameter Adam states are finite:90 actor tensors48,000 steps,6 learner
value and6 opponent-value tensors49,920 steps each. Changed29/30 tensors in
all three actor blocks and6/6 in each critic. L2 movement:discard1.043662,
pick0.748878,koi0.571401,learner value7.374919,opponent value10.772649.

All four new endpoints now total34,078,720 decisions /2,274,630 round endings
/294,469 matches. Warmup is1,310,720 /87,394 /11,094;learner decisions16,891,481,
opponent17,187,239. These are new-study training counters. The archived two
response endpoints'17,039,360 training decisions remain separate dependencies;
none of these counts is fresh strength evaluation or search work.

All four full runner audits and separate stored-Adam audits pass. Actual hashes
of every optimizer receipt input and final checkpoint are reverified. All four
training jobs exit0 with non-overlapping intervals;Win32_Process confirms the
final52764/wrapper50552 trainer is gone. Runner18512/wrapper65160 and three
CPU2 evaluators remain live under launcher23277. All51 protocol-bound inputs
and production SHA2b718198a4f7bc640f457c9fc12311dcb9a8689fba1ffcb09411234ba0920cdb are unchanged.

The fixed160,000-match evaluation continues. No partial outcome selection,
training extension or deployment promotion is performed. The complete-study
independent auditor is prepared but awaits all16 cells;no new method strength
conclusion or manuscript revision is made at this training milestone.


## Final-only learner-study paper export prepared (2026-10-18)

scripts/export_learner_paper_evidence.py requires a terminal study and an explicit
--audit-sha256 matching the final independent audit. It binds the fixed675638eb...
protocol,verifies every audited input,and checks all16 outcome and24 contrast
win/point means and intervals against independent recomputation. It recomputes
the eight-bound primary gate without assuming a positive or negative result.
New training totals and reused response-control training remain separate.

The intended outputs are learner_primary_table.tex (learner-minus-parent and
learner-minus-response),learner_lambda1_table.tex (lambda1-minus-parent and
learner-minus-lambda1),and learner_evidence.json under docs/paper/generated.
The ledger retains all outcomes/contrasts,including those not displayed in the
two compact tables. Different existing output bytes are rejected before writing;
identical regeneration is allowed. The independent audit hash is supplied only
after the real completed audit exists,not inferred from partial results.

Ruff formatting/checks pass. Calling export with a fresh nonexistent output
directory against the actual running study raises the expected incomplete-study
error and creates neither a directory nor evidence files. This validates the
completion gate only;successful full-data export remains pending. All51 bound
study inputs remain unchanged. No result table,PDF revision,new simulation,
training or strength claim is produced by this preparation.


## Pretrained koi-search capability (2026-10-18)

An isolated scripts/published_koi_search.py closes the retained Transformer search-adapter gap without changing production code or the running learner protocol. The original strict CPU2 greedy parent remains the fallback and supplies all card actions. A separate frozen batched actor completes paired stop/continue branches from observation-consistent sampled hidden worlds. Both seats use that continuation; the uniform proposal does not condition on opponent-action likelihoods. Fresh confirmation particles can accept only the selected alternative. See PUBLISHED_KOI_SEARCH.rst for the complete information/inference contract and limitations.

Validation command: `.venv/Scripts/python.exe -m pytest tests/test_published_koi_search.py -q`:12 passed,0 skipped in5.74s,with5 expected upstream Transformer warnings. CPU/CUDA tests compare each actual branch action through independent reference transitions,terminal points and work counts at both root seats; hidden-world substitutions preserve observations/actions/statistics. All non-koi phases retain the original strict fallback. Confirmation routing and budget rejection are covered. Ruff passes after test-only style corrections.

Capability command: `.venv/Scripts/python.exe -m scripts.check_published_koi_search --output runs/published_koi_search_capability`; launcher49769 exits0. Fixed16 greedy-play unguarded koi roots are shared across both parents;32 proposal particles,32 optional confirmation particles,confidence2.5,two warmups and16 timed calls per parent. Source238 decisions/18 round endings/2 completed matches. Parent1 warmup12,620 decisions/957 rounds/128 terminal branches;timed64,888/5,284/1,024. Parent2 warmup12,589/956/128;timed65,630/5,280/1,024. Total search155,727 decisions/12,477 round endings/2,304 branch completions. No fitting or independent strength matches.

Timed means0.3191641813/0.3303635187 seconds,medians0.34126485/0.35461375,maxima0.52851810/0.57487820. No timed proposal or action change;confirmation work is zero. These are small-corpus cost measurements with another CPU evaluator active,not isolated latency or evidence of improvement. Report SHAe18e8fe1e963a01b71a9b48698dc76bd7ce29f4e94c319c6e82ffa9d47823749. All51 live learner-study inputs reverified unchanged. No search strength study or deployment promotion; its next protocol remains dependent on the completed learner-return result.


### Additional direct literature check for the manuscript revision

Primary source https://content.scirp.org/pdf/jcc_2017122809020006.pdf identifies Takaoka,Kawakami and Ooe's2018 Journal of Computer and Communications6:32--41 article,DOI10.4236/jcc.2018.61004. Its Section5 changes UCT choices when ahead to keep match scores near zero and studies player enjoyment; it does not evaluate the present terminal-win adaptation. The draft Related Work adds this objective-dependent Hanafuda precedent. The2019 biological-information follow-up was inspected but not added because it does not substantively inform the current method.

The current public Open Hanafuda AI README was reviewed and the already pinned local snapshot01b4e484ac3d223f4ae8198d2a399f69078174d2 rechecked (commit2026-05-23). The draft now cites its simulation/observable-information/heuristic approach and distinguishes its rule variant and internal benchmark from a head-to-head comparison. The existing COMPARATORS.md review remains applicable. No new opponent was installed,evaluated or ranked. These additions bring the draft citation set to24; final typesetting awaits the complete learner study.


## Pretrained koi-search on-policy root diagnostic (2026-10-18)

The fixed scripts/diagnose_published_koi_search.py collects32 own-policy unguarded koi roots per parent from source matches against an alternating WP/Point whole-match schedule. Both32 and256 particle budgets use all roots,confidence2.5,and independent equal-size confirmation when proposed. Source seeds25930001+ for parent1 and25940001+ for parent2; search seeds25950001+parent_index*1000+root_index. Root collection occurs before branch outcomes. Smaller sample streams are prefixes of larger ones; no independent-replicate claim. No fitted policy or full-match search qualification.

Command: `.venv/Scripts/python.exe -m scripts.diagnose_published_koi_search --output runs/published_koi_search_roots`; session33432 exits0. Parent1 source765 decisions/49 round endings/6 matches;parent2 source533/35/4. Total1,298/84/10,with each final source match unfinished. All four conditions have zero proposals and zero changes;confirmation work is zero. Search work per parent/budget: parent1/32=127,116 decisions/10,347 rounds/2,048 branches;parent1/256=1,022,446/82,944/16,384;parent2/32=137,750/10,866/2,048;parent2/256=1,104,401/87,105/16,384. Total2,391,713 decisions/191,262 round endings/36,864 terminal branches across64 distinct roots and128 budget/root comparisons.

Mean call seconds0.3438674563/0.6300557312 for parent1 at32/256 particles,and0.3580037812/0.6618898063 for parent2. Medians0.34996850/0.63217375 and0.36323070/0.63315000. Cold first calls and concurrent CPU activity are included,so this is not isolated deployment timing. Main report SHA53fe36d4ea385befd06c3b87612b1c665587f8ad2f5216298484d98ca1504a18 retains raw terminal point pairs for every call. Independent standard-library recomputation verifies all128 margins/choices,branch/work sums,and bound identities: receipt SHA0b045fce313508d3128ac99d7a9189d89382690c669955f151aab407a686b769. Among1,024 shared-prefix paired worlds per parent,one point pair differs between batch sizes for each parent;zero win pairs differ. These are continuation numerical differences,not independent samples.

Disposition: there is no observed action-change signal supporting a costly koi-only match study with either setting. No post-outcome confidence reduction,match-strength claim or deployment promotion. This bounded result does not rule out all koi search. Broader card-action search remains a distinct untested direction; the next intervention depends on the completed fixed learner-return comparison. Ruff passes for the diagnostic script. The existing search capability tests remain unchanged and their earlier12-test result is retained.


## Completed learner-return study and revised manuscript (2026-10-18)

Guarded launcher23277 exits0 after all4 training jobs and16 evaluation cells. No checkpoint selection,sample extension,interim ranking or deployment promotion. Protocol675638eb8561f52e2d023c7f8fedc3006eac59a051d32e3b54e8e7cb78e10674 retains all51 bound inputs; production unchanged. Training34,078,720 decisions /2,274,630 rounds /294,469 matches; evaluation160,000 matches /18,542,184 decisions on5,000 shared unique24M deal streams. Reused signed0.95 response training17,039,360 decisions remains a separate dependency. Evaluation round counts were not recorded.

Analysis SHA4934001ea55d09f3248d20bdf36052d2e87534d6579d9095674921a751dedec1. Command `.venv/Scripts/python.exe -m scripts.audit_learner_return_study --output runs/learner_return_study/independent_audit.json` exits0 in7.14s; audit SHA1aaee8bd9885be94ec104b9d4332cf3d6b56d59cc143bd1941795760e2a93b45. It independently checks all raw seat outcomes,exact common seeds,checkpoint/raw/report identities,stored optimizer receipts,and the combined closed-job schedule,then recomputes all16 outcomes/24 contrasts with joint5,000-replicate seed99 resampling. It does not replay every inference or trajectory.

Primary criterion fails:1/8 positive lower win-change bounds. The sole passing comparison is learner versus parent1 against Point. All eight primary means are positive. No substitution of the secondary lambda1/point-margin results for this primary decision,and no use of the101M reserved final range.

### Absolute endpoint outcomes

| Endpoint/opponent | Win score %,95% CI | Candidate-minus-opponent points,95% CI |
| --- | --- | --- |
| parent1/rl-wp-assisted | 51.340000 [50.605000,52.045000] | 2.115800 [1.692115,2.532845] |
| parent1/rl-point-assisted | 50.955000 [50.180000,51.780000] | -1.278600 [-1.771610,-0.770755] |
| parent1_response/rl-wp-assisted | 51.350000 [50.605000,52.090000] | 2.089000 [1.649770,2.540015] |
| parent1_response/rl-point-assisted | 51.355000 [50.560000,52.150000] | -1.504200 [-2.012250,-0.995160] |
| parent2/rl-wp-assisted | 51.080000 [50.325000,51.805000] | 2.077400 [1.654585,2.500225] |
| parent2/rl-point-assisted | 51.155000 [50.360000,51.925125] | -1.464600 [-1.977800,-0.966960] |
| parent2_response/rl-wp-assisted | 50.520000 [49.780000,51.235000] | 1.690200 [1.258390,2.111420] |
| parent2_response/rl-point-assisted | 51.450000 [50.679875,52.235000] | -1.667400 [-2.169200,-1.149780] |
| parent1_learner/rl-wp-assisted | 51.405000 [50.655000,52.150000] | 2.233800 [1.811985,2.648610] |
| parent1_learner/rl-point-assisted | 51.875000 [51.085000,52.665000] | -0.884600 [-1.384605,-0.378380] |
| parent1_lambda1/rl-wp-assisted | 50.530000 [49.775000,51.305000] | 1.965800 [1.514980,2.422420] |
| parent1_lambda1/rl-point-assisted | 50.815000 [50.015000,51.625125] | -1.508800 [-2.013865,-0.978975] |
| parent2_learner/rl-wp-assisted | 51.160000 [50.390000,51.910000] | 2.490200 [2.036165,2.940205] |
| parent2_learner/rl-point-assisted | 51.480000 [50.690000,52.285000] | -1.031200 [-1.526265,-0.538750] |
| parent2_lambda1/rl-wp-assisted | 50.205000 [49.454875,50.935000] | 0.896600 [0.451800,1.351430] |
| parent2_lambda1/rl-point-assisted | 50.465000 [49.685000,51.260000] | -2.579600 [-3.084475,-2.067390] |

### All paired contrasts

Win changes are percentage points. Intervals describe common-deal uncertainty conditional on endpoints,not all training-seed variation.

| Contrast/opponent | Win change pp,95% CI | Point-margin change,95% CI |
| --- | --- | --- |
| parent1_learner-parent1/rl-wp-assisted | 0.065000 [-0.795125,0.895000] | 0.118000 [-0.330205,0.572405] |
| parent1_lambda1-parent1/rl-wp-assisted | -0.810000 [-1.725125,0.155000] | -0.150000 [-0.656010,0.399645] |
| parent1_response-parent1/rl-wp-assisted | 0.010000 [-0.825000,0.890000] | -0.026800 [-0.503200,0.459615] |
| parent1_learner-parent1_response/rl-wp-assisted | 0.055000 [-0.870000,0.940250] | 0.144800 [-0.356820,0.631435] |
| parent1_learner-parent1_lambda1/rl-wp-assisted | 0.875000 [-0.075000,1.780000] | 0.268000 [-0.263605,0.799205] |
| parent1_lambda1-parent1_response/rl-wp-assisted | -0.820000 [-1.830000,0.170000] | -0.123200 [-0.690250,0.420830] |
| parent1_learner-parent1/rl-point-assisted | 0.920000 [0.125000,1.755000] | 0.394000 [-0.074200,0.875410] |
| parent1_lambda1-parent1/rl-point-assisted | -0.140000 [-1.025000,0.760000] | -0.230200 [-0.752410,0.286815] |
| parent1_response-parent1/rl-point-assisted | 0.400000 [-0.425000,1.245000] | -0.225600 [-0.720810,0.269610] |
| parent1_learner-parent1_response/rl-point-assisted | 0.520000 [-0.325000,1.365000] | 0.619600 [0.117155,1.107405] |
| parent1_learner-parent1_lambda1/rl-point-assisted | 1.060000 [0.155000,1.955000] | 0.624200 [0.094585,1.143425] |
| parent1_lambda1-parent1_response/rl-point-assisted | -0.540000 [-1.455000,0.400000] | -0.004600 [-0.569400,0.550425] |
| parent2_learner-parent2/rl-wp-assisted | 0.080000 [-0.760000,0.950125] | 0.412800 [-0.081420,0.918810] |
| parent2_lambda1-parent2/rl-wp-assisted | -0.875000 [-1.835000,0.090000] | -1.180800 [-1.723860,-0.628180] |
| parent2_response-parent2/rl-wp-assisted | -0.560000 [-1.395000,0.245000] | -0.387200 [-0.834605,0.056440] |
| parent2_learner-parent2_response/rl-wp-assisted | 0.640000 [-0.235000,1.540000] | 0.800000 [0.293790,1.315610] |
| parent2_learner-parent2_lambda1/rl-wp-assisted | 0.955000 [0.010000,1.920000] | 1.593600 [1.021200,2.137410] |
| parent2_lambda1-parent2_response/rl-wp-assisted | -0.315000 [-1.320000,0.650125] | -0.793600 [-1.344000,-0.219555] |
| parent2_learner-parent2/rl-point-assisted | 0.325000 [-0.480125,1.165000] | 0.433400 [-0.032610,0.915415] |
| parent2_lambda1-parent2/rl-point-assisted | -0.690000 [-1.615125,0.225000] | -1.115000 [-1.664210,-0.567990] |
| parent2_response-parent2/rl-point-assisted | 0.295000 [-0.505000,1.100000] | -0.202800 [-0.679230,0.267405] |
| parent2_learner-parent2_response/rl-point-assisted | 0.030000 [-0.820000,0.860125] | 0.636200 [0.146170,1.133270] |
| parent2_learner-parent2_lambda1/rl-point-assisted | 1.015000 [0.104875,1.945125] | 1.548400 [1.000765,2.104810] |
| parent2_lambda1-parent2_response/rl-point-assisted | -0.985000 [-1.900000,-0.065000] | -0.912200 [-1.475820,-0.347200] |

Secondary learner-minus-lambda1 win intervals are positive in3/4 matchups,including Point at both parents. Learner-minus-response point intervals are positive in3/4 matchups,including Point at both parents. All four learner-minus-parent point intervals contain zero. The lambda1 arms have negative parent-relative win means in all4 matchups;parent2 also has negative parent-relative point intervals against both opponents. These results distinguish the complete procedures without isolating information from optimization differences.

Paper export command `.venv/Scripts/python.exe -m scripts.export_learner_paper_evidence --audit-sha256 1aaee8bd9885be94ec104b9d4332cf3d6b56d59cc143bd1941795760e2a93b45` exits0. Ledger SHAddcfee08ee8b0bd9ab1656b7e9a9ff9defa08df70db87caeaa8278dc6e4b992e binds all outcomes/contrasts and both generated tables.

Revised manuscript:20 pages,24 cited references,19 embedded fonts,Abstract168 whitespace words. Composition experiments move intact to AppendixF; completed response/learner studies occupy main Section8. Main conclusion starts on page10. All prior numeric tokens are retained across source and generated tables;old citation set preserved plus two direct Koi-Koi references. All7 evidence ledgers reverified. All20 rendered pages reviewed,table/proof pages10/12 inspected in detail;no TeX box/reference/glyph/bounds defects. PDF SHA79dd722cccdaea1491865e280c47cf347ff4b1af08c58a731100d0a3af6c95fc;QA runs/manuscript_learner_qa_v2.json SHA766d718d30b9c450a32aeb232f3569411fecd45fae74d3729f5827b34007cde5. Prior verified19-page PDF retained as output/pdf/manuscript_response_99db5ce0.pdf. The broader user-requested strength/new-method extension remains active; this completed study does not satisfy the new replicated parent-relative win-gain objective.

A final cross-reference correction replaces a stale relative "above" with AppendixF after the composition material moved. Only rendered page16 changed;it was reinspected,and the other19 PNGs match the already reviewed renders exactly. Final QA is the v2 receipt above.

## Fixed-observer self-play capability (2026-10-18)

Question: does keeping each observer's private/public information convention across opponent turns provide a better policy return in self-play? Ordinary per-agent critics are established prior work; see docs/OBSERVER_SELFPLAY.rst and Lyu et al., JAIR77 (2023),295-354, https://arxiv.org/html/2408.14597v1 . No first-general-critic or universally unbiased finite-observation claim.

Isolated implementation: ObserverFeatures builds independent [lane,seat,561] inputs from observation-only scratch projections, including an actual-actor role bit. Scratch states never execute transitions. observer_gae uses fixed-seat rewards and values at every primitive step,with match-reset masks and no round termination. The actor is unchanged. ObserverTrainer uses one shared row-wise MLP and mean value loss over both viewers. Its signed control changes policy advantages only. Parent value weights are copied with a zero role column; allocation preserves the original sampler RNG. Training format is observer_selfplay_v1; ObserverAgent preserves greedy CPU published inference and public endgame assistance.

Commands and validation:

- `.venv/Scripts/python.exe -m pytest tests/test_observer_selfplay_returns.py -q`:9 passed,0 skipped,3.21s. Independent forward-expanded return oracle; exact scalar gradient; CPU/CUDA all-phase Python/native observations and hidden-world isolation; terminal/reset masking. First run had2 fixture tolerance failures because the unchanged signed-GAE implementation computes its eligibility factor in float32 even for float64 test values. The signed-gradient tolerance was corrected to2e-9; algorithm unchanged. Fixed-observer gradient remains checked at1e-15.
- `.venv/Scripts/python.exe -m pytest tests/test_observer_selfplay.py -q`:6 passed,0 skipped,12.74s. CPU/CUDA matched warmup, exact model/Adam/environment/RNG restart for both modes, reference collector rewards/returns, initial parent actor/value weights and CPU RNG equality, unsupported-control rejection.
- `.venv/Scripts/python.exe -m pytest tests/test_observer_selfplay_agent.py -q`:3 passed,0 skipped,5.37s. Parent-equivalent public actions across all phases, hidden-hand/stock/RNG invariance, both capability endpoint actors retained. Expected upstream Transformer warnings only. Ruff and diff checks pass.
- `.venv/Scripts/python.exe -m scripts.check_observer_selfplay --output runs/observer_selfplay_capability_cuda`:session85576 exits0. Fixed128 environments x128 horizon,512 minibatch,3 epochs,width256,T0.05,actorLR3e-6,valueLR1e-4,seed26040001. Both modes use20 value-only+2 policy updates. No strength evaluation or checkpoint selection.

| Mode | Decisions | Rounds | Matches | Value rows | Warmup decisions | Warmup rounds | Warmup matches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| observer | 360448 | 24230 | 3047 | 720896 | 327680 | 22011 | 2767 |
| signed | 360448 | 24228 | 3047 | 720896 | 327680 | 22011 | 2767 |

Total720896 decisions,48458 round endings,6094 completed matches,1441792 value rows. Branch rollouts0. Test fixtures/inference checks are separate from these capability counts. Full model, optimizer, environment, counters, CPU RNG and CUDA RNG digests match exactly after20 warmup updates. Both endpoints have96 finite Adam states:90 actor parameter tensors each192 steps,6 value tensors each2112 steps. Both policies change after warmup. Elapsed update loops36.70/36.30s are capability timings,not isolated deployment latency.

Audit SHAe6a9e88bd6c965a522d8740beb5e9981d10c5f6b35cf03c5f002702bda314580. Observer final SHA f9ad0bfc3533d8e11e9e6980e9aef91ad1b0a23f89784d7492d0e04d3ab0fbb1; signed final SHA ab702d89f03ac5ff6343cf488b9c6f5f126953f09f6e8812d9db7203685e830e. Warmup checkpoints SHA cab7435a019a84308ae4b1998be88ead922b02146d6d4556c67773191d83a849 /41ea5ef848e792e072da566c74c5eea900bf35fd16b0d531715a6ad3f24db88a. Their serialization/configuration differ by experimental mode; compared learning-state contents are identical. Production SHA2b718198a4f7bc640f457c9fc12311dcb9a8689fba1ffcb09411234ba0920cdb and all51 completed learner protocol inputs reverified before/after. No paper rebuild,promotion or reserved101M use. A separately fixed strength comparison remains necessary.

## Fixed observer self-play comparison preparation (2026-10-18)

Protocol docs/OBSERVER_SELFPLAY_STUDY.rst fixes two retained parents x observer/signed policy-return arms. Each520 updates,20 warmup+500 policy,128x128 primitive decisions,512-row minibatches x3 epochs,width256,T0.05,actorLR3e-6,valueLR1e-4,lambda0.95. Training seeds19910001/20010001 match archived ordinary selfplay; initial actor/environment/CPU+CUDA RNG equality is checked before full training,and saved within-parent warmup equality is checked before signed policy updates. New total34,078,720 decisions /68,157,440 value rows. Match/round counts remain actual measured outputs. Archived ordinary-self-play dependency17,039,360 decisions /1,141,291 rounds /146,387 matches is reused separately.

Eight players (four new plus two parents and two ordinary-selfplay controls) x WP/Point x5,000 duplicate pairs on shared27,000,000--27,004,999:160,000 evaluation matches and5,000 independent deal units. Primary all8 observer-minus-parent and observer-minus-signed paired win lower95 bounds positive. All24 contrasts reported; ordinary-selfplay and points secondary.5,000 bootstrap resamples,seed99. No sample extension,ranking,promotion or101M reserved test use. OneGPU trainer/threeCPU2 evaluator limits,source/checkpoint bindings,full optimizer audits and raw seat records.

Runner validation `.venv/Scripts/python.exe -m pytest tests/test_observer_selfplay_study.py -q -s`:3 passed,0 skipped,32.12s. Includes real10-process pipeline,two training arms,eight evaluation cells,matched warmup,artifact/protocol tampering rejection and failed-launch cleanup. Final pipeline work256 training decisions/12 rounds/0 completed matches;32 evaluation matches/4,269 decisions. All12 reduced contrasts computed but not used for strength or configuration selection. Earlier pre-cleanup pipeline also passes3 tests/32.08s and has the same separate256 training decisions and32 matches/4,269 evaluation decisions. The final rerun followed removal of an unused import and unused API flag; both runs remain accounted for.

Final original directory C:/Users/sheeta/AppData/Local/Temp/pytest-of-sheeta/pytest-296/test_real_pipeline_and_tamperi0/pipeline. Faithful ZIP runs/observer_selfplay_pipeline_evidence/pipeline.zip SHA5a2382713ac14e34e0800845060a6fb73ce286e88b52a0ef38b877174b08895f retains original absolute paths;restore to original directory to run unchanged audits. Receipt audit.json SHAccc6798fc0c26f5a00ae023ce0b10eb5fe5d8426a63bec2ddb28383830b4e72a;pipeline protocol SHA84d315dd25908edbad63a7d94bb0745704e805e74d0bcdad0636fcb8f18057dd,analysis SHAbe06165ccedd60b354fc8b173edfc25d37e023145c431f870330fb5f36839693. Ruff and diff checks pass. Full research comparison prepared;no new strength conclusion.

Full launch: `.venv/Scripts/python.exe -m scripts.run_observer_selfplay_study`,session1477 remains active. Protocol runs/observer_selfplay_study/protocol.json SHAed98449175207a42782d1083e9efa69f4c76c6d43af4464587f83c8b8963c3ef,57 bound inputs,code milestone028af02. Initial jobs:parent1_observer GPU trainer3536 and CPU2 evaluators1904/52124/50780 (parent1 against WP/Point,and archived parent1_selfplay against WP). Parent1 initial actor/environment/CPU+CUDA RNG checks pass before updates;first18 warmup updates execute294,912 decisions/19,911 rounds/2,494 matches. These are a live progress snapshot,not final accounting. All57 inputs reverified. No endpoint or new strength estimate yet;do not restart the active directory or edit bound sources.

## Observer self-play independent auditor preparation (2026-10-18)

Prepared scripts/audit_observer_optimizer.py and scripts/audit_observer_selfplay_study.py without changing any running study input. The optimizer auditor derives actor/value Adam steps from fixed updates,warmup,minibatches and epochs; checks every saved parameter and moment,zero-column parent initialization,initial RNG/environment against archived ordinary selfplay,and value-only warmup. It records parameter movement without interpreting it as strength. The result auditor independently reconstructs all seat-score/point pairs,uses joint5000-replicate seed99 bootstrap across all outcomes/contrasts,and verifies primary criteria,matched saved warmups,source identities and closed-job schedule. Full-study execution requires terminal status and per-case optimizer receipts. Explicit --allow-smoke permits only separately labeled reduced-pipeline verification;its qualification result is always null.

`.venv/Scripts/python.exe -m pytest tests/test_observer_audits.py -q`:4 passed,0 skipped,8.26s. Tests use the archived actual two-arm/eight-cell subprocess pipeline;verify saved Adam states and statistics;reject premature execution,wrong value-row totals,misreported win statistics,modified Adam steps and output overwrites. No new training,matches,rollouts or inference replay. Ruff passes.

Applied both auditors to the archived pipeline and retained:

- runs/observer_selfplay_pipeline_evidence/optimizer/parent1_observer.json SHA f77feb34b808b920da861850e73d51e5bdf296c3f6f530f3ef6ef534dbe13fd1.
- Corresponding parent1_signed.json SHA e225b6e0853de8c952cd5d4e2527d87a32886815af246151fd7c68efee238a19.
- runs/observer_selfplay_pipeline_evidence/independent_audit.json SHA92504a6c8b944c0427e0d1a1ce2bd9b51dd0f02e2b8d90aca88f91bf252cc0e7.

The audit verifies existing256 training decisions/12 rounds/0 completed matches/512 value rows and32 evaluation matches/4,269 decisions on2 shared deal streams. All8 absolute outcomes and12 reduced contrasts agree;both warmup checkpoints agree in model,Adam,environment,counters and CPU/CUDA RNG state. Actor tensors each have2 Adam steps,value tensors4. Scope is pipeline verification,not evidence of a stronger policy.

Full launcher1477 was re-polled and confirmed live;native process inspection also finds the training/evaluation worker processes. At the subsequent progress snapshot,parent1_observer reaches109/520 updates,1,785,856 decisions/119,532 rounds/15,266 matches. Three control evaluators hold662/675/668 complete pairs respectively. These partial counts are operational progress only;no outcome-based change,extension or promotion. Full protocol remains SHAed98449175207a42782d1083e9efa69f4c76c6d43af4464587f83c8b8963c3ef with57 unchanged inputs.

## Finite-rollout observer-error decomposition (2026-10-18)

Added docs/OBSERVER_TRACE_IDENTITY.rst and scripts/diagnose_observer_trace_identity.py. With root-payoff rewards r,sign-converted active-observer values W,full remaining return G,and n rollout steps,the standard GAE expansion is A=G0-W0+(1-lambda)sum[k=1..n-1]lambda^(k-1)(Wk-Gk)+lambda^(n-1)(Wn-Gn). The proof is pathwise telescoping after the player-change sign product. Multiplying by the root score and taking expectations separates internal observer-conditioning error from finite-cutoff error. A terminal cutoff has zero boundary error;lambda1 at a nonterminal cutoff generally retains it.

The note states the exact-history tower-property condition and an approximation-error bound. It explicitly separates a player's policy-gradient contribution with the opponent held fixed from the combined shared-parameter zero-sum self-play update,which is not the gradient of the identically zero sum of both rewards. Finite561-feature observations are not proved sufficient histories. Foundations:Schulman et al. GAE https://arxiv.org/abs/1506.02438 and Lyu et al. JAIR77 (2023),295-354 https://arxiv.org/html/2408.14597v1 . No new general GAE/critic-bias theorem claimed.

Command `.venv/Scripts/python.exe -m scripts.diagnose_observer_trace_identity --output runs/observer_trace_identity_cpu.json` exits0 in2.91s. All6,048 rational checks pass:rollout lengths1--6,all player sequences including repeated turns,three reward/value fixtures,two terminal/cutoff cases and lambdas0,0.37,0.95,1. On the existing eight-atom private-bit game,the score-correlated difference E[S(m_A-mu_XA)] is exactly-20301/160000. Full two-step lambda0.95 gives bias-20301/3200000 and gradient-8301/3200000,matching the prior result. A one-step cutoff gives signed gradient-19701/160000 for every lambda,including1;exact root-history values give true3/800 for both cutoff types and every tested lambda. Actual CPU return functions agree across all16 cases,max gradient error1.512542356120472e-9;direct scalar autograd agrees with3/800. The float32 eligibility factor in the existing signed implementation accounts for the tiny discrepancy.

Report SHA830420c471986c33b84bfd3a9a8d2b86f7aaef5bd343637856d4e682b4006d60 binds the new note,diagnostic and existing observer-return implementation. Production SHA2b718198a4f7bc640f457c9fc12311dcb9a8689fba1ffcb09411234ba0920cdb and all57 live observer-study inputs verified before/after. No sampled matches,training,search branches,CUDA computation or manuscript rebuild. This result supports the finite-cutoff interpretation of lambda1,not a causal explanation of its measured Koi-Koi outcome.

Launcher1477 re-polled and remains live. Operational snapshot:parent1_observer194/520 updates,3,178,496 decisions/212,520 rounds/27,201 matches;one trainer and three evaluators still active. No outcome peeking,configuration change,promotion or reserved101M use.

## Final-only observer paper exporter preparation (2026-10-18)

Added scripts/export_observer_selfplay_paper_evidence.py. It pins observer protocol SHAed98449175207a42782d1083e9efa69f4c76c6d43af4464587f83c8b8963c3ef and requires an explicit lowercase final independent-audit SHA. Before creating outputs,it verifies terminal/full-study status,all16 outcomes/all24 contrasts against the independent recomputation,eight primary bounds,160,000 evaluation matches/5,000 shared deals,34,078,720 new decisions/68,157,440 value rows,reused17,039,360 ordinary-selfplay decisions,and exact matched-warmup identities. Point margin keeps its native units;win changes are percentage points. It accepts either a passed or failed primary gate without relabeling the outcome.

The exporter additionally pins the finite-rollout identity report SHA830420c471986c33b84bfd3a9a8d2b86f7aaef5bd343637856d4e682b4006d60 and independently recomputes all16 exact toy expectations from true gradient3/800 and score-correlated error-20301/160000. It preserves the distinction between established GAE algebra,the concrete counterexample and unproven practical strength. Planned outputs:observer_selfplay_primary_table.tex,observer_selfplay_points_table.tex,observer_selfplay_historical_table.tex,and observer_selfplay_evidence.json. Existing differing evidence is never overwritten.

`.venv/Scripts/python.exe -m pytest tests/test_export_observer_evidence.py -q`:3 passed,0 skipped,0.17s. Guards reject running and reduced-pipeline evidence;hand-chosen display fixtures verify percent-point versus raw-point conversion. A direct call against the currently live study also raises 'The observer self-play study is not complete' and creates no generated paper ledger. Ruff/diff checks pass. Full successful export is not yet executed or claimed;it depends on the completed study and final independent audit. No new simulation,training or manuscript/PDF rebuild. All57 running-study inputs reverified unchanged.

Exporter name collision corrected before any paper export:the initial implementation reused the existing scripts/export_observer_paper_evidence.py path,which belongs to the earlier exact observer-bootstrap evidence. Restored that file byte-for-byte to its ledger-pinned SHA1ae51baf49ad85a59b18f3d80bf7e8bf518e07d8aa7937c14bb5e4e1c966fe4a,and moved the new implementation to scripts/export_observer_selfplay_paper_evidence.py. Updated test imports and only the new documentation references. Final3 focused tests pass again,0 skipped,0.15s;ruff passes. Independently rechecked all inputs/outputs in the7 existing paper evidence ledgers (387 unique files),the unchanged PDF SHA79dd722cccdaea1491865e280c47cf347ff4b1af08c58a731100d0a3af6c95fc,and all57 live-study inputs. No existing evidence table,ledger,PDF or training/evaluation artifact was rewritten. The study and original paper remain reproducible.

## Public seat-information diagnostic and first observer endpoint (2026-10-18)

Added scripts/diagnose_observer_seat_information.py, scripts/audit_observer_seat_information.py and docs/OBSERVER_SEAT_INFORMATION.rst. The existing ordinary value input retains relative dealer identity but lacks the absolute seat/dealer context in published actor rows50/51. Swapping both seat labels produces exactly equal corresponding561-value observations and legal/endgame masks while changing only these two published rows. Transformed states are observation probes only; no future transition is executed from them. Source trajectories use own-greedy retained parents,CPU2 singleton,temperature0.05 and paper8,256 decisions each at seeds26080001/26081001.

First command `.venv/Scripts/python.exe -m scripts.diagnose_observer_seat_information --output runs/observer_seat_information` exits1 after parent1's completed trace because numpy.int32 work counters cannot be JSON serialized. It saves256 raw rows and states,17 round endings/2 matches; no parent2 source work is reached. Replaced counter additions with explicit int casts and used a fresh v2 directory, preserving the failure. Successful command with `--output runs/observer_seat_information_v2` exits0 in5.85s. Parent1:256 decisions/17 rounds/2 matches,4 greedy changes (all discard),meanTV0.017141960220205976,maxTV0.4533851444721222,200 roots withTV>1e-6. Parent2:256/16/2,5 greedy changes (all discard),meanTV0.012221446347084192,maxTV0.35407352447509766,217 roots withTV>1e-6. Successful totals512/33/4; including failed executed work768/50/6. Report SHAac30ff58a6b921716ee2b8f5b59c93de5d7e3a818f5c4626e7fc626191bd1684 binds inputs,raw rows and states.

Independent command `.venv/Scripts/python.exe -m scripts.audit_observer_seat_information --output runs/observer_seat_information_v2/independent_audit.json` exits0 in4.32s. All512 original/relabeled pairs exactly match the pinned upstream Python feature builder and differ only at rows50/51; saved raw action-change/TV aggregates agree. Failed parent1 raw rows and states are byte-identical to successful parent1. Audit SHA1d8d9687f2f3e9fba3f1f5e0b3ec699803fecd17dc9e1e20311c0bf2467e2255. This audit performs1,024 within-round history reconstructions,including forced transitions; it adds no new source games,training,CUDA or future rollout branches. All57 live-study dependencies remain unchanged. The finding is missing public policy context,not proof of differing true continuation values,critic-gradient bias or stronger play. A zero-padded public viewer-seat bit is a separately described candidate follow-up; the live561-input comparison is unchanged. No manuscript/PDF rebuild or model promotion.

The first full observer-study training worker terminates0. Parent1_observer:520 updates (20 warmup/500 policy),8,519,680 decisions/568,498 round endings/72,980 matches/17,039,360 value rows. Final checkpoint SHA8a0c64cfba6281f9ac07ceb7d331f3cf4276ad815ab8530f904b22e5eaad4a23; warmup SHA6563b240196b9c5dc08246e5953b094884743aaa20d8a0c689b75a5dc82bbef8. Initial actor/environment/CPU+CUDA RNG match archived ordinary selfplay. `.venv/Scripts/python.exe -m scripts.audit_observer_optimizer --case parent1_observer --output runs/observer_selfplay_optimizer_audits/parent1_observer.json` exits0 in2.79s. All96 actual Adam states are finite and have derived budgets:actor48,000,value49,920; warmup leaves actors unchanged. Receipt SHA72b6a8926ff080a4a0634fb6ab5a03c7033f3e90f7bad3106ed3b9fe30c80bcf. Launcher1477 continues; parent1_signed operational snapshot59/520 updates,966,656 decisions/64,910 rounds/8,274 matches. Three control evaluators remain running. No partial strength outcomes used,study extension,configuration change or reserved101M use.

## Isolated public-seat critic CPU capability (2026-10-18)

Implemented scripts/seat_observer_selfplay.py with public/masked same562-input architecture,zero-padded parent role/seat columns,unchanged published actor,per-observer public features and inherited observer/signed PPO collection/returns/update. The model reconstructs active seat from packed public absolute/relative dealer features; direct560-feature value calls reject missing context. New seat_observer_selfplay_v1 checkpoints retain config/source/optimizer/environment/RNG and exact CPU restart. load_policy supports saved CPU inference. Existing bound training/inference sources are unchanged. This is a candidate information-contract correction,not a new general critic method or a proven sufficient history.

First `.venv/Scripts/python.exe -m pytest tests/test_seat_observer_selfplay.py -q -s` exits1:4 pass/1 fail in10.39s. The failing test incorrectly expected an added input column to learn with an initially zero critic and no completed match reward in two short warmup rollouts. Corrected only the test fixture to give both modes the same nonzero output weights; unchanged implementation then passes5 tests,0 skips,in13.96s. Coverage includes256 reference source positions across all four phases,both viewers,hidden-card/stock/RNG noninterference,seat relabeling,independent rows,exact parent logits,close parent initial values,active packed-feature identity,saved inference,matched observer/signed warmup per context,and exact CPU model/Adam/environment/counter/RNG continuation. Invalid context,legacy formats and tampered source identity are rejected.

Execution accounting:each test invocation generates256 feature-source decisions/16 rounds/2 completed matches. The failed invocation executes public-observer warmup256 decisions/16 rounds/0 matches plus two masked-mode warmup/continuation/replayed-continuation paths of512 decisions/34 rounds/0 matches each (1,280 training decisions/84 rounds/0 matches/2,560 value rows). The completed masked warmup checkpoints confirm256 decisions/16 rounds;the zero-gradient public trajectory is identical through that failure point. The final invocation executes four full512-decision paths,total2,048 training decisions/136 rounds/0 matches/4,096 value rows. Counting both feature-source runs and both training attempts:3,840 executed decisions/252 rounds/4 matches,6,656 training value rows. Exact-resume replay work is counted,not erased.

`.venv/Scripts/python.exe -m scripts.check_seat_observer_selfplay --output runs/seat_observer_capability_cpu` (session75220) exits0. Fixed CPU2 actual-parent capability:both retained parents x public/masked contexts,8 environments x64 steps,256-row minibatches,1 epoch,width256,2 critic-only+1 policy update,seed26092001. Each case executes1,536 decisions/3,072 value rows. Parent1 each103 rounds/8 matches;parent2 each100 rounds/8 matches. Total6,144 decisions/406 rounds/32 matches/12,288 value rows. All96 Adam parameter states finite;value6steps,each actor phase2steps. Initial actor logits match each retained parent exactly;initial values satisfy1e-7 absolute/1e-6 relative tolerance;warmup leaves actor unchanged;policy update changes it;loaded final inference is exact. Public seat-column L1 norms0.08230499178171158/0.09781847149133682 versus masked0/0. No CUDA training or strength evaluations.

Capability audit SHAd6f00fccd056731590d5bb7e9a9a30eb0f55fcf48b1460329b2e482527ba0d59. Finals:parent1_public06d273f8287a9c1981ba4804a7b85a84d8fddd9a76e2bb10e6c606a8bea47601,parent1_masked0f945163003d80783a44526c38dddb557ba00395d370f0f8b34fb7400d13a9a0,parent2_public26321ed672bce7825b61b1f4ed19fda0c03939f0e9a75aac6917419d8288350d,parent2_maskedcefdd7670da0c51dfa55cfe849728149be444561d266653408af69fd747f6a4e. All57 live observer-study inputs verified unchanged before/after. Full GPU capability and strength protocol remain pending the current study;no promotion,manuscript/PDF rewrite or reserved101M use. Launcher1477 remains live;operational parent1_signed snapshot162/520 updates,2,654,208 decisions/177,181 rounds/22,693 matches,with three control evaluations running. No partial outcome-based change.

Saved-state inspection independently checks all four capability finals:each of90 actor parameter states has2 actual Adam steps,each of6 value states has6,all moments finite. Public/masked initial actor,value,environment and CPU/CUDA RNG tensors agree within each parent. Receipt runs/seat_observer_capability_cpu/saved_state_audit.json SHA4389d88b54689087e67bf89c7be334def950e7ac1d8792d0d625247b6884e798;no new simulation. Launcher1477 re-polled and live;all57 inputs remain intact.

## Public-seat checkpoint match-evaluation adapter (2026-10-18)

Added scripts/seat_observer_agent.py and tests/test_seat_observer_agent.py. SeatObserverAgent loads the explicit seat_observer_selfplay_v1 bytes through load_policy,retains checkpoint/config/counter/provenance identities,and inherits NeuralAgent.act with singleton greedy CPU/public-endgame handling. It interoperates directly with evaluate.duel;no production factory or bound observer-study adapter was changed. No policy or critic is trained by loading/inference.

`.venv/Scripts/python.exe -m pytest tests/test_seat_observer_agent.py -q -s` (session14954) exits0:8 passed,0 skipped,10.81s. All4 actual CPU-capability initial checkpoints (2parents x public/masked) match their retained parent on256 source decisions,including all phases. Each trace executes17 round endings/2 completed matches;total1,024 decisions/68 rounds/8 matches. Live state is unchanged by act;opponent-hidden-card assignments,stock order and seed perturbations leave actions invariant. Source seeds26093002/26093003 are reused across contexts.

All4 small trained capability endpoints preserve the saved actor checksum and agree at each candidate decision with direct published actor masked logits,or with the existing public endgame solution. Standard evaluate.duel runs one duplicate pair per endpoint against the greedy fixture:parent1 public/masked each216 decisions,parent2 public/masked each219. Total870 decisions/8 complete matches on2 shared deals26094002/26094003. These are4 duplicate-pair adapter checks on2 shared deal streams,not4 independent deal samples or a strength comparison. Duel does not expose round-ending counts for these matches;no aggregate round count is inferred. No search,training,CUDA validation,checkpoint promotion or reserved101M use. Ruff/diff checks pass;all57 live-study inputs and all5 prior CPU-capability inputs remain unchanged.

## Finite-cutoff manuscript integration and concise conclusion (2026-10-18)

Added scripts/export_observer_trace_evidence.py and generated observer_trace_constants.tex/observer_trace_evidence.json. The exporter pins report SHA830420c471986c33b84bfd3a9a8d2b86f7aaef5bd343637856d4e682b4006d60,checks all bound sources and the production identity,and independently reconstructs true3/800 and score covariance-20301/160000 from the payoff matrix using Fraction arithmetic. All16 terminal/cutoff x observer/signed x lambda expectations and CPU values agree;6048 rational identity cases are preserved. It refuses overwriting different evidence. Command `.venv/Scripts/python.exe -m scripts.export_observer_trace_evidence` exits0;new ledger SHAa926388d445b82216f345ae7a96e94c0341353d53b8bb07779cf3386b183d0ac. No new simulation,CUDA calculation or strength estimate.

Manuscript Abstract/Introduction and Section7 now state that the same exact critic-conditioning failure survives lambda1 at a nonterminal cutoff. AppendixB.2 derives the standard GAE telescoping identity,root-score covariance terms and exact-root-history cancellation,then reports cutoff signed gradient-19701/160000 versus true3/800. It distinguishes a single player's contribution from a shared zero-sum aggregate objective and does not claim sufficient histories for finite Koi-Koi features. The existing terminal example and completed empirical comparisons remain. AppendixB has separate terminal/cutoff/implementation subsections. Limitations and conclusion were condensed to remove repeated caveats while retaining the empirical scope and unmeasured exploitability/human performance. Abstract180 whitespace words;the full conclusion fits page10.

Built with `external/tectonic/tectonic.exe --keep-logs --outdir output/pdf docs/paper/manuscript.tex`. Intermediate visual passes exposed a split conclusion and dense inline fractions;final prose shortening and displayed fractions resolve them. First render logging encountered a Windows stdout encoding error after page8;rerendered with explicit UTF-8 stdout without changing the PDF content. Final PDF21 pages,24 resolved citations,20 embedded fonts;SHAcddde4e37e0eee599dd18052ce7f2513b1a0ad29f4846516aeb4a10adea8cc7f. All21 pages visually reviewed;final pages1-9 pixel-match their earlier reviewed same-turn render,final pages10-21 inspected,with page10/table/conclusion and page13/identity/fractions inspected at full render size. No TeX overfull/underfull,unresolved-reference,missing-glyph or text-bound defects. All previous numeric tokens remain in source/generated tables. All8 ledgers'inputs and declared outputs verified (385 unique bound input files). QA runs/manuscript_trace_qa.json SHA22779a344afb31d3c0034390fdbdf5c55f5f663fc2c5e56fd69a56230acdbbd6. Prior20-page PDF preserved byte-for-byte as output/pdf/manuscript_learner_79dd722c.pdf,SHA79dd722cccdaea1491865e280c47cf347ff4b1af08c58a731100d0a3af6c95fc.

Launcher1477 re-polled live. Three control evaluation workers have terminated0 (parent1/WP,parent1/Point,parent1_selfplay/WP);next three evaluation cells run. No partial strength summaries were used. Parent1_signed operational snapshot415/520 updates,6,799,360 decisions/455,669 rounds/58,460 matches. All57 bound inputs remain unchanged;no protocol modification,promotion or reserved101M use. The paper presents verified finite-game evidence and completed earlier studies,not outcomes of the current observer comparison.

## Public-seat comparison fixed and second observer endpoint audited (2026-10-18)

Added docs/SEAT_OBSERVER_STUDY.rst,SHA23b354fd538e5552d54ea75408420d7a082e1878d9d1211f7ec58c6424ae835c,before inspecting the running study's outcome summaries. It fixes observer returns,the original retained parents,and two562-input contexts (public versus zero-masked seat) with matched architecture/initialization/optimizer/targets/budget. Unchanged parents and the specifically named archived561-input parentN_observer endpoints are controls,not winners selected from the current study. Four520-update arms would execute34,078,720 new decisions/68,157,440 value rows;archived observer training17,039,360 decisions is separate. Sixteen evaluation cells share5,000 fresh28,000,000--28,004,999 duplicate deals,160,000 matches. All8 public-minus-parent/public-minus-masked lower95 win bounds must exceed zero;all24 within-parent paired contrasts and point results remain visible. Joint5,000 whole-deal bootstrap,seed99. No reranking,extension,promotion or101M use. Prerequisites are completed/audited current study,actual CUDA capability and real reduced pipeline. This turn starts no new strength run.

The new protocol distinguishes its warmup contract from the current observer/signed comparison. Public/masked initial full model/environment/RNG states must match. During critic-only warmup,actor/environment/RNG must still match,while critic parameters/targets/moments may differ because their information differs. Both fixed contexts use the same forward dimensions. The public feature is established methodology;the research question is the measured effect of this identified mismatch,not a new general critic architecture.

Independently inspected actual parent1 observer/signed20-update warmup checkpoints. Actor,value,optimizer,environment,counters,runtime,CPU/CUDA/Python/NumPy RNG contents agree exactly;config differs only in policy_returns. Each has327,680 decisions/22,107 rounds/2,770 completed matches/655,360 value rows;1,920 value Adam steps and no actor updates. Receipt runs/observer_selfplay_warmup_audits/parent1.json SHA70af881ae670419187ec5fa51c3dc513398d7e8fb7aedcdc6222f9b2d8ce2ad2 binds both checkpoints,protocol and independent comparison helper. Saved-state inspection adds no simulation.

Parent1_signed then completes520 updates and the worker exits0. Endpoint counters:8,519,680 decisions/572,029 round endings/73,403 matches/17,039,360 value rows;20 critic-only+500 policy updates. Final SHAc6bb70caa70be0c17905c4eb5b948658bc81ab3da0a602e890db9ea73482ae52,warmup SHA4da7cf79792e16079156a6737d22f785ad7ed58433930aa9c6b0dab565e4ac61. Initial actor/environment/CPU+CUDA RNG equality to archived ordinary selfplay passes. `.venv/Scripts/python.exe -m scripts.audit_observer_optimizer --case parent1_signed --output runs/observer_selfplay_optimizer_audits/parent1_signed.json` exits0 in2.72s. All96 stored parameter states have finite moments and correct derived budgets:48,000 Adam steps for each actor tensor and49,920 for each value tensor. Initial-to-final L2 movements:discard0.911258,pick0.666866,koi0.578143,value9.725084. Receipt SHA129c791623031100fd48cd613cb0511c2ac5f02cc673b2b9741c63c21ffc371b.

Both completed parent1 arms total17,039,360 training decisions/1,140,527 rounds/146,383 matches/34,078,720 value rows. These are new study totals,not additional work caused by the audits. Launcher1477 is live and has started parent2_observer;three control evaluation cells have completed and three more remain active. No outcome summaries inspected,configuration changes or new strength conclusions. All57 inputs and the latest21-page PDF remain unchanged.

## Guarded seat-context device capability and exact replay (2026-10-18)

Added scripts/check_seat_observer_device.py and tests/test_seat_observer_device.py. The fixed production CUDA path uses parent1,public/masked contexts,128 environments x128 steps,512 minibatches,3 epochs,width256,20 critic-only+2 policy updates,seed26095001. It checks all256 saved diagnostic positions per context against independent Python features,including active packed-seat reconstruction. Initial actor/value/optimizer/environment/counter/CPU+CUDA RNG signatures must match. At warmup,actor/environment/counters/RNG match but critic states may differ. The last update is re-executed from the preceding saved checkpoint and must match metrics,full model/Adam/environment/counters/RNG exactly. Every stored parameter/moment is finite and actual Adam steps agree with phase counters. Replay decisions/rounds/matches/value rows are added separately to physical work.

GPU scheduling guard requires all four known observer-study training jobs,successful exit codes,valid terminal timings,a matching protocol identity,and successful520-update audits/final hashes. Missing/live jobs remain pending;failed/duplicate/unexpected jobs reject. The non-wait path rejects pending work before creating an output directory or a model. The wait path pins64 inputs and rechecks identities while waiting. It does not infer success from elapsed time or inspect any partial strength summaries.

First `.venv/Scripts/python.exe -m pytest tests/test_seat_observer_device.py -q -s` (session5967) passes3,0 skipped,in13.33s. Added explicit scheduler-protocol identity matching,then reran the three checks (session33789):3 pass,0 skips,in13.25s. Each invocation includes a real reduced CPU path:both contexts,4 environments x64 steps,128 minibatches,1 epoch,1 warmup+2 policy updates plus1 replay per condition. Logical endpoints each768 decisions/50 rounds/4 matches/1,536 value rows;each replay adds256 decisions/16 rounds/0 matches/512 rows. Each invocation totals2,048 physical decisions/132 rounds/8 matches/4,096 rows. Both executed invocations total4,096 decisions/264 rounds/16 matches/8,192 rows;the earlier pass is not erased. Static saved-state observation probes add no new simulation. Initial signatures,warmup actor/environment/RNG and exact resumed final updates all agree;critics differ under the intended information intervention.

Final reduced audit SHA4df957a460039d3171390224baff6a716d6ed98b0952654252b4d53b61adcc58 copied with all checkpoint/protocol/log files byte-for-byte from C:/Users/sheeta/AppData/Local/Temp/pytest-of-sheeta/pytest-303/test_actual_reduced_capability0/capability to runs/seat_observer_device_smoke. All64 inputs verify. Earlier report SHA7acc0ca8474868614f989284477c0562699813996ac5103313af0b774b1292bc belongs to the pre-protocol-guard source and is not the final capability evidence.

Started `.venv/Scripts/python.exe -m scripts.check_seat_observer_device --output runs/seat_observer_capability_cuda --wait`,session20636. It reports waiting_for_existing_gpu_training;only protocol.json exists before the launch receipt is written. Protocol SHAe3679b894abae603fd268b1a1865d34a01b88f3879b0e8df0d73d765091ad894;launch receipt SHA07df12347b9a8305774762c4b0a11207be790f19efed33f191bfe7fb3b6db3df binds this queue to the final CPU smoke audit. GPU capability is not yet executed or passed. Planned logical720,896 decisions plus32,768 replay decisions equals753,664 physical decisions/1,507,328 rows;actual rounds/matches and optimizer budgets await completion. Source/test/trainer/proposed-design inputs are now frozen for this queued job. No new strength training,promotion or101M use.

Original launcher1477 remains active. Parent2_observer snapshot151/520 updates,2,473,984 decisions/165,737 rounds/21,248 matches. Current-study57 inputs and queued-capability64 inputs remain unchanged. The latest21-page manuscript is unchanged and not reissued.

## Seat-context study orchestration and pending real pipeline (2026-10-18)

Added scripts/run_seat_observer_study.py and tests/test_seat_observer_study.py without changing frozen active-study sources. The runner binds the original retained parents,fixed parentN_observer historical controls,their actual optimizer receipts,the seat-capability evidence and the predeclared study design. Full launch rejects an unfinished original study;it additionally requires completed independent source-study audit and CUDA capability with actual96-state value/actor budgets2112/192 and753,664 physical decisions. CPU smoke uses the completed device-smoke evidence. Public/masked configs differ only in seat_context;initial full-state and critic-only warmup actor/environment/RNG contracts are checked. Full initial actor/environment/RNG and padded critic must also match the archived observer initial checkpoint. New checkpoint evaluation uses SeatObserverAgent;historical observer controls use ObserverAgent. Analysis retains all pairwise contrasts and the fixed conjunction of public-minus-parent/public-minus-masked win intervals.

The scheduler allows one trainer and at most three CPU2 evaluators,including the preceding study. Reduced CPU training may proceed during original CUDA work,but all evaluations wait until the prior study is complete and its final analysis identity verifies. The full comparison waits for the prior study before training too. Failed worker launches record failure and clean owned workers. A Windows path-key mismatch was corrected before actual training;Ruff first found an import-order violation,which was fixed. Final Ruff and formatting checks pass.

`.venv/Scripts/python.exe -m pytest tests/test_seat_observer_study.py -k 'not real_pipeline' -q` passes3 tests,0 skips,1 deselected,in5.32s. These check the actual bound CPU capability/fixed smoke plan,the no-output incomplete-prior gate,CPU-training versus evaluator eligibility,prior failure/protocol mutation and failed-launch cleanup. The launch-failure test deliberately creates a failed test directory and no training.

Started `.venv/Scripts/python.exe -m pytest tests/test_seat_observer_study.py -k real_pipeline -q -s`,session44783. Actual artifact directory C:/Users/sheeta/AppData/Local/Temp/pytest-of-sheeta/pytest-305/test_real_pipeline_and_tamperi0/pipeline;protocol SHAc2794a4fab26d5060a4ffbc13eca0fa93ad241ad91c3467bbe2d8bf5cd4b138b binds82 inputs. Both CPU workers exit0. Each uses4 environments x16 steps,32 minibatches,one epoch,one warmup plus one policy update,seed26096001:128 decisions/8 round endings/0 completed matches/256 value rows. Combined actual work256 decisions/16 rounds/0 matches/512 value rows. Each actual value parameter has4 Adam steps;each actor parameter has2. Initial full-state and warmup actor/environment/RNG equality pass;critic-state equality is intentionally not required after treatment. No evaluation matches have yet run.

Session44783 remains live waiting for the preceding study;the planned eight evaluation cells and post-completion tamper checks are PENDING,not passed. Launch receipt runs/seat_observer_pipeline_launch.json SHAbbfd67ba7dd83df863d549122ae4b7eb079230ece5e8d694dfba1be5f261a77f preserves the live handle,path,source/protocol identities and completed training reports. Do not restart on an observation timeout or modify the bound runner/test. Original1477 remains active,queued CUDA capability20636 remains waiting,and all57/64/82 bound inputs verify. No partial strength summaries inspected,new full strength launch,model promotion,101M use or manuscript change.

## Independent seat-context endpoint and result auditors (2026-10-18)

Added scripts/audit_seat_observer_optimizer.py,scripts/audit_seat_observer_study.py and tests/test_seat_observer_audits.py. The endpoint auditor reads actual parent/initial/warmup/final tensors and Adam state,requires one successfully terminated training worker,derives expected actor/value update budgets independently of reported counters,checks zero-padded initial critic import and the masked seat column,and measures parameter movement. It does not invoke the runner training auditor. The final-study auditor independently reads raw duplicate outcomes and recomputes all absolute outcomes/contrasts with joint5,000-resample seed99 bootstrap. It distinguishes concurrent CPU smoke training from original CUDA training,checks closed schedules and matched initial/warmup contracts,and rejects nonterminal studies before writing. Full-study success remains unexecuted because the real pipeline is not closed.

`.venv/Scripts/python.exe -m pytest tests/test_seat_observer_audits.py -k 'not closed_pipeline' -q -s` passes3 tests,0 skipped,1 deselected,in5.12s. Tests inspect both actual reduced-pipeline endpoints,reject an altered loaded Adam step without touching the original artifact,and reject unfinished result analysis without output. Both endpoints have96 stored parameter states:each actor tensor2 actual updates,each value tensor4;moments finite and critic-only warmup verified. Final added-column L1 is0.08054374158382416 for public and0 for masked. This establishes execution of the intervention,not a strength gain. No additional simulation or training was performed.

Independent endpoint reports copied byte-for-byte from C:/Users/sheeta/AppData/Local/Temp/pytest-of-sheeta/pytest-306/seat_optimizer0 to runs/seat_observer_pipeline_optimizer_audits. Public report SHA08d096f2464f0a8d18c24b609a19d3466189c58b609bb216a2d752f3e64bf3ed;masked SHAc3db160336156c7c5f1d37ba0ebe8e606c92d75998c0dab8dd3a76d979f57d0e. Both reports' bound inputs verify. The closed-pipeline statistics/tamper test is deliberately deselected until real evaluations finish;do not describe the final auditor as successfully validated yet.

Specific live handles1477,20636 and44783 were re-polled successfully. Current original-study parent2_observer snapshot384/520 updates,6,291,456 decisions;three CPU evaluation cells remain active. Queued capability and reduced pipeline retain their existing handles;no restart. All57/64/82 frozen inputs verify. Ruff passes for the three new files. Manuscript/model/dealer unchanged;new replicated strength improvement remains unestablished.

## Abstract and Introduction clarity pass (2026-10-18)

Revised only the manuscript opening. Abstract now describes the critic-information mechanism directly and specifies that the retained architecture is the deployed actor. It replaces the unexplained parent-and-response criterion with the actual starting-policy/ordinary-fixed-opponent-PPO comparison. Introduction states the replicated WP improvement and uneven Point transfer directly,shortens the adaptation summary,and keeps the primary learner-return failure alongside the secondary Point/lambda1 gains. Abstract is174 whitespace words (previous180). Related Work and all subsequent source bytes after normalized line endings are unchanged;no current-study outcome is added.

Rebuilt with external/tectonic/tectonic.exe --keep-logs --outdir output/pdf docs/paper/manuscript.tex. Build succeeds;the existing fontconfig environment message does not prevent font embedding. All21 pages rendered at1.3 scale. First rendering command finished the images and comparison,then failed only while printing a ligature-containing text excerpt to cp1252 stdout;no artifact defect or simulation work resulted. Verification rerun uses ASCII JSON output. Pages1/2 are visually reviewed at full render size;pages3--21 have byte-identical pixel samples to the previous reviewed PDF. All20 fonts embed;24 citation keys retained;no missing glyph,unresolved reference,TeX box warning or text-bound overflow. All prior numeric tokens,8 unchanged ledgers and392 unique bound input/output files verify.

New PDF SHA8a405247ec007d4d929e82426d931aae70b1d7a404607bc79d405b615be84175;QA runs/manuscript_opening_qa.json SHA7f0f8610d3dd5f4b73c95856fc74272712fe08bada0f0862b250d842073988cc. Previous PDF preserved as output/pdf/manuscript_trace_cddde4e3.pdf SHAcddde4e37e0eee599dd18052ce7f2513b1a0ad29f4846516aeb4a10adea8cc7f. Specific sessions1477/20636/44783 re-polled live;all57/64/82 bound study inputs remain unchanged. Original parent2_observer snapshot440/520,three original evaluations active. No new strength conclusion,training restart,promotion or dealer change.

## Third observer-study endpoint and both-parent warmup checks (2026-10-18)

Live launcher1477 was repeatedly polled through completion of parent2_observer;no process was restarted. Worker10784 exits0 after520 updates,8,519,680 decisions/567,295 round endings/73,007 completed matches/17,039,360 value rows. Final SHA26443be1424d318cfe667c1a0193fa1a0320eca810679074a3589bd9ecb31aa4;warmup SHA9ced4d2c024f7e8cb9e0d15fc8369bee3039f9ef61af087138e3b65be2a5d85a. Runner checks preserved initial actor/environment/CPU+CUDA RNG equality against archived ordinary self-play. Independent audit_observer_optimizer.run passes in2.76s,receipt runs/observer_selfplay_optimizer_audits/parent2_observer.json SHA84bc90343c1d8c088047429dc67c3d111288fe9e258665395a54b38db9f087a5. All96 actual parameter states have finite moments and derived budgets:48,000 actor and49,920 value Adam updates. Initial-to-final L2 changes are discard0.914437,pick0.678354,koi0.563977,value9.801913.

Three completed training arms total25,559,040 decisions/1,707,822 rounds/219,390 completed matches/51,118,080 value rows. These totals describe executed training and are not a win estimate. Six of16 evaluation cells have exited0;partial outcome summaries remain uninspected. Parent2_signed starts as the fourth and final training arm,while three CPU2 evaluations continue. Queued capability20636 and reduced pipeline44783 retain their live handles.

After parent2_signed saves its20-update warmup,independent saved-state comparison passes for actor,value,optimizer,environment,counters,runtime,CPU/CUDA/Python/NumPy RNG. Config differs only in policy_returns. Both parent2 warmups contain327,680 decisions/21,851 rounds/2,752 completed matches/655,360 value rows,1,920 value updates and no actor updates. Signed warmup SHA4dc9db757e797adfdebe45fec581f1c05367e5d92db841c07d99760a25e9d0ee;receipt runs/observer_selfplay_warmup_audits/parent2.json SHAfe282ff0e9781d7b99b0b2feb583e278dbde0e7db81ffcda3b0b6d3cd62fb72e. Both parents now satisfy the matched full-critic warmup contract. Saved-state inspection adds no simulation. All57/64/82 frozen inputs verify;no experiment change,model promotion or paper revision.

## Temporal-abstraction related-work gap (2026-10-18)

The previous turn was a verified wait:1477/20636/44783 were confirmed live and final training advanced65/520. Current turn re-polls the same handles and addresses a concrete literature gap. Added Sutton,Precup and Singh,Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning,Artificial Intelligence112(1--2):181--211,1999,DOI10.1016/S0004-3702(99)00052-1. Primary paper inspected at https://people.eecs.berkeley.edu/~russell/classes/cs294/f05/papers/sutton%2Bal-1999.pdf,rendered pages1/8/10;the abstract,history-dependent option definition and embedded-SMDP theorem support the temporal-abstraction attribution. Web text extraction is garbled,so actual PDF pages were read visually. Publisher fetch returned403 and author-page fetch timed out;the primary-paper copy was available.

Related Work now connects learner-decision intervals to established temporal abstraction,and identifies information at bootstrap boundaries as the question analyzed here. It does not claim that finite Koi-Koi observations satisfy a semi-Markov sufficiency theorem. Abstract/Introduction and everything after Related Work remain source-identical to the preceding opening revision;no new empirical claim. Bibliography now25 cited references.

Rebuilt21-page PDF SHA6a10c91daec57ab714efb15d752e6193b2d42984c6b31b380f5225c63c782311. QA runs/manuscript_temporal_related_qa.json SHA81c4ed7ad683a650d84936bb67b2c22c401149455e53d137a3932019dd87a6bc binds the inspected primary PDF hash and explanatory scope. All21 pages rendered and contact sheets reviewed;new text/reference pages3/21 inspected at full size. Changed pages2/3/21;other18 pixel-identical to previous review. All20 fonts embedded,25 citations resolved,174-word Abstract unchanged;no TeX box,missing-glyph,reference or bounds errors. Eight evidence ledgers and392 bound input/output files verify,all prior numeric tokens retained. Previous PDF preserved as output/pdf/manuscript_opening_8a405247.pdf.

All57/64/82 active-study inputs remain unchanged. Launcher1477 remains live;parent2_signed snapshot118/520 updates and6/16 evaluation cells complete. Capability20636 and pipeline44783 still wait for their preconditions. No additional simulation from this literature/PDF work,no new strength result or model promotion.

## All observer training complete; seat-context CUDA capability passed (2026-10-18)

Confirmed the same live handles throughout waiting;no restart. Parent2_signed worker27964 exits0 after520 updates:8,519,680 decisions/570,562 round endings/73,080 matches/17,039,360 value rows. Final SHAdcdefa619f09d59749c9172c01db931c5eb46fdd08a70ea3546eb4a680c7404a,warmup SHA4dc9db757e797adfdebe45fec581f1c05367e5d92db841c07d99760a25e9d0ee. Independent audit_observer_optimizer.run passes in2.70s:all96 finite optimizer states,48,000 actual updates per actor tensor and49,920 per value tensor. Parameter L2 movement:discard0.926870,pick0.685623,koi0.567597,value9.889587. Receipt runs/observer_selfplay_optimizer_audits/parent2_signed.json SHA893fad3942103f2edf53c812b1a7610193312c95871a66738e78f85d888db1ea.

All4 original observer/signed training workers are now terminal0,with independent optimizer receipts. Total new training34,078,720 decisions/2,278,384 rounds/292,470 matches/68,157,440 value rows. Both parents' warmups match. Launcher1477 continues the fixed16-cell evaluation;6 cells have completed and3 are running. Partial outcome summaries remain uninspected. Full independent study audit and strength conclusion remain pending all cells.

Queued capability20636 observes the successful completion of all4 GPU training workers,passes its bound prerequisites,and then executes public/masked production-size CUDA validation. Session20636 exits0;do not poll/restart it. Report runs/seat_observer_capability_cuda/audit.json SHA389b10afb3269cf521f76ca6d8771bded80600614f9744d9a27b5c0cfb9309bd. Both contexts complete20 warmup+2 policy updates at128 environments x128 steps,512 minibatches,3 epochs,width256,seed26095001. All256 saved-state feature probes per context match the independent Python construction. Initial signatures agree;warmup actor/environment/counters/CPU+CUDA RNG agree. Critic states differ as intended. Exact resumed last updates match metrics/model/Adam/environment/counters/RNG.

Each logical endpoint360,448 decisions/720,896 value rows/3,046 matches;public24,300 rounds,masked24,298. Each actual replay adds16,384 decisions/32,768 value rows/137 matches;public1,114 rounds,masked1,112. Total physical work753,664 decisions/50,824 rounds/6,366 matches/1,507,328 value rows. These are capability/replay costs,separate from full-study training or strength matches.

Independent saved-state inspection verifies all8 stage-checkpoint hashes and actual96-state Adam budgets:90 actor tensors x192 steps,6 value tensors x2,112 steps,finite weights/moments. Public added-column GPU L1=3.3676018714904785;masked0. Initial model/optimizer/environment/RNG contents and warmup actor/environment/RNG match exactly. First manual inspection incorrectly demanded bitwise equality of a CPU-derived scalar L1 reduction with the GPU-derived scalar;it failed at a2.384185791015625e-7 difference and wrote no report. Follow-up inspection confirmed identical stored data/counters and used absolute1e-6 tolerance only for this cross-device reduction. Exact model/RNG comparisons and same-device replay checks remain exact. No capability or training was rerun. Saved-state receipt SHA2cc53ba857f7e948368bf8321cd9760fb4f9e30d657f750d3bc958a970234d13 records both reductions and the tolerance.

Original1477 and reduced pipeline44783 re-polled live;pipeline still waits for all original evaluations before its8 cells and final tamper checks. All57/64/82 bound inputs verify. Full seat-context strength study is not yet launched:original final evaluation/independent audit and successful actual reduced pipeline remain prerequisites. Latest21-page/25-reference manuscript unchanged;no promotion,dealer change or101M use.

## Planned CUDA initial states match historical controls (2026-10-18)

Added scripts/check_seat_historical_initial.py to exercise the exact upcoming initialization comparison before any full seat study. It requires all original GPU workers terminal0 and the passed CUDA capability,binds69 source/control/capability inputs,and constructs the four planned CUDA configurations from the two original parents. It saves initial checkpoints and applies the upcoming runner's archived-state comparison. No collect,update,forward inference or evaluation call is made;all counters remain zero and optimizers are empty.

`.venv/Scripts/python.exe -m scripts.check_seat_historical_initial --output runs/seat_observer_initial_preflight` exits0 in5.11s,with expected upstream Transformer warnings. Both seeds19910001/20010001 x public/masked pass exact archived actor/environment/CPU+CUDA RNG comparison and exact561-column critic plus zero added column. Public/masked initial actor/value/optimizer/environment/counters/runtime/RNG contents agree within each parent. Report SHA236af212992ad54c4a6b3d3823510ca67c17b7153930ff647498d368b874dbee;all69 inputs verify. Initial checkpoints:parent1_public ebc1f8f52ee324fe26c6329e0c0c8fbf49aee9ee39ffdd46263ef1774a99a929,parent1_masked4b11d36d30a267d58b4364953b5536fe0ec9a08e4d54e21611152a456df5f9ba,parent2_public ce68f31303009fafa38c9b7e2a7fa9f6f939009520346b5a3ad66c849135ba0a,parent2_masked0724840495324d8ec2ca6d870f0f6cb33302d08e14f8dfc6b3465c2aab9780bf. Different file hashes include treatment configs/provenance;the required tensor contents are exactly equal.

Executed training/evaluation decisions,round endings,completed matches and value rows are all0. Constructing initial environments is not counted as completed matches. This confirms the full-seed initialization path,not a new strength result or permission to skip remaining prerequisites. Original1477 and reduced pipeline44783 re-polled live;6/16 original evaluation cells complete. The already completed capability20636 is not restarted. All57/64/82 frozen inputs remain intact. Ruff passes for the new diagnostic;paper and deployed model unchanged.

## Public-seat final evidence exporter prepared

**Seat-study paper exporter prepared:** scripts/export_seat_observer_paper_evidence.py requires explicitly pinned final protocol/audit hashes, complete full-study state and unchanged audited inputs. It checks all16 outcomes/all24 contrasts against independent recomputation, all8 primary bounds and separate new/reused training totals before producing three comparison tables and a full evidence ledger. Three refusal tests pass,0 skipped,0.12s;Ruff/diff checks pass. The successful full-export path remains pending the actual completed seat study;no evidence tables or manuscript changes were produced. All57/64/82/69 current study/capability/pipeline/preflight inputs verify unchanged. Sessions1477 and44783 remain live.

Commands: `.venv/Scripts/python.exe -m pytest tests/test_export_seat_observer_evidence.py -q`; `.venv/Scripts/python.exe -m ruff check scripts/export_seat_observer_paper_evidence.py tests/test_export_seat_observer_evidence.py`. The final command will require `--protocol-sha256` and `--audit-sha256` from the completed full run; no interim estimates are exported. No new simulation, training or match accounting.

## Actor-input reconstruction from public-seat critic features

**Public-seat actor-input coverage checked:** an observation-only decoder reconstructs all300x50 published actor features exactly from the active562-input critic row on1,024 retained probes (512 roots plus seat relabelings),covering all4 phases. The paired561-input collisions still differ only in actor rows50/51;the seat bit restores that distinction. Report runs/seat_feature_reconstruction/audit.json SHAa35a5a3d79626b5eca84edd3e9bdb9207dc394e9784e3f1322e164e8d48d42b6 binds14 unchanged inputs;the prior upstream audit also revalidates. Zero new decisions/rounds/matches,policy inference or training. This establishes measured representation coverage,not sufficient-history or strength guarantees. See SEAT_FEATURE_RECONSTRUCTION.rst. All57/64/82/69 frozen study/capability/pipeline/preflight inputs remain unchanged;the ongoing evaluations and manuscript are untouched.

Command `.venv/Scripts/python.exe -m scripts.check_seat_feature_reconstruction --output runs/seat_feature_reconstruction/audit.json` exits0 in2.31s. Phase counts:discard862,hand capture52,draw capture28,koi82. Decoder takes only one562-vector and uses card groups,public context,timestamps and the independent Python score oracle;expected features come from the native builder. All1,024 original states remain unchanged. Ruff passes. No strength summaries were read or new trajectories generated.


## Observer self-play completed and independently audited (2026-10-18)

Original launcher1477 exits0 with all20 workers (4 training,16 evaluation) successful. Protocol ed98449175207a42782d1083e9efa69f4c76c6d43af4464587f83c8b8963c3ef is unchanged. Analysis SHA55f88985a76580e43bd9eb010f84e243ca127125e998b1605dcd616394dbbc5e. Final-only command `.venv/Scripts/python.exe -m scripts.audit_observer_selfplay_study --output runs/observer_selfplay_study/independent_audit.json` runs after successful completion and exits0;receipt SHA759889ec9ffd6f98f56f8f292f00e590e44dec07bbe489e4657f56aecef78e70. It independently recomputes all16 outcomes/all24 paired contrasts using5,000 joint whole-deal bootstrap resamples,checks both matched warmups,actual optimizer receipts and closed-worker scheduling.

The replicated observer-gain criterion fails:0/8 required win lower95 bounds are positive. All8 intervals include zero. Results below are percentage-point changes with paired95% intervals:

| Parent | Opponent | Observer minus parent | Observer minus matched signed |
| --- | --- | --- | --- |
| 1 | WP | +0.115 [-0.720125,0.920] | +0.150 [-0.720,1.010125] |
| 1 | Point | +0.715 [-0.105,1.505125] | +0.455 [-0.345,1.275] |
| 2 | WP | +0.025 [-0.835125,0.890] | -0.435 [-1.265125,0.400125] |
| 2 | Point | +0.320 [-0.515,1.115] | -0.080 [-0.895125,0.745] |

Observer-minus-historical-selfplay win intervals also all include zero. Two secondary observer-minus-parent point intervals are positive:parent1/WP+0.5384 [0.05338,1.010245],parent2/Point+0.6202 [0.13298,1.089805]. Neither is replicated across both parents for its opponent;all observer-minus-signed point intervals include zero. Both observer endpoints exceed50% against Point on these development deals (51.370% [50.560,52.145],51.100% [50.290,51.885125]),but both retain negative point margins and their parent-relative win intervals include zero. These absolute outcomes do not establish the intervention's advantage. No reranking,budget extension,101M use or deployment promotion.

New training34,078,720 decisions/2,278,384 round endings/292,470 completed matches/68,157,440 value rows. Evaluation160,000 matches/18,571,664 decisions on5,000 shared27M deal streams;duel supplies no round-ending count. Archived ordinary-selfplay work remains separate. Export command `.venv/Scripts/python.exe -m scripts.export_observer_selfplay_paper_evidence --audit-sha256 759889ec9ffd6f98f56f8f292f00e590e44dec07bbe489e4657f56aecef78e70` exits0. Full ledger docs/paper/generated/observer_selfplay_evidence.json SHA067d13a2f0ad3bd6d0f9a18146b3746dbb5c4bbe0ffabcfd2726beea2a499b48 retains every outcome/contrast and emits primary win,point and historical tables. This completes the previously pending successful exporter path;paper integration remains pending.

## Seat pipeline retained and full comparison launched (2026-10-18)

First actual pipeline44783 exits0:1 test passed,3 deselected,0 skipped in10,652.79s including the original-study wait. Closed-pipeline independent test `.venv/Scripts/python.exe -m pytest tests/test_seat_observer_audits.py -k closed_pipeline -q -s` then passes1,0 skipped in6.68s;historical receipt SHA62631ab6e55bc36d13635f0a40644abf881ba85674c67d4a2cf22834ed42089f. At that test session's cleanup,pytest's default retention removes pytest-305 and the original pipeline payload. A subsequent attempt to audit that original path fails FileNotFoundError before output;no receipt is fabricated. The successful historical receipt is retained at runs/seat_observer_pipeline_evidence/first_pipeline_audit_historical.json,but its deleted inputs are not claimed currently verifiable.

Repeat the full real subprocess/tamper test in a fresh explicit workspace: `.venv/Scripts/python.exe -m pytest tests/test_seat_observer_study.py -k real_pipeline -q -s --basetemp runs/seat_observer_pipeline_retained`. Session37190 exits0 with1 passed,3 deselected,0 skipped in41.85s. Permanent payload is runs/seat_observer_pipeline_retained/test_real_pipeline_and_tamperi0/pipeline;never reuse that --basetemp target for another test run. It completes2 actual training workers and8 evaluation workers,then detects changed training counters,opponent identity and protocol before restoring original bytes. No strength criterion is assigned to this reduced run.

Independent saved-state optimizer receipts for the retained run:public e931616c97dc893bdc1c5b51fefdd1d5731cce864be3cfbc617f8a226b4af3d4,masked9ab3aa3e9c60eeed2daf5fd8ba6cfe1a11b92f50d9af14f340b9d0baff19e8ab. Stored under runs/seat_observer_pipeline_retained_optimizer_audits. All96 states each finite,actor2/value4 Adam steps. Final independent raw outcomes/contrasts/schedule audit runs/seat_observer_pipeline_evidence/retained_independent_audit.json SHAe093bd1a803d4637053ffe362a1db887f55d4558d928222f41cd12ed7695c845 passes. Each pipeline executes256 training decisions/16 rounds/0 completed matches/512 value rows,plus32 evaluation matches/3,825 decisions. Both actual invocations total512 training decisions/32 rounds/0 matches/1,024 value rows and64 evaluation matches/7,650 decisions. They reuse the same2 deal streams;do not count4 independent deals.

Launch-prerequisite receipt runs/seat_observer_launch_prerequisites.json SHA27586c3894af342fa5c41df780972b32a81c4eddba9a83cc2b00c675933fc6a9 revalidates all bound inputs of the original final audit,retained reduced pipeline,full CUDA capability and four-config initialization preflight. `.venv/Scripts/python.exe -m scripts.run_seat_observer_study` starts full session47154. Protocol SHA24aaa955f7556177c63531e6cd9951afa510e778f909a3c1c2b3c2e33548243f binds177 inputs. One CUDA trainer and three CPU2 evaluators start;initial cases are parent1_public training,parent1 WP/Point and archived parent1_observer WP evaluation.

The predeclared comparison is unchanged by the now-observed negative result:two original parents x public/masked observer continuation,20 warmup+500 policy updates per arm,planned34,078,720 new decisions and68,157,440 value rows. Sixteen evaluation cells use5,000 fresh28M duplicate deals (160,000 matches). All8 public-minus-parent/public-minus-masked win lower bounds must be positive;historical observer and points are secondary. No intermediate selection,extension or promotion. Full results and actual training counts remain pending. Parent selection,canonical human-play checkpoint and dealer rules remain unchanged.


## Completed observer comparison and seat-input diagnosis in manuscript (2026-10-18)

The verified manuscript now has24 pages,25 cited references and20 embedded fonts;Abstract202 whitespace words. Section9 derives fixed-observer primitive-step returns,states the matched signed-policy control and reports the completed16-cell/160,000-match comparison. None of the8 primary win intervals excludes zero. The main win table is generated from the independent audit;AppendixG keeps secondary point and historical-selfplay comparisons. The full9th evidence ledger retains all16 outcomes/all24 contrasts. The new subsection reports public-seat feature collisions,4/5 changed greedy actions at the two parents and exact300x50 actor-input reconstruction on1,024 probes. Its strength effect is explicitly left to the ongoing separately specified public/masked comparison.

Abstract/Introduction/Conclusion now include these findings without changing the original adaptation or learner-return results. Related Work is byte-equivalent as normalized text;all previous source numeric tokens remain. Reproduction accounts separately for34,078,720 new observer training decisions/68,157,440 value rows,17,039,360 reused historical training decisions and18,571,664 evaluation decisions. Seat-diagnostic work retains the failed serialization collection as additional work;reconstruction adds no simulator steps.

Build command `external/tectonic/tectonic.exe --keep-logs --outdir output/pdf docs/paper/manuscript.tex` succeeds. An overlong evidence-filename sentence was shortened until no overfull/underfull boxes remain. All24 pages rendered and visually reviewed;pages1/11/12/13/22 inspected at full size,with final changed pages1/11/12 rechecked. No missing glyphs,unresolved references,out-of-page text,clipping,overlapping tables or equations. All9 evidence ledgers and the3 seat-diagnostic reports reverify489 unique bound input/output/diagnostic files. The renderer still emits its pre-existing Fontconfig configuration notice;all20 actual PDF fonts embed and render correctly.

PDF output/pdf/manuscript.pdf SHA f606acbb895b313bc592a61245fd4cef2dfbb123aeefb005a2be771ad1ddb961. Source SHA25722e55dc944dc00c2fa5e027f901ccd7567a3fbaa92df01b49d47a7e5d1075. QA runs/manuscript_observer_selfplay_qa.json SHA74a4588ffcb3b62433b903faad0cdc719677d7bd8a67ad2f94ce888db0b29791. Previous21-page PDF retained as output/pdf/manuscript_temporal_6a10c91d.pdf. Full public-seat study47154 remains live and all177 bound inputs verify unchanged;no interim outcome summaries inspected or protocol changes made. No new simulation from this manuscript work.


## Fixed completion-audit queue for live seat study (2026-10-18)

Queued the already prepared independent checks without changing the running study: `.venv/Scripts/python.exe runs/seat_observer_audit_queue.py`,session36650. Queue source SHA69f549ac282438436738dfae03439e95fb3a2cdd15716deeab891f07f3081c8c. Mutable execution receipt is runs/seat_observer_audit_queue.json;it records the immutable study protocol and5 source digests. The one-shot wrapper refuses an existing queue receipt or existing audit/export outputs. It waits for each of the four fixed training workers to end0,invokes audit_seat_observer_optimizer once per endpoint,then waits for the complete study before audit_seat_observer_study and export_seat_observer_paper_evidence. A failed worker,changed bound source or failed check stops the queue with an error receipt. It does not restart workers,inspect interim outcome summaries,select endpoints,change budgets or simulate games.

Confirmed handles47154 and36650 both live with actual50-second polls. Initial read showed164 updates in parent1_public;later197 updates/3,227,648 decisions/216,157 round endings/27,662 matches/6,455,296 value rows. All177 study inputs and5 queue sources verify unchanged. Queue is currently waiting for the first training endpoint;successful final audit/export paths remain pending actual completed workers. This is execution preparation and verified waiting,not a new strength result. Latest verified24-page PDFf606acbb remains unchanged.


## First public-seat endpoint and independent matched warmup (2026-10-18)

First full trainer parent1_public (worker18188) exits0 after520 updates;runner47154 starts parent1_masked (worker35356). Public endpoint:8,519,680 decisions/569,561 round endings/73,093 completed matches/17,039,360 value rows. Final SHAe6fe6726ea4a207834fcd00655be0125ed8ced1f3c8f575da9d55d192ca4cf74,warmup SHA03b83f9db3e3a9e8edbe2c78548da09a2dccf4d5f6f9ead5557b45910fdf71cd. Queue36650 invokes the independent optimizer checker once;receipt runs/seat_observer_optimizer_audits/parent1_public.json SHA6e31d87e3259389f2afe416953bd067690f927cd469dc01202f933c17ac01e37 verifies96 finite per-parameter Adam states,48,000 steps for each actor parameter and49,920 for each value parameter. Initial-to-final L2 movement:discard0.926400944,pick0.691253454,koi0.587061317,value9.602626629. Initial actor and padded critic match the original retained parent;environment/CPU+CUDA RNG also match the archived observer initialization.

After the masked arm saves its20-update warmup,an independent CPU read of the four initial/warmup checkpoints verifies full initial numerical equality (actor,value,optimizer,environment,counters,runtime,Python/NumPy/CPU/CUDA RNG) and warmup equality for actor/environment/counters/runtime/all RNG. Configs differ only in seat_context. Production-source digests agree;Git metadata can differ across documentation-only commits and is not imposed as numerical identity. Each warmup has327,680 decisions/22,107 round endings/2,770 completed matches/655,360 value rows and1,920 actual Adam steps in each of6 value states;no actor optimizer states. All values and moments are finite. Critic tensors differ as intended:public seat-column L1 is2.241960048675537,masked0.0. Receipt runs/seat_observer_warmup_audits/parent1.json SHA09f4bb59b6fb60e6d91f800033056f047fe41bcc1a50614e8f431d134f6bcc35. Inspection adds no simulator decisions,rounds,matches or training.

Both47154 and36650 remain live. The queue records the completed public optimizer audit and waits for parent1_masked;do not manually duplicate queued endpoint/final audits. All177 bound study inputs and5 queue sources remain unchanged. Full comparison,final statistics and successful full seat-paper export remain pending. No interim strength summaries inspected,no model promotion or change to the verified24-page paper.


## Within-round policy-input/action retention and second seat endpoint (2026-10-18)

Added scripts/check_observer_temporal_reconstruction.py and scripts/check_observer_score_reconstruction.py. Their observation-only decoders receive a later562-feature row plus an earlier public round/turn/phase address. Own initial hand is recovered from remaining own cards and recorded own plays; initial field and timestamped public events reconstruct the earlier actor input. Public action records also recover the chosen action. The caller enforces the same played round,including episode identity. Neither decoder receives a simulator state,hidden opponent hand,stock order,earlier observation or actor output.

Both successful commands use `--output runs/observer_temporal_reconstruction/audit.json` and `--output runs/observer_score_reconstruction/audit.json`,respectively,with `.venv/Scripts/python.exe -m scripts.<module>`. Exact comparison succeeds on all3,764 ordered same-round pairs from the512 saved roots:parent1 1,864 pairs/977 inactive-viewer successors,parent2 1,900/991. Earlier phases total3,259 discard/286 hand-capture/82 draw-capture/137 continuation;maximum separation19 decisions. Input/action verification rejects512 not-yet-executed requests and512 different-round requests. Native observation extraction leaves all states unchanged. Zero new decisions,rounds,matches,branches,actor forward/backward passes or training. Ruff passes. Temporal receipt SHA534146a5f301b9fdb617a81fee1493ba8848c5c00d740bc91c2873ff00ae649d;input/action receipt SHAbe39d3c08a47e856aba08496a7ace57a93932f7cc52c5190434433c320d80851. All14/18 bound inputs verify unchanged. See OBSERVER_TEMPORAL_INFORMATION.rst for construction and scope. This is measured retention of earlier policy-score arguments within a round,not a universal sufficient-history,exact-value,unbiased-PPO or strength result. The verified manuscript remains unchanged pending final comparison.

Meanwhile parent1_masked worker35356 exits0 after520 updates:8,519,680 decisions/568,847 round endings/72,993 matches/17,039,360 value rows. Final SHA6ce3db813418676098565d79a35ba889933ac7e56863fb2765ded0555cb6e42a. Queue36650 independently audits actual Adam states;receipt SHA305da3fe7791fef0be9ef4963b248949fc698c65ad0eb202a207b089ca5c650a verifies48,000 actor/49,920 value steps and masked seat-column L1 exactly0. Runner47154 starts parent2_public (worker54120). Two of4 trainers and3 of16 evaluation cells are complete;no worker failure. Both live handles continue and all177 study inputs plus5 queue sources verify unchanged. No interim strength summaries inspected or protocol conditions changed.


## Third seat endpoint and second independent matched warmup (2026-10-18)

Parent2_public worker54120 exits0 after520 updates;runner47154 starts the fourth/final trainer parent2_masked (worker57656). Completed public arm:8,519,680 decisions/568,805 round endings/73,111 matches/17,039,360 value rows. Final SHAcf0ccb1ac494af735598482cbb88ff387adc164d9df52090b14dcc719b02b86a,warmup SHA0c46c104e4ca6f334a438c242eb58171e9f34571a8f23b19699edb2a81f0da92. Queue36650 independently audits its actual optimizer;receipt runs/seat_observer_optimizer_audits/parent2_public.json SHAc5c9e4fabcc902bd9d450c26312bf14537a06445224769b8baa9a22e5a39410a passes96 finite Adam states,48,000 actor/49,920 value steps. Initial-to-final L2 movement:discard0.912969016,pick0.669265448,koi0.575634545,value9.901226992. Final public seat-column L1=4.225125313.

After the masked arm saves warmup.pt,an independent CPU2 read of all four initial/warmup checkpoints confirms full initial numerical equality and warmup equality of actor/environment/counters/runtime/Python/NumPy/CPU/CUDA RNG. Configs differ only in seat_context;all provenance except Git revision agrees. Both warmups have20 updates/327,680 decisions/21,851 round endings/2,752 matches/655,360 value rows. Each has six finite value Adam states at1,920 steps and no actor updates. Critic tensors differ as intended;public seat-column L1=2.2448184490203857,masked=0. Receipt runs/seat_observer_warmup_audits/parent2.json SHA2e701b0864064f23b4338b394870eabb918a9495210876789d7c463b28e65322. This check adds no simulation,training or strength inference.

Three of4 training jobs and6 of16 evaluation jobs now terminate0. The remaining trainer and three evaluators remain live;completion queue waits for parent2_masked. All177 protocol inputs,5 queue sources and9 third-endpoint audit inputs verify unchanged. No interim outcome summaries inspected,no protocol changes,no deployment promotion. Full comparison,final independent paired audit and paper export remain pending. Verified PDF remains unchanged.


## Temporal-information result integrated and opening simplified (2026-10-18)

Revised the manuscript opening to lead with the measured adaptation result,then the exact critic-conditioning failure and its Koi-Koi tests. Abstract decreases from202 to184 whitespace words while retaining the achieved win/latency figures,negative primary outcomes and the positive learner-versus-lambda1 comparison. Introduction combines repeated control-study qualifications into one paragraph and connects those results to the representation diagnostic. The expanded Related Work is unchanged. Every previous source numeric token remains present.

New Section9.4 explains how a later562-feature observation and a supplied public decision address reconstruct an earlier actor input and action. It reports all3,764 exact chronological pairs,1,968 opponent-turn successors,all four root phases and maximum separation19 decisions. The text identifies the policy score explicitly and states the within-round checked scope. AppendixA records3,259 discard/286 hand-capture/82 draw-capture/137 continuation roots,512 unexecuted-action and512 different-round rejections,shared trajectories and zero additional simulation/inference/training. This integrates completed evidence,not an interim seat-strength estimate.

Tectonic build succeeds. All24 pages rendered and visually reviewed;after the final score-notation refinement,pages12--24 rechecked and pages1/2/12/14/21/22 inspected at full size. No overfull/underfull boxes,missing glyphs,unresolved references,page-bound violations or overlapping figures/tables. All20 fonts embed and render;the pre-existing Fontconfig configuration notice remains benign. Actual bibliography has25 cited entries. All9 prior evidence ledgers plus the new temporal/action report bindings verify493 unique files. The two new report identities and all reported counts were checked directly against their saved receipts.

PDF output/pdf/manuscript.pdf SHA0bedb83fb8188ca6f047caf6c03133c2dabd28143954c040ee2dd0313c9c7e5d. Source SHA7def180f63c74205864ed81277557df4a2f9ab1c0c9660d5e0ee25acc6000889. QA runs/manuscript_temporal_information_qa.json SHA95572ce4fce4191c32f7320f4b7d7985907178344d67841390ab5c055e46ad41. Previous24-page PDF retained as output/pdf/manuscript_observer_f606acbb.pdf. The ongoing seat-study inputs and conditions are unchanged;no interim outcome summaries inspected. Last trainer and three evaluators remain active. No new model promotion or simulator work from this revision.


## All seat-study training and endpoint audits complete (2026-10-18)

Final trainer parent2_masked worker57656 exits0 after520 updates. Counters:8,519,680 decisions/570,358 round endings/73,191 completed matches/17,039,360 value rows. Final checkpoint SHA37545a92280b7aa0656265bccf45b020c6b449a635532ccfb71bba78ce4a385d. Queue36650 invokes its independent optimizer audit once;receipt runs/seat_observer_optimizer_audits/parent2_masked.json SHA4edacb220b9a308c6f91a8b3685b6ecb726a9f1f2934521cceb45c7ab6c2c6c7 passes. All96 actual Adam states are finite with48,000 actor steps/49,920 value steps;masked seat-column L1 stays0. Initial-to-final L2 movements:discard0.897750994,pick0.697228298,koi0.580319278,value9.813957572.

All four fixed520-update arms now terminate0 and all four independent endpoint receipts pass. Re-reading their bound training_complete records gives34,078,720 total new decisions/2,277,571 round endings/292,388 completed matches/68,157,440 value rows. Each arm retains20 warmup and500 policy updates. These totals count the four study continuations,not prerequisite smoke/capability work or historical controls. Every input of each endpoint receipt revalidates,as do177 study protocol inputs and5 queue sources. No new simulation or training from this read-only check.

Runner47154 remains live with6/16 evaluation cells complete and three CPU2 evaluators active. Queue36650 switches to waiting_full_evaluation;it will run the final independent paired-outcome audit and paper export only after the whole study completes. No interim strength summaries inspected,no checkpoint selection,promotion or protocol changes. The latest verified24-page PDF remains unchanged. Full strength comparison is still pending.


## Retained Transformer all-action search: bounded saved-root diagnostic (2026-10-18)

Commit8afcd1e adds isolated scripts/published_action_search.py and the fixed diagnostic protocol docs/PUBLISHED_ACTION_SEARCH.rst. This closes the card-action adapter gap for the retained Transformer parents,without altering production,canonical play,initial dealer,live seat-critic study or completion queue. The same observation-consistent worlds are used for each baseline/alternative pair;every branch completes a match with observation-only frozen self-play continuations. The alternative with largest mean win gain must pass a2.5-standard-error margin and then a fresh independent confirmation. This is established Monte Carlo rollout policy improvement,not a new algorithm or guaranteed nonregression.

CPU/CUDA tests replay actual branch decisions and terminal states through the independent Python reference for all three card phases and both acting seats. Hidden-world substitutions preserve observations,actions and records;confirmation only accepts/rejects the previously selected candidate. Related suite18 passed,zero skips,20.43s,11 expected upstream Transformer warnings. Ruff passes. Test simulation is functionality work,not match-strength evidence.

Fixed command `.venv/Scripts/python.exe -m scripts.check_published_action_search --output runs/published_action_search_capability`;session22522 exits0. Before outcomes,the diagnostic selects the first two unforced saved roots per card phase/seat from each parent's existing256-position corpus. Each parent12 roots,24 total;32 common proposal worlds per root,256 fresh confirmation worlds only upon a positive margin. Existing root tensors match their original recorded SHA. No additional source simulation or training. All68 alternative comparisons finish;neither parent proposes or executes a change,so confirmation work is zero. Do not extend this corpus or tune its threshold from the result.

Parent1:220,168 executed branch decisions/15,493 round endings/2,176 completed branches;mean1.361486s,max4.144434s per root. Parent2:233,387 decisions/16,069 rounds/2,176 branches;mean1.336779s,max3.622731s. Total453,555 decisions/31,562 rounds/4,352 branch completions. Repeated baseline trajectories are fully counted;these are not independent source matches. Timings include cold calls and concurrent CPU evaluations.

Report SHAf6f686c33500c3e9f6a2b1eb17d1f907632211aa193523cfb0be8f3794038e6d binds unchanged inputs and raw terminal-point records. Separate standard-library recomputation from both JSONL files verifies all68 means,margins,point differences,selected actions and complete work sums;receipt runs/published_action_search_capability/independent_audit.json SHA03bf0aca56e34f079780c7100afdb0607bda6e485d5b02b697192b3ed85f2a85. There is no playing-strength evidence from this diagnostic. All177 live seat-study inputs and5 queue sources reverify unchanged. Runner47154 remains active with9/16 evaluations complete;queue36650 waits for full completion. The paper PDF is unchanged.


## Round-boundary information diagnostic (2026-10-18)

The within-round decoder does not claim recovery after a round reset. A bounded source-level investigation checks whether this limitation alone justifies adding critic memory. scripts/check_round_boundary_information.py enumerates all8 legal first actions at the first retained source root of each parent,then uses that parent's strict CPU2 public-observation greedy policy until the round ends. There is no search selection,training or new source collection. Every actual transition/result is checked against native CPU;all16 earlier input/action pairs are reconstructed exactly from the last observation before the boundary.

Command `.venv/Scripts/python.exe -m scripts.check_round_boundary_information --output runs/round_boundary_information` exits0;Ruff passes. Parent1 executes128 branch decisions/8 round endings/0 match completions;parent2 executes97/8/0. Total225 decisions and16 round endpoints from2 reused roots,not16 independent deals. All208 packed successor-state entries and both562-feature observer rows coincide for11 distinct initial-action pairs under parent1 and8 under parent2. Thus these19 collision pairs witness actual loss of earlier action information across a reset. The saved branch traces,endpoints and features bind the result;report SHA21c23f7e047e4adebb0b2d7fcadd159c0d33ac51b6a7b775b04e872935651a85 verifies11 unchanged inputs.

This is not evidence of critic bias. Equal full successor states also yield the same future transitions under any fixed state-based continuation and coupled randomness. Forgetting an earlier action does not by itself show loss of predictive information about the future return. Do not infer that recurrent critic memory is required from these collisions or from failure of the within-round decoder across a reset. A memory intervention would require separate evidence of useful predictive information;this probe does not supply it. No new strength comparison or manuscript claim is made. The active seat-critic evaluation and completion queue remain live.


## Public-seat comparison completed and integrated (2026-10-18)

Runner47154 and completion queue36650 both exit0. All four fixed520-update training arms and16 evaluation cells completed without worker failures. Analysis SHAb6e5c7846c1a4bde9745ce3bdf29c865b4fbf6d2824e57b535372a4794de5833;independent audit b5bb55e6469394a9d09a29001efe86114708a6e45528b358f6030155b252b552. The queue runs scripts.audit_seat_observer_study and then scripts.export_seat_observer_paper_evidence with protocol SHA24aaa955f7556177c63531e6cd9951afa510e778f909a3c1c2b3c2e33548243f and that actual audit SHA. The successful exporter writes all16 outcomes,24 contrasts,three tables and evidence ledger9484d19f13ceb25ba39c73427b1c61340d6226229b804067936a9fc7188bd937. Queue receipt960cdfb624178d2fea5da9dec8096e760dcf8de56f2d0f5f73ea44f1e7c91a63;all bound inputs and sources unchanged.

Primary criterion fails:1/8 lower win bounds positive. Public-minus-parent win changes (percentage points,95% intervals):parent1/WP +0.405[-0.415125,1.230],parent1/Point +0.110[-0.685,0.920],parent2/WP +0.145[-0.675,0.955],parent2/Point +0.860[0.070,1.655]. Public-minus-masked:parent1/WP -0.500[-1.320125,0.340125],parent1/Point +0.130[-0.670,0.915],parent2/WP -0.120[-0.985125,0.730],parent2/Point +0.245[-0.560125,1.045125]. All four information contrasts include zero. All four public-minus-archived-observer win intervals also include zero. No endpoint selection,promotion,protocol extension or101M confirmation.

Secondary public-minus-parent point margins are +0.5096[0.03658,0.97062],+0.4634[-0.01162,0.937805],+0.8026[0.34679,1.262205],+0.9800[0.517,1.449005] in the same order. All public-minus-masked point intervals include zero. The masked condition's point improvements over its parent have positive lower bounds in all four matchups,so point gains do not establish an effect of the public-seat input. Absolute public Point win scores are50.685%[49.929875,51.480] and51.345%[50.590,52.085125];their point margins remain negative:-1.6232 and-1.2672. Neither absolute performance nor the one positive primary contrast replaces the preregistered replicated criterion.

New training totals34,078,720 decisions/2,277,571 round endings/292,388 completed matches/68,157,440 value rows. Archived561-input controls add17,039,360 pre-existing decisions/1,135,793 rounds/145,987 matches/34,078,720 value rows as a separate dependency. Evaluation totals160,000 matches/18,586,219 decisions on5,000 shared duplicate deals28,000,000--28,004,999. Final audit independently reproduces all outcomes,contrasts,bootstrap intervals and within-parent warmup equivalence. Existing endpoint optimizer receipts remain valid. All study work is now terminal.

Manuscript integration adds the completed comparison in Section9.5,primary Table8,secondary Tables15/16 and AppendixA accounting. Abstract/Introduction/conclusion explicitly state that the representation repair did not establish replicated win improvement. Related Work is byte-identical after newline normalization;all prior numeric tokens remain. The prior24-page PDF is preserved as output/pdf/manuscript_temporal_information_0bedb83f.pdf. Build command external/tectonic/tectonic.exe --keep-logs --outdir output/pdf docs/paper/manuscript.tex. Bibliography uses a local small font to avoid a nearly empty trailing reference page;final25 pages,25 references,20 embedded fonts and197-word Abstract. No numerical evidence or reference was removed.

Final PDF SHA37883a4104179ba9c7f6a7ea461cc2a85adc9caab20766ee4610556c1e3179b0;source eeaefebacb95d1d204f38bced9ad78e1115e74cd83d277ffeea2c76c3b18b264. QA runs/manuscript_seat_qa.json SHAc839b01cf472b2af8836fea481001849e370a935976cf6e3772a9d3f31cbacd2 verifies576 unchanged bound files. All pages rendered/contact-reviewed;full-size pages1/2/12/13/14/23/24/25 reviewed,including final bibliography reflow. No over/underfull boxes,undefined references,missing glyphs,page-bound violations or observed visual defects. The pre-existing Fontconfig notice does not prevent all fonts from embedding. Initial dealer and playable canonical model remain unchanged. This completes the fixed comparison and its paper integration,not the user's stronger-model goal.


## Joint history-conditioned world generation (2026-10-18)

Reviewed the completed conditional-world diagnostics before choosing the next intervention. Earlier exact known-WP support changed selected-root values,including a sign reversal,but was never integrated into a retained-Transformer deployment adapter. The current uniform Transformer search also omits opponent-action likelihoods. Repeating those diagnostics or treating their selected roots as new strength evidence is unwarranted.

Added isolated scripts/history_conditioned_worlds.py. It enumerates all valid hands only when the public combinatorial bound is<=100,then replays every observed action against each named deterministic observation-only opponent. Own actions are replayed but not scored as opponent evidence. Compatible hand/model pairs retain the model-prior mass before one joint normalization;separate per-model normalization is deliberately avoided. A proposed application uses WP/Point with equal priors and no actual-opponent identity input. The prior must precede the current round's observations,avoiding repeated-history double counting. Empty support,oversized spaces and stale support explicitly refuse sampling. Future stock and deal seeds are freshly randomized after joint hand/model sampling. No production source,canonical model or rule changes.

This is established Bayesian conditioning,not a new general method. The contract in docs/HISTORY_CONDITIONED_SEARCH.rst relates it to existing information-set search/opponent-model work,including Goodman2019 https://arxiv.org/abs/1902.06075 . The module does not implement that paper's re-determinizing tree search. Any further contribution must be established through the concrete Koi-Koi intervention and measured comparison.

Final command .venv/Scripts/python.exe -m pytest tests/test_history_conditioned_worlds.py -q --basetemp runs/history_conditioned_tests_20261018b --junitxml=runs/history_conditioned_worlds_tests.xml passes4,0 skipped,in2.05s with one expected upstream Transformer warning. Ruff passes. Tests cover unequal model priors/overlapping support,zero support,budget refusal before inference,hidden-world invariance,deck validity,fresh stocks/seeds,root preservation and stale support. The real-model fixture fixes parent1 saved root13 with12 candidate hands and verifies all parent/Point compatibility flags by independent native CPU history replay,including the actual generating-parent hand. Earlier pre-format invocation also passed4 in2.41s;both are functionality work,not training or strength evaluation.

Validation receipt SHA7092ead4f980a37047f14178aa63c677397f658ac9d41126765cf945de14f9f0. The completed seat-study artifacts and25-page paper remain unchanged. No simulation study is running. Next work is the terminal-search adapter and CPU/CUDA trajectory/routing checks;any fresh strength protocol must compare with an unconditioned opponent-mixture control to isolate history conditioning from the changed continuation model.


## History-conditioned terminal adapter and fixed capability complete (2026-10-18)

Commit2e4162c implements scripts/history_conditioned_search.py,its correctness tests and the fixed protocol in docs/HISTORY_CONDITIONED_SEARCH_ADAPTER.rst. It retains strict CPU2 parent fallback/public assistance,compares every legal alternative at publicly bounded roots,and keeps one sampled opponent label across paired branches,seat changes and round boundaries. History uses strict CPU WP/Point action compatibility with equal within-round priors;uniform mode uses an independent equal opponent mixture and uniform valid hands. Simulated models are explicitly separate frozen batched approximations. Original production,inference and dealer rules remain unchanged.

Six adapter tests passed initially in16.66s. After closure-binding/style corrections,the combined sampler/adapter/all-action suite passes16,zero skips,in33.01s with13 expected upstream Transformer warnings;Ruff passes. Full CPU/CUDA branches from both root seats replay through the Python reference with all three model routes and multiple rounds. Independent upstream features yield matching routed actions on these probes. Other tests cover fresh confirmation,actual-hidden-world invariance,all alternatives and both refusal paths. Test confirmations are deliberately forced only to exercise routing and are not strength evidence.

The committed fixed command .venv/Scripts/python.exe -m scripts.check_history_conditioned_search --output runs/history_conditioned_search_capability exits0(session53451). Two parents x WP/Point x both seats gives8 source matches on4 shared deal streams31210101/31210102/31210201/31210202. Each source match is completed and every transition checked against native CPU. Each supplies its first unforced,unsolved parent decision with<=100 public hand combinations;all8 cases have a root. Source work940 decisions/60 round endings/8 completed matches. No search action influences source collection.

Both modes run all8 roots with64 proposal worlds,256 conditional fresh confirmation worlds and2.5 standard-error margin. All16 calls complete;both modes propose/change zero actions,so confirmation work is zero. Actual opponent hands are included in their declared strict-model support at every root;this check occurs after action selection. Hypotheses can distinguish model support at some roots,but correct support alone does not establish improved play. History selected mean changes range-0.078125 to+0.015625 in draw-adjusted win units;none passes the fixed lower margin. Do not tune that margin or extend this diagnostic from its outcomes.

Future branch work:parent1/uniform46,735 decisions/3,528 rounds/512 branch completions;parent1/history46,606/3,517/512;parent2/uniform55,050/4,035/512;parent2/history53,830/4,006/512. Aggregate202,221 decisions/15,086 round endings/2,048 completed branches. Historical compatibility replay adds7,798+5,634=13,432 decisions,6,740 opponent calls and no round/match completions. These are distinct from source matches and from training. Four-root elapsed sums:parent1 uniform5.682446s/history14.230178s,parent2 uniform6.120318s/history11.278065s. Per-root observed history costs1.0694--4.6796s;timing includes cold calls/support enumeration,not an isolated service latency guarantee.

Capability audit SHAeb23a374ced89ebda06811e5cf694cf3ff80fae1e81951b110f4319449020efc. New scripts/audit_history_conditioned_search.py independently replays all8 source policy trajectories,checks first-eligible selection,source accounting and terminal points,and recomputes every candidate mean/margin/point difference,selected action and branch-work total from raw paired terminal points. Receipt SHA84cb1f01e99111059b319a7d7b0d6f9d97db1b2066301d9bc3d20242253c007d. Verification repeats940 source decisions/60 rounds/8 match endpoints on the same4 streams;it adds no new independent source deals and does not rerun future branches or every support hypothesis. All bound inputs reverify unchanged.

No process remains live,no new strength study is justified merely by this small functionality sample,and no model/PDF is promoted. The25-page paper is unchanged. Next inspect why action differences remain below uncertainty thresholds and review the existing public round-boundary calibration work before specifying further computation. Keep the completed capability fixed;any new method or strength comparison needs distinct evidence and a fresh protocol.


## Round-transition value construction and full-grid capability (2026-10-18)

Commit ab1db73 adds fresh public-context round starts,RoundWorkspace stopping at the first actual round ending,and finite backward evaluation for both absolute seats/dealers,WP/Point opponents,59 live point totals and8 rounds. No symmetry folding or private next-deal lookup. The resulting model uses independent fresh round deals and frozen batched actors;exact equivalence to the native finite-seed multi-round process is not claimed. Conditional Monte Carlo/truncated rollout are prior methods;novelty requires this concrete construction and measured benefit.

Command .venv/Scripts/python.exe -m pytest tests/test_round_transition_values.py -q --basetemp runs/round_transition_tests_20261018a passes7,0 skipped,in4.08s with4 expected warnings. Reference trajectories check CPU/CUDA,both root seats,forced initial actions,freeze behavior,bankruptcy/final round/draws and next-round endpoints. An independent exhaustive binary-path calculation checks backward values. Ruff passes. Production simulator,actors,dealer rules and canonical model remain unchanged.

Fixed commands .venv/Scripts/python.exe -m scripts.build_round_value_table --parent 1 --samples 2 --seed 32210001 --output runs/round_value_capability/parent1 and corresponding parent2 complete0. Parent1 audit4e82d813d9c413cb4d4541f38c284b2806c5f0ede6923d11c0e31eccd4a973e4:117,252 decisions/7,552 conditional round endings/1,401 true terminal endpoints,4.927178s measured core. Parent2 audit62cc05554239849711d8a432e1a89356cf20ab49a115d8bee045a1966c68a401:118,143/7,552/1,418,4.863394s. Aggregate235,395 decisions/15,104 rounds/2,819 terminal endpoints. Only16 distinct round/deal inputs,shared across all contexts and both parents;zero independent ordinary complete matches and zero policy updates.

New scripts/audit_round_value_table.py separately reconstructs every context's scalar backward recursion from saved public endpoints,verifies all per-round split files against the complete NPZ,and checks all32 groups,terminal/bankruptcy rules,bounds,reported counts and bound inputs/production source. It does not call the producer's backup,rerun neural trajectories or remeasure decision counts/timing. Parent1 independent SHA c63a2aadbf510ed02aa423be9af8c84ca53f38a220f28b1928a9906b600abd75;parent2 43bb249ce4a00855b133650d814a7d6bcc397fc99570d838b0da6bac903e1557. Maximum backup difference0 for both. These two-sample tables are functionality artifacts,not calibrated values or strength evidence.

Fresh fixed estimator protocol docs/ROUND_VALUE_ACCURACY.rst specifies two128-sample tables per parent,independent42M/52M streams,shared-deal bootstrap,and72-context512-replicate terminal references under both native and independently reseeded chance constructions. scripts/run_round_value_tables.py runs the four table cells in fixed order,logs every child and audits each completed table. Stage B implementation must be checked and pinned before reference collection. No outcome-dependent extension,actor training,table selection,policy promotion or101M consumption is authorized by this diagnostic. The stronger-model objective remains open.


## Round-value accuracy execution and search implementation (2026-10-18, ongoing)

Commit0bac769 fixes the four-cell128-sample table protocol;runner17145 executes it sequentially at runs/round_value_accuracy. Inputs/streams/budgets remain unchanged. Completion queue22813 executes scripts.finish_round_value_accuracy after six reference/uncertainty tests pass. It waits for all four table audits,then collects the two parents' fixed72-context512-replicate native/independent terminal references and runs256 shared-deal bootstrap replicates. Stage A totals1,933,312 conditional rounds;Stage B147,456 artificial-context terminal paths. Neither count denotes independent ordinary match evaluation. All runtime work and raw endpoints are retained. No source deal or sample count is increased from outcomes.

Commit68bc465 adds the reference implementation,bootstrap,analysis and completion queue. CPU-only first reference suite3 passed/2 deselected in4.31s;bootstrap test1 passed in1.47s. Final combined CPU/CUDA suite6 passed/0 skipped in10.18s with4 expected Transformer warnings;receipt runs/round_value_accuracy_tests.xml. Reference tests cover repeated round resets,both seats,all three policy routes,freezing,final-round equality,exact Python/native trajectories and accounting. Bootstrap tests compare shared-column weighted backups against explicitly resampled kernels. Timing overlapped the live table builder and is not an isolated throughput measurement.

Commita248de3 adds scripts/round_value_search.py and docs/ROUND_VALUE_SEARCH.rst. The adapter retains exact CPU fallback and strict bounded hand/model likelihoods,simulates only to first round boundaries,then looks up public-context values. Whole-table and particle bootstrap errors are combined;proposal uses A and fresh confirmation uses B. No independent-table-error division by particle count,no actual-opponent identity and no next-round private feature lookup. Table identities must match the parent;the real artifact loader requires completed estimator analysis. This does not make completion an accuracy or strength certificate.

CPU search suite3 passed/2 deselected in2.86s;full CPU/CUDA suite5 passed/0 skipped in4.87s with4 expected warnings,receipt runs/round_value_search_tests.xml. Synthetic fixtures independently reconstruct first-round trajectories,terminal/bankruptcy/draw lookups and joint uncertainty arithmetic;forced test acceptance checks A/B routing and actual-hidden-world invariance. These are functionality results only.

Commit174ee3b fixes the subsequent action-value comparison before execution:16 original-policy source matches on4 shared72M deal streams,first eligible unsolved koi root per case without replacement,512/1,024 proposal/confirmation particles for both methods,plus4,096 independent terminal-reference particles per retained root in8 chunks. Commit6fe8c31 supplies source-policy/raw-statistics/lookup audit. This comparison is not yet running and does not authorize promotion or a complete-match strength claim.

Commit4738c90 records direct prior-art inspection:upstream rl_train.py already uses next-round/points/dealer win tables;the released generation/regression scripts describe continuation selfplay and logistic point fits with a complementary dealer slice. Pipeline/source-file inconsistencies prevent claiming exact regeneration of the shipped historical table from that inspection. Do not describe our construction as introducing continuation tables to Koi-Koi. The contribution candidate is the explicit information-conditioned estimator and demonstrated benefit,which remains to be measured. The25-page manuscript and canonical playable model remain unchanged.


## Round-value accuracy and fixed koi comparison complete (2026-10-18)

All four table cells complete and pass independent scalar audits. Runner17145 exits0;table receipt SHA228147d7a0cab8dbfb1d420f6c262facfacd08453f3c626f922f244103ce7ed4. Total28,952,354 decisions/1,933,312 conditional round endings/370,029 true terminal endpoints;zero independent ordinary complete matches and zero policy updates. Measured runner wall1411.160056s. Each cell uses exactly128 samples for all3,776 contexts;2,048 unique round/seed inputs are shared across contexts and parents. A/B table roles remain fixed.

Completion queue22813 exits0 after both fixed terminal references and analysis. Queue receipt SHAa8ef60bcbc2b4e47d26f275a29987fe67a3185f2f6ac57bd7827c0d1eabb3b4a. Parent1 reference4,955,995 decisions/336,196 rounds/73,728 conditional terminal paths/131,212 reseeded nonterminal rounds;parent2 4,979,588/335,966/73,728/131,105. Aggregate9,935,583 decisions/672,162 rounds/147,456 paths. Native and independently reseeded trajectories share initial deals and agree exactly for all final-round starts. Raw endpoint/split/source checks and accounting pass. These are artificial predeal contexts,not ordinary match-strength evaluation.

Analysis SHAe3eb2f11adf82def79ca4fc61ebf74be9e1aff919a2389060ccca0b9a4f052c9 retains all64 table strata,576 table/reference comparisons and144 chance-model contrasts. A-minus-B signed-value MAE/RMSE:parent1 .063610570/.078879169;parent2 .061677590/.077832319. Adjacent-point contrast disagreement MAE/RMSE:.012627456/.018877065 and.013116021/.020222947. This is agreement between independent estimates,not accuracy against an oracle. Native terminal-reference observed signed-utility RMSE:A .120404005/.119287460 for parents1/2;B .049600014/.054188710. Independent-round reference RMSE:A .105243538/.102974301;B .053643621/.060805931. Divide signed-utility changes/errors by2 for draw-adjusted win units. Neither table is selected from these outcomes. Raw RMSE includes reference sampling error;bootstrap spreads are descriptive and are not simultaneous coverage guarantees.

New scripts/audit_round_value_accuracy.py independently reconstructs all576 per-context estimates/errors/intervals,144 chance contrasts and8 raw error aggregates,plus three fixed complete weighted-backup replicates(0,127,255) per table. All12 weighted backups match exactly(max difference0). Receipt SHA77cbddfb9b6909dbc88e92b784c190260e23b443fd30cbbfd40d48deb8beda24. Remaining253 bootstrap backups per table,aggregate bootstrap spreads and neural trajectories are not independently rerun. No new simulation from this audit.

Fixed command .venv/Scripts/python.exe -m scripts.check_round_search_comparison --study runs/round_value_accuracy --output runs/round_search_comparison exits0(session41597). All16 original-policy source matches finish on4 shared72M streams. Source work1,846 decisions/123 rounds/16 matches. First eligible unsolved koi selection supplies12 roots(7 for parent1,5 for parent2);four absent cases are retained without replacement. Both methods use512 proposal particles and the same conditioned worlds/model labels. Neither proposes or changes any action,so confirmation work is zero and real table-B search confirmation is not exercised by this study.

Terminal proposal work863,165 decisions/68,180 rounds/12,288 completed branches. One-round proposal work31,376 decisions/12,288 conditional rounds/0 terminal endpoints. Independent terminal references add6,908,116 decisions/545,896 rounds/98,304 completed branches(4,096 paired worlds per root). Compatibility replay adds46,460 decisions across the two methods;reference supports are reused without additional historical inference. All twelve alternative-minus-parent reference means are negative,ranging-.409790039 to-.000366211 in win-score units. Their2.5-standard-error lower margins are all negative. This small,correlated root set neither establishes global optimality nor justifies changing thresholds/extending the study.

Per-parent online timing and observed proposal error against the common independent reference:

- Parent1 terminal:7 roots,26.989371s total,win-score RMSE0.006626582,MAE0.004952567.
- Parent1 round:7 roots,15.191770s total,win-score RMSE0.019128171,MAE0.012301017.
- Parent2 terminal:5 roots,18.236142s total,win-score RMSE0.019874232,MAE0.018139648.
- Parent2 round:5 roots,10.747984s total,win-score RMSE0.014317382,MAE0.011497937.

Timing covers act(),including history and inference,but excludes constructors/table loading and the offline table budget. The large simulation-decision reduction is not an equal-size latency gain because strict history replay remains. Round estimates have worse observed error for parent1 and better for parent2;do not report a replicated estimation gain. Search audit SHA633288c149d12880bb4975a9218ce96fc2c53a4695d101220cd25a0d3f7c4281;independent source/raw/lookup/bootstrap audit96bd69476864f0b1df92949ad5ea46729a50b7686d74e3a5f72b26385a39df6a. It replays the same16 source matches,adds zero independent source deals,and does not rerun future neural branches or every compatibility hypothesis. Summary metrics receipt runs/round_studies_summary.json SHAfa19777d6726e0bd0dc7c95ac3c674b29a25344d29f0c1448aa72766cb89e158.

No process remains live,no model is promoted,no101M final-test deals are consumed,and no manuscript is changed. Preserve these fixed budgets and all negative results. The next research target is earlier card selection,not more late-koi particles:at the opening dealer decision there is no observed opponent action,so a uniform valid-hand sampler can represent the ideal shuffled-deal prior without enumerating the huge hand space. That observation is a proposed opportunity,not yet a new implementation or strength result.


## Opening-only card-search implementation (2026-10-18)

Added scripts/opening_search.py with terminal and round-value variants. Search eligibility is restricted to round0/turn0/discard/acting dealer,with eight legal own cards and32 unknown cards. This avoids enumerating10,518,300 candidate hand subsets when no opponent has yet acted. Under the ideal shuffled-deal model,uniform valid-hand rejection and stock permutation sampling suffice;this does not assert exact conditional sampling of the native finite PRNG's seed space. Model prior remains equal WP/Point,with no actual opponent identity. The sampler and original dealer rule are unchanged. Outside the opening,the exact parent and public guard return without search. All seven alternatives to the parent action are compared;existing candidate selection/confirmation remains intact.

Command .venv/Scripts/python.exe -m pytest tests/test_opening_search.py -q --basetemp runs/opening_search_tests_20261018a --junitxml=runs/opening_search_tests.xml passes5,0 skipped,in34.64s with4 expected upstream warnings;session85591 exits0. Tests cover both dealer seats for sampling,initial-deal validity,exact own/public observation equality,actual-hidden-world invariance,all eight card candidates,fresh confirmation,A/B routing,unchanged roots and no-simulation fallback outside the opening. Test margins are forced only to exercise confirmation;boundary values are synthetic. Two lint-only dictionary/set-comprehension rewrites follow the passing tests;Ruff and diff checks pass. No neural strength run or new complete source corpus is produced.

docs/OPENING_SEARCH.rst records the sampling argument,scope and required next comparison. No opening particle/strength study has been specified or started. Keep the completed late-koi and table-accuracy datasets fixed;do not select the better measured A/B table. The canonical playable model and25-page manuscript remain unchanged. All workers are terminal;next work is a fresh fixed opening comparison with independent terminal reference and full accounting.


## Fixed opening selected-action comparison complete (2026-10-18)

Commite297427 fixes the opening comparison before execution. Command .venv/Scripts/python.exe -m scripts.check_opening_search_comparison --study runs/round_value_accuracy --output runs/opening_search_comparison exits0(session34155). All16 source matches complete on four shared81M streams using original strict policies and unchanged dealer selection. Source work1,935 decisions/126 rounds/16 completed matches. Seat pairing supplies exactly8 candidate-dealer openings;the other8 cases are retained as ineligible. No source action is changed by search.

Both methods compare all7 alternatives per root on256 worlds,with matching proposal world hashes/model labels. Every proposal fails its fixed2.5-standard-error margin;zero changes and zero confirmations for both parents/methods. The selected-action union contains6 actions for parent1 and7 for parent2. Exactly4,096 new worlds per selected union action yield106,496 terminal branches. Six of the13 reference means are positive,but all13 fixed lower margins remain nonpositive. Means range-.026855469 to+.013183594 in draw-adjusted win units. No reranking or sample extension follows these results.

Terminal proposals:parent1 1,675,236 decisions/111,812 rounds/14,336 branch completions;parent2 1,689,328/111,753/14,336. One-round proposals:parent1 222,565 decisions/14,336 conditional round endings/6 terminal endpoints;parent2 224,408/14,336/10. Independent references:parent1 5,732,273 decisions/383,660 rounds/49,152 branch completions;parent2 6,706,593/446,188/57,344. Aggregate future computation16,250,403 decisions/1,082,085 round endings;135,168 complete terminal-search/reference branches and28,672 separately counted one-round branches. All history-likelihood work is zero,as specified for the empty-history root. Offline table/calibration work is reused and separately recorded in the preceding study.

Observed error against the independent reference,on the UNION selected by either method,with online act time excluding construction/loading:

- Parent1 terminal:4 roots/6 selected-union actions,74.315459s total;win-score RMSE0.020508176,MAE0.017740885.
- Parent1 round:4 roots/6 selected-union actions,9.546514s total;win-score RMSE0.009607681,MAE0.009241114.
- Parent2 terminal:4 roots/7 selected-union actions,72.221420s total;win-score RMSE0.035080692,MAE0.029872349.
- Parent2 round:4 roots/7 selected-union actions,9.709661s total;win-score RMSE0.004749605,MAE0.004228159.

Round estimates are more accurate on these selected targets at both parent seeds and faster online. This is not an all-action/population error estimate or a match-strength result:the underlying source hands come from four shared deals,and the reference itself is noisy. No formal population-level accuracy replication criterion was fixed. The observed selected-root particle standard errors(.00447--.00609) exceed table-only bootstrap standard errors(.000458--.001274);this identifies a sampling-precision opportunity,not a guarantee that more particles will improve decisions. The two error components are diagnostic and are not added as independent variances in the implemented joint bootstrap.

Producer audit SHAcb0eddc4ed712e49d367937e0e906461fa9d3d632756264c06b2689a4185bb49. Independent audit SHAf1a8d67022c95cec639374f21a5a30fb21dcf1ca1fe6c59f917f40e354bb6e71 replays every original source policy action and opening selection,reconstructs all112 proposal comparisons,checks candidate maxima/confirmation gates/common baselines,and recomputes all13 independent selected-reference estimates. It adds no independent source deals and does not rerun future neural trajectories. Reference/sample and table identities reverify. Summary/precision projection runs/opening_comparison_summary.json SHA36f9e83220035249b61d6f6ff951a3c7393cbd7e0316018cf829a7e8791ef5ba. All workers are terminal.

The canonical model,dealer rule and25-page manuscript remain unchanged;no reserved101M evaluation is consumed. Preserve all fixed budgets and outcomes. Next target informative mid-round card choices:strict finite-model likelihoods can reject a hand/model pair as soon as one observed action disagrees. Investigate this exact short-circuiting to make larger public hand spaces practical before specifying a fresh midgame comparison. This is a proposed computation improvement,not yet implemented or evidence of stronger play.


## Exact likelihood pruning and midround comparison launch

Added scripts/pruned_history_support.py: share observed-action replay across stateless deterministic opponent models and stop a model after its first disagreement. The joint posterior and RNG-coupled sampled worlds match complete replay; no approximate GPU likelihood is used. The complete suffix of rejected hands is outside this routine's validation contract.

Command `.venv/Scripts/python.exe -m pytest tests/test_pruned_history_support.py -q --basetemp runs/pytest_pruned_history_v1 --junitxml=runs/pruned_history_tests.xml` passes4 tests,0 skips,41.99s. Real fixture roots7/32 use strict CPU WP/Point. Command with tests/test_midround_search.py, runs/pytest_midround_v1 and runs/midround_tests.xml passes3 tests,0 skips,18.31s, including CPU/CUDA, both seats, all three alternatives, forced test-only confirmation, hidden-world invariance and public-cap fallback. Expected upstream warnings only. Synthetic boundary tables are functionality tests, not strength.

Frozen before execution: docs/MIDROUND_SEARCH.rst and runner scripts/check_midround_search.py in542b730. Run command `.venv/Scripts/python.exe -m scripts.check_midround_search --output runs/midround_search_comparison --study runs/round_value_accuracy`, session89124. Sixteen complete source matches yield12 eligible first-round fifth-discard opportunities. Uniform/history round search1024/2048, independent all-three-action conditional references4096 each. Results pending; no threshold tuning, extension, promotion or101M use. Auditor2b7528a checks all canonical histories with native transitions, source choice, generated worlds and raw statistics; it does not rerun every likelihood or future neural trajectory.


## Manuscript opening: contribution-first compression

Abstract shortened from197 to157 whitespace words. It now leads with the existing replicated RL-WP adaptation result and exact critic-conditioning counterexample,then states the empirical outcome of the three controlled studies once. Introduction removes repeated diagnostic sample counts and separates update correctness,representation sufficiency and measured strength. No new scientific claim,numeric result or bibliography change. Related Work and all subsequent source content remain identical after newline normalization.

PDF 0fc939c5a22d7cf1c3fd41016dc512434bd48ad803ba459cf09942f1b73480a5;source deead6e47956b6adbfb8625cd756341eeb138c0ead8fc8a9a93c1e9f7ac23183;QA af12e300ad23a3c3449f083068dd6efa3bf99d1dbb2674098271eb9352751151. All576 prior evidence inputs rechecked;all previous numeric tokens remain in the rendered25-page document. Twenty embedded fonts and25 references. All pages contact-reviewed and1/2/7/13/24/25 full-size reviewed, with no visible defects or TeX overflow/underflow/missing-glyph/reference warnings. Tectonic emits its existing nonfatal Fontconfig-config message; rendered fonts are embedded and correct. Prior PDF output/pdf/manuscript_seat_37883a41.pdf retains SHA37883a4104179ba9c7f6a7ea461cc2a85adc9caab20766ee4610556c1e3179b0. The separate midround diagnostic remains running; no new strength claim.


## Midround diagnostic interrupted by shared-baseline divergence

**Midround comparison stopped on a verified inconsistency:** producer89124 exits1 during parent2 case9 terminal references. Repeating the same baseline worlds across alternative batches changes one of512 final point differences from-16 to-22 (particle403). The common-baseline assertion remains intact. Six parent1 rows are complete and preserved;case9 support is saved,but no final comparison or independent audit exists. Waiting audit queue92527 is explicitly stopped after verifying its own Python PIDs;it exits1. No process remains. Failure receipt runs/midround_search_comparison/failure.json SHA36cb949fab6846e527185181c3005dc778b6ba59bc97c851bd732a15a4155821. Next reproduce the divergent continuation and correct batch-dependent inference before any new strength comparison. Do not silently delete the assertion,complete the old protocol with a changed actor or report partial results as a full study. The seven pruning/midround implementation tests remain passed;canonical model,dealer rules and101M deals are unchanged.

The failure is observed,while its numerical cause is still a hypothesis. Both mismatched point endpoints are losses;this does not justify discarding the inconsistency or claiming unaffected unobserved outcomes. All16 source matches and12 selected roots remain saved. Cases1/3/4/5/6/7 are complete under the original producer;both modes change zero actions in those partial rows. Case9 began but its action/reference record is not complete. The separate manuscript opening revision is already verified and unchanged by this failed diagnostic.


## Reproduced batch-dependent action and fixed64 continuation

**Fixed64 continuation verified on the failed case:** dynamic batch inference is isolated as an action-level instability:identical WP features give card18 a tie against32 with446 head rows,but card32 a1.430511474609375e-6 advantage with501 rows. Strict CPU/singleton CUDA choose32. Fixed64 pads phase-head calls to64 rows and chooses18 consistently on this fixture,so it is a different numerical policy,not a CPU-equivalence claim. Four CPU/CUDA order/padding/all-head/baseline tests pass,0 skips,13.06s. Archived case9 then completes all24 paired comparisons:2,621,657 decisions/193,064 rounds/24,576 branch completions in90.688s. Independent trace comparison reruns the same work and verifies1,101,312 shared baseline states/actions and every saved endpoint. Producer 6651189ce88d8c8b32f2a6b3689573385ac7b04be52a336334acdb1875898844;trace d0f7ce5517f1cff94ff7df83823b69bf44765f3a511f2587c6849def4ff15087. Sessions5521/11982/15765 exit0;no process remains. Original midround study stays failed and its partial rows stay preserved. Existing dynamic-batch value tables are not fixed64-calibrated. Next use a declared consistent continuation for any renewed search comparison;no strength claim or promotion.

Reproduction uses the original case9 seed83310010 and support,stopping at the first action divergence:chunk4,continuation call4,particle229,lane458, WP discard. The complete baseline state and native actor features match exactly. That earlier divergence is not claimed to be the sole cause of the original endpoint difference at particle403. Detailed raw legal scores are in runs/midround_batch_divergence/score_inspection.json SHA1e5de14e87dcf7170984ea93ba18cbec127062f65c1eab9ee6ebbf5cc39fa1f0.

The reproduction completed13 paired batches with actual work {"decisions": 1418351, "completed_branches": 13312, "round_endings": 104436},then stopped inside the next batch at its fifth continuation call;the incomplete tail is recorded separately,not added to completed-branch totals. Its saved tensor SHAc8ddf9210b4ac5ae06a61adf762b2731f46da05a616e32214826b260d89557ae. The producer source has two Ruff B023 reports for synchronous inner callbacks capturing labels/worlds; the callback is fully executed before the loop advances. Its already bound source is preserved;ruff --ignore B023 passes. Other new modules and tests pass the ordinary Ruff checks.

`.venv/Scripts/python.exe -m pytest tests/test_fixed_microbatch_continuation.py -q --basetemp runs/pytest_fixed_microbatch_v1 --junitxml=runs/fixed_microbatch_tests.xml` passes4 tests,0 skips,13.06s. All three actors and four live phases are tested under reversed order and sizes1/63/64/65/129. Four expected upstream warnings. The one-state fixed64 probe timings include warmup and are not an isolated speed comparison. The full paired diagnostic and trace replay use no new source deals;their simulation work must each be counted separately.

Commands: `python -m scripts.check_fixed_microbatch_rollout` and `python -m scripts.audit_fixed_microbatch_rollout` using the project venv. The trace audit uses the same continuation implementation and independent state/action/endpoint comparisons and arithmetic;it is not an independent neural implementation or universal floating-point guarantee. See docs/FIXED_MICROBATCH_INFERENCE.rst. The verified manuscript PDF remains 0fc939c5a22d7cf1c3fd41016dc512434bd48ad803ba459cf09942f1b73480a5;no new search result is inserted into it.


## Fixed64 terminal search and actual-match pilot launch

**Fixed64 terminal pilot running:** producer41912 has completed64 strict-parent source matches on16 fresh85M deals,with56 eligible fifth-discard roots and8 earlier-ending cases retained. Uniform/history terminal search both use1024 proposal/2048 conditional confirmation particles,the2.5 margin and fixed64 inference;no boundary-value tables. Each root receives4096 independent conditional terminal references for all three alternatives. After search,the fixed protocol completes128 actual candidate matches (both modes,including unchanged/missing roots),then independent audit queue47601 verifies trajectories,sampling,arithmetic and whole-deal pilot contrasts. Protocol SHA6f769d12d1e303fe20b9082944018900cff61696d9b864b04ac72bd762e949b4. Implementation/protocol3a2a3cf and auditor145a170 are pushed. New terminal routing/invariance tests2 pass,0 skips,35.01s;actual-match intervention tests2 pass,0 skips,2.49s. No result or promotion yet;original failed round-value study,canonical model,dealer rules,verified PDF and101M test streams remain unchanged.

The prospective diagnostic is fixed in docs/FIXED_MIDROUND_TERMINAL_STUDY.rst. Use both parents,WP/Point,both seats,eight replicates;seed85110001+100*opponent_index+replicate. Proposal seeds85210001+case_index;references85310001+case_index. Source work: {"decisions": 7242, "round_endings": 494, "completed_matches": 64}. Sixteen unique deals,not64 independent source deals. Search,reference and actual-pilot work are separate. Both methods use terminal simulation and the verified fixed64 execution policy. No old round table is loaded or relabeled.

`.venv/Scripts/python.exe -m pytest tests/test_fixed_midround_terminal.py -q --basetemp runs/pytest_fixed_midround_v1 --junitxml=runs/fixed_midround_tests.xml` passes2 CUDA integration tests,0 skips,35.01s. Both modes/both seats/all alternatives/fresh forced test-only confirmation/hidden invariance/strict fallback are covered. The likelihood fixture policies are Extremal; strict WP/Point support equivalence remains covered by the previously unchanged pruning tests. Actual-match tests use `tests/test_midround_actual_match.py --basetemp runs/pytest_midround_actual_v1 --junitxml=runs/midround_actual_tests.xml`:2 pass,0 skips,2.49s. Each fixture completes two full native-checked matches and verifies the one declared intervention point.

Producer command `.venv/Scripts/python.exe -m scripts.check_fixed_midround_terminal --output runs/fixed_midround_terminal_study`,session41912. Queue command `.venv/Scripts/python.exe runs/finish_fixed_midround_terminal.py`,session47601. The queue stops if the producer declares failure,then runs `scripts.audit_fixed_midround_terminal` only after completion. Its observation deadline is not treated as evidence that the producer stopped. Each completed reference chunk and each root selection is saved separately; no automatic resume or threshold changes.


## Numerical batching prior-art positioning

Read the official PyTorch2.12 numerical-accuracy documentation and Thinking Machines'10 September2025 article by Horace He and collaborators on18 October2026. Recorded source links and scope in docs/NUMERICAL_BATCH_PRIOR_ART.rst. Batch-dependent numerical results and invariant-reduction methods are established;fixed size does not by itself prove position invariance. Our local contribution here is the saved Koi-Koi paired-rollout failure and the measured fixed64 repair,not a new general inference algorithm. No external kernels installed,no live protocol/dependency or manuscript change,and no new simulation or strength conclusion from this source review.


## Retrospective search round coverage

**Search round coverage:** a hash-checked retrospective audit of the completed Transformer card/koi diagnostics finds no final-round roots:24 card roots and64 unsolved koi roots all occur in rounds1--7. These are shared-trajectory diagnostic positions,not88 independent matches. This leaves final-round card choices outside those two screens;it does not imply a strength gain. Current first-round fixed64 pilot and its conditions remain unchanged. See [coverage counts](SEARCH_ROUND_COVERAGE.rst);report SHA5f3bb5a3d3684116aa1987f4c0affe648efcfb39477ce7614742589bd9073613. No new simulation.

.venv/Scripts/python.exe -m scripts.audit_search_round_coverage verifies the original protocol/input/root hashes and counts public round,turn and phase fields. Ruff passes. Card roots by rounds1--8:parent1 [4,3,1,0,1,2,1,0],parent2 [6,3,0,2,0,1,0,0];koi:parent1 [5,7,4,3,4,3,6,0],parent2 [4,8,3,8,3,4,2,0]. Counts describe only these two existing corpora. Final-round card search is a candidate to assess after the live pilot,not a launched protocol or a post-hoc strength result. Earlier-round opponent-type evidence must be addressed explicitly if extending the current history model.


## Final-round card capability, CPU

**Final-round card capability (CPU):** an isolated uniform-proposal adapter now searches discard and both capture phases only in round8,using the fixed64 WP/Point mixture and unchanged candidate/confirmation logic. Three CPU tests pass,0 skips,13.83s:both seats/all card phases,full Python-reference trajectories,hidden-world invariance,fresh confirmation,strict fallback/public koi guard and invalid-state rejection. Artificial test contexts are not strength observations. Receipt11b3079d1b25dca854e0bc4affa3bd5d7d99e98111cdd97697e5713500fc1558. GPU integration and any prospective strength protocol remain pending until the current pilot completes. [Capability scope](FINAL_ROUND_SEARCH.rst). The two short CPU test runs overlapped the pilot;pilot timings are not isolated benchmarks. No extra GPU job,game change or model promotion.

Command: `.venv/Scripts/python.exe -m pytest tests/test_final_round_search.py -q --basetemp runs/pytest_final_round_v2 --junitxml=runs/final_round_cpu_tests_v2.xml`;session91621 exits0. The initial two-test run67741 also exits0 in13.14s;the final suite adds public-koi/forced-card/terminal tests. Three expected upstream Transformer warnings,zero skips/failures. Ruff passes. Final XML SHA3d2281bdccfe6a7359ba5807930c20d78a6a4fd01c24b03c8346e4b00ca43e31;adapter d4cfa29b50bbb67bd41301ebd0b82991dcc94679c5ed8dfebaccc4faa1f7a7ca;tests b393ade606cba0b615299a57224efafd9118ba5cc46f573e289825d45a3c872f. All live pilot inputs remain hash-identical. The uniform opponent mixture omits historical action likelihoods and is not a full-match posterior. No threshold is chosen from strength outcomes.


## Queued terminal summary and final-round GPU capability

**Post-pilot queue active:** session55682 waits for independent audit47601 to finish,then runs the complete terminal summary and three final-round CUDA integration tests. Receipt runs/terminal_summary_final_round_queue.json;helper runs/finish_terminal_and_check_final_round.py pins its inputs and requires3 passes/0 skips. No extra GPU work starts before the pilot/audit complete. GPU tests are prepared and lint/compile checked,not yet executed. Do not concurrently run the summarizer or reuse its queued output paths. Observation deadlines do not imply that the original jobs stopped.

Command `.venv/Scripts/python.exe runs/finish_terminal_and_check_final_round.py` starts55682,which reports waiting_for_independent_audit. The CUDA suite reuses the CPU information-invariance and guard checks,and separately replays all six phase/seat paired trajectories through the Python reference with device transfers for state comparison. Output logs: runs/terminal_summary_queue.log and runs/final_round_cuda_queue.log;XML runs/final_round_cuda_tests.xml;fresh basetemp runs/pytest_final_round_cuda_v1. The helper exits on upstream noncompletion/failure and never restarts the producer. No prospective strength study is launched.

## Final-round actual-match evaluator capability

**Final-round actual-match evaluator ready:** scripts/final_round_actual_match.py collects every naturally reached, unforced final-round card root and replays one explicitly indexed action. It does not select a root or launch a strength protocol. Early bankruptcy/no opportunity remains a complete recorded match. Four CPU tests pass,0 skips,2.29s:both seats,independent reference trajectories,exact unchanged prefixes/single intervention,unreachable or mismatched roots,illegal actions and early completion. Receipt runs/final_round_actual_capability.json. Two short CPU test runs (2.35s/2.29s) overlap the current pilot;no added GPU work. All42 post-pilot queued input hashes remain unchanged.

Command: `.venv/Scripts/python.exe -m pytest tests/test_final_round_actual_match.py -q --basetemp runs/pytest_final_round_actual_v2 --junitxml=runs/final_round_actual_tests_v2.xml`. Toy policies only;the natural seed19 early-bankruptcy fixture was located by a20-seed Python-reference search. These fixture/test matches are not strength-evaluation observations. The helper returns all eligible roots separately from its complete-match action/work record;packed-state equality is evaluator-only and never supplied as an extra actor input. No protocol/threshold/model/dealer change.

## Fixed64 terminal pilot complete and final-round CUDA capability

**Fixed64 terminal pilot archived:** protocol6f769d12d1e303fe20b9082944018900cff61696d9b864b04ac72bd762e949b4;producer auditbf93ed600f00b21a3a74459556153e856c15cdd1bc2239eaa74c3d18ab8be85d;independent audit34f120767015b0a83115dd42d64e81cd49f18ab9e45149fe8d2d1b10a2294c27;summary70add91731aa02400e1a07817534d1f9759a91ae7a9587b4967dda639f5fe3be. The64 source and128 candidate matches share16 distinct deals;all outcomes retained. Branch simulation totals222,670,284 decisions/16,127,337 round endings/2,084,864 terminal branches. No actual win/point difference,model promotion or101M use. Canonical model,dealer and verified PDF remain unchanged.

See [full results](FIXED_MIDROUND_TERMINAL_RESULTS.rst). Uniform proposals/confirmations:parent1 2/0,parent2 1/1;history:0/0 and2/0. All12 actual win and point contrasts equal zero. Parent1 uniform/history conditional-reference RMSE0.017779494/0.011938428;parent2 0.016888303/0.012876379. Act totals531.789/1570.534s and528.384/1539.475s;not isolated. The single accepted action has positive conditional-reference mean0.0147705 but identical realized terminal points. No gain or population equivalence is established.

Original audit47601 failed at case38 because normalized float64 masses sum to1.000000000000034,whereas its count-ratio comparison expected1.0 within2e-14. Original source is preserved at runs/fixed_midround_terminal_study/failed_auditor_25c9.py;failure and both original queue receipts are preserved. The corrected audit reconstructs equal-prior pair masses independently and requires exact saved-reduction equality;all other checks/tolerances remain. Two regression tests pass,0 skips,2.25s (runs/terminal_posterior_audit_tests.xml). The failed audit adds64 source replays/7,242 decisions/494 rounds and962,007 native history decisions,derived from its checkpoint and full audit;it reached no candidate replay. Corrected full audit5529 exits0.

After the failed queues terminated,the summary was run explicitly. Final-round CUDA integration then passes3 tests,0 skips,5.68s using tests/test_final_round_search_cuda.py and runs/final_round_cuda_tests.xml,with three expected upstream Transformer warnings. Original bound CPU-capability/queue inputs remain intact. Completion receipt runs/fixed_midround_terminal_completion.json. No final-round strength protocol has been launched.

## Final-round prospective pilot protocol and implementation

**Final-round prospective pilot implemented:** [frozen contract](FINAL_ROUND_STUDY.rst) specifies128 strict source matches on32 fresh87M deals,first unforced learner card per phase in round8,and384 separate single-phase candidate matches. Search uses1024 proposal/2048 confirmation particles,unchanged2.5 margin,and4096 independent same-proposal references per alternative. Missing opportunities remain in all comparisons. Two protocol tests pass,0 skips,2.19s;collector/intervention and CPU/CUDA search capability tests remain passed. New producer and independent auditor are lint/compile checked. No fresh source has been collected yet;launch and audit results will be recorded separately. No dealer/model/PDF/101M change.

Producer: scripts/check_final_round_search.py. Auditor: scripts/audit_final_round_search.py independently replays all strict source/candidate choices,first-per-phase selection,uniform hidden-world sampling and WP/Point labels,all alternative/confirmation/reference arithmetic,current-round branch counts,and12 whole-pair bootstrap contrasts. It does not rerun every neural future. Each phase defines a separate one-intervention policy;no combined policy is inferred. The32 source deals are shared across parents and seats. This is a fixed pilot with no automatic deployment or within-pilot budget/threshold extension.

Protocol test command: `.venv/Scripts/python.exe -m pytest tests/test_final_round_protocol.py -q --basetemp runs/pytest_final_round_protocol_v1 --junitxml=runs/final_round_protocol_tests.xml`. All selected source seeds are reserved for the prospective pilot;these tests use only case metadata and synthetic index/state fixtures.

## Final-round pilot launch

**Final-round pilot running:** producer78135 completes128 strict source matches on32 fresh87M deal streams,collecting172 eligible first-per-phase final-round roots. All root searches and all384 candidate matches remain in the fixed protocol. Independent audit queue27155 is live,waiting for the producer;receipt runs/final_round_search_finish.json and log runs/final_round_search_audit.log. Protocol867e2caa5e89e2516a94e46aea522ba63e9f5404d4f58f5639f3f95a5a6ce7d8 binds35 unchanged inputs. Contract/producer/auditor1a746ae are pushed. One root is complete at this snapshot;no outcome claim or promotion. Keep bound files unchanged and poll the same sessions;an observation deadline does not imply the original process stopped.

Command: `.venv/Scripts/python.exe -m scripts.check_final_round_search --output runs/final_round_search_study`. Audit queue: `.venv/Scripts/python.exe runs/finish_final_round_search.py`;it pins all producer inputs and its own source,waits up to6 hours for observation only,and never restarts the producer. All172 roots arise naturally under the declared strict policies. No additional GPU job is running.

## Final-round complete-result summary queue

**Final-round summary queued:** session91434 waits for independent audit27155,then runs scripts/summarize_final_round_search.py once. Receipt runs/final_round_summary_queue.json;log runs/final_round_summary.log. The summarizer verifies audit/input identities and retains all12 actual contrasts,six parent/phase opportunity/error/cost groups,accepted actions,all positive unadjusted reference margins and separate work totals. Source is lint/compile checked;the real summary has not run yet. Do not run it concurrently or reuse the queued output paths. Producer78135 and audit27155 remain unchanged;no extra GPU work.

Command after audit: `.venv/Scripts/python.exe -m scripts.summarize_final_round_search runs/final_round_search_study`. Outputs are fresh summary.json and summary.md in the study folder. The queue pins its own source,the summary script,protocol and35 original inputs;its six-hour observation deadline does not imply upstream termination. No new outcomes or strength claim are generated by preparing this summary.

## Natural final-round search complete (2026-10-19)

**Final-round pilot complete:** producer78135, independent audit27155 and summary91434 all exit0. The128 source and384 candidate matches share32 distinct deals;172 natural roots yield five accepted discard changes. Parent2 versus Point recovers one loss as a win: +3.125pp [0,9.375], point margin +0.4375 [0,1.3125]. Parent1 versus WP improves point margin +0.0625 [0,0.1875] without a win change; every other contrast is zero. All12 win intervals include zero, so there is no replicated gain or promotion. All1,591 independent-audit input hashes reverify. Conditional branches total115,163,667 decisions /8,916,992 round endings /8,916,992 completions. No live worker remains. [Full results](FINAL_ROUND_RESULTS.rst); receipt runs/final_round_search_completion.json. Next investigate fixed-prefix final-round training; no new protocol or model has been launched.

## Fixed-prefix terminal training capability (2026-10-19)

**Fixed-prefix tail training capability passed:** scripts/final_round_training.py trains only complete natural final-round tails with terminal learner-perspective rewards and deploys the unchanged parent in rounds1--7. Five CPU/CUDA tests pass,0 skips,13.00s,including independent transitions,opponent-terminal rewards,frozen weights and exact optimizer/RNG restart. A production-shape CUDA check completes3 updates /5,401 decisions /384 conditional tail completions;zero full matches. Audit aadb1273b63af7961d22fb69ebf010dd718d76d0b6ed640b2454eb726f5bfac2. No strength observation or promotion. [Method,objective and prior art](FINAL_ROUND_TRAINING.rst). JSRL,Backplay and reverse curricula predate the start-state idea;new strength and empirical contribution remain to be established. Next freeze an independent two-parent study.

## Fixed-prefix tail study protocol (2026-10-19)

**Fixed-prefix tail study frozen:** FINAL_ROUND_TRAINING_STUDY.rst fixes1,024 source prefixes per parent on512 shared88M deals,520 updates with128 complete tails per update,and16,384 strict actual matches on2,048 shared89M evaluation streams. Only update520 is evaluated. The primary criterion requires all four parent/opponent win-change lower bounds positive;no checkpoint selection,deal extension or promotion. Producer and independent recorded-action/arithmetic auditor are implemented;two protocol tests pass with no skips. Source collection has not started. Existing capability and canonical policies remain unchanged.

## Fixed-prefix tail study launch (2026-10-19)

**Fixed-prefix tail study running:** producer19319 is collecting the frozen natural88M source prefixes,then runs520 updates for each parent and16,384 strict full evaluation matches. Protocol99c47e73db4dbd08374e8defb95a0689d89a5ba4170f954ba5bb60586da32087 binds57 unchanged inputs;contract/producer/auditor4ff1064 are pushed. At launch verification parent1 has385 prefixes/365 survivors;these are not strength observations. Independent audit queue54117 waits for producer completion,then runs the frozen recorded-action/arithmetic audit and writes a concise summary. Receipt runs/final_round_training_finish.json;log runs/final_round_training_audit.log. Do not restart either worker or change bound inputs. An observation deadline does not stop the producer. Canonical model,dealer,PDF and101M remain unchanged.

## Tail study parent1 training completion (2026-10-19)

**Tail study parent1 training complete:** all1,024 natural source prefixes yield938 surviving round-eight roots and86 early completions. Fixed520 updates complete929,099 conditional decisions (458,275 learner/470,824 opponent) and66,560 conditional final-round completions;these are not full matches. The producer checkpoint audit verifies unchanged prefix/opponent tensors,all96 Adam parameter counters and finite optimizer state. Audit a0314168e25edc6c9cb5479a5cfd3e2c9ca2c12b930c3c5f3307c61f267705ab;final checkpoint 7574187043111a5ed15b1d9c7003590fa18a14bbdc8243dd5b0b8fb16fc9d777. All57 protocol inputs and five checkpoint-audit inputs reverify. Producer19319 remains live collecting parent2 sources;queue54117 remains live waiting. Actual evaluation and independent full-study audit are pending;no strength claim or promotion.
Source prefix work: {"decisions": 105832, "round_endings": 7045, "completed_matches": 86}.

## Both tail-training endpoints complete (2026-10-19)

**Both tail-training endpoints complete; evaluation running:** each parent finishes520 updates and66,560 conditional final-round completions. Combined work is1,860,647 decisions (918,437 learner/942,210 opponent),133,120 conditional rounds/tails and zero full training matches. Natural source banks contain938/918 roots from1,024 prefixes per parent. Both producer checkpoint audits pass:unchanged prefix/opponent tensors,all96 Adam counters and finite optimizer state. Receipt runs/final_round_training_study/training_completion.json;all57 protocol inputs and both checkpoint-audit inputs reverify. Producer19319 now runs16,384 strict full matches on2,048 fresh89M deals,at most three CPU2 workers. Queue54117 remains live waiting for full completion,then independently replays recorded actions and recomputes contrasts. Protocol99c47e73db4dbd08374e8defb95a0689d89a5ba4170f954ba5bb60586da32087 and all conditions remain unchanged. [Frozen study](FINAL_ROUND_TRAINING_STUDY.rst). No strength result,promotion,PDF change or101M use.
Source work: {"decisions": 212212, "round_endings": 14101, "completed_matches": 192}. Parent identities: [{"parent": 1, "survivors": 938, "final_sha256": "7574187043111a5ed15b1d9c7003590fa18a14bbdc8243dd5b0b8fb16fc9d777", "training_audit_sha256": "a0314168e25edc6c9cb5479a5cfd3e2c9ca2c12b930c3c5f3307c61f267705ab"}, {"parent": 2, "survivors": 918, "final_sha256": "06346d9c7127537f64b21b8b1a16a1eb16156fb765a8f7e34141c39803888dd8", "training_audit_sha256": "dd46fbd8b80b8d2b9a1ba02c89c85d7a9ee0c362e3e5afc0dab7f2a9b4d1ca93"}].

## Public hand revelation from existing training prefixes (2026-10-19)

**Public hand-revelation opportunity found:** a separate read-only replay of the2,048 completed training prefixes finds all eight opponent hand discards in446/446/451/450 of512 cases for parent1-WP/Point,parent2-WP/Point. This yields3,982 revealed rounds within1,793 prefixes on512 shared deals. The detector receives public events only;no private hand,held-out evaluation record or neural prediction is used. Two tests pass,0 skips,0.15s. Replay repeats212,212 decisions /14,101 round endings /192 early completions,adding zero independent matches. Artifact 880037d22b20f55cacb55d9cac3186e427d09c656c7416d5c19fe66c8d391cbf. This motivates retrospective actor-input reconstruction and opponent identifiability checks;it is not classifier accuracy or strength evidence. [Scope](PUBLIC_HAND_REVELATION.rst). The short CPU diagnostic overlaps current evaluation;all57 live-study inputs remain unchanged.

## Revealed-hand opponent-input capability (2026-10-19)

**Revealed-hand opponent-input reconstruction passed:** a public-only historical snapshot plus eight subsequently observed hand discards reconstructs744 fixture actor inputs exactly against both native and strict upstream builders,covering both seats/all four phases. Raw/assisted legal masks match;hidden hand reassignment leaves snapshots unchanged. Use before revelation and cross-match/round/actor inputs are rejected. Two tests pass,0 skips,2.85s;no neural inference,identification or strength evidence. Capability ce1346dafa03594d445411873f9cff254531e55c771dddb0bbe25370039fe198. [Scope and prior art](REVEALED_OPPONENT_INPUTS.rst). Next assess finite WP/Point candidate elimination on retained training prefixes;do not change the current tail study. All57 live-study inputs remain unchanged.

## Delayed finite opponent-model elimination capability (2026-10-19)

**Delayed opponent-model elimination implemented:** scripts/revealed_opponent_models.py retains WP/Point candidates across rounds and predicts only after eight public hand discards reveal that round's initial hand. Ambiguous and empty candidate sets remain explicit;opponent identity is never an input. Three tests pass,0 skips,2.08s,including persistent/empty-support behavior and16 exact original-assisted CPU action comparisons across both models/seats/phases. Capability e8611d239775b16dfe2d4ba15aa5c0acbe463d6aa6b98db5d0eeae24a3ebb282. The all2,048-training-prefix identification runner and [fixed scope](REVEALED_OPPONENT_IDENTIFICATION.rst) are lint/compile checked,but have not run. Additional neural inference will wait for the current strict evaluation to finish. No response policy,strength claim or change to the57 live-study inputs.

## Delayed opponent-identification queue (2026-10-19)

**Opponent-identification diagnostic queued:** session81291 waits for successful tail-study completion/audit54117,then runs the fixed all2,048-training-prefix diagnostic once. Receipt runs/revealed_opponent_identification_queue.json binds13 inputs;log runs/revealed_opponent_identification.log. No additional diagnostic neural inference is running yet. Producer19319 remains live;three evaluation cells have completed4,096 matches each and parent2-Point is running. Preserve the current processes and bound files;do not launch duplicate work or infer termination from an observation deadline. No partial win contrasts have been inspected.

## Opponent-identification independent auditor ready (2026-10-19)

**Opponent-identification independent auditor ready:** scripts/audit_revealed_opponent_identification.py independently replays public revelation timing,source endpoints,candidate exclusions,first-identification times,remaining learner opportunities and every reported count. It does not call the new tracker or rerun neural predictions. Two tests pass,0 skips,2.10s,including refusal of candidate resurrection,missing predictions and prematurely dated checks. No real diagnostic result exists yet. After queue81291 completes,run `.venv/Scripts/python.exe -m scripts.audit_revealed_opponent_identification --base runs/final_round_training_study --diagnostic runs/revealed_opponent_identification` once. The57 tail-study inputs and13 identification-queue bindings remain unchanged. Existing producer19319,audit queue54117 and diagnostic queue81291 are retained.

## Fixed-prefix tail study fully audited (2026-10-19)

**Fixed-prefix tail study complete:** producer19319 and independent audit/summary54117 exit0. All16,384 strict matches complete on2,048 fresh deals. Win changes versus parent are+0.122pp [-0.439,+0.684] and+0.537pp [+0.049,+1.050] for parent1 against WP/Point,and-0.122pp [-0.635,+0.391] and-0.098pp [-0.635,+0.391] for parent2. The all-four replicated-gain criterion fails;every point-change interval includes zero. All89 independent-audit input hashes reverify. Training totals1,860,647 conditional decisions /133,120 tails;actual evaluation1,908,865 decisions /127,476 rounds /16,384 matches. No promotion or101M use. [Full outcomes and identities](FINAL_ROUND_TRAINING_RESULTS.rst). Sessions19319/54117 are terminal;do not restart them. Separate diagnostic81291 follows the successful audit using training histories only.

## Delayed public opponent identification completed

Command: `.venv/Scripts/python.exe -m scripts.audit_revealed_opponent_identification --base runs/final_round_training_study --diagnostic runs/revealed_opponent_identification`. Producer81291 and audit exit0. Correct/ambiguous counts by parent1-WP/Point,parent2-WP/Point:428/84,435/77,438/74,445/67;zero wrong or excluded true models. Final roots1,598/1,856 identified. All36,977 original-feature/mask comparisons pass;44,170 predictions. Independent replay212,212 decisions /14,101 round endings /192 early completions,zero new deals or neural inference. All29 independent input hashes verify. Full identities and interpretation: REVEALED_OPPONENT_IDENTIFICATION_RESULTS.rst.

## Public-memory tail capability

Commands: `pytest tests/test_public_memory_tail.py -q` (4 passed,0 skips,14.50s), `pytest tests/test_public_memory_match.py -q` (1 passed,5.44s), `python -m scripts.check_public_memory_tail --output runs/public_memory_tail_capability` (exit0). Initial CPU test mismatch came from comparison-model construction consuming fixture RNG;preserving that RNG fixes matched warmup without changing production code or tolerances. Failed XML retained as public_memory_tail_tests_initial_failure.xml. CUDA production check: public/masked each5,476 decisions,2,571 learner /2,905 opponent,384 conditional tails,0 full matches;all96 Adam states finite,exact shared warmup,frozen prefix/opponents verified. Audit35072349596c1bb731388c0bc3d3afae13e0b8594a890c3ac3040a5168a45acb. Public columns learn;masked stay zero. No strength claim.

## Public-memory comparison protocol

Two protocol tests pass,0 skips,1.72s: fresh balanced91M deal grid,independent stratified bootstrap,zero-cancelling constant strata,conjunctive primary decision,and refusal of premature revelation,changed endpoint and corrupted reconstructed-input hash. Plan preflight verifies126 input identities. Four matched520-update arms;24,576 full evaluation matches on2,048 shared fresh deals. Primary balanced WP/Point mixture win gain must be positive against masked and original parent for both parents. Existing earlier results are not reclassified. No training/evaluation run yet.

## Public-memory full comparison launch

Producer67506: `.venv/Scripts/python.exe runs/launch_public_memory_study.py`;runs the frozen module with UTF8 log runs/public_memory_study.log. Protocol3e1282183196b0f76bfbaa383160304f34377c12d482578bf4cec112a627c7c1,126 input bindings. Four trainers then four strict CPU evaluation cells (three workers). No partial outcomes inspected.

## Public-revelation manuscript revision

Verified27-page PDF,29 references,179-word Abstract,20 embedded fonts. Exporter checks completed tail/identification audits and107 unique bound inputs;combined prior/new QA verifies677 evidence inputs. Both generated tables reproduce audited values. New Related Work covers Bayes' Bluff,Bayesian Policy Reuse,DRON,JSRL. Main text separates identification capability from the ongoing strength experiment and clarifies ordinary PPO cross-round bootstrapping. PDF0b3bd30a9dca1e24dc0b8600ab533d8b0cc2d1168cc2c7b8c049d386a63f3ee3;source36c0779feaea8c5ca5ac3c0f6c197ce1464a4f9c6fa835312897d787edbb8728. All27 final pages contact-reviewed;1/13/14/15/27 full size. Minor layout revision keeps revelation heading and first paragraph together. Old numeric tokens retained;no TeX/ref/glyph/bounds defects. Compilation/rendering CPU work overlaps the current GPU training,so this run is not an isolated latency measurement. Independent audit queue35738 waits on67506 and bindsits protocol/auditor/helper.

## Public-memory first-parent endpoints complete

Producer67506 completed public and masked520-update endpoints:929,883 and929,458 decisions,respectively;66,560 tails each,0 full training matches. Both training audits pass,including original frozen prefix/opponents,audited public-memory root codes,finite96-state Adam andphase step counts. Exact initial/warmup20 comparison passes before parent2-public begins. Endpoint input identities reverified. Independent queue35738 waits for the complete24,576-match evaluation;no partial win outcomes inspected.

## Public-memory all endpoints complete; evaluation started

All4 training arms complete520 updates. Public/masked decisions:parent1 929,883/929,458;parent2 932,153/934,154,total3,725,648. Each66,560 tails;combined266,240 conditional round/tail completions,0 full training matches. All96 Adam states perarm finite with phase-step counts verified;original prefixes/opponents and audited public-code roots agree. Both exact initial/warmup20 comparisons pass before producer67506 starts three-worker strict CPU evaluation. Completion receipt runs/public_memory_study/training_completion.json binds immutable endpoints and reverified inputs. Independent audit queue35738 waits for all24,576 matches. Summarizer is implemented,lintchecked,and verified to refuse this incomplete study;run once after the independent audit.

## Earlier public-memory activation opportunity

Previous goal turn was progress:all4 endpoints completed and independently checked,strict evaluation launched,and27-page manuscript revised/rendered/verified. Current handles67506/35738 re-polled live. New read-only diagnostic `python -m scripts.check_public_memory_activation --output runs/public_memory_activation.json` exits0;two tests pass1.45s,0 skips. All2,048 training prefixes retained. Identified1,746,eligible subsequent fresh rounds1,731,identified without later boundary15,unresolved302. Activation rounds2--8 counts463/387/309/199/138/146/89;51,098 recorded learner decisions remain before round8. Repeatedsource work212,212 decisions /14,101round endings /192earlyfullmatches,zero new independent deals/inference/training. Artifactb9086bbf6fbb82e8f1eef888bff715d868435e88ab16da82c7cdbb4241b2f625. ShortCPUreplay overlaps strict evaluation;not latency evidence. All126 active-study bindings unchanged. No expanded-training implementation/protocol or strength claim yet.

## Public-memory continuation bank materialization

Added scripts/build_public_memory_continuation_bank.py and two tests. Command: `.venv/Scripts/python.exe -m pytest tests/test_public_memory_continuation_bank.py -q --junitxml=runs/public_memory_continuation_bank_tests.xml`;2 passed,0 skipped,1.46s. Ruff format/check pass. Command: `.venv/Scripts/python.exe -m scripts.build_public_memory_continuation_bank --output runs/public_memory_continuation_bank`;exit0 in9.50s. All2,048 prior training prefixes replay exactly to their stored endpoints/counters;root activation agrees with the completed public-timing diagnostic. Banks contain855 and876 int32[208] roots. Public codes come from independently audited identification records;true-opponent labels are separate simulator routing metadata. Later actions only verify the retained source and cannot select/modify the copied root. All37 inputs/two bank outputs reverify. Receipt SHA f2e49203737b4b51e54ab39cd5f73fc669e47d24c68c10081226e224f2556120. Replay repeats212,212 decisions/14,101 round endings/192 early full completions,zero new independent deals or neural predictions. Three synthetic full-match fixtures are correctness work only. This short CPU work overlaps strict evaluation and is not an isolated timing result. All126 live-study inputs remain unchanged. No new learner/study/model promotion. See PUBLIC_MEMORY_CONTINUATION_BANK.rst.

## Complete public-memory continuations: CPU capability

Added isolated scripts/public_memory_continuation.py: fresh paper8 roots in rounds2--8,public singleton memory,448-decision safety bound,full terminal learner returns and per-lane round-end verification. Reuses existing observation conditioning/PPO while preserving all live-study files. New checkpoint format stores optimizer/root RNG/frozen actors/public bank. Command `.venv/Scripts/python.exe -m pytest tests/test_public_memory_continuation.py -k "not cuda" -q --junitxml=runs/public_memory_continuation_cpu_tests.xml` passes2,0 skipped,one CUDA case deselected,in38.63s with two expected Transformer warnings;36592 exits0. Independent reference checks all transitions in a multi-round batch and final reward signs,including opponent-turn endings. Public/masked initial and warmup states match;resume reproduces policy update/model/optimizer exactly;frozen snapshots unchanged. Bank validation and memory tampering checks pass. Ruff passes. XML SHA146137be1184f58c3e80027830468fcf5e1fffbb8897a80d2b5a96a97ef2c627. All126 active-study inputs verify unchanged. CPU fixture work overlaps strict evaluation;no isolated timing claim. CUDA/production-shape checks and online activation adapter remain pending;no strength trial or promotion. Details in PUBLIC_MEMORY_CONTINUATION.rst.

## Public-memory continuation online activation

Added scripts/public_memory_continuation_match.py with public-counter lifecycle,strict CPU inference and reference/native complete matches. Activation occurs at the first fresh round after publicly timestamped identification,including when the opponent deals that round. Memory freezes at activation;unresolved cases use the original parent for all remaining decisions. Command `.venv/Scripts/python.exe -m pytest tests/test_public_memory_continuation_match.py -q --junitxml=runs/public_memory_continuation_match_tests.xml` passes2,0 skips,8.25s with three expected Transformer warnings. Four seat/opponent fixtures compare public/masked initial models against original parent with exact action,terminal and work agreement;independent replay checks first activation and both dealer alignments. A deliberately ambiguous candidate set verifies fallback;reset and invalid public decision addresses reject stale lifecycle usage. XML66d1bf3520e6ce68d2e180c48d7dc0f0c58dc4cc18dcc9384efcf80f65e8bddb. Ruff passes. All126 current-study inputs reverify unchanged. Short CPU checks overlap evaluation;no isolated timing or strength claim. No GUI/default model/dealer change. CUDA/production capability and a prospective new comparison remain pending.

## Continuation production capability prepared and queued

Added scripts/check_public_memory_continuation.py. Requires completed round8 producer,passed independent audit,unchanged evidence and passed real CUDA test;uses855 previously audited parent1 roots,128 environments/3 epochs/minibatch512. Public/masked arms each execute3 logical updates (one critic-only,two policy),with update3 independently resumed from update2 and compared exactly across model/optimizer/RNG/bank. Replay work is separate. All optimizer state/phase steps and frozen ancestry are checked. Two guard/equality tests pass,0 skips,1.39s;XML e1b8c8e1a25c92923cf0c59bc22df9fe9e209e1fa6b05fd060d6fb06d8c2b06b;Ruff passes. Command `.venv/Scripts/python.exe -m runs.finish_public_memory_continuation_capability` launches76467;receipt runs/public_memory_continuation_capability_finish.json initially waiting_for_round8_independent_audit,SHA cdddfcda73bba935221779cc7a1d3eca82a00f1c6a0355580d06b56ee197e1e9. All146 queued inputs verify. It waits for35738 completion,then executes CUDA pytest and the fresh capability run;no CUDA work has started. Observation deadline3h does not terminate67506. No strength protocol or promotion. Preserve76467 and bound sources/tests.

## Public-memory round8 evaluation producer complete

Producer67506 is terminal exit0;all4 cells contain6144 records,total24,576 full matches on the declared2,048 shared fresh deals. Analysis SHA10c018a55a2fd006969aea819e207a59dd5c0f6a4b54241266727c01f655a8f9,all54 output input hashes reverified. Only after full completion were contrasts inspected. Provisional primary mixture public-minus-masked wins(parent1/2):-0.25634765625pp [-0.537109375,+0.0244140625],+0.03662109375pp [-0.244140625,+0.3173828125]. Public-minus-parent:-0.1220703125pp [-0.439453125,+0.18310546875],+0.0732421875pp [-0.23193359375,+0.37841796875]. All18 win and18 point intervals include0;producer primary criterion fails. Independent replay/feature/statistics queue35738 is live and auditing;no independent result yet. Existing summarizer remains unrun until audit completes. Capability76467 is still waiting for that audit;new continuation sources/tests remain bound. No budget extension,model promotion or101M use. These are provisional producer results pending independent verification.

## Public-memory comparison and continuation capability completion

Independent audit35738 exits0 and reproduces all24,576 full matches,all18 contrasts and primary failure. All181 audit inputs reverify. Audit SHA a5a9f350f2388294cdd106758d60eb453a2f03a24ff41f192436a04d4925acd7. Actual evaluation2,857,337 decisions/191,262 round endings/24,576 matches;public identification355,486 strict calls(167,506 WP/187,980 Point). Independent replay verifies300,092 original feature/mask inputs with0 additional neural inference or independent matches. All18 win/point intervals include0. `.venv/Scripts/python.exe -m scripts.summarize_public_memory_study --output runs/public_memory_study` executes once,exit0,5.56s;summary SHA ca6182fb9e687e4c0a23237e161a4bc27de518ffb40e12b32c9b650e583fc985,receipt a566ffd3a66a755d6ff260750a67f909a7cab426d1ad1550bb86f9db8e26c9a9. Results in PUBLIC_MEMORY_RESULTS.rst. No model promotion or101M use.

After that audit,queued76467 completes both commands and exits0. CUDA test passes1,0 skips,2 deselected,in6.55s with two expected warnings;XML fcdb9fa731551eef80cfc6cacb7735b34da4ecfe7df38c1de2a83dfbd1f71744. Real two-arm128-env/3-epoch/512-minibatch capability completes3 updates per arm;each28,782 decisions/14,274 learner/14,508 opponent/384 conditional completions/1,927 conditional rounds. Combined57,564 decisions/768 completions/3,854 rounds. Exact resumed third updates add17,980 decisions/256 completions/1,210 rounds separately. All ordinary full matches/rounds0. Matched initial/warmup model/optimizer/root/RNG equality passes;both96-state Adam ledgers finite and phase counts agree. Public input-column L1 .07289130613207817,masked0. Capability SHA1ba645ade180878ebf5e8803211abe46e1b1cc8161c160100b733b5fb0647bee;all214 inputs verify;finish receipt0c3be9745cc9c0e2e74a781ab304431403bfd06056a7877d80458f9efc7ba0d5. No strength inference from capability. Preserve completed67506/35738/76467;all terminal. Next define fresh multi-round protocol.

## Prospective complete-continuation comparison prepared

Added scripts/run_public_continuation_study.py and scripts/audit_public_continuation_study.py. Four new520-update arms,20 warmup+500 policy,128env/horizon448/3epochs/minibatch512,original parents,public/masked same92110001/92120001 seeds. All855/876 natural public-stopping roots retained;all unsuccessful identification cases remain in full evaluation with original-parent fallback. Evaluation parent/public/masked x both parents x WP/Point x1024 paired deals x2seats =24,576 actual full matches on2,048 fresh93M deals. Bootstrap93310001,10k,stratified equal mixture;same four lower-bound primary conjunction. No old-study extension or101M use. Three protocol tests pass,0 skips,9.24s(one expected Transformer warning),XML runs/public_continuation_protocol_tests.xml. All1,731 roots independently reconstruct from2,048 source histories,with212,212 decisions/14,101 round endings/192 early full completions replayed and no new independent deals. One online fixture and altered-timing/feature/terminal cases verify independent replay rejection. Ruff passes. make_plan preflight executes without training;221 inputs verify,preflight SHA95ce0a01a76c790f5dbdeb0c78028199e61219053167fc7309543cfa4c4815a1. Protocol PUBLIC_CONTINUATION_STUDY.rst. No full run launched at this milestone.

## Complete-continuation comparison launch

Command `.venv/Scripts/python.exe runs/launch_public_continuation_study.py` launches producer46255,logging to runs/public_continuation_study.log. Fresh runs/public_continuation_study/protocol.json SHA95ce0a01a76c790f5dbdeb0c78028199e61219053167fc7309543cfa4c4815a1 matches preflight bytes;all221 inputs verify. Parent1-public starts training. Command `.venv/Scripts/python.exe runs/finish_public_continuation_study.py` launches96354,which waits for full completion then runs the frozen independent auditor and verifies its complete input ledger. Receipt runs/public_continuation_study_finish.json;8h observation deadline never kills the producer. Both handles confirmed live. No paper/manuscript files occur among bound inputs;completed prior-study results may be incorporated during training. No partial evaluation contrast exists or has been inspected. Preserve both handles and bound files;no duplicate launches. Canonical model,dealer and reserved101M unchanged.


## Completed public-memory result incorporated into manuscript

Added scripts/export_public_memory_evidence.py,which pins the completed independent audit/summary receipt,checks all184 source identities and exports all18 contrasts and separate accounting. Generated table SHA584982cfb3ed1dcb629de0f682fcf9a993e84e452a49a08479df1b19b294b2e3;ledger SHA755bf7b2cd630adaa5e1ceea66d7253da71413c00bebf2c50a264c3f99ff8593. Main text reports final-round public-memory primary failure;no ongoing multi-round result is claimed. Abstract shortened to171 words. Related Work retains29 cited references.

Tectonic build passes with28 pages/20 embedded fonts;all756 evidence hashes verify. Initial numeric-retention check flagged the old token `11.` after table renumbering;inspection identified Table11 becoming Table12,not a changed scientific quantity. Failed helper retained as runs/qa_manuscript_public_memory_initial_failure.py. Comparison now strips terminal punctuation;all previous numeric values remain. All28 pages rendered and contact-reviewed;final Abstract reflow changes only pages1--4,whose contact sheet was reviewed again. Final pages1/2/14/15/16/28 inspected full size. No clipping,overlap,box,glyph,reference or page-bound defects. QA runs/manuscript_public_memory_qa.json SHA448ea1700fd94f8139c733240f07869f918b79bcae627641077e4dfd10624f66. PDF c909f25f2574019e05430ce86825199142de9cbb99eb673c9d71d16707b6dbcc;source f48651213672c5da624dda9f7ed0b76b9baf7214196625a5caf9897672d698e3. Previous delivered PDF retained as output/pdf/manuscript_revelation_0b3bd30a.pdf. All221 ongoing-study bindings verify unchanged;producer46255/audit queue96354 confirmed live. Paper work overlaps training and is not isolated latency evidence. No model/dealer/101M change.


## Complete-continuation summarizer prepared during training

Added scripts/summarize_public_continuation_study.py SHA53ae7c2784be2bb3b322de97a513045de15b7f92c7ecb723979fefaad21a08c2. It requires the exact frozen protocol,completed producer,passed independent audit,unchanged input hashes,all18 contrast cells and agreement with producer statistics before writing. Verifies allfour520-update schedules,266,240 conditional multi-round completions,24,576 evaluation matches/2,048 distinct deals,and separately reports source-root replay. Both summary artifacts refuse overwrite. Actual current incomplete study is rejected without writing outputs by `.venv/Scripts/python.exe -m runs.check_continuation_summary_guard` (exit0,1.67s). The first inline invocation failed at shell quoting before importing/executing code;the retained UTF8 helper avoids that issue. Ruff passes. Successful full-summary path remains pending completed study/audit;no results read or produced. All221 study bindings reverify unchanged. Existing producer46255 and independent queue96354 remain the only live jobs;no new computation job launched.


## Closest public-revelation prior art and manuscript clarification

Previous goal turn was progress:completed-memory results and a completion-gated summarizer were committed/pushed. Current producer46255 and independent queue96354 are re-polled live;parent1-public continues its fixed training schedule. No partial evaluation results inspected.

Checked primary full texts: Bayes' Bluff Section3.3 (author-hosted05uai.pdf),Ganzfried/Wang/Chiswick DAI2024 Sections1--3/Algorithm2 (conference-hostedDAI2024_paper_10.pdf),and Rebstock et al. IEEECoG2019 SectionIII/Algorithm1 (arXiv1905.10911v1 HTML). Related Work now distinguishes revealed-card likelihoods,partial-revelation adaptation,and hidden-state inference from the exact historical-input reconstruction and fixed-model identification tested here. New bibliography entries add the latter two works;the DAI manuscript's placeholder DOI/ISBN fields are deliberately omitted. Scope/source details in PUBLIC_MEMORY_PRIOR_ART.rst. No general novelty claim for delayed revelation or default-policy prefixes.

Tectonic compilation and runs/qa_manuscript_public_prior_art.py pass:28 pages,31 cited references,171-word Abstract,20 embedded fonts;all previous numeric values and756 evidence input hashes retained. All28 pages rendered/contact-reviewed;pages3/15/28 checked full size. No clipping,overlap,box,glyph,undefined-reference or page-bound defects. PDF2ffe37c5ef6be499fbc9fd36b161a811eda7be7729fddc3d129e31404ace4440;source5a365f1b7e6e1f1a6366f2e69ca4f053336728bc00e61aeffe11777304805fa9;bib5270a386287ff8645abbc216b39c57c99ab4286bd12179287dae93f4caed0897. QA runs/manuscript_public_prior_art_qa.json SHA80bbecbcbb2e89aba3a9ffcb04b141b1d273c0efc9445c2bdc0cfd3a6bf46e21. Prior delivered PDF retained as output/pdf/manuscript_public_memory_c909f25f.pdf. All221 live-study bindings remain unchanged. Paper compilation/rendering overlaps training and adds no strength observations. Goal remains active pending stronger-policy evidence;no model/dealer/101M change.


## First complete-continuation endpoint and matched warmup verified

Previous goal turn made progress by adding primary-source comparisons to Related Work and verifying the31-reference PDF. Current producer46255 and independent queue96354 are re-polled live. Waits on46255 preserve the running process;parent1-public then completes520 updates and the producer advances to parent1-masked. No evaluation outcomes exist or were inspected.

Command `.venv/Scripts/python.exe -m runs.verify_public_continuation_parent1_first` exits0. It independently reruns the frozen endpoint auditor,compares its report exactly to producer training_audit.json,and reruns matched initial/warmup20 checks. All231 bound inputs verify before and after. Public endpoint SHA e953c46a455ad96b1b518d10c3e704e9cbf651a9a7ff900658f68841b4723f0e;96 finite Adam states,phase-step counts,frozen ancestry/opponents and source-bank joins pass. Counters:4,770,089 decisions(2,350,805 learner/2,419,284 opponent),66,560 conditional multi-round completions,320,027 conditional round endings,zero full training matches. Value steps14,460;discard13,896;pick13,865;koi13,872. Matched initial/warmup actor,value,optimizer,bank,memory,counters and Python/NumPy/CPU/CUDA RNG states agree exactly. Receipt runs/public_continuation_study/parent1_public_completion_verification.json SHA22ac11cb454d1a0074fcf267b24d69a4399cbb6f1bc3525fec2b883fd7b2a583. Verification creates no simulation or neural predictions and overlaps the next arm's training. Otherthree endpoints and prospective full evaluation remain pending;no strength conclusion,promotion or101M use.


## Current-state handoff consolidated during fixed training

The previous turn completed an independently rerun first endpoint and matched initial/warmup check. Producer46255 and audit queue96354 are confirmed live;parent1-masked is training. STATUS had grown to89,972 bytes with contradictory superseded lifecycle statements,including old live handles and old manuscript versions. Replaced that accumulated history with a16,789-byte current snapshot. The complete preceding snapshot remains in pushed commit80a1cf1;EXPERIMENTS and all evidence artifacts remain intact. Original final-evaluation and selected-model sections are preserved byte-for-byte. The current protocol,known failed primary comparisons,current31-reference manuscript,next completion steps,and reserved101M boundary remain explicit. Local links verify and all221 active-study hashes reverify. No bound source,model,protocol,paper,experiment outcome or work count was changed. This is handoff maintenance during an actual running job,not new strength evidence.


## Fixed continuation-root chance diversity measured

Previous turn consolidated the current-state handoff while verifying live producer46255/audit queue96354. Both are re-polled live this turn;parent1-masked continues the frozen schedule. Added scripts/check_continuation_root_reuse.py;Ruff format/check pass. Command `.venv/Scripts/python.exe -m scripts.check_continuation_root_reuse --output runs/public_continuation_root_reuse.json` exits0(66299 terminal). All225 inputs verify. Artifact SHA fcb28bcb12cd28809daa7f563972bab962984122712e33b4a98cf9e8f421d7fc.

Banks855/876 contain820/834 distinct packed states,489/501 seed-episode streams,717/756 initial round-deal keys and equally many deck permutations. Public round/dealer/points/seat/memory contexts548/557. Repeated packed states need not be identical learner observations;no deduplication performed. Completed parent1-public uses all855 roots in66,560 draws,47--111 uses per root,mean77.84795321637426. Replaying only520x128 CPU root-index draws exactly reproduces the final saved bank RNG;Torch/NumPy frequencies agree. No simulator decisions,neural predictions or new independent deals. The continuation copies unchanged seed/episode;reference/native deal code keys future round shuffles by those values and round. Action trajectories may still diverge. Details and a possible separate chance-renewal comparison in CONTINUATION_ROOT_DIVERSITY.rst. No overfitting or strength conclusion follows. Current study bindings and protocol remain unchanged;no new trainer/evaluator or deployment change.


## Public-context chance renewal CPU constructor

Added isolated scripts/continuation_chance.py with immutable public round/dealer/points context and existing reference rejection-sampled dealing. It does not read old hidden cards or chance keys and does not derive opponent identity. It preserves the existing dealer and scoring rules;no live trainer/online agent is modified. Three independent natural-boundary/privacy/complete-trajectory CPU tests pass0 skips in1.37s;final XML3f76bb94d3e06e59ee7b3bf2e8c0106b96022766f4924600742b54039e33cde9. Initial tests also passed3 in1.42s;Ruff initially required TypeError instead of ValueError for two invalid-type guards. Corrected them and reran the affected suite;Ruff now passes. Each test execution includes8 full natural fixture matches and4 conditional continuations with reference comparisons;these are correctness work,not strength evidence.

Command `.venv/Scripts/python.exe -m scripts.check_continuation_chance_bank --output runs/continuation_chance_bank_capability.json` exits0(81690 terminal). All855/876 historical roots reconstruct byte-exactly from their public context and original chance key. Separate capability keys yield855/876 distinct legal fresh permutations with unchanged public contexts/source arrays. Adds1,731 fresh conditional deals and1,731 reconstructions,0 simulator decisions/NN predictions/full matches. Receipt f18357c52be82afa71b6f5d14869d6616c7cc60bcbf41b08b67bd6c482961cfc;all226 bound inputs reverify,including the221 current-study inputs. CPU checks overlap parent1-masked training;no latency claim. CUDA/trainer/resume integration and any new strength protocol remain pending. See CONTINUATION_CHANCE.rst. Current46255/96354 remain live and unchanged;no deployment/101M changes.


## Restartable renewed-root sampler

Added scripts/renewed_root_sampler.py with original-record sampling weights,fresh finite chance keys per draw,separate seat/routing/public-memory metadata,and snapshot identities/cursor/batch lengths/RNG. Restore replays only index draws to reject inconsistent RNG/cursor states;invalid requests cannot advance the schedule. Two tests pass,0 skips,1.51s:serialized exact next-batch equality,independent index schedule,paired draws,unique new keys even for duplicated roots,caller-mutation isolation,distinct public-memory/routing labels,and atomic overrun/tamper rejection. XML9af1d59cd7035cca2de4825207879ae20360bb03d41a7d8dddd5f5f9d46759d1. Initial two-test run passed1.44s;Ruff requested a dictionary literal,then both tests passed1.48s. The final rerun strengthened the metadata fixture so routing labels and public memory differ. All three executions are deal-construction correctness work,zero simulator steps or full matches. Final Ruff passes. Source/test hashes:{"scripts/renewed_root_sampler.py": "967b77c554a325a3b5dd685411ce635bbe486ac0b34cc57f5aec69ccf10f0723", "tests/test_renewed_root_sampler.py": "12c6f81b7dc03316e7e121c2a2c00d0b5b8c499bd4ce0326bfde36639998b202", "runs/renewed_root_sampler_metadata_tests.xml": "9af1d59cd7035cca2de4825207879ae20360bb03d41a7d8dddd5f5f9d46759d1"}.

All221 ongoing-study inputs remain unchanged. Producer46255 and audit queue96354 are live;parent1-masked remains in its fixed520-update schedule. This sampler has not been integrated into a learner or strength run;CUDA/full-trainer-resume checks remain pending. CONTINUATION_CHANCE.rst records the contract. No model promotion,dealer change or101M use.


## First continuation parent complete;renewed-deal learner CPU integration

Producer46255 completes parent1-masked and advances to parent2-public. Independent command `.venv/Scripts/python.exe -m runs.verify_public_continuation_parent1_masked` exits0(25283 terminal),reproduces the endpoint report and matched initial/warmup20 states,and verifies231 inputs. Masked endpoint SHAfe67da80c51298a812bc710049c862558f8bdf011975f4d78f117026281e83cd;receipt c4881c33d09c406b241e4fab9fb8e91a41c35e7aaaa7bc40d3f36eff61a68be8. Its4,746,287 decisions comprise2,341,711 learner/2,404,576 opponent;66,560 conditional completions/320,564 conditional rounds,0 full training matches. All96 Adam states finite;value14,454/discard13,890/pick13,855/koi13,864 steps agree. Parent1's two arms total9,516,376 decisions/133,120 conditional completions/640,591 conditional round endings. This verification adds0 simulation/inference. Strength evaluation remains pending.

Added isolated scripts/renewed_continuation.py,new checkpoint format and finite chance allocation. Reuses existing actor conditioning/PPO,retains complete terminal rewards and frozen prefix/opponents,but renews every sampled round deal. Saves the full root sampler and rejects disagreement between its cursor,fresh-deal count and conditional completions. Command `.venv/Scripts/python.exe -m pytest tests/test_renewed_continuation.py -k "not cuda" -q --junitxml=runs/renewed_continuation_cpu_final_tests.xml` passes2,0 skips,1 CUDA case deselected,in46.40s with two expected warnings;93413 terminal0. Covers reference/native all-transition and terminal-reward agreement,public-memory masking,matched initial/warmup models and chance schedules,exact next-update model/optimizer/metrics/chance-key resume,frozen actors,and corrupt cursor/counter/memory rejection. XML19b7d3eccf220a3289fdfc22ed8041225b3e2b08650a2caed896a00d8e02621f;source732f2b2f1ab5695391f1adce15af677df16bd7d2f4601ac218799ba320b3b90d;test5453023cec3ff233d4573309566a4c2854baae4d5e98ed3337d09e45a91f0d2a.

Initial CPU run54406 also passed2 in45.98s,0 skips/1 deselected;Ruff initially found test import ordering. Fixed imports and added the integrated counter-mismatch guard assertion before the final rerun. Both are correctness/training fixtures,not independent strength samples. All221 ongoing-study inputs remain unchanged;no CUDA test or production capability/new strength run started. Online loader and GPU/full-size checks remain pending. Current46255/96354 remain live;the canonical model,dealer and101M range are unchanged. Details in CONTINUATION_CHANCE.rst.


## Renewed-deal online adapter and deferred CUDA capability

Added scripts/renewed_continuation_match.py and the isolated production checker
scripts/check_renewed_continuation.py. The online adapter inherits the existing
public-event lifecycle and fresh-round activation,loads only inference actors
and configuration,and preserves whole-match parent fallback when identification
is ambiguous. Three tests pass,0 skips,in8.93s with four expected Transformer
warnings;XML a0009da4d30932a9cbc52ecd3d0004b1672baf40e6a494e3c234927f8e4f0d37.
Capability guard/equality tests pass two,0 skips,in1.39s;XML
20b5c51f9a27d98aa150b0683c19307128a709640ffd73d0ba255af5085deaf7.
Commands use pytest on tests/test_renewed_continuation_match.py and
tests/test_renewed_continuation_capability.py with their corresponding run XMLs.
Final targeted Ruff passes. These tests are correctness work,not strength data.

Before capability execution,source review found and corrected a copied bank
path to runs/public_memory_continuation_bank. The ignored queue helper
runs/finish_renewed_continuation_capability.py also checks literal input paths
before waiting. Command `.venv/Scripts/python.exe -m runs.finish_renewed_continuation_capability`
is live as OS Python PID36420,launcher63668;its original tool session output was
lost during compaction,so process and receipt were recovered without relaunch.
Receipt runs/renewed_continuation_capability_finish.json is waiting for the
existing multi-round independent audit;240 pinned inputs all verify. Waiting
receipt SHA c692ee7486c2967ffe8890e67a8fc4ed531757141b023b145ad9c3965a0c5181.
It will run the actual CUDA case,then two production-shaped three-update arms
and exact third-update resumes. The allocation is384 distinct shared chance
keys,768 logical and256 replay constructions;no capability trajectory has yet
run. The8h deadline never kills the preceding producer. New bound sources/tests
must remain unchanged while queued. No prospective strength run or promotion.

Existing producer46255/audit96354 are live(actual Python PIDs10804/32056).
Parent2-public reaches update320/520 after both parent1 arms completed;no partial
evaluation contrasts inspected. All221 old-study inputs remain unchanged.
Dealer selection,canonical human-play weights and reserved101M remain unchanged.


## Chance-renewal strength design recorded before current evaluation

Added docs/RENEWED_CONTINUATION_STUDY.rst before any current multi-round
evaluation file existed. The prospective2x2 comparison retains both fixed-bank
public/masked endpoints and trains renewed public/masked arms from each original
parent. Same root-index/training seeds,budgets and context multiplicities isolate
chance renewal. New chance allocations94110001--94176560/94210001--94276560 are
shared within each parent;266,240 conditional completions use133,120 distinct
keys. All five policies receive40,960 new full matches on2,048 shared95M deals.
Primary renewed-public minus parent/fixed-public win conjunction is distinct
from secondary public-signal and chance-by-memory interaction tests. All pairwise
and opponent-specific results must remain visible;no endpoint selection.

Read-only command `.venv/Scripts/python.exe -m runs.record_renewed_study_design`
checks arithmetic/range disjointness,all240 queued input hashes,the current
training stage and absence of evaluation files. Receipt SHAec855e090728072ddc491a9eadb1c50c3da1bd0538b169a59b310f77d104b6f3.
It adds0 simulator decisions/NN predictions and does not start any strength job.
The new design is not yet a frozen executable protocol;runner/auditor validation
and completed current-study/CUDA-capability prerequisites remain necessary.
No manuscript result,canonical model,dealer mechanism or101M range changes.


## Five-policy paired analysis implemented during fixed training

Previous turn committed/pushed the new online adapter and prospective design,
so it is classified as progress. This turn confirms producer46255,audit96354
and renewal queue PID36420 actually live;the original producer handle is polled
without restarting. Parent2-public continues its unchanged520-update schedule.

Added scripts/renewed_continuation_analysis.py:complete40,960-record grid,shared
95M cases,ten pairwise and one interaction comparison across two parents/three
scopes,paired seat averages and stratified10,000-draw bootstrap. Strength,memory
benefit and interaction decisions require their distinct prespecified both-parent
conjunctions. The loader rejects partial grids,duplicate/wrong deal keys and
invalid score totals;it does not replace trajectory/provenance verification.
No automatic partial-study analysis entry point is added.

Command `.venv/Scripts/python.exe -m pytest tests/test_renewed_continuation_analysis.py -q --junitxml=runs/renewed_continuation_analysis_tests.xml`
passes5,0 skips,in0.47s. Fixtures distinguish positive point margins from win
gains,one-parent/opponent-only improvement from the primary criterion,and memory
benefit from the interaction. Independent scalar bootstrap agrees across257
draws including chunk boundaries. Synthetic40,960-record grid tests do not run
the simulator or neural policies and are not independent strength samples.
XML462ca12e4c1aff98ee385328d6fe9923b7d5aed52a9381ded68b9a36096877b6;
source05dcafbf2939baa564df02bfc7a0ee9131f972cc18d179792447d54f7d15e922;
test5cc69aa810290d08abb900fe0b88f1cb293dba6700b6fa6b535a28c88115c8cb.
Targeted Ruff format/check pass. All240 queued inputs reverify and the original
prospective design hash matches its pre-evaluation receipt. The new runner and
independent auditor remain to implement;no new training/evaluation launched.


## Third fixed-continuation endpoint independently verified

Producer46255 completes parent2-public520 updates and starts parent2-masked.
Command `.venv/Scripts/python.exe -m runs.verify_public_continuation_parent2_public`
exits0(98033 terminal),with one expected upstream Transformer warning. The
endpoint audit reproduces its producer report exactly and all228 inputs pass
before/after;96 finite Adam states and phase counts agree. Final SHA
7beb4e2276d0773d015de8010650af620a0c3bd293b9e4db87333f0f99c49f44.
Actual4,718,538 decisions comprise2,328,594 learner and2,389,944 opponent;
66,560 conditional completions/318,361 conditional rounds,0 full training matches.
Value14,376/discard13,821/pick13,793/koi13,786 optimizer steps. Receipt
ea1e6df4e496bd2e83e94a9c0ec7eea7a7ef9c8186a3df831875d8597347a521.
The verification adds0 simulator decisions or neural predictions. It does not
yet check parent2's paired warmup;that and the remaining endpoint are pending.
Three completed arms total14,234,914 decisions/199,680 conditional completions/
958,952 conditional round endings. No partial evaluation contrasts inspected,
new strength claim,promotion,dealer change or101M use.


## Renewed continuation producer and independent auditor queued

Previous turn implemented/tested paired analysis and independently verified the
third fixed endpoint,so it is progress. Current producer46255,audit96354 and
renewal capability PID36420 are confirmed live;parent2-masked continues its
unchanged520-update schedule. No partial evaluation outcomes inspected.

Added scripts/run_renewed_continuation_study.py and
scripts/audit_renewed_continuation_study.py. The producer requires completed
unchanged prerequisite audits,passed CUDA/resume capability,successful tests
and the pre-evaluation design identity;it implements the stated four new arms
and five-policy40,960-match comparison. The independent auditor reconstructs
every saved root-index RNG and compares it to the fixed control,checks initial
states,source ancestry,chance accounting,finite optimizers,warmup and final520
identity,then replays matches/public evidence and independently recomputes all66
win/point contrast records. No full neural-prediction replay claim is made.

Initial protocol run passed4/failed1 because the synthetic five-deal scalar
bootstrap differed by1.1e-16 from vector summation. Preserved failed XML;changed
only that small-case assertion to1e-15 absolute tolerance and added exact
nonconstant1,024-deal production-shape checks. Five tests then passed2.43s.
Added fixed/renewed initial actor/value/optimizer/global RNG comparison and
corrected a Ruff dictionary-literal finding:six tests passed4.85s then4.62s.
Finally added the real worker's five-format fixture using initial models and
an already used88M deal. Final command `.venv/Scripts/python.exe -m pytest tests/test_renewed_continuation_protocol.py -q --junitxml=runs/renewed_continuation_protocol_tests.xml`
passes7,0 skips,in7.68s with three expected Transformer warnings. All preceding
executions have0 new simulator matches;the final worker fixture adds5 full
correctness matches plus5 reference replays,not new strength observations.
It consumes no93M/95M/101M evaluation deal. Ruff format/check pass.

Final XML88ae130fa534ceea0b27d75520119deb60ffff20251c4afcf4201055e5eeb03a;
runner44073df1608e0811bc1b1f8f16f87db6baf5b5e9d5bf3bd80afbe55cd115f177;
auditorc691069845c880977a600a59cef99c157d680741ec6dae32e53fdab523145f06;
testac94e18b33ed73c96dccd7b6f82800228e50678aba53c661c194856c20c54cb4.
Readiness command `.venv/Scripts/python.exe -m runs.check_renewed_study_readiness`
verifies250 inputs and confirms the actual preflight refuses the live unfinished
comparison. Receipt5708bec872285baa6236a844c1e015650b81f44d8a1d11e06a8151fe26be9d93.

Launched `.venv/Scripts/python.exe -m runs.finish_renewed_continuation_study` once:
session37463,actual Python PID65372,launcher26672. It waits for the preceding
renewed CUDA capability receipt,then runs the new producer and independent audit
sequentially. Waiting receipt runs/renewed_continuation_study_finish.json pins252
unchanged inputs;SHA45a93fa6922e7280c352b71db942365368aec9a8595a4bf036eb93088d8ff253.
It records child PIDs/logs on launch;an8h observation deadline never kills prior
jobs. Do not restart or modify bound sources/tests. No new strength training or
evaluation has started;no promotion,dealer change or101M use.


## Abstract and Introduction focused on measured contributions

Previous turn implemented,tested and queued the renewed comparison;classified
as progress. This turn confirms allfour research processes actually live:
producer46255,audit96354,renewal capability PID36420 and renewal study37463.
Parent2-masked remains in its fixed training schedule;no evaluation contrast
inspected. The publication edit is outside all frozen experiment inputs.

Rewrote only Abstract/Introduction around the established adaptation gain,the
exact critic-information counterexample,and public-input reconstruction. Reduced
repeated experiment counts and combined unsuccessful-repair summaries while
retaining their outcome. Combined frontmatter672->552 whitespace words;
Abstract171->143 words. Related Work and every following source byte are
unchanged,including all empirical results,limitations,appendices and31 references.
Preserved the prior PDF as output/pdf/manuscript_prior_art_2ffe37c5.pdf and
source as runs/manuscript_before_frontmatter.tex. Existing CRLF preserved.
The PDF authoring marker was previously run exactly once and was not repeated.

Build `external/tectonic/tectonic.exe --keep-logs --outdir output/pdf docs/paper/manuscript.tex`
succeeds. The existing Fontconfig configuration notice persists but all20 fonts
are embedded. QA57222 exits0:all756 evidence inputs and252 queued bindings
verify;all prior numeric tokens retained;no over/underfull boxes,missing glyphs,
undefined references/citations or out-of-page text spans. Rendered/reviewed all28
pages in four contact sheets and inspected pages1/2/28 at full size. Final PDF
4bf68f3e5143bd1367f381886b91c8b6586093bc22b3eb970a894829935eacc0;
source08678404210ef1df84d17d0ef9c35cef05e38b3a68e41aa1ee8f37f4c294b84a;
bibliography5270a386287ff8645abbc216b39c57c99ab4286bd12179287dae93f4caed0897.
Final QA runs/manuscript_frontmatter_qa.json SHAfaf681e99a9d37b0989ad99419cc85f3977ccb920aa0f855c1004dd18a02cad0.
No ongoing-study outcome,model,dealer rule or reserved101M data changed.
