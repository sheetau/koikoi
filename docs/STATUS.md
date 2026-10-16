# Current status

Research is active. Public assisted-RL-WP superiority, selected-method independent validation, untouched final testing, the 100M-new-research12 target and manuscript are unfinished. Submission is a later decision. Full commands, hashes and history: [EXPERIMENTS.md](EXPERIMENTS.md). Read DESIGN.md for architecture and RULES.md for game logic.

## Live jobs: verify processes before acting

| Work | Current execution | Next action |
| --- | --- | --- |
| Research12 scale | Session 57088; trainer PID 57432, parent 6624; runs/ppo12_scale_seed30001_captured. Last observed 64,428 updates / 93,022,166 new completed matches. Original source 142c265, never restarted. | Continue fixed 100,000,000-new-match target. |
| Final scale watcher | Session 51102; PID 19092, parent 49408; `python -m runs.watch_scale100m`. Started from protocol 298d59c. | Waits on retained trainer handle; audits and evaluates only after successful target completion. |
| Independent-model search replication | Session 95279; runner PID 10228, parent 29100; `python runs/run_koi_search192_seed1.py`. Search evaluator PID 57536, parent 56492. | Control complete; 1,000-pair captured-search arm running, then paired comparison/audit. |

Evaluator PIDs change between stages. Never duplicate a live pipeline or restart merely because a poll times out. The only remaining training process is the scale run. Koi trainer children 50444/36772/57124 and endpoint runner session 57508 are terminal; all screen reports/audits completed successfully. Earlier scale-52,000 watcher 51551 and search-extension runner 76409 are complete. Old migration directories ppo12_scale_seed30001, relational8_graph_seed50001 and ppo8_seed10001 contain STOP and must not be restarted. All older critic/fixed-opponent/entropy/relational/hybrid endpoints are complete.

```powershell
./.venv/Scripts/python.exe -m koikoi.train --resume runs/ppo12_scale_seed30001/update_00005800.pt --allow-code-change --enable-cuda-rollout --run runs/ppo12_scale_seed30001_captured --target-matches 100000000 --checkpoint-every 1000
```

This records the existing scale command, not an instruction to launch it again. Counts exclude 1,000,464 research12 and 11,141,556 paper8 initialization matches.

## Current evidence and decisions

All results below are development results with paired-bootstrap 95% intervals. Final-test seeds 100,000,000+ are unused. Shared deals across training models are not independent extra pairs.

- Full 1,000-pair koi search on independent MLP seed 10,001: 50.025% (CI 48.2244-51.825%) versus assisted RL-WP; CUDA control 49.375%. Paired gain +0.65 pp (CI +0.125 to +1.2), points +0.246 (CI -0.057 to +0.558). This exploratory nested extension warrants replication, not public superiority or default adoption. Search stays off by default.
- Fixed search settings: uniform 128 particles, horizon 192, margin 2.5, koi-only, deterministic/endgame-assisted CUDA capture. Original 100 plus additional 900 included. All 1,000 control records equal CPU. 9,096 searches / 173 changes / 447,086,592 logical transitions + 294,912 preparation transitions, excluded from training. Summary/audit koi_search192_1000_summary.json and koi_search192_1000_audit.json under runs/ppo8_seed10001_captured.
- Ongoing replication uses seed-1 MLP runs/ppo8_v3/update_00005000.pt, SHA d104ae00824aca114e6cd414c74d02670b9a8de5a009ad4fad557005f66233e5, same 1,000 development pairs and settings. CUDA control 47.825%, all records equal historical CPU eval_5000_guarded.json. Exact-byte alias eval_5000_wp.json supports the live runner; mapping/hash in control_cpu_alias.json. Outputs eval_5000_koi_search192_captured1000.json, difference_koi_search192_1000.json, koi_search192_1000_audit.json. Logs runs/koi_search192_seed1_{control,search,compare}.log. No intermediate selection.
- Scale update 52,000: 75,085,544 new research12 matches; lineage 76,086,008 research12, paper8 11,141,556 separate. SHA 313f2299aba0b3b66252790ac718d614f6d2745d8404940e836b466995777768. Versus transfer initialization: 52.1925% (CI 51.64-52.7676%), 10,000 pairs. Difference from update 35,000: -0.25 pp (CI -0.9675 to +0.4951). Parent advantage persists; further improvement unproven. Reports budget_52000.json, eval_52000_vs_parent.json, difference_52000_vs_35000.json, audit_52000.json. This is not a paper8 public comparison.

## Koi adaptation screen: complete, larger development evaluation queued

Committed implementation 5d4e05d; package SHA 3eb0e894aa14757912a157dd5fa44ffc1e84f83528c677a9a198c93ae8172bcc. Configs ppo8_koi_control.json / ppo8_koi_head.json / ppo8_koi_residual.json compare full-policy continuation, koi-only head adaptation and koi-only head plus residual128. All initialize from runs/ppo8_seed10001_captured/update_00005000.pt (SHA edc44213bc5c5c840297afc33ee64a57b91e9dee48c202535906fc916cde109d), seed 60,001, fresh observation critic256/Adam/RNG/environment, same 1,000 updates / 131,072,000 simulator decisions. Paper8 only. No training endgame guard, no belief loss, entropy 0.02, lambda 0.95. Simulator budgets match; policy-gradient decision counts differ.

| Arm under runs/ppo8_koi_{arm}_seed60001 | New matches | Koi policy decisions | Checkpoint SHA (update_00001000.pt) |
| --- | ---: | ---: | --- |
| control | 1,115,289 | All phases optimized | a225b172f70ae968e02217a0eba2e4ea1bae7ba68cd52f817e72fb3aa41f3b80 |
| head | 1,116,221 | 11,347,406 | 824e8f5392e099aaf0f49c311a2d9acd9f252c18230fac480c5b90baf3714905 |
| residual | 1,115,299 | 11,364,073 | 996443b42fd1f4b91fd096e7292440668ca50c2ef8b67b3cb4682dac424f8f3c |

Each lineage separately adds 11,141,428 paper8 pretraining matches. Actual hashes, counters/metrics, episode sums, finite weights/Adam and 96,000 optimizer steps verified in runs/koi_training_endpoints_audit.json. Specialized card/shared/belief tensors equal the parent exactly and have no Adam state. Do not resume these completed learners.

All original endpoint comparisons and runs/koi_screen_endpoints_audit.json completed. Means/intervals and paired differences were independently recomputed. Head/control: 50.545% (CI 49.9999375-51.0975625%); residual/control: 50.5475% (49.9975-51.125%); residual/head: 49.9125% (49.6525-50.175%), each 10,000 pairs. Neither specialized arm establishes a win-rate gain; do not round near-50% bounds into significance. Both improve point difference against control; residual also improves points against head, which does not meet the win-rate objective.

Public 300-pair results: control 46% (CI 42.75-49.25%), head 48.75% (45.5833-52.1667%), residual 49% (45.75-52.5%). Public paired gains versus control +2.75 pp (CI -1.25 to +6.9167) / +3 pp (-1 to +7.0833). Supplementary comparisons versus initialization on existing shared records: head +0.5 pp (CI -1.25 to +2.25), residual +0.75 pp (-1.25 to +2.75), control -2.25 pp (-6.5833 to +1.9167); per-run difference_vs_initialization.json. No superiority or decline is established from those differences. No candidate adopted or training extended.

Next fixed endpoint: add 700 CPU public pairs per arm, seeds 1,000,300-1,000,999, unchanged model/guards/deterministic/no-search settings. Original 300 remain in combined-1,000 results for every arm. Supplementary native parent duels: each arm versus the initialization on 10,000 guarded deterministic development pairs. Runner runs/run_koi_extension.py, outputs per-run eval_1000_wp_new700.json and ab_vs_initialization_1000.json; summary runs/koi_screen_1000_summary.json. Logs runs/koi_extension_{arm}_{public,parent}.log. Process manifest runs/koi_extension_processes.json. Full protocol in EXPERIMENTS.md. Launch once from committed protocol; no interim selection or further training. Same development deals are not independent extra data across models.

## Verified implementation and limits

Independent Python oracle plus shared C++/CUDA core, GPU PPO/signed match returns, exact restart and initialization lineage, pinned upstream adapter, native paired evaluation, optional beliefs/Q-boosting/frozen opponents/separate critics/search. Actor and rollout policy inputs are observation-only. No native rules changed recently. Full 126 tests passed without skips; only two expected upstream Transformer warnings; Ruff passed. Full-suite session 20815 and production validation 99692 completed.

Optional policy_scope=koi freezes card/shared/belief weights and trains koi/value heads plus optional residual. Policy/entropy/KL/advantage normalization uses koi rows; complete-match GAE and both value losses use all phases. Requires pure self-play match PPO, ppo_value=observation, zero belief loss. Card actions are sampled from the fixed card distribution. koi_width adds a 560 -> width -> width -> 2 tanh residual with zero output initialization. Defaults all/zero preserve old behavior. No strength claim yet.

Twenty dedicated koi tests passed. Actual old-source default equivalence (MLP/FP32 and relational/BF16, two updates, full states/RNG/metrics) passed in runs/koi_default_equivalence.json. Production-shape three-update smokes and fourth eager/captured update equality passed for all three arms in runs/koi_production_equivalence.json, including identical first collections and frozen card tensors. Smoke/duplicate diagnostics are excluded from budgets. Both graph options restore warmup state/RNG; do not silently switch old non-capturable Adam. Current search replica loaded package 3eb0e894... after control package 4ff30860...; its default actor/game/search behavior is unchanged and verified. Scale still uses its original source.

## Prior rejected or inconclusive methods

Do not extend unchanged: relational continuation (public 47.875% on 1,000 pairs despite parent-relative gain), lambda 0.99, entropy 0.005, fixed historical/independent-MLP opponents at 20%, separate observation/privileged PPO critics. Their full endpoints/budgets and negative or inconclusive comparisons are in EXPERIMENTS.md. Wider MLP, explicit features, sampled deployment, Q-boosting, belief/earlier search and on-policy endgame training have no established public advantage.

MLP seeds 1/10,001 at 5,000 updates score 47.825%/49.375% versus assisted RL-WP on 1,000 pairs. Published-koi substitution improves those same pairs by +3.0/+1.8 pp, but additional 3,000 pairs score 49.9417%/50.6833%, without superiority. Relational koi substitution scores 51.4167% on 300 pairs; difference +1.9167 pp (CI -0.9167 to +4.6667), inconclusive. Published koi is an explicit external dependency, not strength attributable solely to our actor.

## Final scale audit and completion gates

The live final watcher requires successful trainer exit and first update crossing 100M new research12 matches, compares latest.pt with final metrics, checks episode sums, rounds = 12 * completed matches + current-round indices, decision arithmetic and finite model/Adam, then exclusively preserves exact bytes as final_100m.pt. This audit logic passed real updates 35,000/52,000 and rejected corrupt counters/nonfinite weights; runs/scale_episode_accounting_audit.json. Never infer completion from a command or watcher alone.

After preservation: budget_100m.json, eval_100m_vs_parent.json (10,000 guarded deterministic development pairs), difference_100m_vs_52000.json, audit_100m.json under the scale run. Logs runs/scale100m_*.log. A below-target/failed exit requires inspection, never automatic restart. Final-scale evaluation is still development, not untouched final testing.

Success still requires public-baseline superiority, independent validation of selected improvements, untouched final evaluation, at least 100M complete research12 training matches and manuscript. Keep the goal active until all are verified. Use .venv/Scripts/python.exe (Python 3.12, Torch 2.12+cu130, NumPy 2.5.4); RTX 5070 Ti 16 GB. Explicit UTF-8 reads, preserve existing line endings, new files UTF-8 without BOM/LF. Runs, external sources and local agent rules are ignored. Commit coherent milestones; avoid repeated status-only commits between them.
