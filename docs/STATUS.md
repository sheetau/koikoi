# Current status

Research is active. Published assisted-RL-WP superiority, the 100M twelve-round match target, final held-out validation and the manuscript remain incomplete. Full history, hashes and commands are in [EXPERIMENTS.md](EXPERIMENTS.md); read it only for the relevant experiment.

## Implementation and validation

Independent Python oracle plus shared C++/CUDA core; paper8 and research12 profiles; GPU PPO with signed match returns; MLP/features/relational/Transformer policies; experimental belief/Q-boosting/search; pinned published-model adapter; paired sequential/native evaluation; koi-policy substitution; exact checkpoints and initialization-lineage budget audit. Actor inputs never include hidden information. Search remains off by default. See DESIGN.md for architecture and RULES.md for game logic.

Captured PPO minibatches and collection are independently optional. Warmup restores state/model/Adam/RNG, and replay reads current weights. FP32/BF16 collection, full updates and resume agree with eager execution, including real scale/relational migrations. Capturable versus old non-capturable Adam can round differently; do not silently switch optimizer execution.

Validation: full 89-test suite passed without skips after the frozen-opponent implementation, including CUDA and upstream checks; two expected upstream Transformer warnings. Two targeted full-update cases also passed after extending coverage to relational/BF16 frozen opponents. Ruff passed. No native game rules changed.

Against actual old source 499990c, two ordinary self-play updates remain bit-identical for MLP/FP32 and relational/BF16: model, Adam, simulator, counters, CPU/CUDA/Python/NumPy RNG and metrics. Report runs/frozen_default_equivalence.json. The ongoing scale process still runs source 142c265; it was not restarted.

Use .venv/Scripts/python.exe: Python 3.12, Torch 2.12.0+cu130, NumPy 2.5.4. Hardware RTX 5070 Ti 16 GB, Ryzen 9900X, about 62 GiB RAM. Preserve UTF-8 and existing line endings.

## Live training jobs (revalidate before acting)

runs/ppo12_scale_seed30001_captured: 23,056 updates / 33,298,483 cumulative new research12 matches last observed. Source 142c265, child PID 57432 (parent 6624), session 57088. Target 100,000,000 new completed matches. Process and advancing finite metrics verified.

```powershell
./.venv/Scripts/python.exe -m koikoi.train --resume runs/ppo12_scale_seed30001/update_00005800.pt --allow-code-change --enable-cuda-rollout --run runs/ppo12_scale_seed30001_captured --target-matches 100000000 --checkpoint-every 1000
```

These counters exclude 1,000,464 research12 and 11,141,556 paper8 initialization matches. Never relaunch a live job. Use immutable numbered snapshots. A run-local STOP requests clean exit after a completed update. Old ppo12_scale_seed30001, relational8_graph_seed50001 and ppo8_seed10001 are terminal migration directories with STOP; do not restart them.

Prior relational/entropy/hybrid trainers and evaluation watchers completed, including sessions 29416, 68401, 33995 and 93839. Those finished runs must not be resumed merely because their process is absent.

New historical-opponent run runs/relational8_history20_seed50001 is live from committed 14d62c0: 377 updates / 427,644 matches / 49,414,144 simulator decisions / 44,492,746 learner decisions observed. Child PID 43060 (parent 19056), training session 95960. Command: `python -m koikoi.train --config configs/relational8_history20.json --initialize runs/relational8_v2/update_00002000.pt --frozen-opponent runs/relational8_v2/update_00002000.pt --run runs/relational8_history20_seed50001 --updates 1000 --checkpoint-every 250`. Process and advancing finite metrics verified.

Final-evaluation watcher session 52845 guards PID 43060 and waits for immutable update_00001000.pt. It then runs 10,000 direct pairs versus control update 1,000, 300 assisted-RL-WP pairs and shared-pair control difference. Outputs ab_1000.json, eval_1000_wp.json and difference_vs_control.json in the history run; logs runs/history20_final_*.log. It fails if the trainer exits without the snapshot. Do not duplicate this watcher.

## Completed evidence and decisions (development only)

| Candidate | Evidence | Decision |
| --- | --- | --- |
| Small MLP, seeds 1 and 10,001, 5,000 updates each | About 11.14M paper8 matches each. Assisted RL-WP: 47.825% (CI 46.05-49.65%) and 49.375% (47.6-51.1256%), 1,000 pairs each. Direct duel 49.975% (49.4025-50.5225%), 10,000 pairs. | No superiority; independent replication complete. |
| Relational continuation, seed 50,001, 3,000 new updates | 3,401,171 new matches + 2,245,436 pretraining. Parent duel 51.27% (50.72-51.825%), 10,000 pairs. Assisted RL-WP 47.875% (46.075-49.625%), 1,000 pairs. | Parent-relative improvement does not generalize to the baseline. Do not extend unchanged. |
| Return trace lambda 0.99 versus 0.95 | Matched 1,000 new updates: 48.12% (47.5699-48.6725%) in 10,000 direct pairs; public baseline 48.75%, 300 pairs. | Reject extension; retain 0.95. |
| Entropy weight 0.005 versus 0.02 | Matched 1,000 new updates: 49.2925% (48.745-49.8375%) in 10,000 direct pairs; public baseline 48.0833% (44.75-51.3333%), 300 pairs. | Reject extension; retain 0.02. |
| MLP cards + published RL-WP koi, two training seeds | First 1,000 pairs improve over each full MLP by +3.0 pp (CI +1.5 to +4.55) / +1.8 pp (+0.3 to +3.375). Additional 3,000 pairs: 49.9417% (48.9083-50.9669%) / 50.6833% (49.6667-51.6919%) versus RL-WP. | Diagnostic improvement, no established public-baseline superiority; no default change. |

All intervals are 95% paired-bootstrap intervals. Shared evaluation deals across training seeds are not independent extra pairs. Additional hybrid seeds 1,001,000-1,003,999 are disjoint from earlier public comparisons but were used in native checkpoint duels. Final-test seeds 100,000,000+ remain unused.

Relational update 3,000 versus 1,500 on 1,000 shared public pairs: -0.675 pp (CI -2.825 to +1.475), so no established win-rate decline either; point difference declines -1.538 (CI -2.8551 to -0.2640). Entropy-screen versus control on 300 shared public pairs: -1.4167 pp (CI -5.5833 to +2.75). Positive point differences alone do not meet the win-rate objective.

Earlier screens: relational beats matched-decision MLP; width-512 MLP loses to relational; on-policy endgame training did not improve; sampled deployment, Q-boosting, explicit features and rollout search have no established advantage. Keep those changes off by default. Details in EXPERIMENTS.md.

## Artifact locations

- Completed relational: runs/relational8_fullgraph_seed50001/update_00003000.pt, SHA 2d4d50a489230ec0080515baf3338957acc6708f839bd92fb4f96df41d971eae. Reports eval_3000_vs_parent.json, eval_3000_wp.json, difference_3000_vs_1500.json. Final checkpoint hash and counters match final metrics.
- Completed entropy screen: runs/relational8_entropy005_seed50001/update_00001000.pt, SHA 03490c82b1ffe5cc56a44f4fc573fabb927bbb0a03ff87c04cc4ca7930d7f51f. 1,136,109 new matches / 131,072,000 decisions; reports ab_1000.json, eval_1000_wp.json, difference_vs_control.json. Hash/counters verified.
- Hybrid reports eval_5000_wp_koi_1000.json, difference_wp_koi_1000.json and eval_5000_wp_koi_new3000.json under runs/ppo8_v3 and runs/ppo8_seed10001_captured. The published koi component is an explicit dependency, not learned strength attributable to the card network alone.
- Prior controlled initialization: runs/relational8_v2/update_00002000.pt. Matched control endpoint: runs/relational8_fullgraph_seed50001/update_00001000.pt. Controlled screens use seed 50,001, fresh optimizer/RNG/counters and same initialization; this is not independent training replication.
- Independent MLP checkpoint runs/ppo8_seed10001_captured/update_00005000.pt, SHA edc44213bc5c5c840297afc33ee64a57b91e9dee48c202535906fc916cde109d. Its historical-source migration was verified bit-identical; report runs/replica_old_revision_migration.json. Do not label it fixed-revision execution throughout.

## Scale strength and budget

Scale update 22,000 versus transfer initialization: 52.275% (CI 51.7325-52.8375%), +1.6925 points, 10,000 development pairs. Shared-pair score improvement versus update 12,000 is +0.855 pp (CI +0.1375 to +1.5925), supporting further development improvement against that parent. Point-difference change +0.3751 (CI -0.1569 to +0.9131) is inconclusive. Reports eval_22000_vs_parent.json and difference_22000_vs_12000.json in the scale run. These research12 results do not establish superiority to the paper8 published baseline.

Immutable update-22,000 audit: 31,771,972 new research12 + 1,000,464 research12 initialization = 32,772,436 twelve-round matches; paper8 initialization 11,141,556 separate. SHA db58b75c2aca40f5a08bc9f9cc57c6149e5c3b9671388ece76c301f491476279. Report budget_22000.json. Simulator and learner decisions coincide for these pure-self-play stages. Resume parent counters and frozen-opponent dependencies are not added again. This is one model lineage, not a project-wide sum.

## Next work and completion gates

Optional fixed-history PPO is implemented, default off. Config opponent_fraction assigns floor(envs * fraction) environments to one fixed checkpoint; learner seats alternate per completed match. Frozen actions are excluded from policy/entropy/belief/KL losses and advantage normalization. A separate trainable observation-only opponent-role critic supplies its signed GAE values; both trainable networks participate in graph rollback. Frozen weights/config/provenance are embedded for self-contained restart. Frozen-opponent pretraining is disclosed separately and excluded from learner ancestry. This is a hypothesis about opponent diversity, not a demonstrated strength gain or a full population/NFSP algorithm.

Live controlled screen: configs/relational8_history20.json, seed 50,001, initialize learner and frozen opponent from runs/relational8_v2/update_00002000.pt. 204 of 1,024 environments use the frozen opponent. The resolved config differs from the prior control only in opponent_fraction 0 -> 0.2; the implementation adds the role critic and yields fewer learner decisions per simulator decision, so this is a method screen, not an isolated opponent-diversity effect. Target 1,000 new updates / 131,072,000 simulator decisions, snapshots every 250. Record actual learner_decisions, expected roughly 90% of total. Run runs/relational8_history20_seed50001; compare with control update 1,000 on 10,000 direct pairs and 300 shared assisted-RL-WP development pairs. Positive evidence requires independent training replication and untouched final evaluation.

Initialized production-shape smoke passed three updates: 3,062 matches / 393,216 simulator decisions / 353,918 learner decisions, finite losses/gradients, runs/relational_history20_smoke. This is execution validation, not strength evidence; duplicated smoke learning is excluded from the controlled experiment.

Keep the scale baseline running. Success still requires beating assisted RL-WP, independent training validation of selected improvements, untouched final-test evaluation, at least 100M complete research12 training matches, and a manuscript. Submission is a later decision. Do not mark complete based on implementation, parent wins or throughput alone.

Production-size frozen-opponent validation also passed: from the initialized three-update smoke checkpoint, ordinary versus captured collection produced exactly the same next update, including learner/opponent-role/frozen weights, Adam, environment, counters and all RNG states. Report runs/frozen_production_equivalence.json; duplicated validation updates are excluded from training counts. Verified package SHA 3f1107914d1c6c5019f1f684c6b7149deaf4982e9f57288533c256747243f72e.

A second fixed-opponent comparison is live: same configs/relational8_history20.json, seed 50,001, learner initialization and 1,000-update target; only --frozen-opponent changes to runs/ppo8_seed10001_captured/update_00005000.pt. Run runs/relational8_history_mlp_seed50001. This MLP has independent training seed/architecture/history, including 11,141,428 paper8 matches disclosed separately. The original historical-parent arm keeps its endpoint. Compare both arms with the same current-self-play control and public opponent, then directly against each other. It tests opponent identity, not diversity alone. No interim strength results informed this choice. Production-shape three-update smoke passed: 3,057 matches / 393,216 simulator decisions / 353,863 learner decisions, runs/relational_history_mlp_smoke; duplicated smoke updates are excluded from the experiment.

MLP-opponent arm started from committed 642e9b2 (same package as 14d62c0); child PID 31848 (parent 59584), training session 59905. Observed 68 updates / 76,467 matches / 8,912,896 simulator decisions / 8,021,039 learner decisions. Process and advancing finite metrics verified; first three updates reproduce the isolated smoke exactly, excluding time/rate metrics. Command: `python -m koikoi.train --config configs/relational8_history20.json --initialize runs/relational8_v2/update_00002000.pt --frozen-opponent runs/ppo8_seed10001_captured/update_00005000.pt --run runs/relational8_history_mlp_seed50001 --updates 1000 --checkpoint-every 250`.

MLP-opponent evaluation watcher session 5205 guards PID 31848, waits for immutable update_00001000.pt, then writes ab_1000.json (10,000 pairs versus control), eval_1000_wp.json (300 public pairs) and difference_vs_control.json. Finally it waits for the original history20 endpoint if necessary (guard PID 43060) and writes ab_vs_history20.json (10,000 pairs, MLP-opponent arm first). Logs runs/history_mlp_*.log. These evaluations are pending; do not duplicate. The original history20 watcher session 52845 remains queued.
