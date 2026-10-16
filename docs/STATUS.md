# Current status

Research is active. Published assisted-RL-WP superiority, the 100M twelve-round match target, final held-out validation and the manuscript remain incomplete. Full history, hashes and commands are in [EXPERIMENTS.md](EXPERIMENTS.md); read it only for the relevant experiment.

## Implementation and validation

Independent Python oracle plus shared C++/CUDA core; paper8 and research12 profiles; GPU PPO with signed match returns; MLP/features/relational/Transformer policies; experimental belief/Q-boosting/search; pinned published-model adapter; paired sequential/native evaluation; koi-policy substitution; exact checkpoints and initialization-lineage budget audit. Actor inputs never include hidden information. Search remains off by default. See DESIGN.md for architecture and RULES.md for game logic.

Captured PPO minibatches and collection are independently optional. Warmup restores state/model/Adam/RNG, and replay reads current weights. FP32/BF16 collection, full updates and resume agree with eager execution, including real scale/relational migrations. Capturable versus old non-capturable Adam can round differently; do not silently switch optimizer execution.

Validation: full 74 tests passed without skips, followed by four targeted resume cases and one new Transformer full-update case; 77 cases collected. Ruff passed. Two expected upstream warnings. Package source unchanged since 142c265 (SHA 2bfd85f83056c95d8593b4bca41ece74981e9d903ab77f171a5ecf8732c7da37). Later milestones changed configs/docs/tests, not learning arithmetic.

Use .venv/Scripts/python.exe: Python 3.12, Torch 2.12.0+cu130, NumPy 2.5.4. Hardware RTX 5070 Ti 16 GB, Ryzen 9900X, about 62 GiB RAM. Preserve UTF-8 and existing line endings.

## Only live training job (revalidate before acting)

runs/ppo12_scale_seed30001_captured: 15,831 updates / 22,862,759 cumulative new research12 matches last observed. Source 142c265, child PID 57432 (parent 6624), session 57088. Target 100,000,000 new completed matches. Process and advancing finite metrics verified.

```powershell
./.venv/Scripts/python.exe -m koikoi.train --resume runs/ppo12_scale_seed30001/update_00005800.pt --allow-code-change --enable-cuda-rollout --run runs/ppo12_scale_seed30001_captured --target-matches 100000000 --checkpoint-every 1000
```

These counters exclude 1,000,464 research12 and 11,141,556 paper8 initialization matches. Never relaunch a live job. Use immutable numbered snapshots. A run-local STOP requests clean exit after a completed update. Old ppo12_scale_seed30001, relational8_graph_seed50001 and ppo8_seed10001 are terminal migration directories with STOP; do not restart them.

All other trainers and evaluation watchers completed, including sessions 29416, 68401, 33995 and 93839. No pending public/hybrid/entropy evaluations remain. The finished relational and entropy runs must not be resumed merely because their process is absent.

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

Scale update 12,000 versus transfer initialization: 51.42% (CI 50.87-51.97%), +1.3174 points, 10,000 development pairs. Shared-pair score improvement versus update 7,000 is +0.2025 pp (CI -0.5226 to +0.985), so later improvement remains unproven. Reports eval_12000_vs_parent.json and difference_12000_vs_7000.json in the scale run. These research12 results do not establish superiority to the paper8 published baseline.

Immutable update-12,000 audit: 17,324,780 new research12 + 1,000,464 research12 initialization = 18,325,244 twelve-round matches; paper8 initialization 11,141,556 separate. SHA 8fb7bcaf7c64717c451a16878e008693506aab64468eaf3f1916bc3bf8bae14e. Report budget_12000.json. Resume parent counters are not added again. This is one model lineage, not a project-wide sum.

## Next work and completion gates

Implement and validate an optional mixed-current/frozen-history opponent experiment before further unchanged paper8 extensions. Motivation: parent-relative gains repeatedly fail to establish public-baseline superiority; this is a hypothesis about improving opponent diversity, not a demonstrated cure or equilibrium guarantee. Frozen-opponent actions must never enter the learner PPO policy/entropy/belief objectives as its own on-policy actions. Handle player-relative value targets correctly, keep opponent identity/seat fixed within a match, count learner decisions separately from simulator decisions, embed frozen weights/provenance in checkpoints, and verify exact ordinary/captured restart. Keep default current self-play unchanged. No historical-opponent implementation exists yet.

Keep the scale baseline running. Success still requires beating assisted RL-WP, independent training validation of selected improvements, untouched final-test evaluation, at least 100M complete research12 training matches, and a manuscript. Submission is a later decision. Do not mark complete based on implementation, parent wins or throughput alone.
