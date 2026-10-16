# Current status

Implementation and research in progress; the published RL-WP strength gate is still unmet. No strongest-AI or publication claim is established. Historical results and earlier plans are in [EXPERIMENTS.md](EXPERIMENTS.md); read that only when relevant.

## Verified implementation

- Independent Python oracle and shared C++/CUDA simulator; versioned paper8 and research12 rules (see RULES.md).
- GPU-resident self-play PPO, native signed GAE, MLP/features/relational/Transformer policies, optional belief loss and separate experimental Q-boosting.
- Atomic checkpoints with exact continuation, source hashes, separate weight-initialization lineage and auditable learning counters.
- Published model adapter, paired confidence intervals, native batched checkpoint evaluation, guarded hidden-state rollout search, optional on-policy endgame masks.
- CUDA Graph minibatch updates with rollback of warmup parameters, optimizer and RNG; FP32/BF16 restart verified.
- Checkpoint budget audit separates rule profiles and pretraining; resume counters are not counted twice. It audits one model ancestry, not the union of all project experiments.

Validation: 66 tests passed on this host, no skips, including CUDA and pinned upstream. Two expected upstream Transformer performance warnings. Ruff passed. Details of independent correctness/restart checks are in EXPERIMENTS.md.

Hardware: RTX 5070 Ti 16 GB, Ryzen 9 9900X, about 62 GiB RAM; local Python 3.12 / PyTorch 2.12.0+cu130 / NumPy 2.5.4. Use .venv/Scripts/python.exe.

## Current jobs (recheck processes and advancing metrics before acting)

| Run | Target | Last observed completed progress | Source |
| --- | --- | --- | --- |
| ppo12_scale_seed30001 | 100,000,000 additional research12 matches | update 3,214 / 4,637,529 matches | 5ddd893 |
| ppo8_seed10001 | 5,000 updates | update 3,935 / 8,765,882 matches | 2f3a366 |
| relational8_v2 | 2,000 cumulative updates | update 1,856 / 2,082,470 matches | b9393cb |

Wider-MLP screening is also live in runs/ppo8_wide_seed40001 from committed a6b9bb6, target 500 updates, snapshots every 100; last checked update 108 / 236,639 matches. Command: `python -m koikoi.train --config configs/ppo8_wide_bf16.json --run runs/ppo8_wide_seed40001 --updates 500 --checkpoint-every 100`.

The scale run is live, not complete. Its fresh counters exclude 1,000,464 research12 and 11,141,556 paper8 pretraining matches, recorded separately. Runs and checkpoints live under ignored runs/. Do not restart live jobs. A run-local STOP requests a clean completed-update snapshot.

```powershell
./.venv/Scripts/python.exe -m koikoi.train --config configs/ppo12_graph.json --initialize runs/ppo12_transfer_v1/update_00000693.pt --run runs/ppo12_scale_seed30001 --target-matches 100000000 --checkpoint-every 1000
```

## Decision evidence (development only)

- Paper8 MLP seed 1, update 5,000: 11,141,556 training matches; 47.825% vs assisted RL-WP over 1,000 pairs, CI 46.05-49.65%. Still below baseline. runs/ppo8_v3/eval_5000_guarded.json.
- Seed 10,001 update 3,000: 46.4167%, CI 43.0833-49.5833%, against assisted RL-WP over 300 pairs. Final replication pending.
- Assisted-training A/B completed 1,000 new updates each. Against RL-WP on 1,000 identical pairs: 49.125% assisted vs 49.175% control; difference -0.05 percentage points, CI -2.35 to +2.225. Direct 10,000-pair duel also inconclusive. Keep endgame training off by default.
- Relational update 1,000 beats MLP update 500 at the same 131,072,000 training decisions: 56.1075%, CI 55.5374-56.6575%, 10,000 pairs. It loses to the much longer-trained MLP update 5,000: 47.648%, CI 47.080-48.220%. This supports further screening, not an isolated architecture effect or superiority to RL-WP.
- Research12 transfer pilot completed 1,000,464 matches and beat its initialization: 52.073%, CI 51.515-52.633%, 10,000 pairs. Scale update 3,000 versus that pilot: 50.6225%, CI 50.0725-51.1901%, +0.8892 points, 10,000 pairs. This is a small development gain; repeated checkpoint selection still requires untouched final-test validation.
- Search, Q-boosting and explicit features have not demonstrated improvement. No candidate has established superiority to RL-WP. Keep search off. Positive point difference alone does not satisfy a match-win objective.

## Next actions and completion gates

Finish final-checkpoint evaluations for replication and relational jobs. Screen a wider MLP with BF16/CUDA Graph at matched decision budgets; this joint capacity/precision candidate is not an isolated architecture ablation. Keep the verified twelve-round scale baseline running alongside method selection.

Before claiming success: exceed assisted RL-WP on paper8, replicate training results, evaluate chosen methods on untouched final-test deals, verify at least 100M completed twelve-round training matches, and prepare the research manuscript. Final-test seeds (100,000,000+) remain unused. Method/ablation selection, belief calibration, archived opponents and reduced-game exploitability are research options, not established results. Submission is a later decision.

Wider candidate config: configs/ppo8_wide_bf16.json (MLP width 512, seed 40,001, BF16, CUDA Graph). Five measured full learning updates after five warmups completed in 3.871 seconds / 10,468 matches while other jobs ran (runs/wide_bf16_benchmark.json). This checks feasibility, not playing strength or isolated throughput. Initial screening target is 500 updates / 131,072,000 decisions, then evaluate against the 500-update width-128 checkpoint and assisted RL-WP before extending.

Koi-decision substitution is now available as `evaluate --koi-agent`: independently choose the card and koi policies, with explicit component metadata, retained pair replay and guard settings. Full-match routing, nested RNG replay and invalid protocol rejection passed; composing RL-WP with itself reproduced the assisted reference on a two-pair identity smoke check. Use this diagnostic to localize the remaining strength gap; no hybrid strength result exists yet.
