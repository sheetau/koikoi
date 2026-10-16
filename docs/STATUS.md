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
| ppo12_scale_seed30001 | 100,000,000 additional research12 matches | update 4,485 / 6,471,309 matches | 5ddd893 |
| ppo8_seed10001 | 5,000 updates | update 4,269 / 9,510,574 matches | 2f3a366 |
| relational8_v2 | 2,000 cumulative updates (complete) | 2,245,436 matches / 262,144,000 decisions | b9393cb |

Wider-MLP screening completed in runs/ppo8_wide_seed40001: 500 updates / 131,072,000 decisions / 1,108,105 matches, source a6b9bb6. It beats the matched-budget small MLP but loses to the relational candidate; further width-only training is deferred.

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

Finish the independent replication evaluation. Prioritize more relational-policy learning following its matched-budget gains. Config relational8_graph.json enables verified CUDA Graph updates, seed 50,001; initialize from relational8_v2/update_00002000.pt and train 3,000 new updates. This resets optimizer/RNG/counters and records the 2,245,436-match pretraining separately; it is not an exact continuation or an isolated capture ablation. Keep the verified twelve-round scale baseline running alongside method selection.

Before claiming success: exceed assisted RL-WP on paper8, replicate training results, evaluate chosen methods on untouched final-test deals, verify at least 100M completed twelve-round training matches, and prepare the research manuscript. Final-test seeds (100,000,000+) remain unused. Method/ablation selection, belief calibration, archived opponents and reduced-game exploitability are research options, not established results. Submission is a later decision.

## Latest screening results

- Relational update 2,000 beats matched-decision MLP update 1,000: 55.035%, CI 54.4699-55.625%, +4.0791 points, 10,000 pairs. Its final public-baseline evaluation is in progress.
- Wider MLP update 500 versus matched-decision MLP update 500: 53.0025%, CI 52.4375-53.5775%. Versus matched-decision relational update 1,000: 46.7625%, CI 46.1625-47.35%. Both use 10,000 pairs. Against assisted RL-WP on 300 pairs: 41.75%, CI 38.25-45.25%. These candidate screens vary precision/architecture/seed, not isolated architecture effects.
- MLP update 5,000 cards + published koi: 52.0833%, CI 49-55.1667%, 300 pairs against assisted RL-WP. Difference from the complete MLP on the same seeds: +2.0833 percentage points, CI -0.4167 to +4.5833. Reverse hybrid: 47.75%, CI 45.3333-50.25%. Koi substitution is suggestive but inconclusive; no default-policy change.
- `evaluate --koi-agent` records both components and their checkpoint metadata, resets nested RNG streams for pair replay, and rejects published twelve-round use or search combinations. Full-match routing and identity smoke checks passed.
- Relational CUDA Graph feasibility: five measured full updates after five warmups took 6.541 seconds under concurrent jobs (runs/relational_graph_benchmark.json), not an isolated performance claim.
