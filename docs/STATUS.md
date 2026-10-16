# Current status

Research is active. The 100M-new-research12 requirement is achieved and audited. Public assisted-RL-WP superiority, independent validation of selected improvements, untouched final testing and manuscript remain unfinished. Submission is a later decision. Full commands, hashes and historical evidence: [EXPERIMENTS.md](EXPERIMENTS.md). Consult DESIGN.md for architecture, RULES.md for game logic.

## Execution: verify actual handles before acting

All training and prior evaluation pipelines are complete. Most recent runner session 59159 (deterministic-card comparison), trainer PIDs 23128/2892, public evaluators34448/44676 and runner 26416 are terminal success. Earlier scale57088/final watcher51102, seed-1 search95279 and koi extension63774 are also complete. Never resume completed learners unchanged. Old migration directories with STOP remain closed.

The full-action 100-pair screen (session 41603, runner 61488/evaluator 11808) is complete and independently audited. Next extension runner runs/run_all_search192_extension.py is prepared but not launched; launch once after committing the fixed protocol below. No new training is selected. Never duplicate live work or infer process termination from a polling timeout.

## Immediate next experiment

Extend the original seed-10,001 MLP full-action search by exactly 900 development pairs, seeds 1,000,100-1,000,999. Model runs/ppo8_seed10001_captured/update_00005000.pt, SHA edc44213bc5c5c840297afc33ee64a57b91e9dee48c202535906fc916cde109d. Assisted RL-WP opponent, deterministic/endgame-assisted CUDA, uniform 128 particles, horizon 192, confidence 2.5, --search-all --search-cuda-graph. Use the validated RNG-exact preprocessing optimization; no algorithm/settings/model change.

Primary endpoint original 100+new 900 =1,000 pairs, retaining the original screen. Compare with existing corresponding CUDA no-search and koi-only segments; verify checkpoint hashes rather than literal agent path slash conventions. Outputs under original MLP: eval_5000_all_search192_new900.json, all_search192_1000_summary.json. Log runs/all_search192_new900.log; manifest runs/all_search192_extension_process.json. No interim selection or final-test seeds. Nested development extension requires independent-model and untouched final validation before adoption.

Completed 100-pair result: all-action 49.25% (CI 43.25-55.25), no-search 43.25%, koi-only 45%. Paired gain versus control +6 pp (CI +2.25 to +10), versus koi +4.25 pp (+0.24375 to +8.75). Absolute superiority remains unproven. Actual hashes and all statistics recomputed in all_search192_100_statistics_audit.json.10,867 searches/103 changes/1,286,922,240 logical plus 2,580,480 preparation transitions, not training. The 100-pair evaluator retained original package be4b1cd0... and was never restarted.

## Latest completed evidence

All comparisons below are development results with paired-bootstrap 95% intervals. Final seeds 100,000,000+ remain unused. Shared deals across models are not independent extra pairs.

- Deterministic-card koi adaptation completed both 1,000-update/131,072,000-decision endpoints, all five 10,000-pair native duels and public 1,000 pairs per arm. Public head 49.25% (CI 47.525-51.025), residual 49.175% (47.399375-50.975). Paired changes versus sampled-card counterparts: +0.125 pp (CI -0.55 to +0.775) and -0.075 pp (-1 to +0.825). Versus initialization: -0.125 pp (-0.875 to +0.625) and -0.2 pp (-1.175 to +0.75). No win-rate improvement; do not extend unchanged or adopt by default.
- Native deterministic-card head versus sampled head 49.925%, versus initialization 49.785% (CI 49.56-50.0000625); residual versus sampled residual 49.86%, versus initialization 49.7025% (49.435-49.9675); residual/head 49.9375%. Keep the unrounded head/initialization bound above 50%. Point gains do not replace the win-rate objective.
- Head endpoint SHA 0faf75b49c4709245e86734bae0a9f3dd5c2af9a6b2c6a4d7741fb2a3c8a9798: 1,122,373 new paper8 matches/8,803,468 rounds/11,387,045 koi decisions. Residual SHA 20847e7608b394474fefe1591388ecae6e6af87b158ba0fc44d284959500f871: 1,120,934 matches/8,786,985 rounds/11,401,936 koi decisions. Each adds 11,141,428 initialization matches separately. All card weights unchanged, no frozen Adam state,96,000 steps on optimized tensors. Actual hashes, config/counters/episode sums/finite tensors and all statistics independently verified in runs/koi_detcards_training_and_duels_audit.json and koi_detcards_final_statistics_audit.json; full runner audit koi_detcards_endpoints_audit.json. No research12 contribution.
- Earlier sampled-card adaptation public 1,000-pair results: full control 47.125%, head 49.125%, residual 49.25%, versus initialization 49.375%. Neither specialization improved win rate. Audits koi_screen_1000_audit.json and koi_parent_duels_audit.json. Do not extend those runs unchanged.
- Koi-only search full 1,000 pairs on original MLP seed 10,001: 50.025% (CI 48.224375-51.825), no-search 49.375%; paired +0.65 pp (+0.125 to +1.2). Replication on seed 1: 48.375% (46.599375-50.2), control 47.825%; paired +0.55 pp (0 to +1.125). Direction agrees but replication interval includes zero; no public superiority or default adoption. Same evaluation deals are not 2,000 independent pairs. Per-model reports/audits in their original run directories.
- Search truncation diagnostic: six deliberately stratified unsolved koi roots across two models, rounds 0/3/6. All 128 worlds x2 actions per root terminate within 192 steps; horizon 192/512 means, confidence margins and selected actions identical. This small diagnostic gives no reason to lengthen the horizon unchanged; it is not a population estimate or full-match guarantee. runs/search_truncation_diagnostic.json; all 1,212,416 transitions excluded from training.

## Verified implementation

Independent Python rules oracle and shared allocation-free C++/CUDA engine; GPU PPO with signed complete-match returns; exact restart/provenance and initialization lineage; pinned published adapter; native paired evaluation; optional beliefs, search, Q-boosting, fixed opponents and separate critics. Actor and rollout inputs contain only own/public observations. No recent rule changes.

Optional policy_scope=koi freezes card/shared/belief tensors and learns koi/value heads plus optional zero-initialized residual. Policy losses/entropy/KL/advantage normalization use koi rows; values/GAE cover all phases. deterministic_cards=true additionally makes fixed card decisions legal argmax while koi remains sampled. Requires pure self-play match PPO, observation critic and no belief loss. Defaults preserve earlier behavior; neither option has established strength improvement.

Full 144 tests passed without skips; two expected upstream Transformer warnings; Ruff passed. Exact default behavior against actual old-source MLP/FP32 and relational/BF16 results, CPU/CUDA restart, frozen parameters and production-shape eager/captured equality verified. Artifacts koi_detcards_default_equivalence.json and koi_detcards_production_equivalence.json. Smoke/diagnostic/search work is excluded from training budgets.

Hidden proposals now share public preprocessing once per root, retaining the exact sequential RNG/rejection/shuffle order and no cross-root cache. Actual old-source comparison: 6,144 identical worlds across both profiles/uniform/weighted modes, including 36 rejection draws; production128/horizon192 full-action root comparisons preserve actions, simulator states, proposal RNG and counters. Audit runs/batched_proposal_equivalence.json. Proposal-only timing improved approximately fivefold; whole-search timing is workload-dependent and does not inherit that ratio. New package SHA20e9d7c361d17f37a1fccca269ea01dcc34ad3a49b60c057f60a0ff1d2bc76d1. Full-suite session64198 completed; no training or rule changes. The completed 100-pair evaluator used the pre-optimization package; the next extension uses this verified optimization.

## Completed 100M scale requirement

Unchanged scale training ended successfully at update 69,267: 100,001,204 new complete research12 matches,1,200,025,665 rounds and18,157,928,448 decisions. Previous update 99,999,773 was below target. Initialization lineage 101,001,668 research12; 11,141,556 paper8 pretraining separate. Do not resume this completed scale run unchanged.

Immutable runs/ppo12_scale_seed30001_captured/final_100m.pt SHA 613ea0d992a54a0a18ab1848f106abe074c5d74640d76c3a86a131448aa29cce exactly equals stable final latest.pt. Actual twelve-round/no-bankruptcy settings, episode sums,12*episodes+partial rounds (11,217), decision arithmetic, finite model/Adam and final metrics checked. Original source 142c265 retained. Reports budget_100m.json, audit_100m.json and audit_100m_independent.json.

Final versus transfer initialization: 51.8125% (CI 51.225-52.3825), points+1.3454,10,000 guarded development pairs/20,000 matches/240,000 rounds. Versus update 52,000: -0.38 pp (CI -1.1200625 to +0.3675), points -0.5739 (CI -1.10863 to -0.0258975). Parent advantage remains; further training gain unproven. All report hashes, means/intervals and paired differences recomputed. This is neither public paper8 superiority nor final testing.

## Other rejected or inconclusive directions

Do not extend unchanged: relational continuation, lambda 0.99, entropy 0.005,20% fixed historical/independent-MLP opponents, separate observation/privileged critics, sampled-card or deterministic-card koi adaptation. Wider networks, explicit features, sampled deployment, Q-boosting, belief/earlier search and on-policy endgame training have no established public advantage. Full evidence is in EXPERIMENTS.md.

Published-koi substitution improves original MLPs on the first 1,000 pairs (+3/+1.8 pp), but additional 3,000-pair results 49.9417%/50.6833% do not establish superiority. It is an explicit external dependency, not strength attributable solely to our actor. The relational substitution diagnostic is also inconclusive.

Keep the goal active until public superiority, independent selected-method validation, untouched final evaluation and manuscript are complete. Use .venv/Scripts/python.exe; explicit UTF-8 reads, preserve line endings, new text UTF-8 without BOM/LF. Runs/external/local agent rules stay ignored. Commit coherent milestones; avoid repeated status-only commits.
