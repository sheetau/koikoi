# Current status

Goal remains active. New100M research12 training is complete; public assisted-RL-WP superiority, independent selected-method validation, untouched final testing and manuscript remain unfinished. Final seeds100,000,000+ are unused. Submission is a later decision.

## Live work: verify handles before acting

Full-action original-model extension completed successfully: runner session17341 and PIDs47200/45000 terminal. Combined original100+additional900 =1,000 development pairs. All report/settings/model hashes, original controls, segment and combined statistics, paired differences and counters independently recomputed by runs/audit_all_search192_extension.py. Evidence: runs/ppo8_seed10001_captured/all_search192_1000_independent_audit.json (SHA2e18f29dee800c1d3190a8bf4546373b75d1d7b753c884135a0dc234a57ef8ac).

Full-action51.15% (95%CI49.325-52.95), points+1.262 (+0.126925..+2.394025). Versus no-search49.375%: paired+1.775pp (+0.774375..+2.75), points+0.772 (+0.221..+1.327). Versus koi-only50.025%: +1.125pp (+0.075..+2.2), points+0.526 (-0.061025..+1.093025). Positive development improvement, not established absolute public superiority. Original100 selection remains part of this nested endpoint.

Running fixed protocol7376e5b: independent original seed1 model runs/ppo8_v3/update_00005000.pt, SHA d104ae00824aca114e6cd414c74d02670b9a8de5a009ad4fad557005f66233e5. Exactly1,000 development pairs seeds1,000,000-1,000,999 against assisted RL-WP, paper8/deterministic/endgame/CUDA/uniform128/horizon192/confidence2.5/full-action/capture/chunk32. Validated execution optimization only; no training, policy change, interim selection or final seeds. Compare existing eval_5000_cuda_control1000.json and eval_5000_koi_search192_captured1000.json. Primary replication endpoint is paired win-rate gain over no-search with95%CI lower bound>0; koi comparison and points secondary. Same deals across models are not extra independent pairs. An inconclusive endpoint is not automatically extended.

Only live experiment: session88215, actual runner PID63652 (parent14620), actual evaluator PID39016 (parent28516); commands and first20 pairs verified. Runner runs/run_all_search192_seed1.py; log runs/all_search192_seed1.log; manifest runs/all_search192_seed1_process.json. Outputs under runs/ppo8_v3: eval_5000_all_search192_chunk32_1000.json, all_search192_1000_summary.json. Verify actual handles before launching or acting; never duplicate or restart on observation timeout. After completion independently audit the endpoint before choosing final protocol. All previous pipelines and training are terminal; do not resume completed training unchanged.

## Verified implementation and latest change

Independent reference/shared C++/CUDA rules, GPU PPO, exact restart/lineage, published adapter and paired evaluation are implemented. Actor and rollout policies see only own/public information. paper8 and research12 remain separate profiles.

Latest implementation28d6397, pushed. Package SHA e9c9f0867a18add428ab62bfd4ccef18e7f389924424b498c00e42b3c6812e94. Optional --search-chunk-size N (default0) requires deterministic CUDA capture. Stops only after every original particle finishes; preserves full-horizon bootstrap otherwise. Logical, actual executed and preparation transitions are separate and excluded from training. Post-terminal scratch states can differ; next root overwrites them completely. Current fixed evaluation does not use this option.

Full173 tests passed without skips (two expected upstream warnings); afterward29 chunk tests passed with strengthened opponent-seat/cache-reuse assertions. Ruff passed. Actual old source/default/chunk comparisons:60 searches across both original MLPs and final100M model preserve values/actions/proposal+torch RNG; default also preserves scratch state. Existing public development pair replay matches both prior outcomes. Evidence: `runs/chunked_search_production_equivalence.json`, `runs/chunked_search_replay_audit.json`. Earlier proposal batching preserves6,144 sampled worlds/RNG: `runs/batched_proposal_equivalence.json`. Timing diagnostics are concurrent, not deployment benchmarks or strength evidence.

## Completed evidence and decisions

- Scale gate:100,001,204 new complete research12 matches at update69,267;1,200,025,665 rounds;18,157,928,448 decisions. Immutable `runs/ppo12_scale_seed30001_captured/final_100m.pt`, SHA613ea0d992a54a0a18ab1848f106abe074c5d74640d76c3a86a131448aa29cce. Actual state, budgets, hashes and statistics independently checked in `audit_100m_independent.json`. Versus transfer parent51.8125% (CI51.225-52.3825); no demonstrated further win gain over52k. Not public superiority; do not resume unchanged.
- Koi-only search1,000 pairs: seed10001 paired+0.65pp (+0.125..+1.2); seed1+0.55pp (0..+1.125). No established absolute public superiority. Seed1 checkpoint `runs/ppo8_v3/update_00005000.pt`, SHA d104ae00824aca114e6cd414c74d02670b9a8de5a009ad4fad557005f66233e5. Its CPU control alias `eval_5000_wp.json` exactly matches historical `eval_5000_guarded.json`.
- Do not extend unchanged: sampled/deterministic-card koi adaptation, relational continuation, lambda0.99, entropy0.005, fixed historical/independent opponents, separate critics. Wider/features/sampled deployment/Q-boosting/belief/earlier search/endgame training have no established public advantage. Published-koi substitution remains inconclusive and an explicit external dependency.

Details are retained in [EXPERIMENTS.md](EXPERIMENTS.md): latest sections cover full-action100+900 and terminal-aware search; earlier sections contain complete metrics, hashes and rejected experiments. Consult [DESIGN.md](DESIGN.md) for architecture changes, [RULES.md](RULES.md) for game logic, [COMPARATORS.md](COMPARATORS.md) for pinned public alternatives and rule incompatibility.

Use .venv/Scripts/python.exe; UTF-8 reads, preserve existing endings, new text UTF-8 without BOM/LF. Keep this handoff compact; record coherent milestones, not every progress poll. Runs/external/local agent rules stay ignored.
