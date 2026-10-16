# Current status

Goal remains active. New100M research12 training is complete; public assisted-RL-WP superiority, independent selected-method validation, untouched final testing and manuscript remain unfinished. Final seeds100,000,000+ are unused. Submission is a later decision.

## Live work: verify handles before acting

Only experiment running: fixed full-action search extension. Session17341, actual runner PID47200 (parent65592), evaluator PID45000 (parent28136), `python -m runs.run_all_search192_extension`. Commands and advancing log verified. All training and earlier evaluation pipelines are terminal success. Never duplicate live work, restart because of a polling timeout, or resume completed training unchanged; STOP migration directories remain closed.

- Model: `runs/ppo8_seed10001_captured/update_00005000.pt`, SHA edc44213bc5c5c840297afc33ee64a57b91e9dee48c202535906fc916cde109d.
- Fixed additional900 development pairs, seeds1,000,100-1,000,999, assisted RL-WP/paper8/deterministic/endgame/CUDA/uniform128/horizon192/confidence2.5/--search-all/--search-cuda-graph. No interim selection, new training, chunk flag or final-test data.
- Primary endpoint retains original100 plus new900 =1,000 pairs. Compare corresponding existing no-search and koi-only100+900 controls. Validate actual checkpoint SHA, not slash conventions in paths.
- Log `runs/all_search192_new900.log`; manifest `runs/all_search192_extension_process.json`. Outputs under model directory: `eval_5000_all_search192_new900.json`, `all_search192_1000_summary.json`.
- Protocol b1f7012; live evaluator retains imported package20e9d7c3... despite subsequent source edits. Evaluation provenance was captured at start. A later aggregation's on-disk hash may be newer; do not attribute it to the live evaluator.

After completion: inspect actual reports/settings/hashes, independently recompute all endpoints and paired intervals, then choose the next strength experiment. A persistent positive full-action effect requires independent-model confirmation before untouched final testing. Keep original records; shared deals across models are not extra independent pairs. Original100 screen:49.25%, paired gain vs control+6pp (CI+2.25..+10), vs koi+4.25pp (+0.24375..+8.75); absolute superiority unproven and prefix reused after method selection.

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
