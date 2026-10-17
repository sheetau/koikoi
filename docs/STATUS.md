# Current status

Goal ACTIVE: build the strongest Koi-Koi AI supported by the measured evidence. The new100M research12 computation target is complete. Both selected warm-PPO models qualify on development, but final superiority is NOT established yet. Final evaluation is RUNNING; latency, final manuscript and evidence archive remain unfinished. Submission is a later decision.

Read this snapshot first. Detailed measurements, negative studies and historical commands are in [EXPERIMENTS.md](EXPERIMENTS.md). The prior full STATUS history remains available at [commit26aeb37](https://github.com/sheetau/koikoi/blob/26aeb37/docs/STATUS.md). Update this snapshot at milestones; append experimental details to EXPERIMENTS instead of duplicating the full history here.

## Live final evaluation: preserve the original processes

- Unified session **17151**, actual runner **45028** (shim61876).
- Final directory: `runs/final_published_test`; `process.json` jobs are keyed by name.
- Actual workers/shims: first warm WP35564/21716; second warm WP64780/58100; first warm Point45896/40536; second warm Point45888/16824.
- All5 actual processes confirmed live. Last observed completed pairs: first WP3330, second WP3190, first Point3160, second Point3270, each targeting10000. These are progress counts, not outcome conclusions.
- Four jobs: both selected training seeds6710001/6720001 x assisted RL-WP/RL-Point, strict CPU2 threads each, paper8, deterministic, endgame assistance, no search.
- Each job10000 duplicate pairs; total80000 played matches on10000 shared unique deal streams, NOT40000 independent deals.
- Final seeds **100000000-100009999 are now consumed/in use**. Never describe them as unused or silently rerun the jobs.
- Primary: BOTH model seeds' WP win-score95% CI lower bounds must exceed0.5. Point comparator is secondary and separate; point means cannot replace wins. No reranking, training, extension or early strength stopping on final outcomes.
- Observation timeout is NOT termination. Poll session17151 or inspect these actual PIDs before any recovery action. Inspect original artifacts on failure; do not launch duplicate workers.

Frozen final protocol SHA256: `4a69cff95079e2bd4cfcda84241f4096149c05cfe408bd11cbfde154eaa060e4`.
Production source SHA256: `4b4360ed783c1d12cb5258e255f4b85f2e30dfa2e111fd8a078cf3439ce612cb`.
Do not change production, final runner/auditor, bound development/comparator files or selected models through final execution/audit. `scripts.run_final_published_test.validate_frozen()` checks the bindings. Full final independent audit does not exist yet.

## Completed development and selected models

`runs/cold_ppo_screen` is terminal; session85269 exited0, former runner57300/final evaluator48436 absent. All9 jobs completed and full independent audit passed. Audit SHA256: `6b8548605b50226738227ce83dba1a2e946efd282461b0fa8d48132a42cfe9ba`.

| Training seed | Warm win score,95% CI | Cold win score,95% CI | Cold minus warm,pp,95% CI |
| --- | --- | --- | --- |
| 6710001 | 51.805% [51.090,52.525] | 50.755% [50.134875,51.395] | -1.050 [-1.875,-0.200] |
| 6720001 | 51.500% [50.764875,52.245] | 50.565% [49.930,51.215] | -0.935 [-1.810,-0.075] |

All rows share5000 development pairs7,200,000-7,204,999. Initial policy gives exactly0.5/zero points on every pair. Earlier warm1000-pair screen was inconclusive; earlier failed search/training experiments remain disclosed. Development selection is not untouched confirmation.

Fixed rule selected **warm**, canonical **seed6710001_warm** (never the better-scoring seed). Warm T0.05/actor LR3e-6; cold T0.01/LR6e-7, so no isolated temperature causal claim. Every model20 critic-only+500 policy updates,8519680 decisions. All4 training totals293897 matches/2287795 round endings/34078720 decisions; warmup included, events may cross the phase boundary.

Selected checkpoints under `runs/cold_ppo_screen/<case>/update_00000520.pt`:

- `seed6710001_warm`: `37d06849261e0c1f0c31a9d2f228388fa1f391b670ce6772093a1c276091cc36`.
- `seed6720001_warm`: `5e8a123d5fafb6ebae8c1ac8d094c9de65bf364381cc10b78208337051e181cc`.

Pinned public comparator revision: `89d789565c7c36c84b74662e43001d960bf80901`. Final protocol binds17 Python files and6 weights from its clean checkout. [COMPARATORS.md](COMPARATORS.md) explains compatibility and claim limits. Public superiority would be against these policies/rules, not every implementation or humans.

## Separate computation target

`runs/ppo12_scale_seed30001_captured/final_100m.pt`, SHA256 `613ea0d992a54a0a18ab1848f106abe074c5d74640d76c3a86a131448aa29cce`, update69267:100001204 NEW research12 matches/1200025665 rounds/18157928448 decisions. `audit_100m_independent.json` passes stored accounting/statistics. This is separate from paper8 selected-model training and cannot be attributed to those pretrained actors. Do not restart the completed scale run.

## Prepared follow-up work (not completed measurements)

**Latency:** `scripts/benchmark_final_latency.py` and independent `scripts/audit_final_latency.py`; frozen `runs/final_latency/protocol.json` SHA256 `ddec2df361cc2e206ab4759ac6f8a477fbf712003ff56dfac8f8887bfdc4083f`. Only protocol exists; no timing run yet. Both models+both opponents, CPU2 threads,1287 cached development states,2 warmup/8 measured passes,41184 timed calls. Mixed1024,64 unguarded per phase,32 guarded groups overlap. Exact final agent.act wrappers/features/inference timed; loading/transitions/checks excluded. No new matches. Actual live-study gate and independent sampling passed; preflight SHA256 `6408908455edec14142eefa15f0a41d4e8b0626f500fb4cc3ce09f8a43181d96`. Keep prepared script/auditor/protocol identities fixed. Project-worker absence checked at pass boundaries; OS noise remains uncontrolled.

**Manuscript:** 8-page English DRAFT in `docs/paper/manuscript.tex`, bibliography and generated evidence; [build notes](paper/README.rst). PDF `output/pdf/manuscript.pdf` SHA256 `adafc7993395263739e734c12ff67dd677126c2399d937150c114b98c425d5d7`. All8 rendered pages reviewed, fonts embedded, tables/counts/citations checked; draft QA SHA256 `5593932840d7d1265b35655ae57f505cd0ef516a9687ac32ada9f894bcf10e6b`. Final outcomes excluded. Portable Tectonic0.17.0 and PyMuPDF1.28.2 are in ignored external/. Do not treat draft as submission-ready.

**Evidence archive:** `scripts/package_research_artifacts.py` implements --plan/--build/--verify; [scope and instructions](ARTIFACTS.rst). Read-only plan371 files/~422MB existing data;11 unfinished inputs missing at last check. No archive built/uploaded. Portable NumPy verifier checks bytes/paths/source/model bindings and independently recomputes final paired statistics; actual development4-endpoint recomputation passes. Full real archive build remains untested pending final inputs. Preflight SHA256 `ae42e6599351e810b913bf4c5263aa080bc5a7be93c63be3b6119bec4e925de1`.

## Next actions, in order

1. Wait on the SAME final session/processes until all4 jobs and runner exit successfully. Then run `.venv/Scripts/python.exe -m scripts.audit_final_published_test`. Inspect complete results and fixed primary/secondary criteria. Preserve failure; do not extend/reselect using these deals.
2. Once final audit passes and original project workers are gone, run `.venv/Scripts/python.exe -m scripts.benchmark_final_latency --run`, then `.venv/Scripts/python.exe -m scripts.audit_final_latency`. No concurrent training/evaluation during measurement.
3. Run `.venv/Scripts/python.exe -m scripts.export_paper_evidence`; revise abstract/results/conclusion/status with audited final results and timing scope. Compile/render/check final PDF. Create `runs/manuscript_final_qa.json` with status=passed,pending_final_results=false and current manuscript/evidence/PDF hashes. Draft QA does not satisfy this gate.
4. Commit coherent final source/manuscript state, then `.venv/Scripts/python.exe -m scripts.package_research_artifacts --build`; independently inspect returned final statistics/conclusions and receipt/archive SHA. Follow the selected-archive limits in ARTIFACTS.rst. Record and push milestones.
5. Audit the original plan's full requirements before any goal-complete claim. If primary superiority fails, retain that result and define genuinely new research with fresh future confirmation; do not relabel existing results as success. Submission requires a later decision.

## Operating/verification reminders

Use `.venv/Scripts/python.exe` (Python3.12/Torch2.12+cu130/NumPy2.5.4), not system Python3.14. Read PowerShell text explicitly UTF8; preserve existing endings, new text UTF8-noBOM/LF. CPU Ryzen9900X, GPU RTX5070Ti16GB. No subagents. Goal remains active during verified waits.

Production is unchanged since the327-test full pass (no skips). The75 final/latency/archive tests pass (43+12+20), with Ruff/diff checks; this is not a new full-suite run. Do not repeat broad tests merely to fill waiting time. Audit scope is identities/accounting/raw statistics, not all-trajectory or all-inference replay. Keep matches, rounds, decisions and search work separate.
