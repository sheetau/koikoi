# Current status

Goal ACTIVE: build the strongest Koi-Koi AI supported by measured evidence. The new100M research12 target, final evaluation, isolated latency and final-results manuscript are complete and audited. Both selected models beat pinned assisted RL-WP on the fixed primary criterion. The separate RL-Point criterion does NOT replicate; both models lose expected points to RL-Point. Local evidence archive and full original-plan completion audit remain unfinished. Submission is a later decision.

Read this snapshot first. Detailed measurements, negative studies and historical commands are in [EXPERIMENTS.md](EXPERIMENTS.md). The prior full STATUS history remains available at [commit26aeb37](https://github.com/sheetau/koikoi/blob/26aeb37/docs/STATUS.md). Update this snapshot at milestones; append experimental details to EXPERIMENTS instead of duplicating the full history here.

## Completed final evaluation

- Original session17151 exited0; actual runner45028 and workers35564/64780/45896/45888 are gone. All4 jobs completed10000 duplicate pairs. Never restart these original directories.
- Directory `runs/final_published_test`; process status `complete_awaiting_audit` is retained by design. Full `independent_audit.json` passes; its SHA256 is `891c765dec23edec33538174433d5105dabc443ca2da1a4f91d4273c965eade8`.
- Summary SHA256 `49a2babbb4e84043ef7101eff683ec7071094261dc2914ee99cb1e51dd7c692f`. Audit independently recomputed raw pair statistics and checked frozen identities/accounting; it does not replay all trajectories/inferences.
- Both model seeds x assisted RL-WP/RL-Point, strict CPU2 threads each, paper8, greedy, endgame assistance, no search. Total80000 matches/9281432 decisions on10000 shared unique deal streams, NOT40000 independent deals.
- Final seeds100000000-100009999 are consumed. No outcome-based retraining, reranking, extension or early stopping occurred. Canonical model remains seed6710001_warm.

| Model seed | Opponent | Win score,95% CI | Point difference,95% CI |
| --- | --- | --- | --- |
| 6710001 | RL-WP | 51.4800% [50.9700,51.9750] | 2.4203 [2.122395,2.7133025] |
| 6720001 | RL-WP | 51.4450% [50.9449375,51.9525] | 2.0221 [1.715885,2.317115] |
| 6710001 | RL-Point | 50.4925% [49.9549375,51.0400625] | -1.8918 [-2.2427125,-1.53859] |
| 6720001 | RL-Point | 51.2875% [50.7200,51.8525] | -1.9603 [-2.3197125,-1.6088825] |

Primary both-model RL-WP win criterion **passes**. Separate both-model RL-Point criterion **fails**; canonical CI includes50%. Do not describe a replicated win against both opponents or superiority in expected points to RL-Point. No universal strongest/human/exploitability claim follows.

Frozen protocol SHA256: `4a69cff95079e2bd4cfcda84241f4096149c05cfe408bd11cbfde154eaa060e4`.
Production source SHA256: `4b4360ed783c1d12cb5258e255f4b85f2e30dfa2e111fd8a078cf3439ce612cb`.
Preserve original production/models, protocols, runner/auditor and bound study/comparator identities for reproducibility.

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

## Latency, manuscript and archive

**Latency COMPLETE:** original session77792 exited0 after final workers were gone; runner29440. `runs/final_latency` has protocol/process/report/raw/independent_audit. All41184 measured calls,10296 warmup,5148 validation on1287 unique cached development states completed; no new matches. Protocol SHA256 `ddec2df361cc2e206ab4759ac6f8a477fbf712003ff56dfac8f8887bfdc4083f`; report `24ec5e4d5574e48a92d590f3f757ef701bfad5cd2daf7cf072dfa8a284f84531`; audit `e3b48a44568597f3cc528396c3a700407054137af4c540e2983a3486434f34cc`; raw `d0b4454a18135c3cbfddf6d4d385d2bf657983f57667253d739fbadaea17fc94`. Mixed mean/median/p95/p99 milliseconds: canonical1.585/1.479/2.036/2.302, second1.595/1.504/2.034/2.286, WP1.974/1.911/2.714/2.925, Point1.926/1.834/2.688/2.901. Exact deployment agent.act path, CPU2 threads. Group overlap/repeated calls are descriptive; project absence checked at pass boundaries, OS noise uncontrolled. Audit checks raw accounting/statistics, not an independent clock/inference rerun. Do not rerun into the completed directory.

**Manuscript COMPLETE for final results:** 9-page English manuscript `docs/paper/manuscript.tex`, final and latency tables generated from26 bound artifacts. PDF `output/pdf/manuscript.pdf` SHA256 `506319f6faaa74bd669e80f0e173d1597eea6924e004893ec03640d6b4413177`; source `9cd591877efe88ffe1bf66bc3bbce606fedfc2af27d1c979d89f78cb1a6fa71f`; evidence `104066c182779b3911fa511269b1dab0f7336e9d02a75e37f261c02c2b22fcb5`. Final QA `runs/manuscript_final_qa.json` passes, pending_final_results=false, SHA256 `a63cee06d64828856ab4cff265cb75b954bb78515ad4a6bef61bda82b2e3e0d5`. All9 latest rendered pages reviewed,14 embedded fonts, no TeX boxes/reference/glyph/bounds defects, citation keys and audited numbers checked. Latest edited pages4/5/9 reinspected; other6 PNGs unchanged from reviewed version. Author/venue and submission are later decisions. [Build notes](paper/README.rst).

**Evidence archive:** `scripts/package_research_artifacts.py` implements --plan/--build/--verify; [scope and instructions](ARTIFACTS.rst). All final inputs now exist, including final manuscript QA. No archive built/uploaded yet. Portable NumPy verifier checks bytes/paths/source/model bindings and independently recomputes final paired statistics. Prepared20 artifact tests passed; actual real archive build/verification is the next action after committing this milestone. This is a selected evidence archive, not full replay of every historical experiment.

## Next actions, in order

1. Commit the final-results/latency/manuscript milestone, then `.venv/Scripts/python.exe -m scripts.package_research_artifacts --build`. Inspect actual archive verification/receipt/SHA; no archive exists yet. Record/push milestone. Keep final PDF/source/evidence hashes matching its final QA.
2. Audit the original plan's full requirements against current implementation, reports, tests and artifacts before any goal-complete claim. Primary superiority passed; retain the secondary failure and all stated limitations. Do not reselect models or relabel consumed final deals.
3. Any genuinely new strength research needs new development/fresh future confirmation. Submission or external artifact publication remains a later decision.

## Operating/verification reminders

Use `.venv/Scripts/python.exe` (Python3.12/Torch2.12+cu130/NumPy2.5.4), not system Python3.14. Read PowerShell text explicitly UTF8; preserve existing endings, new text UTF8-noBOM/LF. CPU Ryzen9900X, GPU RTX5070Ti16GB. No subagents. Goal remains active during verified waits.

Production is unchanged since the327-test full pass (no skips). The75 final/latency/archive tests pass (43+12+20), with Ruff/diff checks; this is not a new full-suite run. Do not repeat broad tests merely to fill waiting time. Audit scope is identities/accounting/raw statistics, not all-trajectory or all-inference replay. Keep matches, rounds, decisions and search work separate.
