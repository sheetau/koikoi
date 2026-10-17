# Current status

The previous research phase completed its measured deliverables: GPU-native implementation,100M research12 computation, two-model final evaluation, isolated latency, final-results manuscript and verified local evidence archive. Both selected models beat pinned assisted RL-WP on the fixed primary criterion. The separate RL-Point criterion does NOT replicate; both models lose expected points to RL-Point. This is a comparator/rule-specific result, not proof of universal strongest play. Submission and external artifact publication remain later decisions.

Read this snapshot first. Detailed measurements, negative studies and historical commands are in [EXPERIMENTS.md](EXPERIMENTS.md). The prior full STATUS history remains available at [commit26aeb37](https://github.com/sheetau/koikoi/blob/26aeb37/docs/STATUS.md). Update this snapshot at milestones; append experimental details to EXPERIMENTS instead of duplicating the full history here.

## Active extension: public expert arbitration

The user requested stronger play, a clearer research contribution and a substantial manuscript revision. Initial dealer selection remains unchanged. The previous final evaluation and archive below are completed historical results; this extension has not yet established a new strength claim.

Implemented a public-observation controller over frozen adapted-win and original-point expert proposals, with duplicate actions collapsed. Full-context and phase-only controllers will be compared with ordinary continued PPO at both existing parent checkpoints, using identical additional decision budgets and fresh development deals. See [ARBITRATION.md](ARBITRATION.md). Source/parent-bound machine protocol is written before the study starts. The playable canonical model remains unchanged.

Validation:47 tests passed across arbitration, published training and published PPO, including CPU/CUDA trajectory equivalence, hidden-state invariance, duplicate-action zero gradients, frozen experts, critic warmup and exact resumed updates. Feasibility only: five updates/81,920 decisions at seed18990001,653 complete matches,7.35 seconds for updates; no strength evaluation. Ruff passes. Study command: `.venv/Scripts/python.exe -m scripts.run_arbitration_study` (fresh `runs/arbitration_study`). Manuscript restructuring and primary-source Related Work expansion are in progress.

The six-arm study is running in exec session75964, with one GPU trainer and up to three CPU evaluators. Production source SHA256 is `2b718198a4f7bc640f457c9fc12311dcb9a8689fba1ffcb09411234ba0920cdb`; do not modify production modules or the frozen runner while this study is active. Analysis is implemented separately in `scripts/analyze_arbitration_study.py` and requires all jobs complete. Four additional tests pass, including actual-parent initial greedy equivalence and corrupted-result rejection.

Manuscript revision:12 references, substantial Related Work, direct Abstract/Introduction, shorter interpretation and reproduction sections. Current11-page PDF SHA256 `4df2ef588e1b88375707d33a73c1b99df1ec686dea6137cc433d46cf8673539f`; current QA `runs/manuscript_arbitration_revision_qa.json` passes all11 visually reviewed pages,14 embedded fonts, citation/box/bounds checks and all26 retained evidence hashes. The new arbitration section explicitly says results are pending. The prior9-page PDF and its historical QA remain preserved in the earlier archive.

A second hypothesis has a tested training-target prototype: same-world paired complete-match outcomes at public expert disagreements. CPU/CUDA reference trajectories, signs, duplicate branches and grouping tests pass (3 tests). CPU feasibility generated64 roots/128 branch completions/15,018 branch decisions; it is not a policy-strength result. See [PAIRED_ROLLOUTS.md](PAIRED_ROLLOUTS.md). Direct outcome-difference regression and embedded deployment are now implemented in `scripts/train_paired_arbitration.py`. The8-test combined pipeline/branch/study suite passes, including a four-match evaluation/statistics round trip. The queued runner `scripts/run_paired_arbitration_study.py` freezes65,536 roots per parent,20 epochs, full/no_match ablation and24 fresh1,000-pair matchups. It waits for the earlier training jobs before GPU work, and for all earlier evaluations before CPU work. No new strength result is available yet.

Execution milestone: parent1_full finished520 updates/8,519,680 decisions/73,390 source matches. Its checkpoint SHA256 is `3eead4c639dc49f66247c51a31b8fb6e04441f32bf7b5ca01e70db8d258da46f`; independent training-record checks pass. Its fixed endpoint is under CPU evaluation while parent1_phase trains. No new endpoint win result has been selected or claimed.

The paired-regression study is queued in exec session28640 (`runs/paired_arbitration_study/process.json`: `waiting_for_previous_training`). Its bound protocol and runner are immutable while queued. Session75964 remains the preceding PPO study. Analysis entry points are `python -m scripts.analyze_arbitration_study` and `python -m scripts.analyze_paired_arbitration_study`; each requires its whole study complete. The paired analyzer checks source-match splits, targets/counters, frozen experts, CPU-recomputed validation MSE and all raw paired comparisons. Its tiny pipeline test passes. Keep both goals active: stronger verified play and the final evidence-based manuscript revision are still unfinished.

## Local human-play delivery

Added `python -m koikoi.play` and the Windows double-click launcher `play.cmd`. The Tkinter desktop client defaults to the hash-checked canonical seed6710001_warm checkpoint with greedy CPU2 inference, public endgame assistance and no search. Original assisted RL-WP/RL-Point and greedy practice opponents are selectable. The UI provides clickable card images, capture choices, koi/stop, public scores/captures, Japanese help, round-result pauses and new matches. It hides opponent hand identities, writes no match records and does not train or alter models. Card images are read from the unchanged pinned upstream checkout; no PySimpleGUI dependency or network service was added.

Validation: `python -m pytest tests/test_play.py tests/test_engine.py tests/test_endgame.py tests/test_upstream.py tests/test_published_ppo.py -q` passed45 with no skips (17 expected upstream Transformer warnings). The8 new tests include20 full controller/reference matches, both capture phases and koi continuation, wrong-turn/illegal-action rejection, hidden-hand/stock view invariance, Tk stop/result/restart/callback cancellation, dense-field layout at1000x750, a full canonical-model match and both original opponents. After final UI adjustments the8 UI/controller/model tests were rerun successfully. Ruff and diff checks pass. Actual Windows launch and card-click/AI-response paths were inspected through Computer Use. These are playability checks, not new strength estimates. Frozen research artifacts and original engine/inference files remain unchanged. README contains Japanese launch/operation/restoration instructions.

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

**Evidence archive COMPLETE:** `dist/koikoi_research_artifacts.zip`,315291737 bytes,372 files/428853553 uncompressed bytes. SHA256 `2ddce4d2033006e332f45d19ed7313f09eff9d11f7e9a625cbd4be4d7ff1f352`; source snapshot `f5cf955ac8f65c935a9f54c13bd0e8f25b6c39de`. Original build session82391 exited0. Portable bytes/member/source/model checks and independent final statistics pass: primaryWP=true, secondaryPoint=false. Receipt `runs/research_artifact_bundle.json` records the actual archive. Independent PowerShell SHA agrees; archived README contains current model instructions and archived PDF matches final QA. Later metadata-only commits do not change this snapshot. Prior ef554ba ZIP and failed path-order ZIP are retained under distinct names. The verifier uses original-platform pathlib ordering; Windows/POSIX regressions cover the discovered ordering bug. [Scope/instructions](ARTIFACTS.rst). Nothing uploaded externally.

## Requirement audit and delivery

- GPU-parallel C++/CUDA core: inspected native/core.h, kernels.cu and BatchEnv dispatch; current full tests exercise reference/CPU/CUDA equality, public observations, masks and signed returns.
- Learning and named baseline: self-play pretrained PPO, two-seed warm/cold ablation, frozen10000-pair primary RL-WP success at both seeds, independent final audit. Secondary Point failure retained; no universal strongest claim.
- At least100M full twelve-round training matches: actual checkpoint config is CUDA/research12/2048 envs/captured rollout; stored counters and sum of2048 environment episode counters agree at100001204. Separate10000-pair/20000-match research12 evaluation against parent scores51.8125% [51.225,52.3825].
- Paper and reproducibility: final-results manuscript/PDF/QA complete, actual archive verified; no journal submission/external artifact publication. Author/venue choices are later decisions, as in DESIGN research gate6.
- Repository practice: root/tests AGENTS and plan/artifact ignores checked; DESIGN/RULES/STATUS/EXPERIMENTS and usage docs maintained. Source/research milestones and delivery metadata are committed separately; the delivered archive identifies its exact source snapshot. GitHub synchronization is checked at delivery.
- Optional belief/search/alternative-learning ideas were investigated with retained negative findings; they are not mandatory additions to the selected no-search policy.

All mandatory implementation/research deliverables above have current evidence. The named baseline requirement is met by the predeclared replicated RL-WP win test, not by claiming success against every opponent. Optional future work includes RL-Point robustness, other rule profiles and new method studies, each with new development/fresh confirmation. Do not reselect models, overwrite original artifacts or reuse consumed final deals for new confirmation.

## Operating/verification reminders

Use `.venv/Scripts/python.exe` (Python3.12/Torch2.12+cu130/NumPy2.5.4), not system Python3.14. Read PowerShell text explicitly UTF8; preserve existing endings, new text UTF8-noBOM/LF. CPU Ryzen9900X, GPU RTX5070Ti16GB. No subagents. All measured studies are terminal; never restart their original run directories.

Final release suite: **460 passed,0 skipped,0 failed,0 errors**,39 expected upstream Transformer warnings,107.92s. JUnit `runs/final_release_tests.xml` SHA256 `cf9925750fddcf9e5bd5da84282591d368e4de87b8d312bf0d6eeb0b11013a1f`. This includes the actual CPU/CUDA/upstream tests and new archive-order regression fixtures. Ruff/diff checks pass. No further broad testing without a new change/failure/concern. Audits cover identities/accounting/raw statistics, not all-trajectory or all-inference replay. Keep matches, rounds, decisions and search work separate.
