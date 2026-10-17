# Current status

The previous research phase completed its measured deliverables: GPU-native implementation,100M research12 computation, two-model final evaluation, isolated latency, final-results manuscript and verified local evidence archive. Both selected models beat pinned assisted RL-WP on the fixed primary criterion. The separate RL-Point criterion does NOT replicate; both models lose expected points to RL-Point. This is a comparator/rule-specific result, not proof of universal strongest play. Submission and external artifact publication remain later decisions.

Read this snapshot first. Detailed measurements, negative studies and historical commands are in [EXPERIMENTS.md](EXPERIMENTS.md). The prior full STATUS history remains available at [commit26aeb37](https://github.com/sheetau/koikoi/blob/26aeb37/docs/STATUS.md). Update this snapshot at milestones; append experimental details to EXPERIMENTS instead of duplicating the full history here.

## Active extension: stronger play and revised manuscript

The user requested stronger play, clearer research contributions and a substantial manuscript revision. Initial dealer selection and the playable canonical model remain unchanged. The previous final evaluation/archive below are completed historical results; the new work is ongoing.

**Arbitration PPO study complete:** session75964 exited0; all six fixed8,519,680-decision training arms and22 evaluations passed the whole-study audit (`runs/arbitration_study/analysis.json`, SHA256 `66f125fe8a2b9245e6cf91e20cbd6121516d76c1f7186846cd1a694ea42aabb2`). Total training51,118,080 decisions /3,425,571 round endings /439,406 matches. Learned selection, phase-only selection and continued PPO each fail the replicated win-gain criterion over their parents against both public opponents. Both phase-only selectors always choose the parent. The full two-parent paired-change figure was generated from audited raw outcomes and visually checked: `output/figures/arbitration_changes.{png,svg,pdf,json}`. No deployment promotion. Details: [EXPERIMENTS.md](EXPERIMENTS.md).

**Paired-outcome regression study evaluating:** session28640, `runs/paired_arbitration_study`. Both65,536-root datasets and all four fixed20-epoch fits are complete and individually audited.The first full selector completed all three1,000-pair matchups:51.050% [49.375,52.700] versus WP,49.650% [48.100,51.300] versus Point,49.450% [47.850,51.075] versus its parent. All raw statistics checked; remaining controls/replication in the24-matchup study are still running. No deployment promotion. Held-out one-action gain over the parent is+0.233pp [-0.094,0.562] for the first full selector and+0.460pp [0.090,0.844] for the second. Full/no_match give similar results; both full selectors have slightly worse MSE than zero prediction. These are single-action diagnostics under frozen continuation, not repeated deployment strength. Both datasets have about9.6% branch win-outcome disagreement versus41.7% point-margin disagreement, motivating investigation of richer score targets after current comparisons finish. Details: [PAIRED_ROLLOUTS.md](PAIRED_ROLLOUTS.md).

**Mechanism diagnostics and static controls:** the first full selector chooses Point at188/523 expert disagreements in2,048 cached random-play states, but only8 disagreements respond to the tested explicit score/round input changes. This is descriptive sensitivity, not causal evidence of improved match decisions. The phase selector greedily chooses its parent expert in all four phases. One near-tie differs from the original deployed parent because freezing requires-grad changes floating-point inference by about1e-6: first-parent Point win outcomes are unchanged and mean points differ by0.002. Both learned controller arms share the same frozen path. `scripts/run_static_expert_controls.py` adds two fixed no-learning card/koi hybrids per parent on the reused22M development deals, running in exec session48159, currently waiting for paired regression to finish. Eight smoke matches and independent report/tampering checks pass; no strength claim. Diagnostics and scripts do not modify bound production or active workers.

**Unlaunched follow-up prototype:** `scripts.train_score_distribution` predicts remaining score distributions and decodes win probability at the current point lead. Shared-encoder expected-point and direct-win controls are implemented. Tail compression preserves win/draw/loss for active paper8 roots. It reuses paired data and adds no simulation; no full-scale fit or strength experiment has been launched. Current paired-regression/static results will determine the next experiment. Details: [PAIRED_ROLLOUTS.md](PAIRED_ROLLOUTS.md).

**Frozen code:** production SHA256 `2b718198a4f7bc640f457c9fc12311dcb9a8689fba1ffcb09411234ba0920cdb`. Do not modify production modules or either study's bound runner/helper scripts while active/queued. Automatic analysis watcher `python -m scripts.watch_extension_audits` is live in exec session98461, directory `runs/extension_audits`. It runs the three existing study analyzers after each study completes and adds both held-out paired-target diagnostics. Do not duplicate these outputs while the watcher is live. Its bound analyzer files must remain unchanged until completion. Tests include CPU/CUDA reference trajectories, hidden-state invariance, exact PPO restart, frozen experts, actual-parent initial actions, whole-match validation splits and a four-match evaluator/audit round trip.47 arbitration/published tests and8 additional pipeline/analysis tests passed; the expanded paired audit pipeline was rerun and passed. These are correctness checks, not strength evidence.

**Manuscript revision ready for further results:**13-page revision includes16 references, expanded Related Work, direct Abstract/Introduction, completed arbitration outcomes with a paired-change figure, and the paired-supervision method under evaluation. PDF SHA256 `16ac69ab47349fe83e2d809db921e064bc18151cc8e7a3a29244ac730ede90db`. `runs/manuscript_arbitration_results_qa.json` passes visual review,18 embedded fonts,16 resolved citations, TeX/bounds checks and all26 historical plus34 extension input hashes. The extension ledger is separate from the preserved historical evidence. Stronger verified play and final manuscript remain unfinished; paired/static matchup outcomes are pending. Prior archived paper and QA remain unchanged.

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
