# Current status

The stronger-play and research extension is active. A new replicated match-win
improvement has not yet been established. Initial dealer selection and the
playable canonical model remain unchanged.

This file is a current snapshot. Detailed commands, failed approaches and work
accounting are in [EXPERIMENTS.md](EXPERIMENTS.md). The complete preceding status
history is preserved at [commit80a1cf1](https://github.com/sheetau/koikoi/blob/80a1cf1/docs/STATUS.md).
Old process handles mentioned in that history are historical, not live jobs.

## Current experiment and next actions

**Multi-round public-memory training complete;full evaluation running:**
producer46255(actual Python PID10804) completed allfour520-update arms and
entered the prespecified24,576-match strict evaluation. Three evaluation workers
were confirmed live as PIDs64032,65900,49308. Log runs/public_continuation_study.log.
Protocol95ce0a01a76c790f5dbdeb0c78028199e61219053167fc7309543cfa4c4815a1
and all221 inputs are unchanged. No partial win contrasts have been inspected.

Independent completion verification63848 exited0. Receipt
runs/public_continuation_training_completion.json SHA
ff847eac8babdfb625deb97c3e6db8f04a92fc535cd151fef7ee8b11df31bb5f
verifies253 inputs,allfour endpoints and both matched initial/warmup20 states.
It reuses unchanged independently verified reports for the firstthree endpoints
and reruns the fourth endpoint and second parent's matched warmup. All96 Adam
states per arm are finite and phase counts agree. Total18,982,723 decisions
(9,364,114 learner/9,618,609 opponent),266,240 conditional completions,
1,277,944 conditional round endings and0 full training matches. This verification
adds0 simulator decisions or neural predictions;it is not strength evidence.

Independent queue96354 waits for the whole producer output before replay/audit;
receipt runs/public_continuation_study_finish.json. Preserve it and the producer.
Renewal capability and the following renewal study remain queued behind that
audit. No new strength outcome or model promotion is established.

The frozen [multi-round protocol](PUBLIC_CONTINUATION_STUDY.rst) uses both
original parents, public/masked memory, all855/876 natural public stopping roots,
520 updates per arm, and24,576 strict full matches on2,048 shared fresh93M deals.
The primary criterion requires public-minus-masked and public-minus-original
balanced-mixture win lower95% bounds to be positive for both parents. Report all
opponent-specific and point outcomes as well. Reserved101M deals remain unused.

Next: preserve producer46255 and independent queue96354 until all24,576
strict matches and the independent audit finish. Then run the prepared
scripts/summarize_public_continuation_study.py once;never overwrite summary.md
or summary_receipt.json. Do not inspect partial contrasts,select checkpoints,
extend the fixed budget or duplicate trainers/evaluators. Preserve all252
renewal-queue input bindings as well as the221 fixed-study bindings. The queued
CUDA capability and renewal comparison can proceed only after their existing
prerequisites succeed.

**Training diversity diagnostic:** the855/876 root banks contain717/756
distinct initial deal permutations. Completed parent1-public samples all855
roots47--111 times(mean77.85) over66,560 continuations;replayed root-index RNG
equals the final checkpoint exactly. No simulation or evaluation was added.
[Counts and possible chance-renewal comparison](CONTINUATION_ROOT_DIVERSITY.rst).
This measures finite chance reuse,not overfitting or strength. Finish the current
comparison before changing its sampling procedure.

**Chance-renewal CPU capability:** an isolated public-context constructor
preserves points,dealer and round while drawing new legal cards/future chance.
Three CPU tests pass,0 skips;all1,731 real roots reconstruct exactly at their
original keys and support1,731 fresh legal conditional deals. Receipt
f18357c52be82afa71b6f5d14869d6616c7cc60bcbf41b08b67bd6c482961cfc.
[Implementation and checks](CONTINUATION_CHANCE.rst). The separate learner and
online loader now pass CPU checks;GPU capability is queued after the current audit.
No new strength trial or modification of the current study has occurred.

**Renewed-root sampling is restartable:** the isolated sampler preserves
source multiplicities and separate public-memory/routing metadata,assigns unique
chance keys within a fixed allocation,and validates saved cursor/RNG consistency.
Two CPU tests pass,0 skips,1.51s;XML9af1d59cd7035cca2de4825207879ae20360bb03d41a7d8dddd5f5f9d46759d1.
The independent renewed learner now passes CPU integration/resume checks;GPU validation remains pending;
the current fixed-root comparison is unchanged.

**Renewed-deal learner passes CPU checks:** two tests pass,0 skips,one
CUDA case deselected,46.40s. Complete multi-round reference returns,matched
public/masked warmup and exact next-update model/optimizer/chance resume pass.
XML19b7d3eccf220a3289fdfc22ed8041225b3e2b08650a2caed896a00d8e02621f.
The new-format online loader passes three tests in8.93s;capability guards
pass two in1.39s,both with0 skips. GPU/production capability and a prospective
strength trial remain pending. Existing46255/96354 and all221 inputs are unchanged.

**Renewal capability queue waiting:** runs/finish_renewed_continuation_capability.py
is live as OS Python PID36420(launcher63668). It pins240 unchanged inputs and
waits for the existing multi-round independent audit before running the actual
CUDA test and two three-update production-shape arms,including exact resume.
Receipt runs/renewed_continuation_capability_finish.json currently says
waiting_for_multi_round_independent_audit;waiting receipt SHA
c692ee7486c2967ffe8890e67a8fc4ed531757141b023b145ad9c3965a0c5181.
The8h observation deadline never terminates the existing producer. Do not
restart the queue or edit its bound new modules/tests. No CUDA stage has run.

**Prospective renewal comparison specified:**
[Design](RENEWED_CONTINUATION_STUDY.rst) separates fixed/renewed chance from
public/masked information using a2x2 comparison and the original-parent control.
It proposes four new520-update arms and40,960 fresh full matches on2,048 shared
95M deals. Primary strength and secondary memory/interaction claims are separate.
The specification was recorded while the current study was still training with
zero evaluation files. Receipt runs/renewed_continuation_design_pre_evaluation.json
SHAec855e090728072ddc491a9eadb1c50c3da1bd0538b169a59b310f77d104b6f3.
Runner,auditor and their CPU/fixture checks are now implemented. A queue waits
for the current comparison and CUDA capability before freezing/executing this
protocol. No renewed strength training or evaluation has started yet.

**Five-policy renewal analysis implemented:** the full-grid loader requires
40,960 records with unique policy/parent/opponent/seat/replicate/deal identities.
It reports all ten pairwise comparisons and the chance-by-memory interaction
for both parents and all three scopes(66 records,win and point intervals).
Strength,public-memory and interaction success remain separate. Five synthetic
tests pass,0 skips,in0.47s;XML462ca12e4c1aff98ee385328d6fe9923b7d5aed52a9381ded68b9a36096877b6.
Scalar stratified-bootstrap calculations agree;incomplete/malformed grids and
one-parent,point-only or one-opponent substitutes for primary success are rejected.
The analysis tests create no simulated matches or strength result. The separate
runner/auditor now pass their own CPU/fixture checks;see the queued milestone below.

**Third fixed-continuation endpoint verified:** parent2-public final
7beb4e2276d0773d015de8010650af620a0c3bd293b9e4db87333f0f99c49f44
passes independent rerun against its producer report,all228 bound inputs,and96
finite Adam states/phase counts. Actual4,718,538 decisions(2,328,594 learner,
2,389,944 opponent),66,560 conditional completions,318,361 conditional round
endings,0 full training matches. Receipt
runs/public_continuation_study/parent2_public_completion_verification.json
SHAea1e6df4e496bd2e83e94a9c0ec7eea7a7ef9c8186a3df831875d8597347a521.
Verifier98033 exited0 and adds0 simulation/inference. This milestone has now
been superseded by the four-arm completion verification above;its receipt
remains unchanged. No strength outcome is available yet.

**Renewal runner/auditor ready and queued:** seven tests pass,0 skips,in7.68s,
including fixed/renewed initial model/optimizer/global RNG equality,real fixed
root-index RNG validation,tamper rejection,independent paired statistics and a
five-policy worker fixture on one already used88M correctness deal. The fixture
adds5 full correctness matches plus5 reference replays,0 fresh evaluation deals.
XML88ae130fa534ceea0b27d75520119deb60ffff20251c4afcf4201055e5eeb03a.
Readiness receipt5708bec872285baa6236a844c1e015650b81f44d8a1d11e06a8151fe26be9d93
verifies250 inputs and confirms actual preflight refuses the unfinished prior
comparison. Queue37463(OS PID65372,launcher26672) now waits for renewed CUDA
capability success,then runs the predesigned study and independent audit in
sequence. Waiting receipt runs/renewed_continuation_study_finish.json pins252
unchanged inputs;SHA45a93fa6922e7280c352b71db942365368aec9a8595a4bf036eb93088d8ff253.
Do not duplicate it or edit its bound new sources/tests/design. Its8h waiting
deadline does not terminate preceding jobs. No new strength outcome or promotion.

**Renewal reporting ready:** scripts/summarize_renewed_continuation_study.py
requires a complete producer,passed unchanged independent audit,all66 result
cells and the full40,960-match grid. It keeps primary strength,secondary memory
and interaction verdicts separate and reports every contrast with work counts.
Four tests pass,0 skips,in1.49s;XMLfd1251a49ab56b880c774dbdeba195b7750229ad932484b0d82ef1ccb659cc4d.
Actual unstarted-study guard refuses output. Run once only after queue37463
finishes successfully;do not overwrite summary.md or summary_receipt.json.
No new study outcome is available. All252 queued inputs are unchanged.

## Completed evidence informing this experiment

**Public-memory round8 comparison independently complete:** producer67506 and audit35738 exit0;all24,576 full matches replay and all18 win/point contrasts reproduce exactly. Primary balanced-mixture gain fails;every win and point interval includes zero. Audit a5a9f350f2388294cdd106758d60eb453a2f03a24ff41f192436a04d4925acd7 verifies300,092 upstream input/mask checks and all181 input hashes. Actual evaluation2,857,337 decisions/191,262 rounds/24,576 matches;355,486 public-model predictions. Summarizer ran once;summary ca6182fb9e687e4c0a23237e161a4bc27de518ffb40e12b32c9b650e583fc985. [All outcomes](PUBLIC_MEMORY_RESULTS.rst). No promotion/extra old budget/101M use. Do not restart67506/35738 or resummarize over existing files.

Public-only identification reconstructs historical inputs after the eighth
opponent hand discard. It identified1,746 of2,048 retained training prefixes,
with302 unresolved and no wrong/empty classifications;1,598/1,856 natural
final-round roots were identified(86.1%). The prefixes share512 deals.
[Identification results](REVEALED_OPPONENT_IDENTIFICATION_RESULTS.rst).
There are1,731 eligible later fresh-round roots,855/876 per parent;1,642 precede
round8. [Bank construction](PUBLIC_MEMORY_CONTINUATION_BANK.rst).

Complete-continuation CPU/CUDA/online/resume capability passed before the current
study. Production audit1ba645ade180878ebf5e8803211abe46e1b1cc8161c160100b733b5fb0647bee
verifies both three-update arms and exact restarted updates;all214 inputs pass.
It is capability evidence,not a strength estimate.
[Implementation and checks](PUBLIC_MEMORY_CONTINUATION.rst).

The preceding unconditioned final-round study completed16,384 strict matches:
one parent/Point cell improved,but the replicated primary criterion failed.
[All outcomes](FINAL_ROUND_TRAINING_RESULTS.rst). Opening,mid-round and final-round
search studies likewise established no replicated win improvement;see
[mid-round](FIXED_MIDROUND_TERMINAL_RESULTS.rst) and
[final-round](FINAL_ROUND_RESULTS.rst) results. CPU singleton and fixed64 GPU
inference remain distinct numerical policies;do not claim action equivalence.
[Numerical analysis](NUMERICAL_BATCH_PRIOR_ART.rst).

Earlier controller,supervised-selector,direct-response,learner-return,
fixed-observer and public-seat comparisons also failed their replicated primary
win-gain criteria. Secondary gains remain reported in the manuscript and
EXPERIMENTS. Static RL-Point koi continuation improved expected point margins
in the measured development comparison,but did not establish a win gain.
None of these studies promoted a new human-play model. Completed studies and
capabilities are terminal;do not restart their directories.

The exact shared-parameter counterexample demonstrates reversed expected
updates under opponent-observation bootstrapping,including a finite cutoff at
lambda1. Learner-decision returns correct the complete finite example.
Public-seat and within-round reconstruction audits identify measured information
loss and verify representations. These are theoretical/representation results;
their empirical fixes did not establish replicated Koi-Koi win improvement.

## Latest manuscript

**Latest verified manuscript:**28 pages,31 references,143-word Abstract and20
embedded fonts. Abstract/Introduction now lead with terminal-match adaptation,
the exact critic-conditioning counterexample and public-input reconstruction.
Repeated experiment counts and separate negative-result summaries are condensed;
combined frontmatter falls from672 to552 whitespace words. Related Work and
all following source bytes remain identical,including every result,table and
qualification. All prior numeric tokens remain in the rendered PDF. All756
evidence inputs and252 live-queue bindings verify. All28 pages rendered/contact-
reviewed,with pages1/2/28 inspected full size;no box,glyph,reference or page-bound
defects. PDF4bf68f3e5143bd1367f381886b91c8b6586093bc22b3eb970a894829935eacc0;
source08678404210ef1df84d17d0ef9c35cef05e38b3a68e41aa1ee8f37f4c294b84a;
QA runs/manuscript_frontmatter_qa.json SHAfaf681e99a9d37b0989ad99419cc85f3977ccb920aa0f855c1004dd18a02cad0.
Previous31-reference PDF retained as output/pdf/manuscript_prior_art_2ffe37c5.pdf.
The prior-art comparisons remain documented in PUBLIC_MEMORY_PRIOR_ART.rst.
Ongoing multi-round/renewal experiments supply no new strength claim yet.

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

## Preserved latency and original release archive

**Latency COMPLETE:** original session77792 exited0 after final workers were gone; runner29440. `runs/final_latency` has protocol/process/report/raw/independent_audit. All41184 measured calls,10296 warmup,5148 validation on1287 unique cached development states completed; no new matches. Protocol SHA256 `ddec2df361cc2e206ab4759ac6f8a477fbf712003ff56dfac8f8887bfdc4083f`; report `24ec5e4d5574e48a92d590f3f757ef701bfad5cd2daf7cf072dfa8a284f84531`; audit `e3b48a44568597f3cc528396c3a700407054137af4c540e2983a3486434f34cc`; raw `d0b4454a18135c3cbfddf6d4d385d2bf657983f57667253d739fbadaea17fc94`. Mixed mean/median/p95/p99 milliseconds: canonical1.585/1.479/2.036/2.302, second1.595/1.504/2.034/2.286, WP1.974/1.911/2.714/2.925, Point1.926/1.834/2.688/2.901. Exact deployment agent.act path, CPU2 threads. Group overlap/repeated calls are descriptive; project absence checked at pass boundaries, OS noise uncontrolled. Audit checks raw accounting/statistics, not an independent clock/inference rerun. Do not rerun into the completed directory.

**Evidence archive COMPLETE:** `dist/koikoi_research_artifacts.zip`,315291737 bytes,372 files/428853553 uncompressed bytes. SHA256 `2ddce4d2033006e332f45d19ed7313f09eff9d11f7e9a625cbd4be4d7ff1f352`; source snapshot `f5cf955ac8f65c935a9f54c13bd0e8f25b6c39de`. Original build session82391 exited0. Portable bytes/member/source/model checks and independent final statistics pass: primaryWP=true, secondaryPoint=false. Receipt `runs/research_artifact_bundle.json` records the actual archive. Independent PowerShell SHA agrees; archived README contains current model instructions and archived PDF matches final QA. Later metadata-only commits do not change this snapshot. Prior ef554ba ZIP and failed path-order ZIP are retained under distinct names. The verifier uses original-platform pathlib ordering; Windows/POSIX regressions cover the discovered ordering bug. [Scope/instructions](ARTIFACTS.rst). Nothing uploaded externally.

The archive is the original release snapshot,not the current extended
manuscript or a bundle of the new studies. Its successful completion does not
complete the active stronger-play objective. No journal submission or external
artifact publication has been performed.

## Operating and verification reminders

Use `.venv/Scripts/python.exe` (Python3.12/Torch2.12+cu130/NumPy2.5.4),not system
Python3.14. CPU Ryzen9900X,GPU RTX5070Ti16GB. Read PowerShell text explicitly as
UTF8;preserve existing endings,new files UTF8 without BOM/LF. No subagents.
Live research jobs at this snapshot:producer46255,audit queue96354,and
renewal capability queue OS PID36420(launcher63668),and renewal study queue
37463(OS PID65372,launcher26672). The producer and audit
actual Python PIDs are10804 and32056 respectively. Confirm processes/receipts
before treating these handles as live later;do not duplicate their work.

Keep full matches,rounds,decisions,conditional continuations and verification
replay separate. Policies must not read hidden hands,stock order,RNG or future
events. Preserve original models,comparators and all bound evidence. Use
targeted checks for changes;do not repeatedly rerun broad suites without a new
failure or concern. Audit scope varies:read each receipt before asserting
trajectory or neural-inference replay. The original release suite passed460
tests with0 skips;that historical result does not replace checks of new work.
