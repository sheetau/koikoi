# Current status

The stronger-play and research extension is active. A new replicated match-win
improvement has not yet been established. Initial dealer selection and the
playable canonical model remain unchanged.

This file is a current snapshot. Detailed commands, failed approaches and work
accounting are in [EXPERIMENTS.md](EXPERIMENTS.md). The complete preceding status
history is preserved at [commit80a1cf1](https://github.com/sheetau/koikoi/blob/80a1cf1/docs/STATUS.md).
Old process handles mentioned in that history are historical, not live jobs.

## Current experiment and next actions

**All six score-auxiliary learning conditions and their independent training
audit are complete. Main strength evaluation is running.** The original
runner20751/PID17896 failed during seed2_none update165 with WinError206 in
os.add_dll_directory. Its queue19901/PID59480 also exited1. Both original
folders and all failure/partial/completed artifacts remain untouched. They are
not live jobs. Detailed previous setup is retained at commit dca731f.

[Recovery design and commands](SCORE_AUXILIARY_RECOVERY.rst). An isolated probe
reproduced the DLL-registration failure after332 registrations of the same
Torch directory; closing registrations restored success. No simulation or
learning was needed for that probe. Each remaining condition ran in its
own process, with at most101 backend instances. Bound learner/native sources,
seeds, budgets and evaluation criteria remain unchanged.

Recovery root: runs/score_auxiliary_recovery. Runner session39422/PID67840
exited0 after seed2_score update520; all six arms are complete. Worker40312 and
launcher62540 are no longer live training jobs.
Completed seed1_none/win/score artifacts are copied
byte-for-byte,except relocated endpoint paths in completion receipts. Seed2_none
reran from its original seed; all recorded numerical metrics, checkpoint
states/RNG/optimizers and28old collections match. Seed2_win/score then ran
once each. No result-dependent selection or additional study arm is introduced.
Recovery seed2_none is complete; its worker15392/launcher67148 has exited0.
An early closed-prefix check passes: all164logged numerical rows,8saved
checkpoints (models/optimizers/RNG) and28collections match exactly,excluding
wall time/git provenance.104immutable files are bound;receipt
c8515b32cdf810d91a365e2c9485399ea18c8dcb370091e4303dd9303a4ff569.
This read-only check adds no training,matches or inference. The scheduled final
full-arm prefix check also passes (receipt0915d3696196f62a5ae503b0240571ded3e1ee16593a6028949a151aa481200d).
Queue session56042/PID42884 completed training_audit and evaluation preparation.
The auditor19968/launcher61320 has exited0. Evaluation runner3068
(launcher67876) is live; current cell workers43464/34188,launchers52844/63148,
parent2_score_wp/point. All eight parent1 cells and the parent2_parent,
parent2_none and parent2_win pairs each completed8,192matches; their workers
have exited. Fourteen of16cells are complete. Raw row hashes and line counts
match the completion receipts. Full
evaluation and independent action/statistical validation remain pending.
Recovery protocol b38aa097e1ce9a4f2989eab445025d7f4388fbf97d7dd0df8722860bddbe5294
binds683inputs,including every original artifact. Six recovery tests pass0skips
(1.67s),two relocation/queue tests pass0skips(1.75s),all synthetic/no simulation.

The original study protocol remains
9ba0bd7b1d187d59269af336537ceab890bf8884a969a9df073dcd626e686f46.
[Main design](SCORE_AUXILIARY_MAIN.rst): two seeds106110001/106210001,original
Point initialization,none/win/score arms,20warmup+500policy updates each;
51,118,080PPO decisions. Each arm collects128complete auxiliary matches every
5policy updates:100cycles/12,800matches;76,800total. None collects without
optimizing. Score adds within-outcome information to the matched categorical
win-only objective. Public features only; terminal margins are training labels.
This extends established auxiliary-learning ideas. Playing gain and substantive
novelty still require results; implementation and audit tests do not establish
those claims. [Component/capability](SCORE_AUXILIARY.rst).

Seed2_none completed520updates/8,519,680PPO decisions/561,106rounds/72,896matches.
Its100auxiliary collections contain1,492,804decisions/98,476rounds/12,800matches/
96,697rows and no auxiliary optimizer steps. Endpoint manifest
ed64f0c2e8e10e1e4f7418a37ebbec986149dfcbc7c120127279a64232ecb16d;
metrics2c2bcd7316f79af3c3cc53a1fde2fda543d38ead4828665b9a3fcf2f256fa9d5.
All748combined recovery/queue input bindings verified unchanged at completion.
The full six-arm training audit passed; main evaluation is running. Do not
restart completed arms or the live queue.

Seed2_win also completed520updates/8,519,680PPO decisions/562,644rounds/
73,092matches. Auxiliary100collections contain1,488,898decisions/98,510rounds/
12,800matches/96,745rows/580,470presentations and4,800Adam calls per phase.
Endpoint manifestb0b2e10a66900dc947a7a895fe4a4f06c3d87b92000c15fe291b17ac8a32dacf;
metricsb89773dd78823578f567feb4cf87ddb595e5ed09ae2db367c270c9518b1d352c.
Its worker41768/launcher11460 exited0. All748recovery/queue bindings and endpoint
file/metric digests verified unchanged at that milestone. The final score arm
is now complete; main strength evaluation has started.

Seed2_score completed520updates/8,519,680PPO decisions/560,337rounds/
72,961matches. Its100auxiliary collections contain1,490,552decisions/
98,259rounds/12,800matches/96,558rows/579,348presentations and4,800Adam calls
per phase. Endpoint manifestd922f0e5bc375612768b6f0aaa0d9e3f13741c999b51f3fd2a618493878a4114;
metricsfe4f4b41e9433270b185d6590849f19d1f2e65e1fc464703c1b151aeeb6b2c8d.
Final endpoint files,manifest,metrics and repeated-prefix digest verify.
Study completion45f7f24a83cfcb99ff09b83e5bada606560aaec1c4e0dab812de4749ba66f486.
Retained totals: PPO51,118,080decisions/3,369,134rounds/437,799matches;
auxiliary8,936,915decisions/589,918rounds/76,800matches. These are training
work counts,not evidence of improved strength. Mechanism diagnostics passed;
strength-result export awaits the complete evaluation and audit.

Restart overhead is counted separately: failed seed2_none executed165PPO
updates/2,703,360decisions,of which164updates were logged. Those logs contain
2,686,976decisions/174,865rounds/22,736matches. Unlogged update165 round/match
counts are unavailable. Auxiliary28collections contain421,608decisions/
27,488rounds/3,584matches. The failure occurred before collection29 environment
creation. Repeating the prefix does not increase independent sample size.

Recovery queue session56042/PID42884 observed exit0 of exact runner67840,creation
FILETIME134369397674275076. Root runs/score_auxiliary_recovery_completion,
protocol81de97227a5a280e31ecbc797bcaeb2e40f8fe189e51443f58191ac2213fc5c7,
748bindings. After successful completion/prefix equality: original independent
training audit,prepare fixed evaluation,run evaluation,independent native and
statistical audit. Training root alone is relocated; the new evaluation protocol
binds recovery provenance before outcomes. Full training audit passed:
08ae5b97cfcc885e9835dfaf33b79552110f21cda440d9d643667bfe34b28e8f,1,143inputs.
It verifies stored parameters/Adam state,all600auxiliary collections via action
replay,reservoir/public-feature/terminal-label agreement and common initial/
warmup states. It does not replay neural actions or gradients. Main strength
evaluation is live under protocolb49a9f06a9757f54e39f089cac1587f21786c6d7c07c786cbb35cc20d8f24c4b
with1,200inputs. Do not restart the live queue or evaluation.

[Evaluation](SCORE_AUXILIARY_EVALUATION.rst): six endpoints plus two retained
parents versus strict assisted WP/Point;4,096paired streams per opponent/both
seats,131,072matches. Seeds109110001/109210001,bootstrap109310001/10,000replicates.
All six simultaneous95% mixture-win lower bounds must exceed0: score-minus-none,
score-minus-win and score-minus-retained at both seeds. All120secondary
coordinates remain descriptive. Candidate seed1_score was named prospectively;
no better-seed selection,promotion or reserved101M consumption.

Completed seed1 models retain their original endpoint manifest digests:
none6686205ff7d8aa08a5e70f04498efb1191de1f191db09378864206050f1a10c1;
win865bcc4a228418a6d9ce590876e3fcecf7ca5b0a57091fc71dbc7304c612f024;
score287505f567c192e6de2c168c058c47ecc05b677a4aaa5f7c41085c5b15d0c663.
Each has8,519,680PPO decisions/12,800auxiliary matches. Full work counts are in
completion receipts and EXPERIMENTS. Completed models alone are not strength
results. Do not rerun closed capabilities/tests while waiting.

[Training audit](SCORE_AUXILIARY_TRAINING_AUDIT.rst),
[mechanism diagnostics](SCORE_AUXILIARY_MECHANISM.rst) and
[export](SCORE_AUXILIARY_EXPORT.rst) are implemented and tested. Main mechanism
diagnostics and independent audit are complete; strength export remains pending.
The recovery reporting adapter supplies exact
commands in SCORE_AUXILIARY_RECOVERY.rst: prepare/run/audit after the full training
audit,then export with the actual completed evaluation audit digest. It changes
only artifact roots/provenance and reports failed-run overhead separately. Three
synthetic tests pass0skips(1.66s),no matches/inference/training. Existing numerical
analysis and six-primary/120secondary criteria remain unchanged. After the
strength evaluation and export,integrate the verified results into the paper.
Interpret the comparisons before deciding the next scientific improvement.
[Completed fixed-state diagnostics](SCORE_AUXILIARY_MECHANISM_RESULTS.rst):
18,270CPU2singleton calls on3,045unforced public rows from25oldvalidation groups;
no new matches,simulator steps or learning. All six endpoints and Point analyzed.
Score-versus-win greedy disagreement331/3,045and249/3,045at seeds1/2. Discard and
both capture entropy are lower for score at both seeds under commonT0.05;koi
entropy differs in direction. These describe fixed inputs,not visitation or
playing gain. Protocol889c2225555e8723040c73d7415a2ccd57979cb1091d17630b8de5eb9c05d1aa;
reporta291e87f5505f26679c02927989bb8ad6bcef287a4d25a58e4c485fd9f9d8fd5;
independent auditd619258403def4da3aa71cbc3b02b45179abe964a73c6ccb70e802da2ea5ab52
passes1,160bindings. Preparation65413,scoring1612,audit1333all exited0. Do not
rerun completed diagnostics. No new tests or training were needed.

[Auxiliary prior-work review](SCORE_AUXILIARY_RELATED_WORK.rst) and three-paragraph
LaTeX/BibTeX integration drafts are ready. Six primary references include PPG,
UNREAL,C51,distributional-loss optimization (Imani/White2018),nonlinear learning
(Lyle et al.2019) and actor/critic specialization/data diversity (Garcin et al.,
ICLR2025). The draft positions the matched target-resolution intervention; it
makes no strength or priority claim. Merge with the completed strength results,
then rebuild/inspect the paper. The fragments are not yet included by manuscript.tex.

The auxiliary methods fragment docs/paper/auxiliary_study.tex now describes
matched losses/heads,collection/training and the fixed evaluation criterion.
Its six-row work table is generated from the audited completion record; the
completed fixed-state diagnostic numbers are included. No strength result or
placeholder is supplied. Merge with final evaluated results and the prior-work
fragment,then render/inspect; final recovery overhead belongs in the exported
accounting. Current PDF/manuscript source is unchanged.

Canonical player,dealer and the latest35page manuscript are unchanged.

## Earlier implementation milestones

The entries below retain their status at the time of each milestone. The current
experiment and completed outcomes above supersede their earlier waiting states.

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

**Bounded hand inference tested on earlier roots:** the isolated sampler and
fixed training-prefix diagnostic pass seven tests total,zero skips. All32
attempts are retained;25 four-card roots have4,845--5,985 possible hands and
128 proposals each. Joint-model pooled card error improves0.156804 to0.149846,
but parent1 worsens;publicly identified single-model error is0.158597. Median
identified-model effective proposals7;14/25 roots have at most10. No replicated
belief gain or strength result. All3,135 valid proposals pass full unpruned
published-policy replay and independent error arithmetic;receipt
2c3b08ca3cfbf19a420576533c34a2bb7db8ce0937d8fb88bfc815580ec1eac4.
[Full diagnostic and accounting](SAMPLED_HISTORY_RESULTS.rst). Existing252/221
queued study inputs are unchanged. Main strict evaluation and its queues remain
live;no model promotion,dealer change or101M use.

**Exact follow-up identifies sampling error:** all23,940 candidate hands at
four fixed development cases were enumerated(21,657 valid). These are only
three distinct own/public roots on two old deal seeds. Exact joint and publicly
identified posteriors lower true-hand card error in allfour cases;128-proposal
marginals can differ by up to0.926829 for one card. Full candidate-set/scalar
arithmetic verification passes;report8feae6976f43329e7816d1ec6a27119229387cc9f804ac8246ceed6266f7c0eb.
The identified parent1/WP support has components of61 and1 under single-card
swaps:local moves alone cannot cross them. Next proposal work should include
global moves and check correlations against the exact tables before strength
use. [Results and work](SAMPLED_HISTORY_RESULTS.rst). No new generalization or
playing-strength result;existing main study and queues continue unchanged.

**Mixed hand proposals implemented and checked:** uniform rejection starts a
stationary global/local chain;global probability0.2 connects local components.
Nine unit tests pass,zero skips,and allfour actual published-model checks agree
with cached exact likelihoods,full replay and hidden-world substitution.
At128 attempts,conditional exact marginal MSE falls25.67%/42.21% at the two WP
roots but rises13.72% at Point(the same public root for both parents). Shared
initialization failure probabilities are26.37%/37.25% at WP;allfailures retained.
This is a standard invariant-kernel construction,not a general new MCMC method
or strength result. [Complete comparison](MIXED_HISTORY_RESULTS.rst). Runtime
receipteaf60c75b07a335c047db0b18f4e854ef5298215589542311f92d629a1d93cb7.
Do not deploy universally:correlation and initialization failure remain material.
Existing252/221 study bindings and the canonical playable policy are unchanged.

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

**Latest verified manuscript:** 35 pages including appendices,34references,
188-word Abstract,467-word Introduction,139-word Conclusion. Main text6,045
whitespace words;Conclusion on page14. The prior structure reduction remains:
detailed return/observer/public-memory studies are retained in AppendicesC--E.

Section6.3 adds the completed Point-initialization study with both primary win
and secondary point tables,calibration design and pre-outcome mechanism.
Abstract/Introduction/Conclusion explicitly report1.77--2.08point gains over
matched WP controls and the lack of replicated additional win improvement.
Original WP-initialized versus Point findings are scoped to those retained
agents. Related Work positions both temperature controls without claiming a
new entropy algorithm. AppendixA records work and full evidence bindings.
All original tables,figures,citations and labels remain.

QA verifies1,581evidence inputs,35contact pages and full pages1,9,10,14,17.
Final wording changes alter only rendered pages1/14,both rechecked. No TeX
box/glyph/reference warnings,replacement characters or missing numbered
references;all21font names are embedded. A decimal-line overfull warning in
the first build was fixed by displaying six-digit divisors while retaining
full precision in the bound calibration report. No new simulation or training.
PDF SHA871b8afce615954c789a463a784158d4e273a70ca33c2f65d472cf67873f6bc6;
TeX SHAee5a7766e11e1a3743b26bc6eb2aab5a94e1ac17e6c1c4cabef6cb309473092a.
QA runs/manuscript_point_20261020_qa.json SHA
 d2de912a02ab580648abf11d65abc8d5175ee97341498ee2fc1b567817a3fdb0.
Prior PDF/TeX/QA/guide: runs/manuscript_point_previous.
Renders: runs/manuscript_point_render. Renderer uses system Python3.14,
external/pdf_render PyMuPDF and Pillow;training uses .venv Python3.12.
The PDF-skill artifact marker already ran earlier;never repeat it.

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
The koi-temperature study and its finalizer are complete,as are renewal,
posterior-search and boundary-residual jobs. No live job is retained here.
Their old process handles are historical;never restart their directories.

Keep full matches,rounds,decisions,conditional continuations and verification
replay separate. Policies must not read hidden hands,stock order,RNG or future
events. Preserve original models,comparators and all bound evidence. Use
targeted checks for changes;do not repeatedly rerun broad suites without a new
failure or concern. Audit scope varies:read each receipt before asserting
trajectory or neural-inference replay. The original release suite passed460
tests with0 skips;that historical result does not replace checks of new work.
