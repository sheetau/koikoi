# Current status

The stronger-play and research extension is active. A new replicated match-win
improvement has not yet been established. Initial dealer selection and the
playable canonical model remain unchanged.

This file is a current snapshot. Detailed commands, failed approaches and work
accounting are in [EXPERIMENTS.md](EXPERIMENTS.md). The complete preceding status
history is preserved at [commit80a1cf1](https://github.com/sheetau/koikoi/blob/80a1cf1/docs/STATUS.md).
Old process handles mentioned in that history are historical, not live jobs.

## Current experiment and next actions

The root-allocation/step-size factorial study is running in session 92368,
worker PID 33408, under runs/priority_branch_study. Training, training audit,
evaluation and evaluation audit run sequentially in that process. Do not start
a duplicate. [Fixed design and prior work](PRIORITY_BRANCH_GRADIENT.rst).
Protocol SHA256 4543224c95c1ec5e9c24cfa0ed6f4450eda956e3d8e5bf79f125bd9fa105e79b.
Both parents receive uniform/priority roots x Adam1e-5/1e-4,128updates per arm.
The predefined candidate is priority/high; all8endpoints and both parents will
be evaluated on fresh118M GPU development deals. No mid-run selection.

The new sampler computes priorities from public action probabilities, retains
one weighted-reservoir root per complete match and applies its exact conditional
inclusion correction. Uniform mode reproduces the prior implementation exactly.
Six estimator/sampler/restart tests pass on CPU/CUDA. A complete reduced8-arm
training/evaluation/audit pipeline also passes, including80evaluation matches.
These are capability results, not strength evidence.

Completed saved-data coverage analysis: runs/branch_update_coverage. Both old
shared arms reconstruct exactly at all16saved actor/Adam checkpoints.256replayed
optimizer steps,0new simulations/trained endpoints. About30%of sampled roots have
max action probability>=.99 but contribute only0.0145%/0.0163%of empirical weighted
logit-gradient energy. A fixed public sensitivity proxy gives finite-pool second-
moment ratios0.6455/0.6654; these do not guarantee future variance reduction.
Final shared actors differ from their parents at1/64and3/64fixed diagnostic roots.
Summary SHA8c873e04c1ea39fe430784393527aed8174a3cc9de9283dc7e4f11eab6815f20.
This analysis is closed; do not repeat it. The recent VSR preprint and active
importance-sampling literature are explicitly credited in the new design.

A separate saved-pool categorical-sensitivity analysis is complete:
[Exact bound and results](CATEGORICAL_SENSITIVITY.rst). Enumerating at most
128 sign partitions gives the exact maximum root-logit gradient norm for
bounded terminal outcomes. Three independent numerical/closed-form tests pass.
Replacing the Gini proxy by this bound lowers the hypothetical saved-pool
second moment a further 6.34%/4.11%; no new simulations, updates or independent
samples. This is neither a full-network variance guarantee nor a strength
result. It does not change the running study or justify an automatic new grid.
Do not repeat runs/categorical_sensitivity.

## Completed uniform-root branch-gradient screen

**The six-arm complete-match branch-gradient screen and its independent audits
are complete. All six primary simultaneous win-change intervals include zero.
No study job remains live. The stronger-play/research objective remains active.**
[Complete signal and match results](BRANCH_POLICY_GRADIENT_RESULTS.rst).

Both retained parents were trained with shared-action branches, independent-action
branches and root-sampled REINFORCE. Each arm received128updates/8,192source
matches. All six final endpoints and both parents were evaluated on fresh GPU
paired development deals. No checkpoint selection, promotion or reserved101M use.

Primary shared-minus-parent win changes in percentage points, simultaneous95%:
parent1 -0.7080[-1.8799,+0.4639]; parent2 +0.0366[-1.1353,+1.2085].
All four shared-minus-control intervals also include zero. The matched-source
screen does not establish a gain from the branching or coupling intervention.

Source training:5,722,980decisions/384,049round endings/49,152matches.
Additional branches:9,685,484decisions/706,597round endings/157,465completions.
Evaluation:3,787,953decisions/254,310round endings/32,768matches on2,048distinct
paired streams. Branch simulations and replay are not independent fresh matches.

Training audit6c95abeba7d2a2a447298c9402c4f72cbaa341c14501c2d07e97fb35b9e03658.
Evaluation audit7d87dfd2a6a71eae875fa12f641f12092078966cd7714ab54f2d59d0dc5475f3.
They replay all recorded actions and verify roots/labels/counters/primary statistics.
A separate32-trajectory baseline check agrees on all3,843CPU singleton action
requests. This subset does not replace a complete strict-CPU evaluation.
Training session64742/PID19268 and queue15164/PID55284 exited0. Their artifact
roots runs/branch_policy_gradient_training and runs/branch_policy_gradient_completion
are closed. Do not rerun completed commands or overwrite outputs.

The earlier frozen-root diagnostic did find lower conditional root-logit variance
(shared/independent0.29824and0.64505). Post-hoc full-actor gradients also have lower
conditional variance(0.620943and0.341429), using the same eight replicates and
4,019,715parameters. This is signal evidence, not a replicated match-win gain.
Both diagnostic roots remain closed: runs/branch_policy_gradient_diagnostic and
runs/branch_parameter_variance. Parent models and dealer remain unchanged.

The next fixed intervention is the root-allocation/step-size study above.
Do not describe variance reduction as stronger play or as invention of all-action
policy gradients. Assess root-sampling efficiency and update/coverage limitations
before committing to another training budget; a larger learning rate is not yet
shown to solve the problem. Read existing round-continuation/Q-boosting/search
results before proposing a supposedly new variant. The paper has a verified 38-page opening revision: Abstract 160 words and
Introduction 436 words, with empirical limits consolidated after the main
contributions. Related Work onward is unchanged; all 3,313 evidence bindings
verify. QA: runs/manuscript_priority_opening_20261020_qa.json.
PDF SHA256 40e14e9132163e154704581c75bef31ffbe5c52e2115d126c6d053fd7fc3d0a3.

## Recently completed score-auxiliary study

**The six-arm score-auxiliary study, independent training/evaluation audits,
fixed-state diagnostics and result export are complete. The primary win-gain
criterion fails: all six simultaneous intervals include zero. No study job is
live. The broader stronger-play and research objective remains active.**

[Complete results and all comparisons](SCORE_AUXILIARY_RESULTS.rst).
The two-seed comparison holds the 65-logit head and initialization fixed while
adding within-outcome margin likelihood to outcome supervision. All three
conditions begin from original Point. Each receives20warmup+500policy updates
and100auxiliary collections of128matches; none collects without optimizing.
Public inputs only; terminal margins are training labels.

Primary equal-mixture win changes in percentage points, simultaneous95%:

- Seed1 score-none: -0.1953 [-1.1108,+0.7202].
- Seed1 score-win: -0.0305 [-0.9607,+0.8997].
- Seed1 score-parent: +0.8209 [-0.2281,+1.8699].
- Seed2 score-none: +0.3510 [-0.5326,+1.2345].
- Seed2 score-win: -0.1709 [-1.0401,+0.6983].
- Seed2 score-parent: +0.6805 [-0.3703,+1.7314].

All24 descriptive score-versus-none/score-versus-win win/point intervals also
include zero across both seeds and all opponent scopes. All18 point contrasts
of none/win/score against retained WP-initialized parents have positive marginal
lower bounds; the no-auxiliary control shares this benefit. These results do
not identify an auxiliary-learning gain. Small gains or harms remain compatible
with the intervals. No promotion, better-seed selection or reserved101M use.

Retained training:51,118,080PPO decisions/3,369,134round endings/437,799matches.
Auxiliary collection:8,936,915decisions/589,918round endings/76,800matches.
Evaluation:131,072matches/15,240,199decisions/1,014,387round endings on8,192
shared deal streams, both seats. Six primary simultaneous comparisons and
all120descriptive marginal quantities were fixed before evaluation.

Recovery queue session56042/PID42884 exited0 after training_audit, prepare,
evaluate and evaluation_audit. Auditor30044/launcher64380 and all evaluator
workers have exited. Export session18695 also exited0. Never restart these
completed jobs or overwrite their outputs. Detailed setup and progression are
preserved in EXPERIMENTS and at commit ca3fe92.

Evaluation protocol:
b49a9f06a9757f54e39f089cac1587f21786c6d7c07c786cbb35cc20d8f24c4b.
Training audit passed1,143bindings:
08ae5b97cfcc885e9835dfaf33b79552110f21cda440d9d643667bfe34b28e8f.
Evaluation audit passed1,235bindings:
cf5f8fc9c5873dca98893ae3abac13722ec4e71f7c091f98464e5ca59d868ceb.
It replays all recorded actions through native CPU, checks initial deals,
legality/work/points and independently reproduces all intervals. Replay is
validation work, not new independent matches; neural inference is not replayed.
Export evidence binds1,825inputs:
a8f3ac919fba4cfe1e10f0e3caa1fb23b5542fcb5e76beac2d10cfa5ecbaf311.
The complete export remains in runs/score_auxiliary_results. Unrounded values
are retained there; the result document reproduces every contrast.

[Recovery provenance](SCORE_AUXILIARY_RECOVERY.rst): original runner20751 and
queue19901 failed with WinError206 after repeated DLL-directory registration.
Original artifacts remain untouched. Recovery uses fresh processes without
changing numerical sources/seeds/budgets. The repeated164logged updates,
8checkpoints and28collections agree exactly. Failed-run overhead is separate:
2,703,360executed PPO decisions; only164updates have round/match logs, giving
174,865rounds/22,736matches. Auxiliary overhead421,608decisions/27,488rounds/
3,584matches. Missing update165 round/match counts are not imputed. Repeated
prefix work adds no independent evidence. All numerical/source protocols and
failure receipts remain preserved. Do not rerun closed capabilities or audits.

[Completed fixed-state diagnostics](SCORE_AUXILIARY_MECHANISM_RESULTS.rst):
score/outcome policies disagree on331/3,045and249/3,045unforced public inputs.
Score entropy is lower in discard and both capture phases at both seeds under
commonT0.05; koi entropy differs in direction. These policy-output changes did
not establish an auxiliary match-performance gain. The diagnostic adds18,270
CPU2singleton calls and no new matches. Independent audit:
d619258403def4da3aa71cbc3b02b45179abe964a73c6ccb70e802da2ea5ab52.
Do not repeat the completed mechanism commands.

The auxiliary results and six-reference prior-work review are integrated into
the verified 38-page manuscript. The complete primary table is in Section 6.4;
methods and work are in Appendix L. See the Latest manuscript section below.

Canonical player and initial dealer selection remain unchanged. Stronger
replicated play and substantive novelty are still open requirements.

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

**Latest verified manuscript:** 38 pages including appendices, 40 references,
184-word Abstract, 514-word Introduction and 164-word Conclusion. Conclusion
on page 15. The main account adds the auxiliary comparison in Section 6.4;
its detailed loss, training, work and diagnostics are in Appendix L.

Table 8 contains all six simultaneous primary comparisons. All intervals
include zero; all 24 auxiliary win/point comparisons against matched controls
also include zero. The point advantage over retained WP parents includes the
no-auxiliary condition. Abstract/Introduction/Conclusion keep these results
separate from the established adaptation gain and analytical information
findings. Related Work adds six verified auxiliary/distributional references
and corrects the abbreviated account of retained-Transformer search coverage.
All earlier tables, figures, citations and labels remain.

The table exporter verifies 1,833 inputs. Final QA verifies 3,313 bindings,
all six primary table rows and six auxiliary work rows. All 38 pages were
visually reviewed; full pages 1,3,11,15,18,34,35,36,37,38 were checked. Final
render changes on 11,35--38 were rechecked. Removing duplicate diagnostic
interpretation resolves the short continuation before References. All 24 font
names are embedded; no TeX box/glyph/reference or bibliography warnings remain.
No new simulation, inference or training was used for this paper update.
PDF SHA d4e6471cb00840cf9b773f0dc39f043db09a1782a42bd6fa0d85c1bfdd114758;
TeX SHA 85c19bbd05057f3f021d96b27f0a7aede22b9cd012bdab9aa5af89a64aba9625.
QA runs/manuscript_auxiliary_20261020_qa.json SHA
0b3ed9c3301e6ea6125583ce5c0894b1a410638575f53ac84801a0632d06fc9a.
Prior PDF/source: runs/manuscript_auxiliary_previous.
Renders: runs/manuscript_auxiliary_render. Renderer uses system Python3.14,
external/pdf_render PyMuPDF and Pillow; training uses .venv Python3.12.
The PDF-skill artifact marker already ran earlier; never repeat it.

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
