# Current status

The stronger-play and research extension is active. A new replicated match-win
improvement has not yet been established. Initial dealer selection and the
playable canonical model remain unchanged.

This file is a current snapshot. Detailed commands, failed approaches and work
accounting are in [EXPERIMENTS.md](EXPERIMENTS.md). The complete preceding status
history is preserved at [commit80a1cf1](https://github.com/sheetau/koikoi/blob/80a1cf1/docs/STATUS.md).
Old process handles mentioned in that history are historical, not live jobs.

## Current experiment and next actions

**Koi-temperature comparison is running:** four original-WP initializations,
two matched seeds x common-temperature/calibrated-koi conditions, each with
20 critic-only + 500 policy updates. Learning rates, architecture, returns and
budgets are fixed; only koi temperature differs within each seed. Deployment
uses the original greedy singleton CPU path. [Frozen design](KOI_TEMPERATURE_STUDY.rst).

Production-size capability and independent saved-state audit pass: matched
initial weights/critic/Adam/environment/RNG, stationary warmup actor, later
actor updates and exact resumed update22. Corrected capability has 720,896
logical training decisions / 48,508 rounds / 6,093 matches; replay adds 32,768
decisions / 2,217 rounds / 291 matches. Earlier checker failure is preserved
separately: its fixed arm trained 360,448 decisions / 24,254 rounds / 3,052
matches before a metadata-iteration error; calibrated training had not started.
This is execution capability, not strength evidence. Independent capability SHA
0f37f1c43a10082bbbf6f892a1d28341e528c657751fb47332fa3825e1f7d0f1.

Training uses seeds 96110001 and 96210001, from original released WP rather
than retained A/B. After all four endpoints finish, evaluate them and A/B on
2,048 duplicate deals per opponent, both seats: 49,152 full matches on 4,096
shared development deals (97M). Primary criterion requires all four mixture
win lower bounds to exceed zero: calibrated-minus-fixed and calibrated-minus-
retained-parent at both seeds. Full trajectory/optimizer/statistics audit and
summary are queued. No outcome-based retuning or endpoint selection. The
sampling change also affects entropy/gradient scale, so this does not isolate
visitation alone. The calibration itself is not a new general algorithm.

Study protocol binds 91 inputs, SHA
b9075ac261ce6a3e681a64fa7845f031850ef20fba60f618435b3ad108a74d19;
it matches the prelaunch receipt. Original producer tool handle was lost,
but actual driver PID59656 (launcher54836) is live. At the current snapshot,
seed1_fixed has completed and its worker20924/launcher18484 are gone.
seed1_calibrated also completes; worker16036/launcher9328 are gone.
seed2_fixed also completes; worker64436/launcher55036 are gone.
seed2_calibrated is training (worker25144,launcher5464); no evaluation is complete.
Finalizer session64818 / PID36936 (launcher36764) is waiting; its eight-input
protocol SHA f6180c6f94acadb02d7f907f1d145df24577189cf50d429b532bf6a41c7c6f87.
Inspect process/finish_process and actual OS processes before any continuation;
never start duplicate workers or overwrite this directory. All study-bound
code, tests, XML and design are now frozen.

New checks: four adapter tests pass (actual CUDA included), three evaluation/
analysis tests pass, and three independent-auditor tests pass, all zero skips.
The auditor tests cover paired-seat cancellation, primary failure despite
point gains, per-head Adam counters and corruption rejection. Independent
optimizer inspection also passes actual initial/warm20/update100 checkpoints.
No new replicated improvement has yet been established. Canonical human-play
model, dealer, reserved101M and the latest verified paper/PDF are unchanged.

**First endpoint independently checked:** seed1_fixed finishes520 updates,
8,519,680 decisions /571,167 round endings /73,290 training matches. Warmup20
accounts for327,680 decisions /22,078 rounds /2,769 matches. Value Adam takes
49,920 steps; every actor head takes48,000. All saved schedules, finite weights,
initial original-WP identity, unchanged warmup actor, later actor changes and
final520/final.pt equality pass. Final checkpoint SHA
 e9c63e565982472994643625e4ef3bd574e97e54a125b157148818f54b95b156.

Early audit d7b87fcd5ca952a2c549c14c75b1e6b7cd26c6da91e32402a194e82a25a53f70
binds107 inputs at seed1_fixed/early_training_audit.json. It also verifies
fixed/calibrated initial actor,critic,Adam,environment,counters,all RNG and
runtime equality; configs differ only in koi_temperature. No forward call or
new training is performed by this audit. This is training integrity,not playing
strength. Full final match replay/statistics remains queued after all arms.

**Second endpoint independently checked:** seed1_calibrated finishes520
updates,8,519,680 decisions /571,529 round endings /73,374 training matches.
Warmup20:327,680 decisions /22,077 rounds /2,767 matches. Value Adam49,920;
every actor head48,000. All saved stages,initial ancestry,warmup stationarity,
later actor changes,finite state and final checkpoint identity pass. Checkpoint
9de43715d4b1805c45d8519d21dfb5b78ff15eb9c0524a927059220229e6ea4e;
early audit3669edfb5650d60dbae8ca584a23306e81fc04d62c6593f4d0d887023b8940c3
binds106 inputs. First paired training seed is complete; the second seed's
fixed/calibrated training and all strength evaluation remain outstanding.

**Third endpoint independently checked:** seed2_fixed finishes520 updates,
8,519,680 decisions /571,267 round endings /73,479 training matches. Warmup20:
327,680 decisions /22,041 rounds /2,779 matches. Value Adam49,920; each actor
head48,000. All saved stages,original-WP ancestry,warmup stationarity,head
changes,finite states and final checkpoint identity pass. Checkpoint
08a1293b6c560bf79b389b5d6ebcaf973f17e1cb8250592131c82f615f79d94c;
early audite9d906190b32d9008cc0dcc6eda5023080a5a1cadbba2c0b1a6bc42d8e69498c
binds107 inputs. It also verifies exact fixed/calibrated initial actor,critic,
Adam,environment,counters,all RNG and runtime at seed2. Only koi_temperature
differs in config. The fourth learner and all strength evaluation remain pending.

**Final-policy mechanism diagnostic prepared, not run:** seven policies
(original WP, retained A/B and four final learners) will be scored on common
public inputs after all four training endpoints complete. One population is
3,258 observations from25 old validation matches,3,045 unforced; another is
all512 saved final environment snapshots. The latter were produced by the
rollout preceding the final optimizer update, not a fresh rollout of the final
policy. Neither population estimates complete training visitation. Report
common-temperature and declared-training-temperature probabilities separately
from greedy changes; no simulation, outcome labels or checkpoint selection.
[Design and primary-source positioning](KOI_TEMPERATURE_MECHANISM.rst).

Two checks pass,0 skips,2.73s: independent binary entropy / score-rescaling
counterexample and actual public-feature hidden invariance plus deployment
action equality on four phase examples. Diagnostic protocol binds98 inputs,
SHA a25292a03b0d343870fd2e3b3c829bd2834aa41a3cdf6f467bb6877fd571c941.
Do not edit its bound script,design,tests or XML. A one-shot local queue now
waits for all four complete.json receipts, then runs the frozen diagnostic:
session10737,actualPID13764,launcher9832. Source is the ignored local file
runs/koi_temperature_mechanism_queue.py; queue protocol SHA
fb9c21fc65153fed0ec07d6d0b99a748af432107492dbbe47258896710555edd.
Do not launch a duplicate diagnostic. After completion, inspect its raw-score/
scalar accounting before using findings. Main producer,calibrated worker and
both waiting queues are live; no strength result is available yet.

The independent mechanism auditor is also ready: `python -m
scripts.audit_koi_temperature_mechanism --output runs/koi_temperature_mechanism`
AFTER the scorer completes. It checks every population/phase,public masks and
guards,recorded-score argmax and scalar probabilities/gaps/changes without
rerunning neural predictions. Two tests pass,0 skips,1.30s. Independent scalar
arithmetic matches all16 old calibration groups; receipt
c094bc5039ace2d0c46104b17757c1eda4cff40ff4c39c31fd780c53caafb3d8
binds11 inputs and adds0 neural calls/steps. This is an old-data cross-check,
not the still-pending final-policy diagnostic or any new strength evidence.

**Renewed-chance comparison independently complete:** allfour520-update arms,
40,960matches and full replay/statistics audit pass. Producer58904/queue65372
are gone;queue37463 exits0. The predeclared strength,public-memory and interaction
criteria all fail. No endpoint is promoted;the final-confirmation gate returns
qualified=false and101M remains unused. [All66contrasts](RENEWED_CONTINUATION_RESULTS.rst).

Primary mixture-win changes(pp):parent1 new-minus-original -0.903[-1.917,+0.098],
new-minus-fixed-public -0.110[-1.062,+0.842];parent2 -0.183[-1.136,+0.769] and
+0.134[-0.903,+1.172],respectively. Of66win contrasts,two exclude zero,both
negative against WP. Fifteen secondary point intervals exclude zero;none
substitutes for a failed primary or mechanism criterion.

Protocol b380344727a3ac878e1c542b1754d6eac478ae66269cd662c60c0cd85dec46aa;
audit0f920e4be44cf833964975f3f7784846eb556c32f215011597a5cc892480da0a;
summary4f7fbccb5f97c88d54792083ac39b7818b07a8fcddde5053f6c567611f91ee98.
Summarizer62894 exits0 and ranonce. Preserve completed records;do not restart
or overwrite summaries. Study used316initial/323queue bindings.

New training19,033,187decisions/266,240conditional completions/1,276,266conditional
round endings,0fullmatches;133,120shared distinctchancekeys. Evaluation
4,754,060decisions/318,260round endings/40,960matches on2,048shared95Mdeals.
Public-model predictions399,624. Replay verifies290,156historical inputs with0new
neural calls/independentdeals;root replay separately212,212dec/14,101round/192endings.

The manuscript includes the completed renewal result. Public identity
conditioning has failed across the three completed memory studies. Exact
current-hand inference now improves descriptive prediction error, but two
completed action-search diagnostics provide no stronger-player result.

**Complete exact-hand diagnostic verified:** all 25 eligible roots finish;
seven ineligible attempts remain in coverage. Session 67641 exits 0. There are
20 distinct public observations from eight shared development deals. Seventeen
new exact tables and eight cache uses cover 141,645 candidates and 138,848
valid hands across records. Mean Brier error: uniform 0.155551, history 0.136122,
identified history 0.131641; both conditioned variants improve at 23/25 roots.
Both parents and both opponent groups improve descriptively. These are old
development roots, not held-out evidence or a strength gain.

New inference takes 969.04 seconds, 484,673 hypothetical historical transitions
and 451,774 strict CPU calls, plus 953 source replay decisions. Independent
candidate/scalar/group/work verification passes, binds 91 inputs and adds
953 replay decisions with zero neural calls. It does not rerun all likelihood
networks. Report 24dfb447432d512a0a7df5f40ea9cb344c72a581007b37a524c78759b5f70a20;
audit 1dcb809be800cec9b528cd464fcf7448687a45ba0d5bdd7d1731c3757aad2341.
[Complete results](SAMPLED_HISTORY_RESULTS.rst).

**Both exact-belief action diagnostics independently complete:** sessions
76166 and 48350 exit 0. No live computation remains. The first comparison searches
all 25 roots under four conditions with 128 proposal / 512 confirmation caps and
margin 2.5. All 100 choices retain the parent; all proposal margins are nonpositive.
There are 49 positive-mean proposals, but zero confirmations under this gate.

A separately frozen follow-up confirms those 49 fixed alternatives with fresh
512-particle samples, retaining the same final margin. One history-only action
changes (parent 2 / WP, case 0). Its actual-source completion changes a draw to
a loss, point difference 0 to -38. All other choices retain the parent. Neither
study establishes a strength gain. All choices precede actual-hidden assessment;
these are conditional old-root completions, not independent full-match trials.
[Results and accounting](PREPARED_BELIEF_ACTION_RESULTS.rst).

First search: 7,802,486 hypothetical decisions / 95,232 completed branches /
573,452 round endings, 428.36s. Actual assessment: 25 conditional completions,
1,777 decisions / 145 round endings, 100 reused lookups. Follow-up: 3,816,893
hypothetical decisions / 50,176 completed branches / 282,000 round endings,
95.56s. It reuses 25 actual completions and adds one with 18 decisions / two
round endings. Each producer separately replays 1,906 source-prefix decisions.
Each independent auditor adds 953 prefix decisions and replays only the new
actual completions. No new historical neural calls or evaluation deals.

First audit c39cdb170eaefd936c4ee35f827cdfb5d42f7b02d7458dafa93b445d7569e6c2
binds 143 inputs; follow-up audit
a65cd98b73b8d56b95a9d94a26467cb81ed66eb9af947f4f7367e75f9830586b binds 153.
Both verify scalar search arithmetic and recorded completion actions; neither
independently reruns all hypothetical neural policies. Preserve finished runs.
Three CPU adapter checks, one actual CUDA/reference check, and two checks for
each auditor pass with zero skips. The CUDA check ran twice, both passed.

**Boundary-residual estimator diagnostic complete:** session 83797 exits 0;
50 cells (all 25 old roots x uniform/identified-history) and independent scalar
verification pass. The standard two-level correction reduces sample variance,
but not enough to pay for the cheap extra trajectories. Mean variance ratios
are 0.937784 / 0.934917; variance-times-recorded-seconds ratios 1.188984 / 1.221901.
Every one of the 50 time-cost ratios exceeds one. No action or policy is selected.
[Full results](BOUNDARY_RESIDUAL_RESULTS.rst), [prior art](BOUNDARY_RESIDUAL_PRIOR_ART.rst).

New work: high 4,202,458 decisions / 308,866 rounds / 51,200 terminal paths;
cheap 1,529,919 decisions / 204,800 rounds / 2,528 true terminal endpoints;
reference 16,823,174 decisions / 1,235,907 rounds / 204,800 terminal paths.
Recorded simulation/lookup times are 101.83 / 26.13 / 408.58s, excluding world
construction and prior inference. Producer and auditor each replay 953 source
decisions. Zero actual completions, new evaluation deals or training updates.
Protocol 2095651ec1a1027e23aac7e1fe6b9a2dc5f89ec43da16620a546e8e181fd941b
binds 198 inputs; audit
e74599970f1b0d4fe69a61e5bfee086353f6f4b6d570c1922c6e7df4e51fbba2 binds 253.
Do not rerun or overwrite this completed experiment.

A separately labelled post-hoc allocation calculation predicts mean best
variance-cost ratios 0.992473 (time) / 0.996522 (decisions). These optimistic
sample-moment estimates are not independently measured gains; no coefficient
is adopted. Receipt f87983c730d115b56efb27746c037da81141fd35153ce29548ca71e644256d1c
binds 257 inputs, with zero new simulation. The return identity and game-planning
variance reduction are prior work. This route is not prioritized for learning.

**Completed calibration underlying the live comparison.** Existing original-WP
scores on old random-policy public states show much more deterministic koi/stop
sampling than discard at common temperature 0.05. A score-only calibration on
75 whole source matches sets koi temperature 0.21989584747027907 to match the
normalized discard entropy; discard/capture retain 0.05. On 25 separate old
source matches, koi normalized entropy changes 0.199169 -> 0.446049 (discard
0.452271). This measures sampling probabilities, not actual training visitation
or strength. No outcome labels enter calibration. [Details](PHASE_EXPLORATION_CALIBRATION.rst).
Receipt d191ca3bafe0c3bb1f268f612a3dfb80f2223dc97292ed87d0571c6024487e0a;
two analytical tests pass, zero skips, 1.23s; no neural calls or simulation.

The isolated adapter and matched comparison are now implemented, verified and
running as described above. The old warm/cold study changed LR as well as
temperature, so it does not resolve this hypothesis. Keep 101M unused and do not
describe any new replicated strength gain before completed evaluation/audit.

**Fixed-root multi-round comparison independently complete:** producer46255
and audit96354 exited0. All24,576 strict matches and all18 contrasts reproduce.
Every win interval includes zero;the replicated primary criterion fails.
Parent1 public-minus-masked point intervals are positive for the mixture and
Point;parent1 masked-minus-parent intervals are negative in those same scopes.
These secondary point effects do not establish a replicated win gain.
[All outcomes](PUBLIC_CONTINUATION_RESULTS.rst). Audit
c1694086e2f91a7b13005324b409957cdd4aee4f84d09e2640e3e01a8601c322 verifies278inputs,
146,184 original historical inputs/masks and matched initialization/warmup.
The summarizer ran once;summary76f1352be16817c76e7c45fdeabf3cb97474581596809ded7cd50bc78a11eb29.
Do not restart the completed jobs or overwrite their summary.

Training totals18,982,723 decisions(9,364,114 learner/9,618,609 opponent),266,240
conditional completions,1,277,944 conditional round endings,0 full training
matches. Actual evaluation2,856,503 decisions/191,108 round endings/24,576matches
on2,048 shared93M deals,plus201,568 strict public-identification predictions.
Independent replay repeats these records with0 new neural calls or independent
deals. Root verification separately replays212,212dec/14,101round/192match endings.

**Renewed CUDA/production capability passed:** one actual CUDA test,0 skips,
7.528s(XML8b25274a538870705bb188159e75bbf613a9b243246839a35dd8383af1bd1262).
Two three-update production arms and exact next-update resumes pass;all96 Adam
states per arm are checked,initial/warmup states match,public memory weights
change and masked weights stay zero. Capability audit
bae671fbb2c674a888c8014baee4c4090d74523594ee363e8ee7ecc1c285aada verifies306inputs.
Logical capability work57,550dec/768conditional completions/3,844conditional
round endings;resume replay19,186dec/256completions/1,276round endings,0 full
matches. There are384 distinct shared chance keys,768 logical and256 replayed
conditional deal constructions. Queue PID36420 completed successfully. These
are capability checks,not strength evidence. Canonical model and dealer unchanged.

**Confirmation conditions fixed before renewed evaluation:**
[Prospective specification](RENEWED_CONFIRMATION.rst) permits only the completed,
independently audited and summarized study's replicated primary strength result
to qualify both renewed-public520 endpoints. Secondary/one-parent findings do
not substitute. Proposed confirmation uses three policies,4,096 paired deals
per opponent,8,192 shared fresh101M deals and98,304 matches;it is not running.
Memory/interaction claims remain separate and no automatic promotion occurs.
The evaluator/auditor/reporting are now implemented and verified below; execution
remains conditional on the completed development primary result.

Read-only qualification gate passes4 tests,0skips,1.39s;XML
6559213974f60c48c94ae315e566979a05a0f2169291a7dc107624cfbfe0c9d8.
Actual unfinished-study refusal and zero current evaluation files are recorded
in runs/renewed_confirmation_pre_evaluation.json SHA
 a9547cb8a9601b40e162988dca2bd1515e4b696bc4749601806d869dda89a934,
with321 immutable input bindings. This preparation runs0 simulation/neural calls
and consumes0 final deals. Main producer58904 and queue65372 remain live.

**Final-confirmation analysis implemented:** the three-policy loader requires
all98,304 record identities and rejects missing,duplicate,wrong-policy/deal or
invalid score records. It retains18 paired contrasts and18 absolute-performance
cells and requires allfour primary win bounds. Six synthetic tests pass,0skips,
1.33s;independent scalar checks cover4,096-pair variable samples and chunk
boundaries. XML4f709888979e2a8d295b1a7c356a7076238a4c6e4598c7fb6a06fd0571c75a78.
Synthetic rows and integer seed allocations are not played matches;0simulation
or neural calls. No final evaluator/auditor has been launched or final deal used.

**Final-confirmation workflow verified before development evaluation:**
New runner,auditor and summarizer implement the fixed98,304-match allocation,
strict singleton CPU play,full trajectory/public-input replay and independent
scalar absolute/contrast statistics. Eleven tests pass,0skips,5.18s;XML
 a651e732a9c3f62bee001fd89b44350a03aadc8db2dca0bd1e41607607859c5a.
Actual unfinished development is refused before an output/deal is created.
Receipt runs/renewed_confirmation_workflow_pre_evaluation.json SHA
 f81c80195f2394312309f7c9c88f048ffb21fae1e80f0006e410a7cef45a5385
binds336inputs while95M evaluation files remain0. Do not alter these files
without explicitly revising and revalidating that recorded capability.

Correctness worker uses initial models only and one already-consumed88M deal.
Both seats/three formats reproduce original actions and public evidence. Two
test attempts total12actual matches,1,380decisions,96round endings;the separate
replay repeats those counts with68historical input checks. First attempt failed
only in receipt counter aggregation after play/replay;its records are preserved.
Zero fresh confirmation deals,zero trained-candidate strength evaluation and no
promotion. After a qualifying completed development study only,run in order:
`python -m scripts.run_renewed_confirmation --output runs/renewed_confirmation`,
`python -m scripts.audit_renewed_confirmation --output runs/renewed_confirmation`,
`python -m scripts.summarize_renewed_confirmation --output runs/renewed_confirmation`.
None is queued or launched. A failed development primary leaves101M untouched.

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

**Latest verified manuscript:**29pages,31references,145-word Abstract,20embedded
fonts. Includes the completed40,960-match renewed-chance comparison and four
primary contrasts;strength,memory and interaction replication allfail. All66
contrasts/accounting remain in the bound ledger. Abstract/Introduction and
Conclusion agree with fixed/renewed outcomes. Conclusion shortened182->128words
in three paragraphs;body results and all previous numeric tokens retained.
Bibliography contents/font size unchanged;local item spacing removes an
isolated final reference page. New exporter verifies375inputs;full QA verifies
956evidence inputs and66live diagnostic bindings. All29pages rendered/contact-
reviewed,pages1/15/16/17/29 fullsize;no layout,glyph,reference or page-bound defects.
PDFea8073cbd5197820fb6f737055eb324adea49f0907f1f4eb325a103b074a9e6b;
sourcebee2dc10a52f02e9ec168cc800f9ae80b89a425da6e6ab283877ac7a5c568b56;
QA runs/manuscript_renewal_qa.json SHA038c62b5048139662f1788d4b68860c74c3f1e15d9f845800f4b7c4378a3b745.
Prior PDF output/pdf/manuscript_before_renewal_dfa9122e.pdf is preserved. Current-
hand inference remains a development diagnostic and supplies no new playing-
strength claim. Canonical human model and dealer are unchanged.

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
Live work is the koi-temperature study and its waiting finalizer described
at the top. Renewal, posterior-search and boundary-residual jobs are complete.
Their old process handles are historical; never restart their directories.

Keep full matches,rounds,decisions,conditional continuations and verification
replay separate. Policies must not read hidden hands,stock order,RNG or future
events. Preserve original models,comparators and all bound evidence. Use
targeted checks for changes;do not repeatedly rerun broad suites without a new
failure or concern. Audit scope varies:read each receipt before asserting
trajectory or neural-inference replay. The original release suite passed460
tests with0 skips;that historical result does not replace checks of new work.
