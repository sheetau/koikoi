# Current status

The stronger-play and research extension is active. A new replicated match-win
improvement has not yet been established. Initial dealer selection and the
playable canonical model remain unchanged.

This file is a current snapshot. Detailed commands, failed approaches and work
accounting are in [EXPERIMENTS.md](EXPERIMENTS.md). The complete preceding status
history is preserved at [commit80a1cf1](https://github.com/sheetau/koikoi/blob/80a1cf1/docs/STATUS.md).
Old process handles mentioned in that history are historical, not live jobs.

## Current experiment and next actions

**Longer adaptation and recovered evaluation are independently complete.**
[All results](PUBLISHED_DURATION_RESULTS.rst). Session2601 exits0 after all12cells
and the full native-action/statistics audit. Training, evaluation and recovery
are terminal; never restart them. Original failed queue/evaluator receipts remain
preserved. The recovery repeats four incomplete cells and copies eight complete
cells unchanged. All5,753 preserved prefix rows match repeated inference exactly.

The predeclared four-contrast simultaneous95% win criterion fails:

| Seed | Long minus short (pp) | Long minus retained (pp) |
| --- | --- | --- |
| 99110001 | +0.5798 [-0.3191,+1.4788] | +0.5249 [-0.3906,+1.4404] |
| 99210001 | +0.8942 [+0.0005,+1.7878] | +0.6104 [-0.2847,+1.5054] |

Long Point win scores are50.9460% [50.0916,51.8127] and52.2217%
[51.3489,53.0945],with descriptive marginal95% intervals. Long WP scores are
52.7283% and51.8066%. All four opponent-specific marginal lower bounds exceed50%,
but these secondary results do not replace the failed primary criterion or
prospective confirmation. Point mean margins remain negative:-1.1873/-0.3379;
the second interval includes zero. Both long mixture point margins improve over
both short/retained comparators with positive descriptive paired intervals.
No replicated win improvement over retained policies is established.

Training:131,727,360decisions/8,804,257rounds/1,133,134matches/8,040updates;
short prefixes included once. Evaluation:98,304matches/11,376,824decisions/
762,598round endings on8,192shared deals. Native replay checks those recorded
actions;it adds no independent deals or neural action selection. Additional
recorded recovery duplicates:5,753matches/667,309decisions/44,637round endings.
Unrecorded interrupted work remains unknown. Recovery capability separately
repeats8fixture matches/918decisions/64round endings,not strength evidence.

Evaluation protocol76e1b6af987e21f4e05e270aaabcd85ee9a89e68621589d1802a3a7543b24f18.
Recovery protocolbf1163339fdb52303344f118c92bb5c75204eea7e0211cdc295ab3ad9fca86c8.
Passed auditf193d0ac7ce3aa63071f3f65fb81d6b809329944012b4fa48d30c5a9e0367b31.
Summary74d03c23bb41a2fce755a1aee885119592de4bb527e8ae9f7ce9f870544e0050.
Recovery completion9640e4891cd7e70c64f3d5c63baf6b9dae5a9a76316743c7fd8663a62c70757b.
Complete-only recovered export runs/published_duration_results/evidence.json
b4e7bc2566deb3718aaf4c0dd19ac7d270750fe511f933a13e461c0b11c06212 binds197inputs;
all dependencies/output hashes verify. Do not rerun or overwrite completed exports.
Recovery sources/tests and all frozen inputs remain unchanged.

No endpoint is promoted; canonical player,dealer and101Mconfirmation are unchanged.
No duration-study job is live. This is greater-compute adaptation,not a novel algorithm.
The manuscript now includes the completed duration result (latest PDF below).
Do not extend this closed comparison or repeat closed temperature/return sweeps.

**Point initialization training is complete; the fixed main evaluation is live.**
[Specification](POINT_INITIALIZATION_STUDY.rst),[result](POINT_INITIALIZATION_CALIBRATION.rst).
The old public CPU-score corpus supplies75calibration/25descriptive validation
source groups,excluding forced choices. Four Point divisors match WP0.05phase
mean normalized entropy on calibration:0.16808073347585306,
0.19483521603032863,0.21949649006919622,0.23576122751596976.
Validation entropy differences remain;do not refit or claim equal exploration
on all states. No outcomes,new inference,simulator steps or learning are used.
Independent scalar entropy checks all24means to1e-12;three tests pass0skips,
Ruff passes. Report runs/point_initialization_calibration.json SHA
869ab1e45e7e8a5569ff20edaea55fdc5e2a2464d6d6a32a9c50df188bd701f3 binds9inputs.
Keep the calibration source,test,specification and corpus unchanged.

Implemented scripts/phase_temperature_ppo.py without changing frozen core or
prior-study sources. Nine tests pass0skips:both WP/Point all0.05 controls exactly
match original CUDA updates;per-phase scaling,invalid configs,Point warmup,
full next-update restore and two full-match greedy-policy equivalence pass.
The training runner's three failure/preservation tests also pass0skips.

Production capability runs/point_initialization_capability is terminal:
3arms x20warmup+2policy updates with128x128rollouts. Initial critic/env/optimizer/
RNG match across all arms;Point actor matches between its two arms. Saved-state
inspection confirms original pinned weights,all nine20/21/22snapshots,counters,
finite tensors,actual Adam steps/moments and exact resumed endpoints.
Unique work:1,081,344decisions/70,512round endings/8,968training matches;
repeated verification:49,152decisions/3,220round endings/409matches. These are
capability executions,not strength evidence. Independent audit
f8d3036e9af6b55b21cf94275515fd0c19f61b737539931133f1e38b56538367 binds94inputs;
producer receiptbe2b1c01539138582eaeeb89d7c9fb6262089f0b0efe1c9f3b91ac6835d02c1e.

Main training root runs/point_initialization_study is terminal. Session26829
exits0 after all six520-update learners; actual23676 is gone. Never restart it.
All six20-update warmups and520-update endpoints pass saved-state audits.

| Arm | Decisions | Round endings | Training matches |
| --- | ---: | ---: | ---: |
| seed1_wp_fixed | 8,519,680 | 571,625 | 73,440 |
| seed1_point_fixed | 8,519,680 | 562,510 | 73,153 |
| seed1_point_matched | 8,519,680 | 566,059 | 73,381 |
| seed2_wp_fixed | 8,519,680 | 572,238 | 73,449 |
| seed2_point_fixed | 8,519,680 | 562,166 | 72,862 |
| seed2_point_matched | 8,519,680 | 564,384 | 73,084 |

Total51,118,080decisions /3,398,982round endings /439,369training matches /
3,120updates. Included warmup:1,966,080decisions /128,492rounds /16,375matches.
Each endpoint has valueAdam49,920steps and each actor head48,000steps.
Last endpoint e96385a464231b56612827cef2c6944aca20f86e67bd8f07374e76eddb425e6e;
last standalone audit056a4490bdcb90d74dd352bcbd69ae10e74e03137904c029ee02a365042b82a7.
Complete receipt14a61deb6a030701c9879650bbf27c0b59d218a898f3967ef01126cbf137d620.
Training protocoldefd0e993e2c85efb844164b0c1edfa4ac6bae917c31ea6a6eb33d7ba4618048
and its102inputs remain frozen. The immutable complete receipt's evaluation
field records its creation-time state, not the subsequent evaluator status.

Both seeds' actual initial-state comparisons pass. Within each seed all three
arms share initial critic, optimizer, environment, counters, runtime and four
RNG states; the two Point actors are identical. Three-arm comparison receipts:
seed1 639678df15809acbd0fc965f701a607124f50377963b0d1f730db8eee098b454;
seed2 ef28204f585796f30897dacbaea887d03347bdcf66f3069b771624ea39717fcd.
All12standalone audits exactly equal the audits embedded by main preparation.
Saved-state checking adds no inference, simulation or new training; it is not
playing-strength evidence. No main outcome summary is available yet.

[Fixed main design](POINT_INITIALIZATION_MAIN.rst):seeds102110001/102210001,
three arms WP0.05/Point0.05/Pointcalibrated,each20warmup+500policy updates.
Total51,118,080decisions including1,966,080warmup once. Only totalupdate520 is
an evaluation endpoint. No resets,intermediate outcome checks or extensions.
Following all training,eight policies play131,072matches on8,192shared WP/Point
deals. Primary:Pointcalibrated-minus-WPfixed andminus-retained at both seeds,
one four-contrast simultaneous95% family;all lower bounds must be positive.
Report120secondary win/point coordinates. If qualified,fixedcandidate is
seed102110001's Pointcalibrated;101Mconfirmation remains separate andunused.

**Evaluation capability and completion queue are ready and frozen.**
New scripts/evaluate_point_initialization.py implements the fixed16cells,strict
singletonCPU2 inference and exclusive launch/completion/failure receipts.
No periodic status-file replacement. A failed worker stops owned process trees;
restarting the same evaluation is refused. Native CPU replay and separate scalar
bootstrap code verify all120secondary coordinates andfourprimary simultaneous
bands. The checker covers20warmup and520final saved stages before main evaluation.

Ten evaluation/checkpoint tests pass0skips in8.78s:paired scheduling,corrupted
record rejection,scalar bootstrap agreement,zero-SE/single-seed gate rejection,
training completion gating,owned-worker failure cleanup,andfive corruptions of
the actual first warmup snapshot. Four queue tests pass0skips in1.47s,including
an early completion receipt while the exact producer is still live.

Capability root runs/point_initialization_evaluation_capability is terminal;
session17700 exits0. All16cells x4matches finish:64matches/6,607decisions/
433round endings on4shared deals,using original-initialization fixtures and
retained parents. Independent recorded-action/statistics audit passes,with no
new inference or independent matches from replay. Capability isnotstrength.
Protocol d1d7646baaa00074240431c9deb1394b1e8fe51a83d0bf6022a5d906aa57d343;
audit0e87ebb17f564d06711e00a0d225a6497be0e7eac3452363476e4f56dbfa958a binds149inputs;
summary0185b290e35940374b3d0e56e4fd322ad04cf95083bf2b62ca100b39de2c6e04.
First actual warmup audit passes:327,680decisions/22,036rounds/2,770matches,
actorunchanged,valueAdam1,920steps,actorAdam0. Receipt
runs/point_initialization_study/seed1_wp_fixed/warmup_audit.json
SHA12c8f04ae4dc91588c66dd72e9a82c7c2a0e2853d39c54c4cb9b7912eea30d86.
It verifies only that saved prefix,not training convergence or playing strength.

Completion queue runs/point_initialization_completion remains live,session25193,
launcher60448/actual42304. Its exact held producer handle reports exit0 and the
matching training completion passes. Preparation exits0; evaluation is running.
Queue protocol3419c9da7ec674828b8865ea61dddc6e253600586093c9a23c8d2d940864ac8a
binds156unchanged inputs. It owns evaluation followed by the independent audit;
do not launch another evaluator or queue. It never promotes a model.

Main evaluation root runs/point_initialization_evaluation now exists.
Protocol SHA1df245b6fea1799c0ed9f91518642ccb3d1e4bf9c0f7e682a1208819e609b7e7
binds187inputs and all12training audits. The16cells use4,096paired deals per
opponent,131,072planned matches, and the previously fixed bootstrap family.
Evaluator launcher6284/actual53480 started04:45:17JST and remains live.
The first fourteen cells are complete: all four parent1 policies and the retained
parent2, parent2_wp_fixed and parent2_point_fixed policies against WP and Point
with8,192matches per cell. Completed-cell producer totals are114,688matches /
13,327,226decisions /889,231round endings. The two new Point-initialized cells
add16,384matches /1,905,828decisions /126,931round endings. Receipt hashes,
raw-file hashes,complete row counts and protocol/checkpoint identities agree.
This is producer completion,not the later native-action/statistics audit or
strength evidence. No partial outcome means were inspected.

At09:00JST the final workers are OS-live: parent2_point_matched_wp
launcher64044/actual31196(start08:55:00),and parent2_point_matched_point
launcher61980/actual61892(start08:59:25). The preceding workers53044/39900
have exited. Fourteen of16cells are complete; do not rerun them. The remaining
two cells are owned by evaluator53480.
Monitor evaluation_complete.json/evaluation_failed.json in the evaluation root,
and complete.json/failed.json in the completion-queue root. Track actual process
handles and newline-terminated row counts without interpreting partial outcomes.
Final results and the main native-action/statistics audit remain pending.

The complete-only Point result exporter is ready:
scripts/export_point_initialization_results.py. It requires the pinned training
and queue protocols,matching queue completion and explicit passed-auditSHA,
all16completed cells,all6training arms/all12saved-state audits,andunchanged raw
records/dependencies. It validates all120secondary coordinates andfourunrounded
simultaneous primary bands,then emits all24absolute and36contrast win/point rows,
plus primary results and full-precision JSON. Main work and replay are separate;
capability work remains in the ledger. No export selects or promotes a model.

Ten synthetic tests pass0skips in1.29s;Ruffpasses. They cover complete output,
units,non-overwrite,missing/failed/changed evidence,omitted/nonfinite coordinates,
zero-SE and all-four qualification. The actual live study is correctly refused
before output directory creation. No main result table or summary isexported.
After the queue and full audit complete,run once into a fresh directory:

    python -m scripts.export_point_initialization_results --output runs/point_initialization_results --audit-sha256 <completed-main-audit-sha256>

XML runs/point_initialization_export_tests_final.xml SHA
43d77aa1f183edad22b48c73c93b040f3d427241161cbbf7dca63e3fa0ae2110.
ExporterSHA7a282ec61e819ffdb6982bbe08fbed4528ec77da809e2fb53f514de0ea5f647f.
All156frozen queue inputs remain unchanged. Queue42304 and evaluator53480 are OS-live.

Next monitor queue session25193/actual42304, evaluator53480 and its live workers.
Training session26829 finished successfully; do not poll or restart it.
Freeze both training and evaluation/queue sources,tests and XML receipts. If a
wait expires,verify these exact processes;never restart solely from timeout.
The full main result ispending. Main seed inventory scans3,642JSONfiles with956
seed-bearing files,no scalar seed collisions or unreadable files;it does not
prove absence of unrecorded/implicit schedules. ScanSHA
 dfa5688894853738b608b0c79c530027cc6a0e7dc0854d54b1b932244aa98e68.
Initialization alone is not a new method;the broader objective stays unachieved.

**Completed final-policy Point mechanism diagnostic.**
[Results](POINT_INITIALIZATION_MECHANISM_RESULTS.rst) report all six endpoints
and three initial reference settings on the old25validation source groups:
3,258rows,3,045unforced choices. Six models make18,270singletonCPU2 calls;
originalWP/Point scores are reused. Zero new matches, simulator steps or learning.
The producer session64947 exits0; the independent scalar/population audit passes.

At both seeds and every phase, calibratedPoint has higher entropy at its declared
divisor than fixedPoint, but lower entropy under common0.05 and larger raw top-two
score gaps. Its declared entropy falls below its initial calibrated value in
all eight phase/seed comparisons. The calibration's probability effect persists
on these inputs; this does not establish equal exploration or better decisions.
Greedy calibrated-versus-fixed Point disagreement is232/200 of2,578discard
inputs and3/6 of215koi inputs. All phases/paired counts are reported, without
confidence intervals or strength conclusions from the reused correlated inputs.
Main full-match outcomes have not been inspected or used in this diagnostic.

Root runs/point_initialization_mechanism; protocol
cb4fdf6236f7e6cb74210ce663bac29a6b2536e5aff41fbc320ad21290a83b04 binds194inputs.
Report36e66a1aaea7f0179a505558ca2d254dcb827bbe31b4060cc76c03d910373f6a;
audit892486fa2f44efcdca5d93978cfce913fc8e8ff2bdbeccd2573fd3ad1aca257f binds198.
Three focused tests pass0skips(1.49s);Ruffpasses. TestXML SHA
0d423bb7580bf08489d92d2170b3bd4738787f51a828649c96033b4038538fab.
Freeze these completed sources/tests/inputs and do not rerun their networks.
Main evaluation and canonical deployment are unchanged; full results are pending.

Completed [initialization diagnostic](PUBLISHED_INITIALIZATION_COMPARISON.rst):
all28stored published configs use WP. Archived Point scores atT0.05 have lower
normalized entropy(koi0.028503 versusWP0.199169 on215old inputs). A naive
initialization swap also changes exploration. No new arm selected before the
fixed duration evaluation completes.

Completed training-log diagnostic:late approximate KL stays nonzero in both
seeds;pooled entropy changes0.504221 to0.481882 and0.500358 to0.504529
(short/long endpoint last100updates). This establishes neither convergence nor
an actor-gradient diagnosis. Report runs/published_duration_training_diagnostic.json,
e42ebf46118ef3064d56ceba3048e79c59bc99db1f40e74f5f48c5612e753d75.
See EXPERIMENTS for metric definitions and limitations.

**Koi-temperature comparison independently complete; improvement criterion fails.**
All four520-update learners and all49,152 full matches finish. Finalizer
session64818 exits0 after the complete training-state,recorded-action and
statistics audit. Producer59656,evaluation workers58596/61292/60952 and
finalizer36936 are gone. No live job remains in this study;never restart it.
[All results](KOI_TEMPERATURE_RESULTS.rst).

Primary mixture-win changes,percentage points with95% paired intervals:

| Learner | Calibrated minus fixed | Calibrated minus retained parent |
| --- | --- | --- |
| seed1 | -0.1953 [-1.1108,+0.7446] | +0.0122 [-0.8972,+0.9094] |
| seed2 | -0.7751 [-1.7274,+0.1770] | -0.7202 [-1.5991,+0.1588] |

All18 paired win-change intervals contain zero. Four secondary point-change
intervals exclude zero;these do not replace the failed primary criterion.
No checkpoint or temperature is reranked,extended or promoted. Reserved101M
confirmation deals and the canonical human-play agent remain unchanged.

Training:34,078,720 decisions /2,285,698 round endings /293,524 matches,
including20 critic-only+500 policy updates per arm. Warmup is included:
1,310,720 decisions /88,244 round endings /11,088 matches. Evaluation:
5,706,119 decisions /383,234 rounds /49,152 full matches on4,096 shared97M
deal streams. Replay repeats those evaluation events;it adds no independent
matches or neural action selection. All saved training stages and both
within-seed initial-state/RNG matches pass. No training replay is claimed.

Protocol b9075ac261ce6a3e681a64fa7845f031850ef20fba60f618435b3ad108a74d19
binds91 inputs. Independent audit
c1c89bf669f48490b51a40e6cebb35cf88a115b232fa95c6fe9f0ae1be625b45
binds169;summary247a230efbb275f58658be6f89c6a789fa9cc9c4b4c26ad2de43673c21b9e26a
binds171. Preserve runs/koi_temperature_study and completed finalizer receipts.

The guarded exporter runs once after finalizer completion,with the explicitly
pinned audit hash. runs/koi_temperature_results contains all18 absolute rows,
all18 contrasts and the four primary rows,with unrounded CSV and readable RST.
Export evidencee24777f4ac07c4cf32955a170902aea789c56ae960d43ea609c3fdbb57cb7877
binds174 inputs. Every exported CSV identity/value and output hash agrees with
the completed summary. This is presentation verification,not another analysis.
Four exporter tests pass0 skips;capability and earlier heading correction are
recorded in EXPERIMENTS.md. Do not overwrite completed exports.

**Final-policy mechanism diagnostic independently complete:**
Seven policies score3,045 old unforced validation inputs and481 unforced final
snapshots,adding24,682 singleton CPU calls and0 simulator steps or updates.
Independent population/mask/scalar verification passes with0 new neural calls;
it does not rerun the networks. [Findings](KOI_TEMPERATURE_MECHANISM_RESULTS.rst).
On215 common validation koi states,declared-temperature normalized entropy is
0.253215/0.253104 for calibrated versus0.200316/0.165775 for fixed controls.
Raw score gaps are1.462182/1.388870 versus0.590015/0.618418. Greedy changes
against originalWP are17/24 versus20/22. Those probability/score effects did
not establish a match-win gain. Snapshot koi results contain only39 states.

Mechanism protocol a25292a03b0d343870fd2e3b3c829bd2834aa41a3cdf6f467bb6877fd571c941;
report ad65e505430c175d78a51e32455f2be43824d18158be2ee1dd482f4ae22b04c4;
audit bcda39c6f5abef07a79332676a33df841cd344072aff9a27966bd50b985ae214.
Bindings98/108/113. Preserve these completed outputs. The focused
KOI_TEMPERATURE_PRIOR_ART.rst adds primary policy-distillation,discrete-SAC and
target-entropy literature and a concise candidate Related Work paragraph.
No new general algorithm claim follows from adjusting temperature.

**Directional return diagnostic independently complete; reversal not established.**
[All70 coordinates](DIRECTIONAL_RETURN_RESULTS.rst). Both retained CUDA models
complete8,192 sampled-self-play matches on4,096 shared98Mdeals:952,157 actual
decisions,64,067 rounds,17,907 model-forward batches,0 updates. A separate128-
match capability passes first. Both parent parameter digests remain unchanged.
CPU replay verifies every state/public-input hash,legal mask,action and work
output. Separate forward TD expansions,scalar scores and both-seat sums pass;
maximum advantage difference8.881784197001252e-16. No neural-inference replay
is claimed,and replay adds no independent matches.

One70-coordinate simultaneous95% bootstrap family preserves whole shared deals,
both parents and seats. All10 MC-mean and all30 estimator-minus-MC bands include
zero. The predeclared continue-direction reversal fails at both parents.
Seven GAE-mean coordinates are positive;these do not establish positive true
policy gradients or greedy deployment gains. This is inconclusive about
estimator equality,not proof of absence of bias. No return repair or model is
selected from the result.

Main protocol cae47d239af584ea62f47a97d879ecf83d7108870ca975b6a34db63a6c4ee911
binds42 inputs. Report5085458b15947062550d3a0732cb6a30d02bd9f29b988909c6085f50c4b7f059;
audit0aa47825725adf0ed1d7142edf1214c42c0653cfea555e1a72511104c608da3c;
summary5c63a84a8735520c73065e3a7e919f465ea5ddcea6b969794202430a6cf1c0c9.
Five tests pass0 skips,Ruff passes. Producer84684 and auditor82239 exit0;
no directional-diagnostic job remains. Preserve completed artifacts;do not restart.

The subsequent duration study above tests a different factor:training budget.
Do not repeat the closed return-decay or temperature sweeps. Existing paired
rollout selectors already used same-world branch outcomes,including larger-data
and opponent-preserving controls;do not relabel that method as a new direction.
The stronger-play objective remains active and unachieved. Canonical play,
dealer and reserved101M confirmation are unchanged. The later manuscript
revision below incorporates this completed diagnostic.

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

**Latest verified manuscript:** 33 pages including appendices, 34 references,
159-word Abstract, 425-word Introduction and 122-word Conclusion. The main text
now reaches its Conclusion on page13 rather than page18. Its whitespace word
count falls from7,484 to5,466 (27.0%). Detailed direct-opponent/return,
fixed-observer/seat and final-round/public-memory studies move to AppendicesC--E.
The main text retains their primary outcomes and explicit links to every detail.

Abstract, Introduction and Conclusion remain unchanged. Every original table,
figure, citation and label remains; the reference titles now distinguish the
moved appendices. Paragraph comparison finds only one changed original prose
paragraph: its final forward pointer becomes an explicit appendix reference.
New main summaries retain the failed replicated-win findings and distinguish
fixed-critic return controls from the learner-decision procedure's optimization
changes. No new playing-strength or algorithm claim is introduced.

QA rechecks1,353 evidence inputs and all156 live Point queue inputs. All33pages
are contact-reviewed and pages10--13 are inspected at full size. All34numbered
references are present,21unique font names are embedded,and there are no TeX
box,glyph,undefined-reference or replacement-character issues.
PDF SHA b9de94e68c9b1324924d461272bebed0bae24ab48e2155136a65d4810857c88b;
source SHA fcdebef32f65ebf6ba19e6c2707a7379cf64b4eb240d4a50ce911d6132b84355;
bibliography dd702f28fb9c7d49ec33ceaabd19672268e6fa3c57d5aadadec61292ceb7ad08.
QA runs/manuscript_structure_20261020_qa.json SHA
d202cc996f93ead6070321a72f50b636b5d38dcd373c7a25666cb7c4db2520b2.
Prior PDF/TeX/QA/guide: runs/manuscript_structure_previous.
Renders: runs/manuscript_structure_render. Renderer uses system Python3.14,
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
