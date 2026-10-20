# Current status

The stronger-play and research extension is active. A new replicated match-win
improvement has not yet been established. Initial dealer selection and the
playable canonical model remain unchanged.

This file is a current snapshot. Detailed commands, failed approaches and work
accounting are in [EXPERIMENTS.md](EXPERIMENTS.md). The complete preceding status
history is preserved at [commit80a1cf1](https://github.com/sheetau/koikoi/blob/80a1cf1/docs/STATUS.md).
Old process handles mentioned in that history are historical, not live jobs.

## Current experiment and next actions

The fixed prospective512 comparison is complete and independently exported.
All512sources/128deal streams are included;237eligible roots,275missing roots.
The four-contrast development strength gate fails. Controller-minus-parent
actual win-score differences are +0.3906pp[0,+1.5625] and0pp[-1.9531,+1.9531]
for parents1/2, with nominal simultaneous95% intervals. Controller-minus-root-only
is -0.1953pp[-.7812,0] and0pp[0,0]. Zero observed differences do not prove equality.
All-source model-conditional gains versus parent are +.2619/+.3246pp;
adoptions41/42 versus root-only28/28. See PROSPECTIVE_INCUMBENT_RESULTS.rst.
Protocol700e11608e3ced60919ca7ef31da460785ff9391459ac83c7e8a16f4a1a0c170;
summaryc80bbcbc67b1923632ea2480bacb06df8066f1e587ad83e843458fc782298f1e.
Original session89944/PID66280 is finished. Do not restart or extend this cohort.

Completion pipeline40713/PowerShellPID49904 completed the original export and
started the prewritten all-source opponent-only certificate comparison.
State runs/incumbent_completion_pipeline.json is running_certificate_comparison;
queue runs/certified_tail_replay/queue.json,PID35056. Four workers/32chunks.
Do not duplicate export/freeze/launch. Pipeline will audit after computation.
After its audit passes, run scripts.export_certified_tail_replay once; the
pipeline does not call this final report exporter. Reused-cohort computational
results are descriptive, with no fresh strength gate or confidence interval.
No new sources, cap extension or reserved101M use.

The opponent-only certificate resolves final-turn losses with public marginD<-2,
without simulated transitions. Nine tests pass; synthetic costs374to50 and
698133to152994 with identical exact values/maps. The frozen comparison applies
it to both arms under the original equal maximum allowances (not equal realized
work/time). Common completed policies must match; every adoption has independent
full Python replay. See CERTIFIED_TAIL_REPLAY.rst and
OPPONENT_LAST_TURN_CERTIFICATE.rst. Runner6tests/exporter4tests pass,0skips.

An isolated SymbolicFinalTurnPlanner adds an own-win regionD>2 and a checked,
versioned unchanged-parent continuation for omitted keys. It defines a complete
program, with all other missing keys rejected. Twelve tests pass,0skips.
Synthetic fixed9034to1258 and optimized9082to1306decisions; cap1258perphase
recovers parent204/275 and candidate461/550. Full independent Python replay
executes omitted decisions. See SYMBOLIC_FINAL_TURN_POLICY.rst.
This capability is not active in either fixed comparison and has no natural
coverage evaluation scheduled. Inspect the ongoing certificate comparison first.

Current manuscript includes the complete prospective result and its cost table,
as well as the previously closed exploratory64-source witnesses. Abstract and
Conclusion now distinguish the prospective conditional gains from the failed
actual-win criterion. Related Work has44references including POMDP safe policy
improvement (AAAI2023). PDF38pages;Abstract159/Introduction389/Conclusion169words.
All38pages rendered/contact-reviewed,full1/2/16/17 inspected,27fonts embedded,
no TeX/layout/bibliography warning; existing Fontconfig notice only.
PDF4c32ce1413ea29f27c5a42acd4b9abe7a053c08e97b47100f499b957dd50b21b.
QA runs/manuscript_focused_opening_qa.json; bound source/summary/export/table record
docs/paper/generated/prospective_incumbent_evidence.json. Prior audits reused;
no full old-corpus rehash or training rerun.

The symbolic program now has a separate adoption/three-tail assessment adapter,
scripts/symbolic_tail_assessment.py. It independently replays parent/candidate
programs with their leaf rule intact, executes divergent fixed/contingent roots,
and separately counts symbolic parent calls within all parent inference calls.
Four tests pass,0skips,2.35s; Ruff passes. Both seats, exact values, source/fallback
identity, independent action-call counters and mismatched selection/binding
rejection are covered. This adapter has no natural evaluation launch. The frozen
opponent-only comparison's95 inputs remain unchanged;10/32chunks complete,
queue35056 and all four workers confirmed live during this milestone.

Next: complete/audit/export the already running certificate comparison, then
decide the next implementation from its coverage and computational results.
Keep the broader stronger-play goal active. Canonical play and dealer unchanged.

### Completed diagnostics and supporting history

**The eight-arm root-allocation/step-size study is complete. Its predefined
strength criterion fails; one candidate-minus-parent primary interval is
wholly negative. All training/evaluation audits pass. That study has no live job.**
[Full results, all comparisons and work](PRIORITY_BRANCH_RESULTS.rst).
Session 92368 / PID 33408 exited0. Do not restart runs/priority_branch_study.
Training protocol SHA256 4543224c95c1ec5e9c24cfa0ed6f4450eda956e3d8e5bf79f125bd9fa105e79b.

Priority/high minus parent, percentage points with simultaneous95% intervals:
parent1 -0.3906[-2.2583,+1.4771]; parent2 -2.0874[-3.9551,-0.2197].
Priority/high minus uniform/high: parent1 +3.2349[+1.3672,+5.1025];
parent2 -0.6226[-2.4902,+1.2451]. All four lower bounds were required positive.
The first-parent advantage over a weaker high-rate control does not establish
improvement over the retained parent or replicated priority benefit.
No promotion or reserved101M use.

Training source:7,619,431decisions/512,651rounds/65,536matches,1,024updates.
Branches:19,943,108decisions/1,458,051rounds/326,506completions.
Evaluation:4,783,517decisions/318,537rounds/40,960matches,2,048distinct streams.
Training audit8fb3967cfb3af9e4fd0c50e7e9b4e2cb407faa0fb059aff89c94826fda19c5d7.
Evaluation auditf6dafc4c4d7609026822ce268ae8efd13b8710ebc778e9656f72674887da01c8.
Priority roots reduce confidence>=.99 from about30%to5--6%, but constant sampled
branch outcomes remain about68--70%. Larger updates did not produce stronger play.

**Full-source Monte Carlo study is complete; its six-contrast criterion fails.**
[All outcomes and work](FULL_SOURCE_RESULTS.rst). Session98245 / PID58904 exited0;
runs/full_source_study is closed. Both transition/statistical audits pass.
Both parents x full/root/branch x rates1e-5/1e-4 received128updates/64matches.

Full/high minus parent, percentage points with simultaneous95% intervals:
parent1 -1.1719[-3.2227,+0.8789]; parent2 -0.6592[-2.7100,+1.3916].
Full/high minus root/high: +2.3315[+0.2808,+4.3823] and
+3.8940[+1.8433,+5.9448]. Full/high minus branch/high:
+2.5757[+0.5249,+4.6265] and +1.4893[-0.5615,+3.5400].
Using all source decisions outperforms the high-rate root-sampled controls at
both parents, but does not establish improvement over the retained parents.
Low-rate full-minus-parent descriptive means are +0.2075pp and +0.0122pp.
No endpoint is promoted and no reserved101M data is used.

Source:11,272,771decisions/772,262rounds/98,304matches.
Branches:9,536,856decisions/708,187rounds/157,583completions.
Training:1,536updates/3,711,114gradient rows,1,806.783383s including I/O.
Evaluation:6,598,037decisions/446,749rounds/57,344matches;2,048distinct streams.
Protocol d878e4c8307f711e4fb1a3460502bf1fa6de353ede85b168f81a1d725ab00447.
Training audit58279e548d54d872dca9495c246a265d181999430199fe6591a570d339612525.
Evaluation audit e30fd79fe943c27d4ee1b24612571b456f688809de5b2c797373ec86e7ddfb38.
Complete-only export runs/full_source_results_export.json binds all tables and
metric counts. No full optimizer/neural-inference replay is claimed.

Core tests8 and reduced12-arm capability1 passed before launch; no skips.
The actual completed export now passes too. Do not rerun this closed comparison
or automatically add another root-allocation/learning-rate grid. The standard
full-trajectory REINFORCE control closes a missing comparison; it is not a new
algorithm. The broader stronger-play objective remains active.

Finite belief-tail capability is complete:64sources/16distinct deal streams,
16eligible roots,48missing,0budget refusals. Session14083/PID59532 exited0.
[All roots, costs and work](BELIEF_TAIL_RESULTS.rst). No live capability job.
Every root-action contingent policy passed independent Python transition replay;
source/actual-tail traces and bindings passed the complete-only export audit.
Six core tests pass,0skips. Implementation/protocol commit c282020.

Parent1:0exact expected gain at7roots. Parent2:1/110gain at1of9roots, mean
1/3520(+0.028409pp) over its32sources. Fixed-continuation gains0at both.
All actual win outcomes unchanged. This establishes one model-conditional
future-decision opportunity, not prospective replicated match-win improvement.
It does not justify a large strength evaluation of this three-draw gate.
Sources7,629decisions/511rounds/64matches; native verification separately same.
Planning660,758decisions/255,492terminal branches; independent policy replay
650,714decisions/248,760terminal completions. Actual assessment114decisions/
48conditional completions. Full separate history/export audit counts in report.
Protocol fc7a318560d7b294a41d057fa349de84d45ba1fbef64b2fb61b845598e52173f.
Complete320758c9c0fb79367423509c1f715d752e97680e9ae9473bf8b443195c616f7a.

Deferred chance equivalence is complete:all16closed roots/32fixed+optimized
plans match every eager rational value,action,wholepolicy and represented work.
Session14613/PID66820 exited0. No live job. Physical transitions fall660,758to
134,091(-79.7065%);terminalbranches255,492to118,941. Initialrows63,000to80;
maximumfrontier407. Observationnodes126,181 andowndecisionnodes4,706 unchanged.
All134,091new transitions independently match Python. Instrumented time8.113025s
includes5.686892s oracleverification;transition reduction is not latency speedup.
[Results](DEFERRED_BELIEF_TAIL_RESULTS.rst). Three coretests pass,0skips,94.47s,
including a full four-draw eager/deferred comparison and nonunit priors.
Protocol a451f2605fb4e02fa16f532101d38f2babc79e56857da7cb8d7b3416a38f7de8.
Complete7ad4de1a0bd4b1dffff83ec55934cc84bbb9a7975b1a764f224dd05b8e5cd95a.
The eager source and all its closed-study bindings remain unchanged.

The five-draw capability check is complete:64sources/16freshstreams,
18complete roots,15node-budget refusals,31missing. Session26188/PID26632 exited0.
[All results/refusals/costs](EARLIER_BELIEF_TAIL_RESULTS.rst).
Independent weighted Python replay verifies every completed root-policy value;
export separately replays source/prefix/actual-tail traces and checks bindings.
Parent1 gains79/8580at1of8completedroots (all32source mean+0.028773pp);
parent2 gains68/54945at1of10 (mean+0.003868pp). Parent1gain is from future
optimization;parent2gain is already obtained by the fixed-continuation control.
All actualmatch outcomes unchanged. No replicated match-win improvement.

Source7350decisions/496rounds/64matches; nativeverification separatelysame.
Completed planning6175862physicaldecisions/5427006terminalbranches;refused
partialplanning7894917/6922198. Oracle4967546decisions/4362358completions;
actualtails168/54. All represented-prefix units and support/exportwork separate
inreport. Protocol5cd1ab2345649ed83656c5f81cefd2cf44d2036a4c7b2cf2464aa4ff7da701fa.
Complete5357cb51bd8696db1824bec20aa4a62cdc8cfb3b842259ac28c269f436f5a9fc.
Do not restart this closed study or automatically extend its sourcebudget.

Terminal aggregation comparison is complete:all18completed five-draw roots,
36fixed/optimizedpolicies per method match every value/action/wholepolicy,
physical/representedwork andadaptercalls. Session99274/PID68548 exited0.
[Paired results](TERMINAL_REDUCED_TAIL_RESULTS.rst). No live job.
Alternatingmethodorder(9firsteach),separatefreshcaches;fixedthenoptimizedwithin
method. Control99.440871s,reduced63.783373s (-35.8580% observedtotal);maximumplan
10.236094s versus8.502455s. Both5513147physicaldecisions/4856793terminalbranches;
this comparison adds11026294decisions/9713586branches,0sources. Groupednodes
5023163all versus339711live. Ownnodes89673unchanged. Previous independentpolicy
oracle reused through exactmap/valueequality;no newfull Pythontransitionreplay.
Bothcap2Mphysical;control2Mallnodes/reduced500klive,nonbinding onthese18roots.
The15previouslyrefusedroots were notsolvedhere. No newstrengthgainclaimed.
Protocol857196060cd8f3670676533f2883c58806940adbda86c4f0e4d72aa3d2de43e6.
Completebca0c2086ba5a520c763de362354dc87cbf20d6a623d452ec68b4bee838cc835.
Two newcoretests pass,0skips,2.99s. Results committed/pushed at1b9e1c2.

Ceiling pruning Stage1 is complete:all18saved completed roots/36policies retain
chosenaction/value/wholepolicy;everycomputedrootvalue/mapmatches. Independent
Pythonreplaypasses.20plansomitrootactions,allwithchosenvalue1. Seven coretests
pass,0skips,2.76s. Session46597/PID22668 exited0;no liveStage1 job.
Pairedtimecontrol65.597451s/ceiling35.732193s (-45.5281%);physicaltransitions
5513147/3535665 (-35.8685%). Oracle3404854decisions/3003624terminalcompletions.
Stage1adds9048812planningdecisions/7976613terminalbranches,0sources;oraclework
separate. Alternatingorder,freshseparatecaches,matchedcaps. No newstrengthgain.
Protocolaae5eec7dea9637385180462a96c00ad8b0f059355a5d3305a7b6c62a52bc140.
Complete6efe77307aa016c8d6d4329f73540c0bf0dc3928262dfdfdbcdc019ee2d24308.

Ceiling recovery Stage2 is complete: all 15 old refusals were attempted under
500k live-node / 2M physical-transition caps; 10 completed and 5 remain refused.
Session3418 / PID42192 exited0. No live study. The original cohort now has
28 completed, 5 refused and 31 missing roots among 64 sources / 16 deal streams.
[Audited recovery and combined results](CEILING_RECOVERY_RESULTS.rst).
No source matches were added; this is an exploratory computational extension.

Combined optimized-minus-parent conditional expected win gains over all 32
sources per parent are +0.6715pp and +0.5905pp. Every actual match-win difference
is zero. Sources18 and50 have identical root states on the same Point deal,
although earlier match action prefixes differ. Both exact optimal Qs are
25249/28600. Their large conditional gains are not independent replications.
At this root, both parents and the optimized policy choose action0; the fixed-
continuation optimizer chooses21. The improvement requires future decisions.

Recovery planning:12,283,153 decisions / 10,912,399 terminal branches completed;
9,999,960 / 8,761,604 in partial refused plans. Independent Python oracle:
9,970,387 decisions / 8,844,642 terminal completions. Actual tails:86 decisions /
30 conditional completions; exporter separately replays these same tails.
History replay42,540 decisions. All work categories are separate; no new samples.
Refusals0,4,20,36,52 all hit physical caps, with zero ceiling cutoffs in their
partial work. Four fail during fixed continuation, one during optimization.
Do not automatically raise budgets or extend source collection.
Protocol ca89d632709be3f25b5ce1c0ffe7f6f71628d9d4ad257c5c898b3a1f6e465eb9.
Complete fa7c01ef7751624558c94c61cec01bbe306bbf0218ae1999c937d576ce27227c.
Report002a12d637ac9d7e3443e674463696ff0d76e0d5040344d1cc8a46fefa6ddcba.
Receipt runs/ceiling_tail_recovery_export.json binds the exporter and old/new
artifacts, verifies all exact values and computed/omitted action sets, and
independently replays actual suffixes. Export succeeds; lint/format pass.
Goal active; canonical play/PDF unchanged, reserved101M unused. New replicated
match-win improvement remains open. A saved-data public-outcome bound check is now complete (session71770 exited0).
[Bound and proof](PUBLIC_TAIL_BOUNDS.rst); [coverage](PUBLIC_TAIL_BOUND_COVERAGE.rst).
Seven tests pass,0 skips,1.93s. Every possible next stock card and all subsequent
legal choices are checked on eight late fixtures at three synthetic margins;
this is finite verification, not exhaustive hidden-deck enumeration.
Public captures plus remaining declaration opportunities give conservative
final margin intervals; strict one-sided intervals certify a fixed win/loss.
No rules or production policy change. No new simulation/model-call/source work
in the saved-data coverage check (unit-test transitions are correctness work).
Coverage:25,801/305,425 fixed learner keys;25,801/299,564 optimized learner keys;
13,220/552,339 opponent keys. Counts are per-root/method unique keys, unweighted.
25,715 of25,801 certified learner keys belong to sources2/34, already certified
at the root. Only86 learner keys elsewhere qualify. None of the six positive-
gain roots has certified saved keys. No remaining refused root is certified;
no partial-tree interior cache exists. These results do not justify integrating
this coarse bound as the next recovery optimization. No speedup was measured.
Complete06dc246eb69c23621dd71e15b80922785acffb0218b8fe279eede22556db44a7.
Report43df90f6bb98deba1c11b126a9f80ab96805f0c64198521d97a57d2e4a129030.
The analysis is closed. Any tighter bound needs new reasoning and verification;
do not simply repeat this check or raise the closed recovery budget.

Passive five-refusal profiling is complete (session82737/PID17896 exited0).
All old counters/calls/stopping stages reproduce exactly. Every parent-first
root action completes; later comparisons exhaust the unchanged physical cap.
[All root actions and costs](REFUSED_TAIL_PROFILE_RESULTS.rst).
Additional planning11,922,254 transitions /10,460,295 terminal branches; history
6,615 decisions (3,255 teacher calls /105 guarded). No new source matches.
96.62998% of physical transitions occur on turns14/15. Profiled source36 includes
its complete fixed control before optimized refusal. No new strength result.
Completecd59d2d09e21d94b0502ec59efebe5469416731351658e8e72bf353dc0543f73.
Protocol668b3cb3b49e79257cd7d7599f2cb70dc59c0bb11d49f38fdb93a0b3c5a09f42.
Bound support-world and partial-cache artifacts are available for reuse.

Added BoundedBeliefPlanner: resolved value plus remaining mass bounds a candidate
continuation; prune only when it cannot beat an exact incumbent. Root exact Qs
and upper bounds are disjoint; chosen contingent policies remain complete.
Ten tests pass,0 skips,3.32s: hidden information, nonunit masses, actual partial
cutoffs below the ceiling, propagated cutoffs, both-seat native comparisons and
independent Python policy replay. [Proof and protocol](BOUNDED_BELIEF_TAIL.rst).
The paired all28-root comparison is complete (session66474/PID52964 exited0).
All56 chosen policies/values, computed root maps and partial upper bounds pass;
all physical/adapter-call monotonicity checks pass. [Results](BOUNDED_TAIL_RESULTS.rst).
Control13,896,524 / bounded12,422,529 physical decisions (-10.6069%); terminal
branches12,333,528 /11,003,741. Total155.243126s /139.910682s (-9.8764%).
Maximum individual plan21.064108s /21.528018s; not every plan is faster.
There are26 root bounds below1 and2,330 partial candidate cutoffs across trees.
New paired work26,319,053 decisions /23,337,269 terminal branches,0 source matches.
Prior independent Python oracle evidence reused through whole-policy/value
identity; no new full Python/neural replay or strength result. Export audit passes.
Protocol110dd1c58a9d89e5073dd0835852fc9930d8b5777b13141afb1e90dd58268df5.
Complete7770f4879c856119ffcca0547544fcaf547b3d39241396375a5f449eef9938c2.
Report2a93fa7eff67fff0842c35c3a2922628fcea79ebdc8957d184e8e901c87ce718.

Bounded recovery is complete (session14768/PID28228 exited0); no live job.
All5 attempts retain budget refusals:0/36 in optimized,4/20/52 in fixed planning.
Source0 newly completes its fixed control: action12 Q161723/240240 versus parent
39 Q1711/2640, exact gain3011/120120 (+2.5067pp locally). Source36 fixed gain0.
[Audited attempts and work](BOUNDED_RECOVERY_RESULTS.rst). No actual tails were
assessed because no full fixed+optimized comparison completed. The original
28complete/5refused/31missing cohort and combined realized results are unchanged.
Source0's feasible fixed improvement is not yet an executed controller gain.
Completed planning3,725,271 decisions/3,303,427 terminalbranches; partial refused
9,999,949/8,754,799. Independent Python oracle3,419,559/3,031,396. No new source
or history-support replay. Fixed planning52.533664s;oracle111.643012s;partial
375.385298s. All bindings/plan values/upperbounds/counters pass export audit.
Protocol31da8e14da6b2fb3c669b541fa75b532f620f0d8120d456bb05f8d4da96ee2b4.
Complete05c3b1ce713f975c7f60dd1167a88abc645150ad0348e565c6daae9fa5a99201.
Report13937c64654d93e15a7abd28151d75641193f96e71afb2046ee5c5d49261d804.
Do not restart this closed study or automatically raise its budgets.

Added separate IncumbentBeliefPlanner to retain only completed root policies
when a later candidate exhausts the cap. Exact Qs, unfinished upper bounds,
refusal and optimality gap remain distinct. No incomplete policy is executable.
[Semantics and adoption requirement](INCUMBENT_BELIEF_TAIL.rst). Three tests
pass,0 skips,2.18s: a retained Q3/4 policy improves on the Q1/4 parent while
leaving gap1/4, with caps before/during another candidate; native full plans and
Python oracle match, empty refusals retain accounting. This adds no study or
production model change. Next implement the controller's same-belief exact
parent-value comparison and validate incumbent execution on the saved cohort,
with a fixed resource schedule and frozen policies before any actual assessment.
The full-refusal comparison stays unchanged. Retaining feasible incumbents is
standard anytime search, not a general algorithm priority claim. Goal remains
active; replicated empirical win improvement, deployment and paper update open.

The previous goal turn made progress: bounded comparison/recovery completed and
incumbent retention was implemented/tested. A separate selection/execution
controller now binds hypothesis arrays/weights, parent/opponent identities,
viewer, horizon and root observation. Require the exact unchanged-parent value
from fixed continuation; strict gains only, equal candidates prefer fixed.
Fresh schedule:fixed then optimized,2M decisions/500k live/depth32 each; skip
optimized if parent unvalued or fixed value1; no transfer between phase caps.
Execution uses own/public keys and rejects missing keys/wrong root/seat/identity.
Four tests pass,0 skips,1.64s. [Protocol](INCUMBENT_CONTROLLER.rst).
Saved-plan execution audit is complete (session96041 exited0). All33 available
roots /66 parent-controller tails pass execution and separate trace replay.
Seven strict adoptions (parent1:5,parent2:2);31 missing and3 unvalued available
roots retain parent. [All64 selections and work](INCUMBENT_EXECUTION_RESULTS.rst).
Conditional gains over32sources:576449/76876800 (+0.749835pp) and2699329/457142400
(+0.590479pp). All realized match-win changes remain0. This adds source0's fixed
improvement to the adopted policies, not a fresh empirical strength result.
Actual214decisions/66conditionalcompletions; separate replay same. Parentactor
calls130/opponent84. No new search/history/source work or training. This audit
did not run the full new search schedule or reconstruct unsaved partial plans.
Completec5bc93283539291ecd67cb7af7ae634131a98bf90e8fb4932fc460f596606eb4.
Protocol5e806932b7cd4f4794700b6462e46cff6cc0423bebd08e62da75a7c53a4bcf9e.
Report11b2146894da59e12f8b7373308b47e7708d65f80940b682c9259a5a1f1fa0d4.

Actual retained-incumbent evaluation is complete (session18992/PID25980 exited0).
All five roots ran both phases. Previously attempted phases reproduce exact work
with zero new adapter calls; optimized phases4/20/52 add explicit new work.
Parent references match the profile. Two old complete fixed policies reuse their
oracle evidence; five newly retained root policies pass independent Python replay.
[All attempts and combined results](INCUMBENT_RECOVERY_RESULTS.rst).
Source0 now adopts an optimized continuation of rootaction39:Q55113/80080 versus
parent1711/2640, gain4819/120120 (+4.0118pp locally). Search remains capped with
within-optimized-phase gap24967/80080. The chosen policy itself is complete.
Other4 roots retain parent. All10 actual parent/controller tails and separate
replays pass; point margins and wins are unchanged. Seven total adoptions remain
in the original64source/16stream cohort. Combined conditional gains per32sources:
204203/25625600 (+0.796871pp) and2699329/457142400 (+0.590479pp); realized gains0.
Completed-phase work3,725,271decisions/3,303,427terminalbranches; capped-phase
work15,999,927/14,074,152 (19,725,198/17,377,579 combined). New independentoracle
5,732,910/5,040,902;represented oracle68,628,933decisions/13,590,720completions.
New adapters38,319parent/12,029opponent. Actual46decisions/10completions, separate
replay same; actualactor24parent/22opponent calls. No newsource/history/training.
Fixed107.516820s,optimized199.313384s,oracle187.030644s with reused caches;
these are not a paired speed comparison. Complete-only export audit passes.
Protocol2365bf2ff39b69e38ba973f18da10fdb265a9e9e351b0dd958497e1cf4517b94.
Complete76c56afcc94ea8aaa2d7116766d193161a65676b09ede430b8e33b23eea49380.
Report0192337e0ef33d4f85829acc7db54a8810e92314ef19ac442bbba69920d810aa.
No live job. Do not repeat this closed diagnostic or automatically raise caps.

[Expected-match guarantee and prior work](INCUMBENT_GUARANTEE.rst) states the
conditional-expectation policy-switching argument under the declared model,
including correct posterior and committed continuation assumptions. A finite
check passes1test,0skips,1.40s:conditional gain1/4 with gateprobability3/10 gives
populationgain3/40, can lose on an individual world, and reverses under a changed
true prior. This does not prove superiority under the finite seed distribution.
Primary HSVI2004, SPIBB2019 and MCTS-SPIBB2023 sources were read and credited;
anytime policies/bounds and safe baseline improvement are established ideas.
No new-general-algorithm or world-best claim. Manuscript/PDF remain unchanged.
Next fix a prospective comparison on unused development deals, with parent and
root-only control, explicit matched total computation, source budget and cost/
feasibility checks. No new seeds or source budget have been allocated yet.
Do not infer population effect size or replication from the16 saved streams.
Goal remains active; canonical play, dealer and reserved101M remain unchanged.

A saved-gradient block analysis is also closed: runs/branch_gradient_blocks.
It verifies all four previous full-gradient totals by splitting 768 output
weights,3biases and4,018,944encoder parameters. It does not support adopting
head-only updates: signal estimates are uncertain at parent1 in both blocks.
No new simulations, forwards/backwards, updates or independent samples were added.
Do not repeat it, and do not automatically launch another priority/head-only grid.

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
result. It did not change the now-completed study or justify an automatic new grid.
Do not repeat runs/categorical_sensitivity.

New primary-source prior-art review identifies OSCC (arXiv2609.33543v2,
16 October 2026), which already studies policy-gradient covariance and
cost-aware coupling selection. Its C3 requires branch-local policy draws;
the local shared-action sampler uses a different coupling class. The
[condition/proof and exact example](POLICY_COUPLING_CONDITIONS.rst) show why
fresh shared policy innovations can preserve marginal path laws and expected
root gradients, while same-draw reuse within a branch can bias them. Three
tests and the exact rational check pass; no new Koi-Koi simulations or training.
This is a scope clarification, not a new variance-reduction or stronger-play claim.
The eight-arm study completed without changing its bound sources.

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

The root-allocation/step-size follow-up above is now also complete.
Do not describe variance reduction as stronger play or as invention of all-action
policy gradients. Assess root-sampling efficiency and update/coverage limitations
before committing to another training budget; a larger learning rate is not yet
shown to solve the problem. Read existing round-continuation/Q-boosting/search
results before proposing a supposedly new variant. The latest38-page paper
retains its clearer160-word Abstract and436-word Introduction and now includes
an exact finite-game gradient figure. See Latest manuscript for current QA.


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

**Latest verified manuscript:**38pages,40references,160-word Abstract,
436-word Introduction and164-word Conclusion, now on page16. The opening
foregrounds adaptation, critic conditioning and public-input recovery.
A new Figure1 on page12 shows the exact complete-game expected update versus
lambda: true gradient3/800, reversal at6567/6767 and the lambda.95 negative
update. Every101 plotted coordinate is independently checked against the prior
trace identity. This visualizes the existing counterexample, not new simulation.

Only that figure and its prose cross-reference changed; reversing those edits
reconstructs the preceding TeX byte for byte. Existing tables, references and
Abstract/Introduction remain intact. The auxiliary comparison and all other
measured claims retain their previous evidence. The latest branch/full-source
studies are separate technical reports, not newly asserted results in this PDF.

New figure inputs/outputs were rehashed. Prior3,313-binding validation is reused,
not rerun. Pages1--11 are pixel-identical; changed pages12--38 were rendered and
reviewed in contact sheets, with pages12/13 and the actual figure PDF at full
size. All27font names are embedded; no page-bound, TeX box/glyph/reference or
bibliography warnings remain. No new training or simulation was used for the
figure; the simultaneous full-source study is separately counted above.
PDF SHA a896c9a995a8d0156585f24c493f2f721db7ed9e97f3d214f4a092901275286e.
TeX SHA aefd5280af68bf1de2e2fe7325999cdd6a3123b53298ece498cef0d2c7ac5176.
QA runs/manuscript_observer_figure_qa.json SHA
ab2a982de405a2a36b1aff0d0d1fa9514464a168dfeb78b770fd4013ce39abdc.
Previous PDF/source: runs/manuscript_observer_figure_previous.
Renders: runs/manuscript_observer_figure_render. SystemPython3.14 with
external/pdf_render handles rendering; .venv Python3.12 with external/plotting
creates the figure without changing training dependencies.
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
