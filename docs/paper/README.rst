Manuscript working files
========================

This is an English research manuscript, not a submitted or published paper.

Current verified revision: 39 pages including appendices, 44 cited references,
159-word Abstract, 389-word Introduction and 169-word Conclusion. The main text
ends on page19. The title is "Match Adaptation and Public-Information Planning
for Koi-Koi Transformers." Abstract and Introduction state the adaptation,
public-information controller and exact-critic contributions directly.

Section9 explains complete contingent policies, retained completed candidates
under search caps and the conditional-expectation guarantee. It includes the
closed exploratory64-source witnesses and complete prospective512-source
comparison. The latter does not meet its predefined actual-win strength gate.
Section9.5 now defines executable certificate compression: an explicit history
map plus an unchanged-parent rule in the public own-final-turn win region.
It presents the synthetic capped-recovery witness and completed opponent-only
comparison, which reduces transitions by2.0191/1.2086% without changing selections
or outcomes. The whole-cohort symbolic comparison is running and supplies no
results to this PDF. Related Work includes HSVI, SPIBB, MCTS-SPIBB and the2023
POMDP safe-improvement paper. The guarantee uses conditional expectation under
the specified model; it is not a new general safe-improvement theorem.

Appendix A is a concise evidence map and reproduction guide. Its former1,601-word
ledger is preserved, modulo line endings, in ``reproduction_ledger.tex``.
Detailed negative results and work accounting remain in that companion ledger
and ``docs/EXPERIMENTS.md``.

Latest QA: ``runs/manuscript_symbolic_method_qa.json``. All39pages were rendered
and contact-reviewed; changed pages18/19 inspected at full size. All27fonts embed;
no page-bound, TeX box, glyph, reference or bibliography warning remains. The
existing Fontconfig notice is unchanged. ``generated/symbolic_method_evidence.json``
binds the new method and completed comparison. Every474eligible arm selection
descriptor is unchanged in the opponent-only comparison. Earlier experiment
audits are reused without rehashing the complete old corpus.
Current PDF: ``output/pdf/manuscript.pdf``; SHA256
f6a7f81b0684b2708649a772c2413ca01deea8ca97ecb726e546afd994e0b66a.
The preceding PDF is ``output/pdf/manuscript_before_symbolic_method_4c32ce14.pdf``.

The stronger-play objective remains active. Current job state is in
``docs/STATUS.md``; later historical revision notes are not a live-job inventory.
Both retained models pass the original RL-WP criterion. Additional replicated
win improvement and the separate RL-Point criterion remain unestablished.
Initial dealer selection and canonical human-play deployment are unchanged.

From the repository root, regenerate the numerical tables and evidence ledger::

    .venv/Scripts/python.exe -m scripts.export_paper_evidence
    .venv/Scripts/python.exe -m scripts.export_score_auxiliary_paper_evidence

The exporter verifies fixed input hashes and the hashes of development reports.
If the final independent audit is absent, the final table remains empty and
the ledger explicitly says draft_final_results_pending. An existing incomplete
or invalid final audit is an error, not a reason to use a provisional summary.
This exporter does not replace the experimental auditors. The tracked generated
files permit inspecting the manuscript without the ignored local run directory.
Audited latency inputs produce the mixed-state timing table and are bound in
the same evidence ledger, including the raw timing file's digest.

Compile with a TeX engine supporting the listed ordinary LaTeX packages.
The local build uses the portable Tectonic 0.17.0 Windows MSVC release::

    external/tectonic/tectonic.exe --keep-logs --outdir output/pdf docs/paper/manuscript.tex

Create output/pdf first if needed. Tectonic downloads its font/package bundle
on the first build; later builds use the cache. It runs bibliography and
cross-reference passes automatically. The official archive SHA256 checked
against the GitHub release API was
f61ce51f0b0ade1015b7de7ef368541c5424e9756ecbd0d7af97d6d48030845f.
The executable and downloaded tool dependencies stay in ignored external/.
PDFs and review images are ignored build outputs. A conventional LaTeX/BibTeX
workflow from docs/paper is also possible but has not yet been exercised here.

Before submission, complete the author and venue information and arrange the
permitted artifact release. Code is public, but local model files and run
artifacts remain ignored. The gated local packaging workflow is documented in
docs/ARTIFACTS.rst; it requires current final manuscript QA and a clean committed
source snapshot. Submission and external artifact publication are separate
decisions, and a local ZIP is not an external archival release.

Source verification
-------------------

The method description is checked against koikoi/published_train.py,
koikoi/published_actor.py, signed_gae and minibatch_step in koikoi/train.py,
the fixed cold/warm protocol, and docs/RULES.md. The cited original Koi-Koi
paper was read from the pinned repository PDF. PPO and GAE bibliographic
records were verified against https://arxiv.org/abs/1707.06347 and
https://arxiv.org/abs/1506.02438 on 2026-10-17. Development, earlier-screen,
feature-compatibility, unsuccessful-search, diagnostic and scale evidence
remain distinct in generated/evidence.json.

Completed latency measurement
-----------------------------

scripts/benchmark_final_latency.py has a frozen local protocol at
runs/final_latency/protocol.json (SHA256
ddec2df361cc2e206ab4759ac6f8a477fbf712003ff56dfac8f8887bfdc4083f).
The original final study exited successfully and passed its full audit before
these commands were executed once::

    .venv/Scripts/python.exe -m scripts.benchmark_final_latency --run
    .venv/Scripts/python.exe -m scripts.audit_final_latency

The measured operation is the same CPU agent.act call used by the final
evaluation, including its public endgame wrappers, feature construction and
inference. Agent loading, reference construction, simulator transitions,
correctness checks and process inventories are excluded from call timing.
The four agents use the same 1,287 cached development states, with two warmup
passes and eight measured passes. Agent order rotates so each agent occupies
each ordering position twice. Reports separate a 1,024-state mixed sample,
64 unguarded states from each of four phases, and 32 guarded states. These
groups overlap; repeated timings are descriptive, not independent confidence
samples or a deployment-policy state distribution.

The runner rejects a live/unaudited final study and checks for project training
or evaluation workers at measured-pass boundaries. It never terminates other
processes and refuses to overwrite an existing timing run. Unrelated operating
system activity is not controlled. The independent auditor verifies cached
sampling, identities, legality, repeated action stability, phase/guard groups,
work counts and raw timing summaries; it does not independently remeasure the
wall clock or rerun inference. This task simulates no matches and consumes no
additional final deals.

All 41,184 measured calls completed and the independent audit passed. Mixed-state
mean/median/p95/p99 milliseconds are 1.585/1.479/2.036/2.302 for canonical seed
6710001, 1.595/1.504/2.034/2.286 for seed 6720001,
1.974/1.911/2.714/2.925 for RL-WP, and 1.926/1.834/2.688/2.901 for RL-Point.
The outcome audit SHA256 is
891c765dec23edec33538174433d5105dabc443ca2da1a4f91d4273c965eade8;
the timing audit SHA256 is
e3b48a44568597f3cc528396c3a700407054137af4c540e2983a3486434f34cc.
Never rerun these commands into the completed original directories.


Completed arbitration figure
----------------------------

The13-page extension revision includes the completed arbitration PPO comparison.
Its tracked generated/arbitration_changes.pdf is copied from the audited plot::

    .venv/Scripts/python.exe -m scripts.plot_extension_results --base runs/arbitration_study --study arbitration --output output/figures/arbitration_changes

The plot requires the complete passed study audit, independently recomputes raw
paired outcomes and records input/output hashes. generated/arbitration_evidence.json
binds the source plot receipt, protocol, raw outcomes, audit and copied figure.
The historical generated/evidence.json remains unchanged. Current manuscript QA
is runs/manuscript_clarity_revision_qa.json; paired-regression and static-control
match results remain pending. The historical archive is preserved separately.


Clarity and classification-prior-work revision
--------------------------------------------

The18 October revision keeps13 pages and expands the bibliography to18 cited
works. Abstract/Introduction lead with the measured adaptation result and the
identical-initial-greedy-policy comparison; the temperature/learning-rate study
is described as a joint intervention. The fixed historical final test is
explicitly separated from later development. Repeated defensive qualifications
were shortened while preserving opponent, budget and uncertainty scope.

Primary sources checked on18 October2026: Lagoudakis and Parr's ICML2003 paper
at https://cdn.aaai.org/ICML/2003/ICML03-057.pdf; Langford and Zadrozny's ICML2005
paper at https://icml.cc/Conferences/2005/proceedings/papers/060_Relating_LangfordZadrozny.pdf;
and Dimitrakakis and Lagoudakis's journal record at
https://link.springer.com/article/10.1007/s10994-008-5069-3. The last entry now uses
Machine Learning72:157--171 rather than an arXiv-only record.

Latest QA is runs/manuscript_clarity_revision_qa.json. PDF SHA256:
a6538307993938f84291ebe757a12e8dc2cc7f99d74523ac467302fbe548e56a.
All13 pages were visually inspected;18 fonts are embedded and18 citations resolve.
The26 historical and34 arbitration evidence hashes remain verified and unchanged.
The previous PDF is preserved as output/pdf/manuscript_arbitration_16ac69ab.pdf.
Paired-regression/static-control and data-scale comparisons remain ongoing; this
revision introduces no unverified stronger-model result.


Completed paired-supervision results revision
--------------------------------------------

The latest14-page manuscript includes the completed paired full-match comparison
and moves48-endpoint fitting diagnostics to an appendix. Regenerate its table and
identity ledger with::

    .venv/Scripts/python.exe -m scripts.export_paired_paper_evidence

The exporter recomputes raw paired outcomes and checks the completed audit, then
binds117 inputs including48 fitting checkpoints. The original26-input evidence
and34-input arbitration ledgers are retained. Latest QA is
runs/manuscript_completed_paired_qa.json. PDF SHA256:
075411f59901a0418417dc8608d5d6c23aa729051ec324ea0eeb315e0bfefd1f.
Seven changed pages were visually reviewed; the remaining seven renders match
earlier reviewed images byte-for-byte. All18 fonts are embedded,18 citations
resolve, and numerical/accounting/identity checks pass. The previous clarity PDF
is retained separately. Static-hybrid and opponent-preserving studies are still
running; their pending results are excluded from the manuscript's findings.


Completed static and continuation-target revision
------------------------------------------------

The current15-page manuscript adds a four-row static Point-continuation table,
with all effects recomputed from raw paired outcomes. It integrates completed
opponent-target/switch-budget diagnostics in the appendix and adds primary
DAgger/AggreVaTe positioning to Related Work. Regenerate the additional table
and59-input ledger with::

    .venv/Scripts/python.exe -m scripts.export_followup_paper_evidence

The historical26/34/117-input ledgers remain unchanged. New followup ledger SHA
is a72b8ebd49b69d6bed3bb7c78b519e864c5cc36de68880377522b12511b6967d.
Current PDF SHA c5d7c21532e318f11c66fb4932f454e17899f20c3a9970479c99c65eef39a88e.
All15 pages visually reviewed;18 embedded fonts,20 cited references, no TeX
box/reference/glyph warnings or page-bound defects. QA SHA
 e10ea1c7e7c26b294e9b421cdee298d842179429a95b5346c623b7c70efb1e74.
Previous14-page PDF preserved as output/pdf/manuscript_paired_075411f5.pdf;
original archive unchanged. Direct-response study outcomes are not yet included.

New primary records checked18 October2026:
https://proceedings.mlr.press/v15/ross11a.html and
https://arxiv.org/abs/1406.5979. They position iterative data collection and
cost-to-go learning relative to this project's fixed-source regression.


Main-text structure revision
----------------------------

The current PDF retains15 pages overall and20 references, but the conclusion
moves from page12 to10. Main-source whitespace tokens decrease4,925 to3,775
(about23%); this count includes TeX tokens and is an editorial measure, not an
experimental metric. Dedicated appendices now contain simulation/timing detail
and controller derivations. Repeated explanations are removed while every prior
numeric token, generated table input, equation and citation key remains present.

PDF SHA90f9d9f812f7e5656ec981664f50abb8edcc14d5af554edfd09fb7321587f17c.
QA runs/manuscript_structure_qa.json SHA9b817ce74800e74fd1dfed42995038fabb47a33b8127d3d19fbd7e6510122e0f:
changed pages1--14 visually reviewed; page15 is byte-identical to its prior
reviewed render;18 embedded fonts; all4 evidence ledgers and numeric/input/
reference preservation checks pass. No TeX box, missing-reference/glyph or
page-bound defects. Previous version: output/pdf/manuscript_followup_c5d7c215.pdf.
The active response study is still excluded from the reported outcomes.


Observer-conditioning revision
------------------------------

The title is now Adapting Published Koi-Koi Transformers: Match Wins and Critic
Conditioning. Section7 states the finite-game example; AppendixB derives the
score-function expectation, lambda threshold, controls and learner-decision tail
handling. The proof concerns one gradient coordinate before PPO clipping and
empirical normalization. Existing Koi-Koi match results continue to use the
original signed estimator. The new estimator's strength is not asserted here.

Baisero and Amato (AAMAS2022,44--52) and Lyu et al. (AAAI2022,9396--9404) were
checked against the primary proceedings/arXiv sources on2026-10-18. They establish
the related partial-observation critic-conditioning problem. The example is a
specific alternating-observer construction, not a new general gradient theorem.

Regenerate its tracked evidence with::

    .venv/Scripts/python.exe -m scripts.export_observer_paper_evidence

The exporter verifies pinned CPU/CUDA diagnostics, production/diagnostic source
identities and an independent Fraction calculation. It adds no simulation or
training. generated/observer_evidence.json SHA
5b5419a725ff72e01678b65fbfc273fc813bcea3fec8730cc7d2cb783a780623.
The preceding15-page PDF remains in
output/pdf/manuscript_structure_90f9d9f8.pdf. New PDF SHA
 eeff7b6d9d98595692b32408d3ebae66dc49b6b06b6b8d92baf070dc5a740ef7.
QA runs/manuscript_observer_qa.json SHA
7ffb1430d37ce5acd0c1c1063f1e5d5a08af956654d8eda5a56533137595fea1:
all17 pages reviewed, equations on8/12 inspected at full size,20 fonts embedded,
22 references resolved,191 abstract whitespace words,conclusion on11. All prior
numeric tokens and generated outcome tables remain,all5 evidence ledgers verify,
and no TeX box,missing-glyph or page-bound errors remain. The live original study
and seven guarded-launcher verification inputs were rechecked unchanged.


Shared-parameter descent revision
--------------------------------

The Abstract,Introduction,main theory section and conclusion now state the
stronger exact shared-parameter result:the expected signed-return update is a
strict local descent direction in the specified scalar policy family. AppendixB
retains the original independent-logit example and derives the scalar gradient,
negative directional derivative andlambda threshold. Existing empirical numbers
and cited prior work are retained. New Koi-Koi strength results are still pending.

The updated exporter verifies the retained scalar report, its source identities
and an independent Fraction projection of the original two-coordinate gradients.
It binds fresh CPU verification and the previously recorded CUDA calculation
without claiming a new CUDA run. Updated generated/observer_evidence.json SHA
f7472951c106e3a069e09ad89628fba75d85f82351738f92c64306a411338b6d.

PDF SHA6d7406c923befcf78d5d6a89b3351f0e51ed63d1ca0540cdc037a2fa62f4ca6b.
QA runs/manuscript_shared_observer_qa.json SHA
3c97e47e27e60ff86a2e90a3610234938119591dc1d58502c60070eb0d200dc5:all18 pages visually reviewed,
with pages1/8/12/13 inspected in detail;19 fonts embedded,22 references resolved,
199 abstract whitespace words,conclusion on11. Numeric tokens and all5 evidence
ledgers verify. Final build has no TeX box,reference,glyph or page-bound defects.
The preceding17-page PDF remains as output/pdf/manuscript_observer_eeff7b6d.pdf.
Both live experiment protocols and their29/51 bound inputs are unchanged.


Completed response comparison and observation terminology
--------------------------------------------------------

Section8.4 now reports the completed fixed-opponent response study:four trained
endpoints plus two parents,twelve matchups,120,000 matches on5,000 shared fresh
23M deal streams. The generated table shows response-minus-parent and
response-minus-selfplay win changes with95% intervals. None has a positive lower
bound; the main text also identifies the negative parent1/WP response-versus-
selfplay interval and the three negative point-margin intervals. All underlying
outcomes and contrasts remain in the bound evidence. No learner-return strength
result is claimed while that study continues.

The new exporter verifies the pinned independent audit and every source/input
identity,then checks the displayed contrasts against its independently
recomputed values. Regenerate with::

    .venv/Scripts/python.exe -m scripts.export_response_paper_evidence

Ledger generated/response_evidence.json SHA
dc4a0d1e42ce10285b4d53b55ceac0b9660fb25af4b2711395dcc4b3d972d106.
The observation terminology now consistently includes the acting player's own
hand and public state; it does not call the private own hand public information.
This follows reference.observe,native observe and the560-feature value input.
Public endgame assistance and publicly released opponents retain those terms.

PDF SHA99db5ce08838111024016e8800bcdfced949cb3f1e752dd14a80b5449042787b.
QA runs/manuscript_response_qa.json SHA
8b84da1e3e4b48017dd32e6e464f19a5406bc23f32e1eb28d101a4294690ee1b:all19 pages reviewed,
with table/proof pages11/13 inspected in detail;19 embedded fonts,22 references,
202 abstract whitespace words,conclusion on12. All prior numeric tokens and
all6 evidence ledgers verify; no TeX box,glyph,reference or page-bound defects.
Prior18-page PDF:output/pdf/manuscript_shared_6d7406c9.pdf. All51 live learner-study
inputs remain unchanged; the actual launcher session23277 remains live.


Finite-cutoff extension
-----------------------

The current21-page manuscript includes the exact nonterminal-cutoff extension
in Abstract,Introduction,Section7 and AppendixB.2. It separates internal
observer-bootstrap errors from the final boundary term using the standard GAE
expansion. In the existing scalar game,lambda1 at a one-step cutoff still gives
-19701/160000 instead of true3/800. Exact root-history conditioning removes the
score covariance;finite Koi-Koi features are not asserted to satisfy that
history condition. Existing empirical results remain unchanged.

Regenerate the exact constants and bound ledger with::

    .venv/Scripts/python.exe -m scripts.export_observer_trace_evidence

The exporter verifies the pinned6048-case rational report and independently
recomputes all16 toy expectations from the payoff matrix. Ledger
``generated/observer_trace_evidence.json`` SHA
 a926388d445b82216f345ae7a96e94c0341353d53b8bb07779cf3386b183d0ac.
The original observer_evidence exporter and ledger remain unchanged.

Final PDF SHA
 cddde4e37e0eee599dd18052ce7f2513b1a0ad29f4846516aeb4a10adea8cc7f.
QA ``runs/manuscript_trace_qa.json`` SHA
 22779a344afb31d3c0034390fdbdf5c55f5f663fc2c5e56fd69a56230acdbbd6.
All21 pages reviewed;24 references and20 embedded fonts;Abstract180 words.
The condensed conclusion fits page10;the displayed cutoff derivation is on13.
All previous numerical tokens and8 evidence ledgers verify;no TeX box,glyph,
reference or page-bound defect remains. Previous20-page PDF is preserved as
``output/pdf/manuscript_learner_79dd722c.pdf``. The ongoing observer study remains
outside this revision until its fixed evaluation and audits complete.


Opening clarity revision
------------------------

Abstract and Introduction now lead with the measured match-win adaptation and
the precise critic-information counterexample. The text names the controls
instead of relying on an unexplained parent-and-response criterion,clarifies
that deployment retains the actor architecture,and shortens the adaptation
summary. The primary negative learner result and secondary Point/lambda1
result are both retained. No new experiment outcome is introduced.

PDF SHA8a405247ec007d4d929e82426d931aae70b1d7a404607bc79d405b615be84175.
QA runs/manuscript_opening_qa.json SHA
7f0f8610d3dd5f4b73c95856fc74272712fe08bada0f0862b250d842073988cc.
All21 pages rendered;the changed first2 pages were visually reviewed,and
pages3--21 are pixel-identical to the previous reviewed PDF. Abstract174 words,
24 references,20 embedded fonts. All prior numeric tokens,8 evidence ledgers
and their392 unique inputs/outputs verify. Related Work and all following
source text are unchanged. No TeX box,missing-glyph,reference or bounds defects.
Previous PDF:output/pdf/manuscript_trace_cddde4e3.pdf.


Temporal-abstraction attribution
--------------------------------

Related Work now cites Sutton,Precup and Singh (Artificial Intelligence112,
181--211,1999) for options and semi-Markov temporal abstraction. Primary PDF
pages1/8/10 were visually checked. The learner-decision connection is stated
as temporal organization;the analysis concerns critic information at interval
boundaries. Finite Koi-Koi observations are not claimed to be sufficient states.

PDF SHA6a10c91daec57ab714efb15d752e6193b2d42984c6b31b380f5225c63c782311.
QA runs/manuscript_temporal_related_qa.json SHA
81c4ed7ad683a650d84936bb67b2c22c401149455e53d137a3932019dd87a6bc.
All21 pages rendered and reviewed;changed pages2/3/21,other18 pixel-identical
to prior review. New paragraph/reference inspected at full size. Abstract174
words,25 cited references,20 embedded fonts;8 ledgers/392 input-output files
verify. No numerical or empirical change. Previous PDF preserved as
output/pdf/manuscript_opening_8a405247.pdf.


Fixed-observer results and public-seat diagnosis
------------------------------------------------

The24-page revision adds the completed fixed-observer self-play comparison
(Section 9),its primary win table,and point/historical controls (AppendixG).
All8 primary intervals include zero. The same section reports the diagnosed
public-seat ambiguity and exact actor-input reconstruction on1,024 probes.
The separate public/masked strength comparison is still running.

Regenerate the completed observer tables and full ledger with::

    .venv/Scripts/python.exe -m scripts.export_observer_selfplay_paper_evidence --audit-sha256 759889ec9ffd6f98f56f8f292f00e590e44dec07bbe489e4657f56aecef78e70

Ledger generated/observer_selfplay_evidence.json SHA
067d13a2f0ad3bd6d0f9a18146b3746dbb5c4bbe0ffabcfd2726beea2a499b48.
PDF SHA f606acbb895b313bc592a61245fd4cef2dfbb123aeefb005a2be771ad1ddb961.
QA runs/manuscript_observer_selfplay_qa.json SHA
74a4588ffcb3b62433b903faad0cdc719677d7bd8a67ad2f94ce888db0b29791.
All24 pages rendered/reviewed;25 references,20 embedded fonts,202-word Abstract.
Nine evidence ledgers and the seat-diagnostic reports verify489 unique files.
Prior numerical tokens and Related Work remain;no TeX box,glyph,reference or
page-bound defects. Prior PDF:output/pdf/manuscript_temporal_6a10c91d.pdf.
