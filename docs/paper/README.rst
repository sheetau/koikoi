Manuscript working files
========================

This is an English research manuscript, not a submitted or published paper.

Current revision: Abstract and Introduction now lead with the measured
contribution; Related Work contains12 primary-source references spanning
Koi-Koi, self-play, policy composition, information-set search and evaluation.
The public expert arbitration extension is explicitly marked as under evaluation.
Its outcomes are not included in the completed adaptation results. The previous
9-page final-results manuscript and QA remain preserved in the existing archive;
the current working PDF is a separate revision.
The final two-model/two-comparator evaluation and isolated latency measurement
are complete and independently audited. Both selected models clear the fixed
RL-WP win criterion; the separate RL-Point criterion does not replicate, and
both models have negative expected point differences against RL-Point.
Author information and venue formatting remain decisions for submission.

From the repository root, regenerate the numerical tables and evidence ledger::

    .venv/Scripts/python.exe -m scripts.export_paper_evidence

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
