Manuscript working files
========================

This is an English research manuscript, not a submitted or published paper.
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
