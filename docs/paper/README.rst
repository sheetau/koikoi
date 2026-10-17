Manuscript working files
========================

This is an English research draft, not a submission-ready or published paper.
The final two-model/two-comparator evaluation is running. The draft describes
completed development evidence and prospectively fixed final criteria without
transcribing interim final scores. Update its status, abstract, results and
conclusion only after the complete independent final audit.

From the repository root, regenerate the numerical tables and evidence ledger::

    .venv/Scripts/python.exe -m scripts.export_paper_evidence

The exporter verifies fixed input hashes and the hashes of development reports.
If the final independent audit is absent, the final table remains empty and
the ledger explicitly says draft_final_results_pending. An existing incomplete
or invalid final audit is an error, not a reason to use a provisional summary.
This exporter does not replace the experimental auditors. The tracked generated
files permit inspecting the draft without the ignored local run directory.

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

Before submission, finish the frozen final evaluation and its raw-record audit,
measure isolated decision latency, revise all pending-result passages, complete
the author and venue information, and assemble the licensed local artifact
release. Code is public, but local model files and run artifacts are currently
ignored and are not an archival release. Submission and external artifact
publication are separate decisions.

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
