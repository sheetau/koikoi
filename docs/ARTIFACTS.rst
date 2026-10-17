Local research evidence archive
===============================

The archive is prepared by scripts/package_research_artifacts.py. It is a local
deliverable, not an upload or publication. At present the final comparison and
latency measurement are unfinished, so no final archive has been built.

Inspect the required files without loading models or creating an archive::

    .venv/Scripts/python.exe -m scripts.package_research_artifacts --plan

After final evaluation, latency auditing, manuscript revision and PDF review::

    .venv/Scripts/python.exe -m scripts.package_research_artifacts --build

The builder requires terminal successful study status, complete independent
audits, refreshed final manuscript evidence, a final PDF QA record, and a clean
committed source snapshot. The final PDF QA record is
runs/manuscript_final_qa.json and must bind the current PDF, manuscript source,
and generated evidence hashes with status=passed and pending_final_results=false.
The current draft QA file does not satisfy this gate.

Contents and boundaries
-----------------------

The selected archive contains the committed source snapshot, both selected
model checkpoints, all four cold/warm endpoints and their critic-warmup
snapshots, training configurations/metrics, the initial comparator checkpoint,
raw development/final outcomes and audits, frozen protocols, comparator source
files/weights/licenses, the cached timing corpus and its measured raw records,
the separate100M checkpoint/budget evidence, and the final manuscript sources
and PDF. Historical JSON files bound by the original final seed scan are
preserved. It does not include every exploratory checkpoint, the Python
environment, compiled native libraries, compiler/font caches, Git internals,
unrelated upstream projects, or the original journal PDF/artwork.

Each archive member has a SHA256 and byte count in ARTIFACT_MANIFEST.json.
Files stream into a temporary ZIP and are verified before the final name is
assigned. A failed partial ZIP is retained; existing archives are never
silently replaced. Default destination is dist/koikoi_research_artifacts.zip,
which remains ignored by Git. The separate local receipt
runs/research_artifact_bundle.json records the completed archive's SHA256.
Record that receipt/hash in the research ledger before transferring the file.

Portable verification
---------------------

The verifier uses Python and NumPy, without PyTorch, CUDA, native compilation,
model deserialization, or extraction of archive files::

    python scripts/package_research_artifacts.py --verify path/to/archive.zip

Run it from a trusted copy of the source. It checks exact archive membership,
file hashes/sizes, frozen protocol bindings, production-source identity, and
independently recalculates the four final pair means, percentile-bootstrap
intervals, and separate primary/secondary replication criteria. It checks
each raw seed and seat-relative score/point sign. Whole duplicate pairs are
resampled with5000 replicates and seed99. It does not independently replay
training, inference, trajectories, or timings, and a matching self-contained
manifest alone is not an authenticity signature; compare the externally
recorded archive digest as well.

Original protocols are preserved byte-for-byte. Their original Windows
absolute paths are mapped under the recorded original_root to relative ZIP
members for verification. Paths outside that root, absolute archive members,
parent traversal and case-colliding entries are rejected. This mapping does
not rewrite the experimental record or claim that every original audit CLI
is relocation-aware.

Reproducing execution
--------------------

For actual inference/training, use a fresh checkout of the project at the
revision recorded in ARTIFACT_MANIFEST.json. Follow README.md and
requirements-windows.lock.txt to prepare the recorded runtime and native
compiler. Fetch the full pinned upstream checkout through
python -m koikoi.upstream --fetch before restoring archived run artifacts.
The external/ source subset inside the archive has no Git metadata and is
not by itself a replacement for the clean pinned upstream checkout required
by the production loader. Compare its file hashes with the fetched checkout.
Network access is therefore normally needed for execution dependencies;
portable archive verification itself does not need network access.

Copy the archived runs/ artifacts into that fresh workspace without changing
their bytes. The selected canonical model is
runs/cold_ppo_screen/seed6710001_warm/update_00000520.pt; the second frozen
model is seed6720001_warm at the same update. Their exact identities and
settings are in the preserved final protocol. Training the selected recipe
anew uses the existing configuration and a new output directory, for example::

    python -m koikoi.published_train --config runs/cold_ppo_screen/seed6710001_warm/config.json --run runs/reproduced_warm_6710001 --updates 520

That command performs a new experiment; it is not automatically executed by
the archive tool and is not a continuation of the measured model. Numerical
identity across different hardware/software is not promised. Do not tune a
reproduction against the already consumed final deals or silently replace
the archived reports. New strength claims require a separately specified
evaluation. Full historical-study replay may require additional exploratory
artifacts beyond this selected archive.

Keep THIRD_PARTY_NOTICES.md and the upstream LICENSE with the included source
and weights. The archive adds no new license grant or author attribution of
its own. Submission and external artifact publication remain separate steps.
