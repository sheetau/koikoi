Exporting the complete auxiliary comparison
===========================================

The result exporter is implemented and tested. Main learning is still running;
no main result export exists. This document describes presentation, not outcomes.

scripts/export_score_auxiliary_results.py exports all six primary simultaneous
win contrasts and all 120 secondary absolute/paired win and point coordinates.
It determines qualification from the unrounded simultaneous lower bounds,
including score-minus-outcome-only at both seeds. A point gain, a favorable
marginal interval or one successful seed cannot replace a missing primary gain.
Degenerate primary intervals remain undefined and cannot support qualification.

The text uses percentages for absolute win scores and percentage points for win
differences. JSON preserves the original fractions and unrounded estimates.
PPO and auxiliary decisions, rounds and matches are listed separately for all
six arms, together with auxiliary retained rows and example presentations.
The JSON also preserves every training counter, including optimizer call counts.
Warmup enters PPO totals once. Capability and repeated verification work remain
in the experiment ledger rather than being added to unique main-training totals.

The exporter requires the completed four-stage queue, all sixteen evaluation
cells, the complete six-arm training audit and the exact final evaluation audit
digest. It checks bound artifacts before creating its output directory. It
refuses incomplete, changed or capability evidence and existing exports.
This reuses the already-computed statistics; it adds no inference or matches.

After queue completion, inspect the passed evaluation audit and supply its
SHA256 to the following command::

    .venv/Scripts/python.exe -m scripts.export_score_auxiliary_results --output runs/score_auxiliary_results --audit-sha256 <completed-audit-sha256>

The default study and queue are runs/score_auxiliary_evaluation and
runs/score_auxiliary_completion. Outputs are results.rst and evidence.json.
The fixed mechanism diagnostic is separate. A passing development criterion
identifies the prospectively named seed1_score for a separately prepared
confirmation; export does not promote a model or consume reserved 101M deals.

Validation
----------

Ten focused tests pass, zero skips, 1.59s. All values in these tests are synthetic
fixtures. Tests cover complete export and non-overwrite, percentage-point units,
all six primary and 120 secondary coordinates, distinct work streams, missing
completion/cells, failure receipts, changed raw results or training audit, wrong
audit/queue identities, missing outcome-only improvement, zero variance,
nonfinite results and capability rejection. Ruff and diff checks pass.
The final source change after pytest only restores ordinary prose spacing.
No policy inference, simulation or learning is performed by these tests.

Digests:

* Exporter: 11f04bd13030f4f158ad79ae16fa628c7509190d10aa53af07bb38919e4298e2.
* Tests: 92892ed97e39215b3df962c7966781717d1a67fa27812f947a7c37106cd6117c.
* Test XML: c3efbe9a1c654da23ad6eefc3183591e2d44b7bfa211d550d6655518bdda6508.

End-to-end export of actual main results remains pending. Its tables and the
mechanism report will support the eventual manuscript revision; no new strength
claim has yet been added to the paper.
