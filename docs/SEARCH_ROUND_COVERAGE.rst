Round coverage of completed Transformer search diagnostics
=========================================================

A retrospective check reads the exact roots retained by two completed
diagnostics. It verifies their original input and root hashes, then counts
rounds without rerunning policies, selecting new roots or inspecting new deals.

====================== ====== =======================
Diagnostic             Parent Roots by round 1 to 8
====================== ====== =======================
All card-action phases  1      4, 3, 1, 0, 1, 2, 1, 0
All card-action phases  2      6, 3, 0, 2, 0, 1, 0, 0
Unsolved koi decisions  1      5, 7, 4, 3, 4, 3, 6, 0
Unsolved koi decisions  2      4, 8, 3, 8, 3, 4, 2, 0
====================== ====== =======================

Neither corpus contains a final-round root. The card diagnostic selected the
first two unforced roots per card phase and seat; it did not stratify rounds.
The koi diagnostic selected unsolved decisions, so public assistance also
affects which decisions are eligible. These counts describe these two
corpora, not every previous search experiment. Multiple roots can come from
one match; the 88 roots are not 88 independent matches.

The absence identifies a coverage gap for final-round card choices. Such
search needs no subsequent-round simulation. It does not establish larger
action gains, better estimates or a stronger policy. The running fixed64
midround study deliberately uses first-round roots and remains unchanged.
After its complete results and costs are available, final-round card search
can be considered under a separate prospective protocol. Any history-based
extension must state how it handles information about opponent identity from
earlier rounds; simply calling a first-round prior a full-match posterior
would be incorrect.

Reproduction::

  .venv/Scripts/python.exe -m scripts.audit_search_round_coverage

Output: runs/search_round_coverage.json, SHA256
5f3bb5a3d3684116aa1987f4c0affe648efcfb39477ce7614742589bd9073613.
No new simulator decisions, training, actual matches or strength evidence.
