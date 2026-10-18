Natural final-round search: completed pilot
==========================================

The fixed study in FINAL_ROUND_STUDY.rst completed and passed independent
source/candidate replay and arithmetic audits. All 128 source matches and
384 separate single-phase candidate matches share 32 distinct deals.
There are 172 natural roots: 116 discard, 33 hand-capture and 23 draw-capture.
Twelve source matches end before round eight; they remain in the comparisons.

Five accepted changes are all discard choices: four for parent 1 and one
for parent 2. Parent 2 versus RL-Point recovers one loss as a win among 32
matches: win change +3.125 percentage points, whole-pair bootstrap 95% interval
[0, 9.375]; point-margin change +0.4375 [0, 1.3125]. Parent 1 versus RL-WP
has no win change and point-margin change +0.0625 [0, 0.1875]. Every other
win/point contrast is zero. All twelve win intervals include zero. This is
one recovered win, not replicated strength, population equivalence or a
reason to promote a model. The three phase candidates are separate policies,
not a combined three-intervention policy.

The recovered match is parent 2, Point, learner seat 0, seed 87110103:
first eligible discard 35 -> 28 changes final points [29,31] -> [36,24].
Independent uniform-reference mean advantage is 0.03173828125, with the
unadjusted 2.5-SE lower margin 0.01867638937661075. References use the same
approximate hidden-world mixture, not an opponent-history posterior.
One accepted parent-1 change has a nonpositive independent reference margin;
all accepted and positive-reference choices are retained in summary.json.

Source work is 14,407 decisions / 985 round endings / 128 completed matches.
Candidate work is 43,212 decisions / 2,955 round endings / 384 matches.
Search and reference work is 115,163,667 decisions / 8,916,992 round endings /
8,916,992 terminal branch completions, including confirmation. Independent
replay repeats source/candidate work; it adds no independent deals. Branches
are conditional final-round continuations, not new full matches.

Observed act time totals 264.441 seconds and 225.572 seconds for the two
parents' discard roots. All five accepted changes take 4.93--7.09 seconds.
This is recorded experiment timing, not an isolated deployment benchmark.
No new model, dealer rule, manuscript or reserved 101M evaluation is changed.

Artifacts and identities
------------------------

All artifacts are under runs/final_round_search_study/:

* protocol.json: 867e2caa5e89e2516a94e46aea522ba63e9f5404d4f58f5639f3f95a5a6ce7d8
* audit.json: 4df01f66e968623a6d74507c2594cc5f3d1c8dd71482b6212971e6992737b655
* independent_audit.json: 4f7dcc2f26bb962c1398281e4d0456975f1650e0b5a90a010266f95f66560fc9
* summary.json: 860125d0acf516c5ef07b5e8e76c5b3296ad2d527054168b83d639cb3ba46078

All 1,591 independent-audit input identities were reverified at completion.
Producer 78135, audit queue 27155 and summary queue 91434 all exit zero.
Do not restart or poll these historical sessions. Completion receipt:
runs/final_round_search_completion.json.

Commands: scripts/check_final_round_search.py,
scripts/audit_final_round_search.py, scripts/summarize_final_round_search.py;
the exact frozen invocations and inputs remain in protocol and process records.

Next question
-------------

The observed opportunity is final-round discard choice. A separate proposed
follow-up fixes rounds 1--7 and trains from naturally reached round-eight
states to terminal match outcomes. This is not an extension of the completed
pilot. It needs its own implementation, controls and prospective evaluation.
