Deferred chance expansion for exact tails
========================================

The eager three-draw capability check found one conditional improvement among
sixteen eligible natural roots. Reaching earlier decisions requires representing
larger tails without first copying every possible unused stock prefix.
``scripts/deferred_belief_tail.py`` keeps one initial row per compatible
hand/model and expands chance only when a simulator transition actually draws
a stock card. It preserves exact observation-contingent policy optimization.

Mass conservation
-----------------

For n remaining stock cards and d possible future draws, assign a compatible
hand/model row mass prior(model)*n!/(n-d)!. The eight unused stock cards give
n-d=8. This integer represents all equally likely ordered d-card prefixes.
No stock order is committed by its canonical representative.

At a transition without a draw, retain the row and mass. At a draw, create
one successor per remaining stock card and divide the row's mass by n.
The resulting integer is prior(model)*(n-1)!/(n-d)!. Each successor has the
same remaining unused-suffix multiplicity as direct prefix enumeration.
If the match terminates before d draws, its unexpanded mass contributes to
the terminal value. No unseen suffix is materialized or silently dropped.

Induction over transitions equates the mass of every visible history with
the eager prefix representation. Therefore the two planners give identical
action values, maximizing actions and complete contingent policies when they
use the same deterministic policies and tie rule. Both enforce one learner
action per own/public history. Stock sets and hypothesized opponent hands
remain internal simulator data.

The draw trigger covers discard without a two-card capture choice, and
hand-capture selection. All actual captures, yaku, scoring, legal actions and
termination remain in the existing engines. Every new native transition in
the verification runs is compared with the independent Python engine.
No production game rule or policy adapter changes.

Accounting and limits
---------------------

Physical simulator rows count executed transitions. Represented-prefix counters
sum integer mass across those transitions; they equal the eager branch counters
for a single model with unit prior. For nonunit prior they are weighted work,
not additional actual simulation or independent samples. Report both separately.
Observation-node counts remain the same as in eager enumeration.

Public hand/draw caps, a frontier-row cap, node budget and depth limit bound
the computation. A limit raises an explicit refusal. Exponential information
tree growth remains; deferred enumeration alone does not solve arbitrary
full-match or game-theoretic optimization. This is a representation improvement
to established finite POMDP planning, not a new general planning algorithm.

Verification
------------

Three tests pass, no skips, in 94.47 seconds. Both-seat three-draw fixtures use
nonunit prior mass and compare every rational root value, selected action,
entire root-policy map and weighted branch count with the eager implementation.
All deferred native transitions match the Python oracle. Hidden root/RNG
invariance, unchanged inputs, incomplete support and computation caps are checked.

A four-draw fixture independently constructs the complete eager prefix space.
Its fixed and optimized policies and every rational action value match deferred
enumeration. Differential checks exercise all four decision phases. This test
validates the extra draw; it is not a natural-root strength measurement.

The next check is fixed to all sixteen completed roots of the closed capability
corpus, with both fixed and optimized continuations. It loads their already
bound support tables and eager policies, constructs deferred hypotheses, and
uses fresh strict policy calls. Every deferred native transition is replayed
with Python; all root policies/values and represented branch counts must equal
the saved eager results. Initial rows, frontier maximum, actual/represented
work and verification overhead are reported. No roots are selected by gains,
no new source matches are added, and this creates no new strength sample.

Command, using a fresh output directory::

    .venv/Scripts/python.exe -m scripts.check_deferred_belief_tail

The fixed default output is ``runs/deferred_belief_tail_equivalence``. Preserve
the closed eager implementation and its input bindings. No earlier-tail
natural-root evaluation is specified by this numerical-equivalence check;
that evaluation will use a separate fixed protocol.
