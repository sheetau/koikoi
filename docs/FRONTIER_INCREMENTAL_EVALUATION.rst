Prospective incremental evaluation of the selected frontier controller
====================================================================

Compare the selected next-discard frontier hybrid with the existing symbolic
controller on 4,096 fresh independent source matches: 1,024 for each combination
of the two retained parents and WP/Point opponents. The primary question is
whether the added fallback improves expected terminal win score for both
parents under a fixed equal-opponent mixture. Report effect sizes as well as
the decision of this test.

Selection and comparison
------------------------

The old controller runs its fixed and, when applicable, optimized phases under
the original two-million-transition caps. The hybrid preserves every completed
old selection. Only parent_unvalued invokes the new fallback, using the skipped
second-phase allowance. Its public boundary is the next learner discard,
computed from the root turn. Prefix and all local attempts share that second
allowance. All budgets and entry rules are fixed by selected_frontier_controller.py
and frontier_incremental_design.py before source collection.

The comparison is hybrid minus old controller, not hybrid minus the neural
parent. When the fallback is not invoked, both algorithms execute the same
program and the incremental value is exactly zero. No absolute old-policy
value oracle is required to establish that equality. When the fallback is
invoked, the old controller executes the neural parent. Independent Python
prefix reconstruction and full terminal replay of both policies for every
adopted local patch give the exact incremental value through the verified
reach-weighted decomposition. Every source contributes to the denominator,
including missing roots, support refusals, no-gain cases and identical programs.

Fresh source law
----------------

Use the previously verified logged source harness: an independent uniform
learner seat, initial dealer 0, and independent Fisher--Yates permutations
rejected only by the canonical valid-deal rule. Persist the seat and each
accepted deck before use. The paper8 rules and parent/opponent checkpoint and
inference identities are unchanged. Parent play before the first eligible
identified-model five-draw root makes the source prefix common to both arms.

No development source, saved deal or old cohort row becomes a new sample.
Chance duplicates arising independently are valid IID observations and are not
resampled. The source audit checks every recorded transition, public activation,
actor action, accepted-deck ledger and posterior construction. The preserved
8,192-source cohort stays closed. A source, support, program or audit failure
stops the new study's inference and preserves its artifacts; it never selects
a replacement source. Run exactly the prescribed cases, without outcome-driven
extension, early stopping, threshold changes or policy changes.

Statistical reduction
---------------------

For each source, let Delta be the independently verified hybrid-minus-old
conditional expected win score, in [0,1]. Identical/fallback cases contribute
zero. Under the matching source law, averaging Delta estimates the full-match
expected incremental gain. Thresholds are fixed at::

    1/1000000, 1/100000, 1/10000, 1/1000, 1/100, 1/10, 1/2.

Apply the existing exact binomial-tail lower-bound routine. Allocate family
error 1/20 equally to four populations and then seven thresholds, yielding
1/560 per threshold; round each probability lower bound downward on a 32-bit
dyadic grid. Integrating the resulting survival lower bounds gives simultaneous
95% lower bounds on all four means. The two fixed equal-opponent mixtures
inherit simultaneous bounds by averaging their population bounds.

The primary criterion is that both parent-mixture lower bounds are strictly
positive. Report all four population means/bounds and whether their individual
lower bounds are positive, even when the joint primary criterion passes.
Report paired realized score differences separately as descriptive outcomes;
the conditional-mean interval is not an interval for the realized sample mean.

This incremental contrast does not retest or relax the old requirement that
both controller-versus-parent mixture lower bounds exceed 0.1 percentage points.
That earlier joint practical-gain criterion failed and remains reported as such.
A positive new increment alone establishes neither a large practical effect,
absolute superiority over every opponent nor an equilibrium guarantee.

Execution and accounting
------------------------

Before sampling, require complete method/integration reviews and a verified
selected-controller pipeline, then freeze code, tests, runtime, checkpoints,
this design and the ordered case list. Use four workers with fresh processes
per 16-source chunk. Persist complete programs before oracle or actual-tail
execution; record all missing/refused sources rather than excluding them.

Keep source matches/rounds/decisions, source replay, support reconstruction,
original fixed/optimized planning, prefix construction, local proposal/reference
work, independent prefix replay, local value oracles, actual old/hybrid tails
and recorded-action replay in separate ledgers. Node/task counts retain their
different meanings. Never label reused value evidence as newly executed work.
The final summary is available only after all 256 chunks pass independent audit.
