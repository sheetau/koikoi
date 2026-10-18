Fixed64 terminal pilot results
==============================

The complete pilot did not change any actual match outcome or final point
margin. All 56 eligible fifth-discard roots and the eight cases without that
opportunity are retained. The 64 parent matches and 128 candidate matches
share 16 distinct deals; they are not 192 independent deals.

Uniform search proposed three changes and confirmed one, for parent2 against
WP. History-conditioned search proposed two changes and confirmed none.
The confirmed action has a positive independent conditional-reference mean
of 0.0147705 draw-adjusted win probability (2.5-SE lower margin 0.00553598).
Its actual match nevertheless ended with exactly the parent's final points.
All twelve paired pilot win/point contrasts are zero. The degenerate empirical
bootstrap intervals describe the observed pairs, not a population equivalence
certificate. Eight alternative actions have positive unadjusted 2.5-SE
reference margins; these are model-conditional diagnostics, not eight wins.

History conditioning lowers the measured all-action error against its own
conditional terminal reference for both retained parents:

========== ============= ============= ============= =============
Parent     Uniform RMSE  History RMSE  Uniform time  History time
========== ============= ============= ============= =============
1          0.017779494   0.011938428   531.789s      1570.534s
2          0.016888303   0.012876379   528.384s      1539.475s
========== ============= ============= ============= =============

Each row covers 28 roots and 84 alternatives. Times sum the act calls; they
are not isolated benchmarks. Four short CPU capability-test runs overlapped
the pilot (13.14s, 13.83s, 2.35s and 2.29s). Uniform error also includes its
omission of historical evidence. A more accurate conditional estimate alone
did not produce a better deployed policy in this pilot.

Accounting and verification
---------------------------

* Source play: 7,242 decisions, 494 round endings, 64 matches.
* Candidate play: 14,485 decisions, 988 round endings, 128 matches.
* Search and reference futures: 222,670,284 decisions, 16,127,337 round endings,
  2,084,864 terminal branch completions. Confirmation work is already included.
* Exact history pruning executed 678,995 model calls. The corresponding
  unpruned full-replay count is 1,520,324, a derived comparison, not another
  executed workload.

The producer and independent source/trajectory/statistics audit pass.
The audit verifies all actual choices, histories, sampling and arithmetic;
it does not rerun every support-likelihood inference or future neural branch.
Fixed64 continuation trace checks provide separate evidence for the latter
implementation. See runs/fixed_midround_terminal_study/summary.json and
independent_audit.json for identities and individual results.

The first independent audit failed on case38's saved model marginal:
1.000000000000034 versus the mathematical count ratio 1.0. The former sums
2,324 individually normalized float64 masses. The old check allowed only
2e-14 absolute error. Its source and failure receipts are preserved. The
corrected check independently constructs equal-prior pair masses and requires
exact equality of the saved reduction; it does not widen the shared tolerance.
Two regression tests pass, including rejection of a one-ULP modification.
The entire audit then passes. The failed attempt additionally replayed the
64 source matches and 962,007 native history decisions before stopping; no
candidate-match replay occurred in that failed attempt.

Decision
--------

Close this first-round pilot without promotion, threshold adjustment or
additional draws from its deal pool. Final-round card search is the next
candidate because previous diagnostic coverage omitted that round and its
continuations contain no later-round rollout. The separate adapter passes
three CPU and three CUDA tests; the natural-match intervention evaluator
passes four CPU tests. Those tests establish execution, not strength.
A new prospective protocol must fix its source deals, root selection,
missing-opportunity handling, search budget and actual-match comparison before
collecting outcomes. No final-round strength study has started at this milestone.
The canonical playable model, dealer rule and reserved101M deals are unchanged.
