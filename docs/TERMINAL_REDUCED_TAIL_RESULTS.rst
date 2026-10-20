Terminal aggregation: equivalence and paired cost
=================================================

Terminal aggregation preserves every rational value, selected action and complete
contingent policy at all 18 completed roots of the five-draw capability check.
Both methods also make identical physical transitions and policy-adapter calls.
The comparison covers 36 fixed/optimized policies per method, with no new source
match or strength measurement.

.. list-table:: Complete paired comparison
   :header-rows: 1

   * - Quantity
     - Deferred control
     - Terminal aggregation
   * - Planning seconds, total
     - 99.440871
     - 63.783373
   * - Largest single-plan time, seconds
     - 10.236094
     - 8.502455
   * - Physical decisions
     - 5,513,147
     - 5,513,147
   * - Physical terminal branches
     - 4,856,793
     - 4,856,793
   * - Learner decision nodes
     - 89,673
     - 89,673
   * - Grouped observation nodes
     - 5,023,163, including terminal groups
     - 339,711 live groups

Observed total planning time falls by 35.8580%. Each method runs first at nine
roots and uses a separate fresh policy cache. Within each method, fixed planning
precedes optimized planning. Actors use strict singleton CPU inference with two
threads. One paired execution per root establishes this measured implementation
comparison, not a precise deployment latency distribution.

Each method executes the same 5,513,147 transitions; together the new comparison
adds 11,026,294 physical transitions and 9,713,586 terminal branches. The reduced
method avoids grouping terminal observations and directly sums their weighted
outcomes. The earlier independent Python policy replay verified the identical
saved policies and values; that bound evidence is reused here. The new both-seat
unit tests separately exercise independent replay and computation-cap failures.

Both methods have a 2,000,000-transition cap per plan. The control's all-node cap
is 2,000,000 and the reduced method's live-node cap is 500,000. No cap binds on
these completed solutions. The 15 roots refused by the earlier study are outside
this comparison. Their recoverability and any additional gain remain unmeasured.

The five-draw strength evidence is unchanged: one conditional gain per parent,
with no realized match-win change. See EARLIER_BELIEF_TAIL_RESULTS.rst. This
computational improvement does not itself establish a stronger deployed agent
or algorithmic priority.

Command: ``python -m scripts.check_terminal_reduced_tail``.
Session99274 / PID68548 exited0. Implementation/protocol commit:b42d720.
Output ``runs/terminal_reduced_tail_equivalence`` is closed and contains every
root's order, timing, work and adapter-call counts.
Protocol SHA256:
``857196060cd8f3670676533f2883c58806940adbda86c4f0e4d72aa3d2de43e6``.
Complete SHA256:
``bca0c2086ba5a520c763de362354dc87cbf20d6a623d452ec68b4bee838cc835``.
Input bindings, method-order balance and aggregate counters were checked after
completion. No training, reserved101M use, canonical promotion or PDF rebuild.
