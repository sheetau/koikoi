Partial-action bound recovery results
====================================

All five refusals were attempted: 0 complete fixed+optimized comparisons and 5 remaining refusals. No source matches or history-support decisions were added.
Limits remained500k live groups/2M physical transitions per plan,
4096 frontier rows/depth32. Support worlds were reconstructed exactly
from the bound profile artifacts; policy caches started fresh.

.. list-table:: Every attempt
   :header-rows: 1

   * - Source
     - Status
     - Interrupted stage
     - Parent Q
     - Completed fixed Q
     - Optimized Q
     - Reason
   * - 0
     - budget_refused
     - optimized
     - 1711/2640
     - 161723/240240
     - -
     - Physical transition cap exceeded
   * - 4
     - budget_refused
     - fixed
     - -
     - -
     - -
     - Physical transition cap exceeded
   * - 20
     - budget_refused
     - fixed
     - -
     - -
     - -
     - Physical transition cap exceeded
   * - 36
     - budget_refused
     - optimized
     - 0
     - 0
     - -
     - Physical transition cap exceeded
   * - 52
     - budget_refused
     - fixed
     - -
     - -
     - -
     - Physical transition cap exceeded

Completed fixed plans remain valid even when subsequent optimization
is refused. Their gains are feasible model-conditional opportunities,
not completed fixed-versus-optimized comparisons or actual win gains.
The protocol assesses actual tails only after both plans complete.

.. list-table:: Completed fixed-policy opportunities
   :header-rows: 1

   * - Source
     - Exact fixed-minus-parent Q
   * - 0
     - 3011/120120
   * - 36
     - 0

.. list-table:: Additional work
   :header-rows: 1

   * - Category
     - Counter
     - Count
   * - completed_plans
     - decisions
     - 3725271
   * - completed_plans
     - round_endings
     - 3303427
   * - completed_plans
     - completions
     - 3303427
   * - completed_plans
     - live_observation_nodes
     - 357711
   * - completed_plans
     - own_decision_nodes
     - 306318
   * - completed_plans
     - chance_rows_added
     - 3303477
   * - completed_plans
     - represented_decisions
     - 52445376
   * - completed_plans
     - represented_round_endings
     - 11200860
   * - completed_plans
     - represented_completions
     - 11200860
   * - completed_plans
     - terminal_rows_valued
     - 3303427
   * - completed_plans
     - terminal_mass_valued
     - 11200860
   * - completed_plans
     - ceiling_cutoffs
     - 0
   * - completed_plans
     - pruned_actions
     - 0
   * - completed_plans
     - continuation_cutoffs
     - 1
   * - completed_plans
     - bounded_actions
     - 1
   * - refused_partial
     - decisions
     - 9999949
   * - refused_partial
     - round_endings
     - 8754799
   * - refused_partial
     - completions
     - 8754799
   * - refused_partial
     - live_observation_nodes
     - 1056133
   * - refused_partial
     - own_decision_nodes
     - 915988
   * - refused_partial
     - chance_rows_added
     - 8749414
   * - refused_partial
     - represented_decisions
     - 129850083
   * - refused_partial
     - represented_round_endings
     - 28450134
   * - refused_partial
     - represented_completions
     - 28450134
   * - refused_partial
     - terminal_rows_valued
     - 8754799
   * - refused_partial
     - terminal_mass_valued
     - 28450134
   * - refused_partial
     - ceiling_cutoffs
     - 611
   * - refused_partial
     - pruned_actions
     - 611
   * - refused_partial
     - continuation_cutoffs
     - 1202
   * - refused_partial
     - bounded_actions
     - 1202
   * - oracle
     - decisions
     - 3419559
   * - oracle
     - round_endings
     - 3031396
   * - oracle
     - completions
     - 3031396

.. list-table:: Elapsed components (not a paired speed comparison)
   :header-rows: 1

   * - Stage
     - Seconds
   * - fixed
     - 52.533664
   * - oracle
     - 111.643012
   * - refused_partial
     - 375.385298

Actual continuations and independent Python policy replay are separate
from source matches. Represented prefix counts are not physical simulations
or independent data. The prior28 completed and31 missing roots remain in
the original64-source cohort; remaining refusals retain the parent under
this comparison protocol. No new empirical strength claim, promotion,
reserved101M use or manuscript change follows from these diagnostics.

Complete SHA256: 05c3b1ce713f975c7f60dd1167a88abc645150ad0348e565c6daae9fa5a99201.
Protocol SHA256: 31da8e14da6b2fb3c669b541fa75b532f620f0d8120d456bb05f8d4da96ee2b4.
