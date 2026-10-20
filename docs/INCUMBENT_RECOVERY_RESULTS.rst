Retained-incumbent recovery results
==================================

All five former full-comparison refusals run the actual capped
fixed-then-optimized controller. Every fixed parent reference matches
its previously completed root evaluation. Previously attempted phases
reproduce exact work with zero new adapter calls; three optimized
phases are explicitly new work. No source/history replay was added.

.. list-table:: All five frozen selections and actual tails
   :header-rows: 1

   * - Source
     - Mode
     - Parent Q
     - Selected Q
     - Exact gain
     - Within-phase optimum gap
     - Parent margin
     - Controller margin
   * - 0
     - optimized
     - 1711/2640
     - 55113/80080
     - 4819/120120
     - 24967/80080
     - 12
     - 12
   * - 4
     - parent
     - 0
     - 0
     - 0
     - None
     - -20
     - -20
   * - 20
     - parent
     - 1585/52624
     - 1585/52624
     - 0
     - None
     - -24
     - -24
   * - 36
     - parent
     - 0
     - 0
     - 0
     - None
     - -24
     - -24
   * - 52
     - parent
     - 28007/1420848
     - 28007/1420848
     - 0
     - None
     - -24
     - -24

.. list-table:: Combined original cohort, means over32 sources per parent
   :header-rows: 1

   * - Parent
     - Adoptions
     - Conditional gain
     - Realized gain
     - Changed outcomes
   * - 1
     - 5
     - 204203/25625600
     - 0
     - 0
   * - 2
     - 2
     - 2699329/457142400
     - 0
     - 0

The selected policy is complete and exactly evaluated even when its
phase hits a cap. A positive optimum gap leaves the best possible policy
unresolved. A fixed-phase gap says nothing about optimizing later choices.
Parent fallback remains explicit if no strict exact gain is available.

.. list-table:: Additional search and independent policy-replay work
   :header-rows: 1

   * - Category
     - Counter
     - Count
   * - completed_search_phases
     - decisions
     - 3725271
   * - completed_search_phases
     - round_endings
     - 3303427
   * - completed_search_phases
     - completions
     - 3303427
   * - completed_search_phases
     - live_observation_nodes
     - 357711
   * - completed_search_phases
     - own_decision_nodes
     - 306318
   * - completed_search_phases
     - chance_rows_added
     - 3303477
   * - completed_search_phases
     - represented_decisions
     - 52445376
   * - completed_search_phases
     - represented_round_endings
     - 11200860
   * - completed_search_phases
     - represented_completions
     - 11200860
   * - completed_search_phases
     - terminal_rows_valued
     - 3303427
   * - completed_search_phases
     - terminal_mass_valued
     - 11200860
   * - completed_search_phases
     - ceiling_cutoffs
     - 0
   * - completed_search_phases
     - pruned_actions
     - 0
   * - completed_search_phases
     - continuation_cutoffs
     - 1
   * - completed_search_phases
     - bounded_actions
     - 1
   * - capped_search_phases
     - decisions
     - 15999927
   * - capped_search_phases
     - round_endings
     - 14074152
   * - capped_search_phases
     - completions
     - 14074152
   * - capped_search_phases
     - live_observation_nodes
     - 1656450
   * - capped_search_phases
     - own_decision_nodes
     - 1445974
   * - capped_search_phases
     - chance_rows_added
     - 13989164
   * - capped_search_phases
     - represented_decisions
     - 186957900
   * - capped_search_phases
     - represented_round_endings
     - 46000503
   * - capped_search_phases
     - represented_completions
     - 46000503
   * - capped_search_phases
     - terminal_rows_valued
     - 14074152
   * - capped_search_phases
     - terminal_mass_valued
     - 46000503
   * - capped_search_phases
     - ceiling_cutoffs
     - 1281
   * - capped_search_phases
     - pruned_actions
     - 1281
   * - capped_search_phases
     - continuation_cutoffs
     - 1662
   * - capped_search_phases
     - bounded_actions
     - 1662
   * - new_oracle_physical
     - decisions
     - 5732910
   * - new_oracle_physical
     - round_endings
     - 5040902
   * - new_oracle_physical
     - completions
     - 5040902
   * - new_oracle_represented
     - decisions
     - 68628933
   * - new_oracle_represented
     - round_endings
     - 13590720
   * - new_oracle_represented
     - completions
     - 13590720

.. list-table:: Other additional work
   :header-rows: 1

   * - Category
     - Counter
     - Count
   * - new_adapter_calls
     - new_parent_adapter_calls
     - 38319
   * - new_adapter_calls
     - new_opponent_adapter_calls
     - 12029
   * - actual_tails
     - decisions
     - 46
   * - actual_tails
     - round_endings
     - 10
   * - actual_tails
     - completions
     - 10
   * - separate_trace_replay
     - decisions
     - 46
   * - separate_trace_replay
     - round_endings
     - 10
   * - separate_trace_replay
     - completions
     - 10
   * - actual_actor_calls
     - parent_actor_calls
     - 24
   * - actual_actor_calls
     - opponent_actor_calls
     - 22

.. list-table:: Elapsed components; caches reused, not a speed comparison
   :header-rows: 1

   * - Stage
     - Seconds
   * - fixed
     - 107.516820
   * - optimized
     - 199.313384
   * - new_oracle
     - 187.030644

The64 original sources /16 streams remain the denominator. These five
selections replace their earlier saved-plan selections; the other28
available roots and31 missing roots are unchanged. Conditional tail
completions and reused action caches are not independent samples.
No prospective win-rate replication, training, deployment, reserved101M
evaluation or manuscript rebuild is claimed.

Complete SHA256: 76c56afcc94ea8aaa2d7116766d193161a65676b09ede430b8e33b23eea49380.
Protocol SHA256: 2365bf2ff39b69e38ba973f18da10fdb265a9e9e351b0dd958497e1cf4517b94.
