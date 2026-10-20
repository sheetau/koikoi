Prospective contingent-policy comparison
========================================

The fixed development strength gate does not pass. All512 sources and128 deal streams are included.

.. list-table:: Paired improvements
   :header-rows: 1

   * - Contrast
     - Actual win-score difference (pp)
     - Simultaneous95% interval (pp)
     - Model-conditional difference (pp)
   * - parent1:controller-parent
     - +0.3906
     - [+0.0000, +1.5625]
     - +0.2619
   * - parent1:controller-root_only
     - -0.1953
     - [-0.7812, +0.0000]
     - +0.0210
   * - parent2:controller-parent
     - +0.0000
     - [-1.9531, +1.9531]
     - +0.3246
   * - parent2:controller-root_only
     - +0.0000
     - [+0.0000, +0.0000]
     - +0.0254

The actual score is win=1, draw=1/2, loss=0. Two seats are averaged within a deal; intervals resample deal clusters within opponent. The conditional column uses the declared finite belief and is a secondary measure, not an observed win-rate gain.

Opportunity statuses: {'missing_root': 275, 'evaluated': 237}.

Adoptions: {'parent1:root_only': 28, 'parent1:controller': 41, 'parent2:root_only': 28, 'parent2:controller': 42}. Changed actual win scores: {'parent1:root_only': 2, 'parent1:controller': 1, 'parent2:root_only': 4, 'parent2:controller': 4}.

.. list-table:: Search costs under equal maximum allowances
   :header-rows: 1

   * - Arm
     - Physical decisions
     - Planning seconds
     - Parent calls
     - Opponent calls
     - Capped phases
   * - controller
     - 227608204
     - 8508.238
     - 2261097
     - 733960
     - 53
   * - root_only
     - 150686381
     - 6565.889
     - 1928011
     - 687613
     - 16

Both arms had at most4M native transitions and1M live nodes per eligible root. Realized work differs. Caches started empty for each arm; four concurrent workers make elapsed timings operational costs, not isolated latency benchmarks.

.. list-table:: Work accounting
   :header-rows: 1

   * - Category
     - Counts
   * - source
     - {'decisions': 59512, 'round_endings': 4002, 'completions': 512}
   * - native_source_verification
     - {'decisions': 59512, 'round_endings': 4002, 'completions': 512}
   * - history
     - {'replay_decisions': 478512, 'candidate_decisions': 239395, 'opponent_decisions': 239117, 'teacher_calls': 231362, 'guarded_decisions': 7755, 'feature_checks': 0, 'feature_teacher_calls': 0}
   * - planning
     - {'decisions': 378294585, 'round_endings': 330675612, 'completions': 330675612, 'live_observation_nodes': 40478303, 'own_decision_nodes': 33866483, 'chance_rows_added': 329389102, 'represented_decisions': 4870265843, 'represented_round_endings': 1577997484, 'represented_completions': 1577997484, 'terminal_rows_valued': 330675612, 'terminal_mass_valued': 1577997484, 'ceiling_cutoffs': 410907, 'pruned_actions': 410928, 'continuation_cutoffs': 24352, 'bounded_actions': 24352}
   * - actual_tails
     - {'decisions': 2514, 'round_endings': 711, 'completions': 711}
   * - tail_replay
     - {'decisions': 2514, 'round_endings': 711, 'completions': 711}
   * - oracle_physical
     - {'decisions': 61821861, 'round_endings': 54051761, 'completions': 54051761}
   * - oracle_represented
     - {'decisions': 741008054, 'round_endings': 176513040, 'completions': 176513040}

Additional export source replay: {'decisions': 59512, 'round_endings': 4002, 'completions': 512}. Additional export tail replay: {'decisions': 2514, 'round_endings': 711, 'completions': 711}. These are verification work, not new matches.

Every adopted value pair was independently replayed with Python rules during the study. This export checks policy bindings, selection, stored oracle evidence, all actual traces and exact aggregate arithmetic. It does not rerun search or hypothetical neural inference. No additional model training, canonical-model promotion or reserved101M use occurred.

Protocol SHA256: 700e11608e3ced60919ca7ef31da460785ff9391459ac83c7e8a16f4a1a0c170.
Summary SHA256: c80bbcbc67b1923632ea2480bacb06df8066f1e587ad83e843458fc782298f1e.
