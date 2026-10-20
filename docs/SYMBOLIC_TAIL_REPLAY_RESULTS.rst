Executable symbolic endgame programs: complete-cohort comparison
===============================================================

The opponent-only and symbolic programs use the same512 saved sources and128 deal streams. No new source matches are added. Missing roots stay in each parent's256-source denominator. Both arms use the original maximum allowances and fresh caches. Actual win scores are1 for wins, 1/2 for draws and0 for losses. Outcome gains below are versus the unchanged parent.

.. list-table:: Policy outcomes
   :header-rows: 1

   * - Version
     - Parent
     - Arm
     - Adoptions
     - Exact conditional mean
     - Conditional pp
     - Exact actual mean
     - Actual pp
   * - opponent_only
     - 1
     - controller
     - 41
     - 3060065042423/1168288734412800
     - +0.2619
     - 1/256
     - +0.3906
   * - opponent_only
     - 1
     - root_only
     - 28
     - 216561437171/89868364185600
     - +0.2410
     - 3/512
     - +0.5859
   * - opponent_only
     - 2
     - controller
     - 42
     - 1735115616569/534527465472000
     - +0.3246
     - 0
     - +0.0000
   * - opponent_only
     - 2
     - root_only
     - 28
     - 140019678697/46793650176000
     - +0.2992
     - 0
     - +0.0000
   * - symbolic
     - 1
     - controller
     - 42
     - 3097041720623/1168288734412800
     - +0.2651
     - 1/256
     - +0.3906
   * - symbolic
     - 1
     - root_only
     - 28
     - 216561437171/89868364185600
     - +0.2410
     - 3/512
     - +0.5859
   * - symbolic
     - 2
     - controller
     - 44
     - 22864683476447/6948857051136000
     - +0.3290
     - 0
     - +0.0000
   * - symbolic
     - 2
     - root_only
     - 28
     - 140019678697/46793650176000
     - +0.2992
     - 0
     - +0.0000

.. list-table:: Coverage changes
   :header-rows: 1

   * - Arm
     - Counts
   * - controller
     - {'eligible': 237, 'new_parent_values': 5, 'lost_parent_values': 0, 'new_adoptions': 3, 'lost_adoptions': 0, 'greater_gain': 3, 'smaller_gain': 0}
   * - root_only
     - {'eligible': 237, 'new_parent_values': 3, 'lost_parent_values': 0, 'new_adoptions': 0, 'lost_adoptions': 0, 'greater_gain': 0, 'smaller_gain': 0}

.. list-table:: Planning work and coverage
   :header-rows: 1

   * - Version
     - Arm
     - Counts
   * - opponent_only
     - controller
     - {'parent_valued': 226, 'adoptions': 83, 'parent_adapter_calls': 2261097, 'opponent_adapter_calls': 733960, 'capped_phases': 53, 'decisions': 223012606, 'round_endings': 194285581, 'completions': 194285581, 'live_observation_nodes': 24037478, 'own_decision_nodes': 20144734, 'chance_rows_added': 193501094, 'represented_decisions': 2893161027, 'represented_round_endings': 935966869, 'represented_completions': 935966869, 'terminal_rows_valued': 194285581, 'terminal_mass_valued': 935966869, 'ceiling_cutoffs': 410852, 'pruned_actions': 410863, 'continuation_cutoffs': 24227, 'bounded_actions': 24227, 'certified_opponent_rows': 510622, 'certified_opponent_mass': 4595598}
   * - opponent_only
     - root_only
     - {'parent_valued': 233, 'adoptions': 56, 'parent_adapter_calls': 1928011, 'opponent_adapter_calls': 687613, 'capped_phases': 16, 'decisions': 148865177, 'round_endings': 129973229, 'completions': 129973229, 'live_observation_nodes': 15975289, 'own_decision_nodes': 13721749, 'chance_rows_added': 130184184, 'represented_decisions': 1970688014, 'represented_round_endings': 635613813, 'represented_completions': 635613813, 'terminal_rows_valued': 129973229, 'terminal_mass_valued': 635613813, 'ceiling_cutoffs': 55, 'pruned_actions': 65, 'continuation_cutoffs': 125, 'bounded_actions': 125, 'certified_opponent_rows': 202356, 'certified_opponent_mass': 1821204}
   * - symbolic
     - controller
     - {'parent_valued': 231, 'adoptions': 86, 'parent_adapter_calls': 2261367, 'opponent_adapter_calls': 755336, 'capped_phases': 43, 'decisions': 203294310, 'round_endings': 166957488, 'completions': 166957488, 'live_observation_nodes': 20771555, 'own_decision_nodes': 16510614, 'chance_rows_added': 176042217, 'represented_decisions': 3332442050, 'represented_round_endings': 967686766, 'represented_completions': 967686766, 'terminal_rows_valued': 166957488, 'terminal_mass_valued': 967686766, 'ceiling_cutoffs': 414248, 'pruned_actions': 414261, 'continuation_cutoffs': 29203, 'bounded_actions': 29203, 'certified_own_win_rows': 9797947, 'certified_own_win_mass': 88181523, 'certified_opponent_loss_rows': 510622, 'certified_opponent_loss_mass': 4595598}
   * - symbolic
     - root_only
     - {'parent_valued': 236, 'adoptions': 56, 'parent_adapter_calls': 1917292, 'opponent_adapter_calls': 698749, 'capped_phases': 10, 'decisions': 120421328, 'round_endings': 99346358, 'completions': 99346358, 'live_observation_nodes': 12480593, 'own_decision_nodes': 10144363, 'chance_rows_added': 104935971, 'represented_decisions': 2064578207, 'represented_round_endings': 619625331, 'represented_completions': 619625331, 'terminal_rows_valued': 99346358, 'terminal_mass_valued': 619625331, 'ceiling_cutoffs': 57, 'pruned_actions': 69, 'continuation_cutoffs': 130, 'bounded_actions': 130, 'certified_own_win_rows': 5380859, 'certified_own_win_mass': 48427731, 'certified_opponent_loss_rows': 202356, 'certified_opponent_loss_mass': 1821204}

Physical transitions, represented simulated completions and certified rows/mass are separate measures. The opponent_only implementation calls its loss counts certified_opponent_rows/mass; the symbolic version calls them certified_opponent_loss_rows/mass and adds certified_own_win_rows/mass. Program-entry counts below count entries across root programs, not unique histories or simulated work.

.. list-table:: Verification, conditional execution and program structure
   :header-rows: 1

   * - Category
     - Counts
   * - oracle_physical
     - {'decisions': 74652353, 'round_endings': 65460267, 'completions': 65460267}
   * - oracle_represented
     - {'decisions': 857295308, 'round_endings': 200605680, 'completions': 200605680}
   * - oracle_inference
     - {'new_parent_adapter_calls': 26090, 'new_opponent_adapter_calls': 0}
   * - actual_inference
     - {'learner_decisions': 1483, 'parent_adapter_calls': 1166, 'opponent_adapter_calls': 1031, 'symbolic_parent_adapter_calls': 9}
   * - common_program_structure
     - {'common_completed_root_values': 978, 'retained_explicit_entries': 15646702, 'omitted_certified_entries': 3217165}
   * - actual_tails
     - {'decisions': 2514, 'round_endings': 711, 'completions': 711}
   * - tail_replay
     - {'decisions': 2514, 'round_endings': 711, 'completions': 711}
   * - export_tail_replay
     - {'decisions': 2514, 'round_endings': 711, 'completions': 711}

Symbolic parent calls are included in total parent adapter calls. Independent oracle calls are reported separately from planning and actual-tail calls. Every adopted program's parent and selected value was replayed through terminal scoring with Python rules, including omitted actions. Common completed root values and retained actions matched; omitted histories were checked against the public certificate.

.. list-table:: Operational planning times
   :header-rows: 1

   * - Version
     - Arm
     - Seconds
   * - opponent_only
     - controller
     - 7076.146
   * - opponent_only
     - root_only
     - 5493.811
   * - symbolic
     - controller
     - 7599.165
   * - symbolic
     - root_only
     - 5745.183

Saved controls and concurrent workers do not provide an isolated latency benchmark. Lower per-branch cost can enable additional phases, so it need not lower total scheduled work.

The original prospective strength gate failed and remains unchanged. These reused-cohort comparisons are descriptive, without a fresh strength gate or confidence intervals. Conditional improvements depend on the specified belief and opponent model; they are distinct from observed match-win changes.

This export verifies completed audit bindings and exact aggregate arithmetic. It adds no simulation or inference.

Protocol SHA256: f8e2d65097b327102dde82c5ec89a338ae53c3c441b8bdf92b58846b8cc4d3c5.
Audit summary SHA256: 13d1e4b952e48778006b934108084c803b9929d5c66a85f279cc0cfbe434a6e9.
