Final-opponent-turn certificate: complete-cohort comparison
==========================================================

Both implementations use the same saved512 sources/128 deal streams. This comparison adds zero source matches. All missing roots and support refusals remain in the256-source denominator per parent. Gains are relative to the unchanged parent. Actual scores count wins1, draws1/2, losses0.

.. list-table:: Policy outcomes
   :header-rows: 1

   * - Version
     - Arm
     - Parent
     - Adoptions
     - Exact conditional mean
     - Conditional pp
     - Exact actual mean
     - Actual pp
   * - Original
     - controller
     - 1
     - 41
     - 3060065042423/1168288734412800
     - +0.2619
     - 1/256
     - +0.3906
   * - Original
     - controller
     - 2
     - 42
     - 1735115616569/534527465472000
     - +0.3246
     - 0
     - +0.0000
   * - Original
     - root_only
     - 1
     - 28
     - 216561437171/89868364185600
     - +0.2410
     - 3/512
     - +0.5859
   * - Original
     - root_only
     - 2
     - 28
     - 140019678697/46793650176000
     - +0.2992
     - 0
     - +0.0000
   * - Certificate
     - controller
     - 1
     - 41
     - 3060065042423/1168288734412800
     - +0.2619
     - 1/256
     - +0.3906
   * - Certificate
     - controller
     - 2
     - 42
     - 1735115616569/534527465472000
     - +0.3246
     - 0
     - +0.0000
   * - Certificate
     - root_only
     - 1
     - 28
     - 216561437171/89868364185600
     - +0.2410
     - 3/512
     - +0.5859
   * - Certificate
     - root_only
     - 2
     - 28
     - 140019678697/46793650176000
     - +0.2992
     - 0
     - +0.0000

.. list-table:: Matched computational allowances
   :header-rows: 1

   * - Version
     - Arm
     - Native decisions
     - Physical completions
     - Represented simulated completions
     - Certified rows
     - Certified mass
     - Parent values
     - Capped phases
   * - original
     - controller
     - 227608204
     - 198881179
     - 940562467
     - 0
     - 0
     - 226
     - 53
   * - original
     - root_only
     - 150686381
     - 131794433
     - 637435017
     - 0
     - 0
     - 233
     - 16
   * - certified
     - controller
     - 223012606
     - 194285581
     - 935966869
     - 510622
     - 4595598
     - 226
     - 53
   * - certified
     - root_only
     - 148865177
     - 129973229
     - 635613813
     - 202356
     - 1821204
     - 233
     - 16

.. list-table:: Coverage changes
   :header-rows: 1

   * - Arm
     - Counts
   * - controller
     - {'eligible': 237, 'new_parent_values': 0, 'lost_parent_values': 0, 'new_adoptions': 0, 'lost_adoptions': 0, 'greater_gain': 0, 'smaller_gain': 0}
   * - root_only
     - {'eligible': 237, 'new_parent_values': 0, 'lost_parent_values': 0, 'new_adoptions': 0, 'lost_adoptions': 0, 'greater_gain': 0, 'smaller_gain': 0}

The controller retains separate2M-transition/500k-live-node phases; root-only uses4M transitions/1M nodes. Both start with empty caches. Completing a previously unavailable parent value can enable an additional phase, so reduced branch cost need not reduce total scheduled work. Symbolic resolutions are separate from simulated terminal completions.

.. list-table:: Operational planning times
   :header-rows: 1

   * - Version
     - Arm
     - Seconds
   * - original
     - controller
     - 8508.238
   * - original
     - root_only
     - 6565.889
   * - certified
     - controller
     - 7076.146
   * - certified
     - root_only
     - 5493.811

The original runs provide saved computational controls. These are separate concurrent runs, not an alternating isolated latency benchmark.

.. list-table:: Additional verification and conditional execution
   :header-rows: 1

   * - Category
     - Counts
   * - oracle_physical
     - {'decisions': 61821861, 'round_endings': 54051761, 'completions': 54051761}
   * - oracle_represented
     - {'decisions': 741008054, 'round_endings': 176513040, 'completions': 176513040}
   * - oracle_inference
     - {'new_opponent_adapter_calls': 0}
   * - actual_tails
     - {'decisions': 2514, 'round_endings': 711, 'completions': 711}
   * - tail_replay
     - {'decisions': 2514, 'round_endings': 711, 'completions': 711}
   * - export_tail_replay
     - {'decisions': 2514, 'round_endings': 711, 'completions': 711}

The original prospective strength gate failed. It is unchanged by this extension. The reused-cohort means above are descriptive; they carry no fresh confirmation claim or new confidence intervals.

Completed common root values and complete learner maps matched. Every adoption's parent and selected values were independently replayed under Python rules, including omitted opponent moves. Complete-only audit checked all bindings, selections, common maps, actual traces and work totals. This export verifies bound artifacts and exact aggregate arithmetic; it adds no simulations or inference calls. The conditional values require the specified belief and opponent model and are separate from source-win changes.

Protocol SHA256: 8b91c528ca2f0883821a3435104058bdf7435cce4951b95a52b0a69f0bd53e40.
Audit summary SHA256: 8ac783d5873c626436e373488e3cf8af6a96192473a1b3c585979e047299e9d0.
