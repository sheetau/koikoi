Ceiling-pruned tail recovery
===========================

All fifteen earlier refusals were attempted. 10 roots now complete;
5 remain refused. The original cohort therefore has
28 completed roots, 5 remaining refusals and 31 missing roots
among 64 source matches on 16 distinct deal streams. No source match was added.
The limits changed from 500,000 all nodes to 500,000 live nodes plus 2,000,000
physical transitions per plan. This is an exploratory computational
extension, not a new prospective strength evaluation.

.. list-table:: All fifteen attempts
   :header-rows: 1

   * - Source
     - Parent
     - Opponent
     - Status
     - Parent Q
     - Fixed Q
     - Optimized Q
     - Actual point margins P/F/O
     - Refusal reason
   * - 0
     - 1
     - wp
     - budget_refused
     - -
     - -
     - -
     - -
     - Physical transition cap exceeded
   * - 1
     - 1
     - wp
     - complete
     - 3339/5720
     - 3339/5720
     - 51767/85800
     - -4/-4/-4
     - -
   * - 4
     - 1
     - wp
     - budget_refused
     - -
     - -
     - -
     - -
     - Physical transition cap exceeded
   * - 18
     - 1
     - point
     - complete
     - 20707/28600
     - 23433/28600
     - 25249/28600
     - 8/8/8
     - -
   * - 20
     - 1
     - point
     - budget_refused
     - -
     - -
     - -
     - -
     - Physical transition cap exceeded
   * - 21
     - 1
     - point
     - complete
     - 1
     - 1
     - 1
     - 10/10/10
     - -
   * - 25
     - 1
     - point
     - complete
     - 1
     - 1
     - 1
     - 10/10/10
     - -
   * - 26
     - 1
     - point
     - complete
     - 14177/77220
     - 14177/77220
     - 26051/123552
     - 0/0/0
     - -
   * - 34
     - 2
     - wp
     - complete
     - 1
     - 1
     - 1
     - 66/66/66
     - -
   * - 36
     - 2
     - wp
     - budget_refused
     - -
     - -
     - -
     - -
     - Physical transition cap exceeded
   * - 50
     - 2
     - point
     - complete
     - 59641/85800
     - 70903/85800
     - 25249/28600
     - 8/8/8
     - -
   * - 52
     - 2
     - point
     - budget_refused
     - -
     - -
     - -
     - -
     - Physical transition cap exceeded
   * - 53
     - 2
     - point
     - complete
     - 1
     - 1
     - 1
     - 12/12/12
     - -
   * - 55
     - 2
     - point
     - complete
     - 1
     - 1
     - 1
     - 24/24/24
     - -
   * - 59
     - 2
     - point
     - complete
     - 0
     - 0
     - 0
     - -24/-24/-24
     - -

.. list-table:: Combined means over all 32 sources per parent (percentage points)
   :header-rows: 1

   * - Parent
     - Fixed minus parent Q
     - Optimized minus fixed Q
     - Optimized minus parent Q
     - Realized fixed minus parent
     - Realized optimized minus parent
   * - parent1
     - +0.297858
     - +0.373643
     - +0.671502
     - +0.000000
     - +0.000000
   * - parent2
     - +0.414051
     - +0.176428
     - +0.590479
     - +0.000000
     - +0.000000

Missing/refused roots retain the parent in both arms. Earlier 18 completed
policies/outcomes are reused through exact Stage 1 equivalence. Q values
use the declared uniform valid-deal and publicly identified deterministic
opponent model. Every computed recovery value passes independent Python
chance/policy replay. Omitted root actions remain unevaluated, certified
unnecessary only when a chosen value reaches 1. These source-history means
are descriptive; no population confidence interval or new win criterion
is asserted. All plans were frozen before actual-tail assessment.

.. list-table:: Additional work only
   :header-rows: 1

   * - Stage
     - Counts
   * - new_source_matches
     - 0
   * - history_support
     - {"candidate_decisions": 21840, "feature_checks": 0, "feature_teacher_calls": 0, "guarded_decisions": 770, "opponent_decisions": 20700, "replay_decisions": 42540, "teacher_calls": 19930}
   * - completed_planning
     - {"ceiling_cutoffs": 6280, "chance_rows_added": 10894389, "completions": 10912399, "decisions": 12283153, "live_observation_nodes": 1135960, "maximum_frontier_rows": 504, "own_decision_nodes": 891635, "pruned_actions": 6282, "represented_completions": 91364670, "represented_decisions": 192678705, "represented_round_endings": 91364670, "round_endings": 10912399, "terminal_mass_valued": 91364670, "terminal_rows_valued": 10912399}
   * - refused_partial_planning
     - {"ceiling_cutoffs": 0, "chance_rows_added": 8746177, "completions": 8761604, "decisions": 9999960, "live_observation_nodes": 1069651, "maximum_frontier_rows": 299, "own_decision_nodes": 932639, "pruned_actions": 0, "represented_completions": 27422442, "represented_decisions": 130410738, "represented_round_endings": 27422442, "round_endings": 8761604, "terminal_mass_valued": 27422442, "terminal_rows_valued": 8761604}
   * - independent_policy_oracle
     - {"completions": 8844642, "decisions": 9970387, "round_endings": 8844642}
   * - represented_oracle_units
     - {"completions": 83231280, "decisions": 175814550, "round_endings": 83231280}
   * - actual_tails
     - {"completions": 30, "decisions": 86, "round_endings": 30}
   * - export_tail_audit
     - {"completions": 30, "decisions": 86, "round_endings": 30}

.. list-table:: Additional elapsed components
   :header-rows: 1

   * - Stage
     - Seconds
   * - fixed
     - 89.601113
   * - optimized
     - 60.557837
   * - support
     - 41.517469
   * - independent_oracle
     - 321.447137
   * - refused_partial
     - 366.289361

Completed fixed work preceding an optimized refusal remains in the work
ledger; the incomplete comparison contributes no intervention gain.
Represented prefix units are not actual simulations or independent data.
Policy replay uses cached observation-keyed opponent actions; actual tails
use fresh strict calls. Export replays all new actual tails and reuses the
bound original source/prefix audit. No training, reserved101M evaluation,
canonical promotion or manuscript rebuild occurred.

Recovery protocol SHA256: ca89d632709be3f25b5ce1c0ffe7f6f71628d9d4ad257c5c898b3a1f6e465eb9.
Recovery complete SHA256: fa7c01ef7751624558c94c61cec01bbe306bbf0218ae1999c937d576ce27227c.
