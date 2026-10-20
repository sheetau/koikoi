Finite belief-tail capability results
====================================

All 64 source matches completed on 16 distinct deal streams. Exact plans
completed at 16 roots; 48 sources had no
eligible root and 0 roots exceeded a computation cap.
Sources use two retained parents, assisted original WP/Point and both seats.
This fixed capability check measures conditional opportunities and cost.
Its actual-tail outcomes do not constitute a prospective strength test.

.. list-table:: All completed roots
   :header-rows: 1

   * - Source
     - Parent
     - Opponent
     - Seat
     - Worlds
     - Parent Q
     - Fixed Q
     - Optimized Q
     - Actual point margins parent/fixed/optimized
   * - 2
     - 1
     - wp
     - 0
     - 90
     - 1
     - 1
     - 1
     - 16/16/16
   * - 7
     - 1
     - wp
     - 0
     - 1980
     - 27/44
     - 27/44
     - 27/44
     - 0/0/0
   * - 15
     - 1
     - wp
     - 1
     - 4950
     - 1
     - 1
     - 1
     - 4/4/4
   * - 19
     - 1
     - point
     - 0
     - 2970
     - 112/165
     - 112/165
     - 112/165
     - -4/-4/-4
   * - 22
     - 1
     - point
     - 0
     - 1980
     - 7/33
     - 7/33
     - 7/33
     - 4/4/4
   * - 27
     - 1
     - point
     - 1
     - 270
     - 1
     - 1
     - 1
     - 20/20/20
   * - 30
     - 1
     - point
     - 1
     - 180
     - 1
     - 1
     - 1
     - 12/12/12
   * - 32
     - 2
     - wp
     - 0
     - 720
     - 61/72
     - 61/72
     - 61/72
     - 40/40/40
   * - 34
     - 2
     - wp
     - 0
     - 90
     - 1
     - 1
     - 1
     - 24/24/24
   * - 40
     - 2
     - wp
     - 1
     - 1980
     - 331/792
     - 331/792
     - 1691/3960
     - 10/10/10
   * - 42
     - 2
     - wp
     - 1
     - 5940
     - 17/5940
     - 17/5940
     - 17/5940
     - -22/-22/-22
   * - 46
     - 2
     - wp
     - 1
     - 36630
     - 1
     - 1
     - 1
     - 22/22/22
   * - 49
     - 2
     - point
     - 0
     - 990
     - 1
     - 1
     - 1
     - 34/34/34
   * - 51
     - 2
     - point
     - 0
     - 1980
     - 0
     - 0
     - 0
     - -24/-24/-24
   * - 54
     - 2
     - point
     - 0
     - 1980
     - 0
     - 0
     - 0
     - -20/-20/-20
   * - 62
     - 2
     - point
     - 1
     - 270
     - 1
     - 1
     - 1
     - 16/16/16

Q is the exact draw-adjusted match-win expectation under the declared
uniform valid-deal and publicly identified deterministic-opponent model.
Every root action's entire contingent policy was replayed in every world
with independent Python transitions. Opponent predictions in that replay
came from recorded observation-keyed caches; neural calls were not repeated.
All plans were frozen before actual-world assessment. Export independently
replays source matches, selected prefixes and all actual tails.

.. list-table:: Means over all 32 sources per parent (percentage points; missing/refused roots contribute zero)
   :header-rows: 1

   * - Parent
     - Fixed minus parent Q
     - Optimized minus fixed Q
     - Optimized minus parent Q
     - Realized fixed minus parent
     - Realized optimized minus parent
   * - parent1
     - +0.000000
     - +0.000000
     - +0.000000
     - +0.000000
     - +0.000000
   * - parent2
     - +0.000000
     - +0.028409
     - +0.028409
     - +0.000000
     - +0.000000

These conditional-value means describe the sampled source histories; they
are not population estimates with a validated confidence interval. Source seats
and parents share deals. No new independent sample is created by world expansion.

.. list-table:: Work accounting
   :header-rows: 1

   * - Stage
     - Counts
   * - sources
     - {"completions": 64, "decisions": 7629, "round_endings": 511}
   * - native_source_verification
     - {"completions": 64, "decisions": 7629, "round_endings": 511}
   * - history_support
     - {"candidate_decisions": 3422, "feature_checks": 0, "feature_teacher_calls": 0, "guarded_decisions": 200, "opponent_decisions": 3373, "replay_decisions": 6795, "teacher_calls": 3173}
   * - planning
     - {"completions": 255492, "decisions": 660758, "observation_nodes": 126181, "own_decision_nodes": 4706, "round_endings": 255492}
   * - independent_policy_replay
     - {"completions": 248760, "decisions": 650714, "round_endings": 248760}
   * - actual_tail_assessment
     - {"completions": 48, "decisions": 114, "round_endings": 48}
   * - export_source_and_root_audit
     - {"completions": 64, "decisions": 9586, "round_endings": 623}
   * - export_tail_audit
     - {"completions": 48, "decisions": 114, "round_endings": 48}
   * - refused_partial_planning
     - {}

.. list-table:: Measured planning cost
   :header-rows: 1

   * - Stage
     - Total seconds
     - Maximum root seconds
   * - fixed
     - 1.736486
     - 0.878361
   * - optimized
     - 1.346955
     - 0.718470
   * - support
     - 7.140921
     - 1.231978

Fixed planning runs first; optimized planning reuses its policy caches.
Times exclude independent policy replay and are not isolated throughput
comparisons. Support cost is additional. No training, production-model
promotion or reserved101M evaluation occurred. The broader strength
objective remains open. See BELIEF_TAIL.rst for the fixed protocol.

Protocol SHA256: fc7a318560d7b294a41d057fa349de84d45ba1fbef64b2fb61b845598e52173f.
Complete SHA256: 320758c9c0fb79367423509c1f715d752e97680e9ae9473bf8b443195c616f7a.
