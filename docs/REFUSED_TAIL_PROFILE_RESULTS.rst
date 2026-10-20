Passive profile of remaining tail refusals
=========================================

All five failures reproduce the closed recovery's work, policy-call
counts and stopping stage exactly. Every parent-first root action
completes; exhaustion occurs in later action comparisons. Source36
also reproduces its complete fixed control before optimized refusal.

.. list-table:: All evaluated root actions
   :header-rows: 1

   * - Source
     - Plan
     - Action
     - Status
     - Exact Q if complete
     - Physical decisions
   * - 0
     - fixed
     - 39
     - complete
     - 1711/2640
     - 695596
   * - 0
     - fixed
     - 9
     - complete
     - 36503/80080
     - 526822
   * - 0
     - fixed
     - 12
     - budget_refused
     - -
     - 777573
   * - 4
     - fixed
     - 21
     - complete
     - 0
     - 1232401
   * - 4
     - fixed
     - 1
     - budget_refused
     - -
     - 767580
   * - 20
     - fixed
     - 1
     - complete
     - 1585/52624
     - 1523282
   * - 20
     - fixed
     - 6
     - budget_refused
     - -
     - 476711
   * - 36
     - fixed
     - 21
     - complete
     - 0
     - 865730
   * - 36
     - fixed
     - 1
     - complete
     - 0
     - 641991
   * - 36
     - fixed
     - 46
     - complete
     - 0
     - 414573
   * - 36
     - optimized
     - 21
     - complete
     - 0
     - 1683463
   * - 36
     - optimized
     - 1
     - budget_refused
     - -
     - 316532
   * - 52
     - fixed
     - 1
     - complete
     - 28007/1420848
     - 1417683
   * - 52
     - fixed
     - 6
     - budget_refused
     - -
     - 582317

.. list-table:: Successful physical transitions by turn and phase
   :header-rows: 1

   * - Turn (zero based)
     - Phase
     - Decisions
   * - 11
     - 0
     - 2938
   * - 11
     - 2
     - 92
   * - 11
     - 3
     - 401
   * - 12
     - 0
     - 30600
   * - 12
     - 2
     - 803
   * - 12
     - 3
     - 13271
   * - 13
     - 0
     - 226597
   * - 13
     - 1
     - 41833
   * - 13
     - 2
     - 2541
   * - 13
     - 3
     - 82706
   * - 14
     - 0
     - 2626400
   * - 14
     - 2
     - 25403
   * - 15
     - 0
     - 8123748
   * - 15
     - 1
     - 744921

The final two turns account for 11,520,472/11,922,254 transitions (96.6300%).
Phase0 is discard,1 hand capture,2 draw capture,3 koi. Root action
values are exact only for completed actions; no value is assigned
to the interrupted candidates. There is no new complete optimized
solution or actual-tail assessment, and no new strength result.

.. list-table:: Additional diagnostic work
   :header-rows: 1

   * - Counter
     - Count
   * - decisions
     - 11922254
   * - round_endings
     - 10460295
   * - completions
     - 10460295
   * - represented_decisions
     - 153638226
   * - represented_completions
     - 32982282

.. list-table:: Additional history-support work
   :header-rows: 1

   * - Counter
     - Count
   * - replay_decisions
     - 6615
   * - candidate_decisions
     - 3255
   * - opponent_decisions
     - 3360
   * - teacher_calls
     - 3255
   * - guarded_decisions
     - 105
   * - feature_checks
     - 0
   * - feature_teacher_calls
     - 0

Planning, represented prefixes and history replay are separate counters.
No source match, training update or independent sample was added.
Timing includes instrumentation and is not a speed comparison. Saved
support worlds and partial action caches are bound for subsequent checks.

Complete SHA256: cd59d2d09e21d94b0502ec59efebe5469416731351658e8e72bf353dc0543f73.
Protocol SHA256: 668b3cb3b49e79257cd7d7599f2cb70dc59c0bb11d49f38fdb93a0b3c5a09f42.
