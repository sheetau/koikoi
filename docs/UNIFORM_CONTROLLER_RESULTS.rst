Uniform-deal controller evaluation
==================================

Complete audited source matches: 8192.
Protocol SHA256: bc5fce90a1fefb146f766e2be7177c67992292307cf896413b525ba1183dbfc9.
Joint prespecified criterion: FAILED.

Primary values are conditional-expectation improvements over each unchanged
parent under independent uniform valid round deals and a randomized learner
seat. All missing-root and fallback sources remain in the denominator.
The four lower bounds are simultaneous at least95% under the declared law.
No observed-win, absolute-superiority or arbitrary-opponent claim follows.

.. csv-table:: Expected improvement by parent and opponent
   :header: Population,Sources,Mean(pp),Lower(pp),Exact lower

   0:wp,2048,0.223729,0.095296,2046464751/2147483648000
   0:point,2048,0.340125,0.130832,5619171503/4294967296000
   1:wp,2048,0.195892,0.083806,1799711101/2147483648000
   1:point,2048,0.232651,0.093521,4016697921/4294967296000

.. csv-table:: Equal WP/Point mixtures
   :header: Parent,Mean(pp),Lower(pp),Lower exceeds0.1pp

   0,0.281927,0.113064,True
   1,0.214272,0.088663,False

Success requires positive lower bounds in all four populations AND a mixture
lower bound above0.1percentage point for each parent. Exact fractions decide
the criterion. Display rounding has no role in the decision.

.. csv-table:: Realized conditional-tail differences (descriptive only)
   :header: Population,Controller minus parent(pp)

   0:wp,0.048828
   0:point,0.830078
   1:wp,0.268555
   1:point,0.146484

Work accounting
---------------

Source matches, native verification, history reconstruction, planning,
full Python value verification, actual tails and their audit/replays
are separate. Simulated completions and represented mass are not new
source observations. Bounded entropy draws exclude discarded internal
seeded auto-deals. Adapter calls include public endgame wrappers.

.. code-block:: json

   {
     "actual_inference": {
       "learner_decisions": 17669,
       "opponent_adapter_calls": 12172,
       "parent_adapter_calls": 14392,
       "symbolic_parent_adapter_calls": 81
     },
     "actual_tail_replay": {
       "completions": 7756,
       "decisions": 29841,
       "round_endings": 7756
     },
     "actual_tails": {
       "completions": 7756,
       "decisions": 29841,
       "round_endings": 7756
     },
     "audit_actor_verification": {
       "opponent": 472703,
       "parent": 477417
     },
     "audit_history_reconstruction": {
       "candidate_decisions": 3926939,
       "feature_checks": 0,
       "feature_teacher_calls": 0,
       "guarded_decisions": 129870,
       "opponent_decisions": 3849857,
       "replay_decisions": 7776796,
       "teacher_calls": 3719987
     },
     "audit_identification": {
       "point": 102773,
       "wp": 91173
     },
     "audit_source_replay": {
       "completions": 8192,
       "decisions": 950120,
       "round_endings": 63678
     },
     "audit_tail_execution": {
       "completions": 7756,
       "decisions": 29841,
       "round_endings": 7756
     },
     "audit_tail_inference": {
       "learner_decisions": 17669,
       "opponent_adapter_calls": 12172,
       "parent_adapter_calls": 14392,
       "symbolic_parent_adapter_calls": 81
     },
     "audit_tail_replay": {
       "completions": 7756,
       "decisions": 29841,
       "round_endings": 7756
     },
     "history_support": {
       "candidate_decisions": 3926939,
       "feature_checks": 0,
       "feature_teacher_calls": 0,
       "guarded_decisions": 129870,
       "opponent_decisions": 3849857,
       "replay_decisions": 7776796,
       "teacher_calls": 3719987
     },
     "oracle_inference": {
       "opponent": 5605754,
       "parent": 7969610
     },
     "oracle_physical": {
       "completions": 990849188,
       "decisions": 1136752186,
       "round_endings": 990849188
     },
     "oracle_represented": {
       "completions": 3724047360,
       "decisions": 16281250241,
       "round_endings": 3724047360
     },
     "planning": {
       "bounded_actions": 598521,
       "ceiling_cutoffs": 8011394,
       "certified_opponent_loss_mass": 509358978,
       "certified_opponent_loss_rows": 56595442,
       "certified_own_win_mass": 1064982456,
       "certified_own_win_rows": 118331384,
       "chance_rows_added": 3466794305,
       "completions": 3313552121,
       "continuation_cutoffs": 598521,
       "decisions": 4014276202,
       "live_observation_nodes": 418805100,
       "maximum_frontier_rows": 1002496,
       "own_decision_nodes": 338634574,
       "pruned_actions": 8011602,
       "represented_completions": 19992342195,
       "represented_decisions": 73189809414,
       "represented_round_endings": 19992342195,
       "round_endings": 3313552121,
       "terminal_mass_valued": 19992342195,
       "terminal_rows_valued": 3313552121
     },
     "planning_inference": {
       "opponent": 21885433,
       "parent": 43350284
     },
     "sampling": {
       "accepted_deals": 63678,
       "attempts": 64519,
       "bounded_draws": 3032393,
       "seat_bounded_draws": 8192
     },
     "source_identification": {
       "point": 102773,
       "wp": 91173
     },
     "source_native": {
       "completions": 8192,
       "decisions": 950120,
       "round_endings": 63678
     },
     "sources": {
       "completions": 8192,
       "decisions": 950120,
       "round_endings": 63678
     }
   }
