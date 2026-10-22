Fresh incremental frontier-controller evaluation
===============================================

Complete independently audited source matches: 4096.
Contrast: selected frontier hybrid minus existing symbolic controller.
The same fixed cohort retains 15 sources acquired before the gate-scope repair;
the remaining 4,081 sources are acquired after the corrected freeze.
The primary values are exact conditional expected win-score increments.
Every missing-root, refused-support and unchanged-policy source is included.
The four population lower bounds form one simultaneous 95% family.

.. csv-table:: Expected incremental gains
   :header: Population,Sources,Mean(pp),Lower(pp),Exact lower

   0:wp,1024,0.000677034432509,6.64356241468e-05,285338833/429496729600000
   0:point,1024,0.00189202637006,2.73724400904e-05,23512747/85899345920000
   1:wp,1024,0.0010677759066,1.10018327832e-05,1476641/13421772800000
   1:point,1024,0.000500209467106,7.94267829042e-06,68227087/858993459200000

.. csv-table:: Fixed equal-opponent mixtures
   :header: Parent,Mean(pp),Lower(pp),Positive lower bound

   0,0.00128453040128,4.69040321186e-05,True
   1,0.000783992686854,9.47225553682e-06,True

Primary criterion (both mixture lower bounds positive): PASSED.
All four individual population lower bounds positive: True.

The previous controller-versus-parent 0.1pp criterion remains a separate
failed test. A positive incremental result does not establish that practical
threshold or absolute superiority over arbitrary opponents.

.. csv-table:: Realized score differences (descriptive)
   :header: Population,Hybrid minus old controller(pp)

   0:wp,0
   0:point,0
   1:wp,0
   1:point,0

Source outcomes and separate work ledgers
----------------------------------------

Planning physical transitions and represented belief mass are separate.
Source matches are not augmented by oracle or replay completions.
Learner executor calls do not purport to count underlying neural calls.

.. code-block:: json

   {
     "status_counts": {
       "evaluated": 1943,
       "missing_root": 2153,
       "new_v2": 4081,
       "old_no_strict_gain": 1257,
       "old_parent_unvalued": 60,
       "old_strict_exact_gain": 626,
       "positive_increment": 26,
       "retained_v1": 15
     },
     "work": {
       "audit_actor_verification": {
         "opponent": 236839,
         "parent": 238901
       },
       "audit_history_reconstruction": {
         "candidate_decisions": 2019730,
         "feature_checks": 0,
         "feature_teacher_calls": 0,
         "guarded_decisions": 61234,
         "opponent_decisions": 1973461,
         "replay_decisions": 3993191,
         "teacher_calls": 1912227
       },
       "audit_identification": {
         "point": 50953,
         "wp": 46315
       },
       "audit_local_physical": {
         "completions": 10558665,
         "decisions": 12067877,
         "round_endings": 10558665
       },
       "audit_local_represented": {
         "completions": 26983440,
         "decisions": 79200636,
         "round_endings": 26983440
       },
       "audit_prefix_physical": {
         "completions": 49757,
         "decisions": 370024,
         "round_endings": 49757
       },
       "audit_prefix_represented": {
         "completions": 53441190,
         "decisions": 1054701450,
         "round_endings": 53441190
       },
       "audit_source_replay": {
         "completions": 4096,
         "decisions": 475740,
         "round_endings": 31819
       },
       "audit_tails_actual_work": {
         "completions": 3886,
         "decisions": 14730,
         "round_endings": 3886
       },
       "audit_tails_calls": {
         "learner_executor_calls": 8728,
         "opponent_adapter_calls": 6002
       },
       "audit_tails_replay_work": {
         "completions": 3886,
         "decisions": 14730,
         "round_endings": 3886
       },
       "history_support": {
         "candidate_decisions": 2019730,
         "feature_checks": 0,
         "feature_teacher_calls": 0,
         "guarded_decisions": 61234,
         "opponent_decisions": 1973461,
         "replay_decisions": 3993191,
         "teacher_calls": 1912227
       },
       "new_v2_sampling": {
         "accepted_deals": 31701,
         "attempts": 32086,
         "bounded_draws": 1508042,
         "seat_bounded_draws": 4081
       },
       "new_v2_source_execution": {
         "completions": 4081,
         "decisions": 474005,
         "round_endings": 31701
       },
       "new_v2_source_identification": {
         "point": 50802,
         "wp": 46106
       },
       "new_v2_source_native": {
         "completions": 4081,
         "decisions": 474005,
         "round_endings": 31701
       },
       "planning_fixed": {
         "bounded_actions": 1045,
         "ceiling_cutoffs": 513,
         "certified_opponent_loss_mass": 92117700,
         "certified_opponent_loss_rows": 10235300,
         "certified_own_win_mass": 199088316,
         "certified_own_win_rows": 22120924,
         "chance_rows_added": 782562185,
         "completions": 750130958,
         "continuation_cutoffs": 1045,
         "decisions": 901397514,
         "live_observation_nodes": 95415911,
         "own_decision_nodes": 77530057,
         "pruned_actions": 624,
         "represented_completions": 4921889059,
         "represented_decisions": 17969961518,
         "represented_round_endings": 4921889059,
         "round_endings": 750130958,
         "terminal_mass_valued": 4921889059,
         "terminal_rows_valued": 750130958
       },
       "planning_inference": {
         "opponent": 11065979,
         "parent": 19226731
       },
       "planning_optimized": {
         "bounded_actions": 267815,
         "ceiling_cutoffs": 4040237,
         "certified_opponent_loss_mass": 125282754,
         "certified_opponent_loss_rows": 13920306,
         "certified_own_win_mass": 244468035,
         "certified_own_win_rows": 27163115,
         "chance_rows_added": 873963441,
         "completions": 842555785,
         "continuation_cutoffs": 267815,
         "decisions": 1013583010,
         "live_observation_nodes": 103549323,
         "own_decision_nodes": 83216947,
         "pruned_actions": 4040237,
         "represented_completions": 4599906995,
         "represented_decisions": 16350752854,
         "represented_round_endings": 4599906995,
         "round_endings": 842555785,
         "terminal_mass_valued": 4599906995,
         "terminal_rows_valued": 842555785
       },
       "planning_prefix": {
         "arrival_mass": 374975370,
         "chance_rows_added": 278731,
         "completions": 49757,
         "decisions": 370024,
         "expanded_tasks": 802,
         "pending_mass": 0,
         "represented_completions": 53441190,
         "represented_decisions": 1054701450,
         "represented_round_endings": 53441190,
         "round_endings": 49757,
         "terminal_mass": 53441190
       },
       "planning_proposal": {
         "bounded_actions": 2427,
         "ceiling_cutoffs": 182275,
         "certified_score_loss_mass": 78775663,
         "certified_score_loss_rows": 2950358,
         "certified_score_win_mass": 31446587,
         "certified_score_win_rows": 136285,
         "chance_rows_added": 81845858,
         "completions": 78251806,
         "continuation_cutoffs": 2427,
         "decisions": 92499292,
         "live_observation_nodes": 8750290,
         "own_decision_nodes": 7902228,
         "pruned_actions": 182275,
         "represented_completions": 227461690,
         "represented_decisions": 764380162,
         "represented_round_endings": 227461690,
         "round_endings": 78251806,
         "terminal_mass_valued": 227461690,
         "terminal_rows_valued": 78251806
       },
       "planning_reference": {
         "certified_loss_mass": 18175452,
         "certified_loss_rows": 480435,
         "certified_win_mass": 29626002,
         "certified_win_rows": 80599,
         "chance_rows_added": 16621488,
         "completions": 16083550,
         "decisions": 19004982,
         "expanded_tasks": 1860914,
         "represented_completions": 40834326,
         "represented_decisions": 153423378,
         "represented_round_endings": 40834326,
         "round_endings": 16083550,
         "terminal_mass": 40834326,
         "terminal_rows": 16083550
       },
       "retained_v1_sampling": {
         "accepted_deals": 118,
         "attempts": 119,
         "bounded_draws": 5593,
         "seat_bounded_draws": 15
       },
       "retained_v1_source_execution": {
         "completions": 15,
         "decisions": 1735,
         "round_endings": 118
       },
       "retained_v1_source_identification": {
         "point": 151,
         "wp": 209
       },
       "retained_v1_source_native": {
         "completions": 15,
         "decisions": 1735,
         "round_endings": 118
       },
       "sources": {
         "completions": 4096,
         "decisions": 475740,
         "round_endings": 31819
       },
       "tails_actual_work": {
         "completions": 3886,
         "decisions": 14730,
         "round_endings": 3886
       },
       "tails_calls": {
         "learner_executor_calls": 8728,
         "opponent_adapter_calls": 6002
       },
       "tails_replay_work": {
         "completions": 3886,
         "decisions": 14730,
         "round_endings": 3886
       }
     }
   }
