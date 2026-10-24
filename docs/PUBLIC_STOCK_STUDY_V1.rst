Public-stock development protocol v1
====================================

This protocol separates engineering validation, adapted-policy qualification,
rule screening, and independent confirmation. It does not extend any prior
cohort. The three-factor semantics are fixed in PUBLIC_STOCK_RULES.rst.

Training
--------

The matched matrix has eight profiles and two initialization/sampling seeds,
911 and 912. Each model receives 1,024 PPO updates, 512 environments and 64
decisions per environment per update: 33,554,432 decisions, or 536,870,912 for
the matrix. Count completed rounds and matches separately. The two initial
deal-seed blocks start at 814,000,000 and 815,000,000; episodes advance only
after terminal matches. Rules share these blocks for matched training design.

Each policy has 660 public/own inputs, two 256-unit tanh layers, 50 masked
action logits and a scalar observation-only critic. The reward is terminal
match win (+1), loss (-1), or draw (0), with no point-based shaping. PPO uses
three epochs, minibatches of 4,096, learning rate 0.0003, clipping 0.2,
entropy coefficient 0.01, signed GAE lambda 0.99 and discount 1. All phases,
including the additional stock choice, count as decisions. This matches
decision samples and network updates, not completed games or elapsed time.
The final checkpoint is used; there is no best-checkpoint selection.

Qualification
-------------

Evaluate both control policies on 256 new duplicate streams starting at
816,000,000, against a card-weight greedy policy and the released GPU-batched
RL-WP without endgame assistance. The greedy policy stops whenever possible;
at hand or stock choice it maximizes immediate weighted capture. This is a
quality diagnostic, not a new comparison to the historical assisted baseline.
Before interpreting a rule screen as evidence about strong AI, both controls
must reach at least 65% against greedy and 45% against unassisted RL-WP.
These are engineering gates on point estimates, not significance claims.
If a gate fails, record the failure and improve policy adaptation under a
new protocol rather than declaring a strength-maximizing rule.

Computation screen (conditional on qualification)
-----------------------------------------------

Use 64 new duplicate streams starting at 817,000,000, two learner seeds,
all eight rules and budgets 0, 2, 8, 32 hidden-world particles per legal root
action. This gives a no-search baseline and a 1:4:16 rollout-budget ratio.
Each arm faces its own frozen adapted policy without search. The horizon is
32 decisions including the root action. Every rollout policy sees only its
own observations; nonterminal leaves use that policy's bounded public critic.
The root compares paired advantages on common hidden worlds and subtracts
one sample standard error; it keeps the baseline unless a positive lower
score is available. This is a heuristic guard, not a safety certificate.

The primary screen quantity is win-score improvement at budget 32 over budget
0, taking the smaller improvement across training seeds. Select the noncontrol
profile with the largest such value; resolve exact ties by profile name.
Report all eight curves, including adverse or nonmonotonic outcomes. Require
positive gains for both selected seeds and a larger worse-seed gain than the
control before launching a selected-rule confirmation. Otherwise close the
screen as unsuccessful and do not repurpose its streams for confirmation.

Confirmation (reserved; not authorized by a screen ranking alone)
---------------------------------------------------------------

If the screen gate passes, fix the selected rule and compare it with control
on 512 new duplicate streams starting at 818,000,000. Retain both learner seeds
and all four budgets. Use the same opponent definition and budget settings.
The four primary inequalities are positive high-budget gain for each seed
and a positive selected-minus-control difference in that gain for each seed.
Use 20,000 paired deal-cluster bootstrap replicates and a one-sided 98.75%
lower bound per inequality (nominal Bonferroni 95% family). Do not extend,
pool, or reselect after seeing this cohort. A successful result still needs
cross-opponent and stronger-policy replication before a default-rule change.

Audit and limits
----------------

Store checkpoint hashes, exact protocol, actions, final states, deal IDs and
separate counters. Independently replay every scored trajectory through
Python. Sampling uncertainty is distinct from exact conditional certificates.
Wall time includes inference and audit overhead unless explicitly separated;
it is not a controlled latency benchmark. These newly trained actors are
not the project's two stronger published-policy adaptations. Passing the
qualification gate alone does not establish parity with those parents.
