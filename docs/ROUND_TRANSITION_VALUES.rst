Round-transition policy values: construction and capability protocol
==================================================================

Question and distinction from earlier work
-----------------------------------------

The bounded history-conditioned search produced no changes on its fixed eight
source roots. That small result neither establishes that search is ineffective
nor licenses lowering its threshold. Its complete-match samples include all
later-round randomness. Test a different estimator that stops at the first
round ending and evaluates the public context before the next deal.

The earlier round_boundary_calibration experiment fit a seat-symmetric,
two-coefficient logistic model to old MLP self-play outcomes. It did not improve
held-out Brier loss over the card-aware network and was not adopted. The present
construction instead samples complete one-round transitions for every live point
total and backs them up with fixed policies. It does not reuse those fitted
coefficients, old MLP actors or their evaluation deals.

Conditional expectation and truncated rollout are established techniques, not
new general algorithms. See Art B. Owen, Monte Carlo theory, methods and examples,
Chapter8, https://artowen.su.domains/mc/Ch-var-basic.pdf , and Dimitri Bertsekas,
Rollout, Policy Iteration, and Distributed Reinforcement Learning (2020),
https://web.mit.edu/dimitrib/www/dp_rollout_book.html . A Koi-Koi contribution
would require the concrete construction and measured estimator/policy benefit.

Public context and declared chance model
---------------------------------------

For one retained parent, keep all axes: round index0..7,parent absolute seat0/1,
opponent model Point/WP,player-zero points1..59,and absolute dealer0/1. Total
points remain60. This gives3,776 contexts per parent. Do not fold seats or dealers
by symmetry: the published actors can differ under seat relabeling. Policies
are fixed,frozen and observation-only,with public endgame assistance.

At a new round, cards and within-round action histories reset. Sample a fresh
valid deal for each context/replicate and run exactly that round. The transition
records retain next points,dealer,round and true match-terminal flag. Values use
those public fields only; next-round private cards are not looked up or evaluated.
Terminal match utility is sign(points0-30),with sign reversed for parent seat1;
draws are zero. A nonterminal round ending uses the next-round table.

The backward recursion defines an independent-fresh-deal round model. The native
multi-round simulator derives later deals from a finite shared seed; no proof of
its statistical independence from all earlier observations is asserted. The
table therefore makes a chance-model approximation as well as finite-sample
and batched-policy approximations. Keep the actual match evaluator unchanged.

Under an ideal independent-deal model and exact continuation values, replacing
a terminal return by its conditional expectation at the boundary preserves its
mean and cannot increase its variance. This statement does not automatically
apply to an estimated table,or guarantee reduced paired-action error/stronger
play. Shared table error,point-context coupling and candidate selection must be
measured. Independent confirmation data cannot reuse an adaptively selected
value-table estimate without accounting for that dependence.

Implementation
--------------

scripts/round_transition_values.py supplies fresh public-context starts,a CPU/CUDA
workspace that freezes each lane at its first round ending,and float64 backups
over every context. The workspace retains the true next-round/terminal endpoint;
its temporary terminal flag is scratch control only. No policy acts on the new
deal before the boundary value is applied. Count simulated decisions,round
endings and terminal match endpoints separately. Artificial conditional starts
do not constitute complete independent matches from the standard initial state.

scripts/build_round_value_table.py uses the fixed batched parent/WP/Point routing
already verified in the history-search adapter. For a given round/sample,the
same valid deal is reused across score,dealer,seat and opponent contexts. This
common randomness is deliberate and must be retained when estimating table
uncertainty. Round seeds are seed+1000000*round_index+sample_index. Sampled public
endpoints,per-round files,backed-up values,model/source identities and work are
saved. No actor training,point interpolation or seat-symmetric fitting occurs.

Correctness checks
------------------

Seven focused tests pass on CPU/CUDA: public-context/deal construction,reference
trajectory equality through the first round ending from both parent seats,
freezing completed lanes,bankruptcy/final-round handling,terminal draws,and
input preservation. Exhaustive binary-outcome paths independently verify the
entire backup logic across selected rounds/points/dealers/seats/models. A
nonterminal eighth-round record and invalid live point totals are rejected.
These tests do not establish table accuracy or policy improvement.

Fixed capability run, before large computation
---------------------------------------------

Run both retained parents,each with exactly TWO samples per context,batch256,
base seed32210001. Both parents share the sixteen (round,seed) deal inputs.
Each parent simulates7,552 conditional rounds;the pair15,104. Report true
terminal-match endpoints separately;independent complete matches and policy
updates are both zero. Two samples are intentionally only a full-grid plumbing
and cost check. Do not use this noisy table for deployed search or strength claims.

After committing the implementation and this protocol,run sequentially:

  .venv/Scripts/python.exe -m scripts.build_round_value_table --parent 1 --samples 2 --seed 32210001 --output runs/round_value_capability/parent1
  .venv/Scripts/python.exe -m scripts.build_round_value_table --parent 2 --samples 2 --seed 32210001 --output runs/round_value_capability/parent2

Require unchanged inputs,all32 round/seat/model groups per parent,7,552 actual
round endings per parent,bounded values,and independent reconstruction of the
saved public-transition backups/accounting. Preserve every output. Do not extend
the sample budget from intermediate values. A later estimator-accuracy protocol
needs fresh independent deal streams,shared-deal uncertainty analysis and a
matched full-match-rollout reference before any strength comparison.
