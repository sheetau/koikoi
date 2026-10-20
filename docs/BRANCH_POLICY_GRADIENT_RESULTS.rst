Complete-match branch-gradient results
======================================

Sharing continuation-action uniforms reduced the observed conditional gradient
variance for both retained parents. The subsequent six-arm learning screen did
not establish a win-rate improvement: all six primary simultaneous intervals
include zero. No model is promoted. The signal and playing-strength results are
reported separately below.

Design and scope
---------------

Each parent supplies 64 fresh complete self-play matches. One unforced decision
is reservoir-sampled per match and weighted by that match's eligible-decision
count. Every legal first action is simulated to actual match termination.
Both continuation policies use their own public observations and T=0.05.

Both conditions share each root's hidden world and future round-deal stream.
The experimental condition also shares continuation-action inverse-CDF uniforms
by root and relative simulator step. The control uses independent action
uniforms. There are eight continuation replicates per root and condition.
No root, phase or replicate is selected according to its outcome.

The planned metric is the trace of conditional root-logit gradient variance.
It includes the eligible-decision weight. A post-hoc analysis uses the same
saved outcomes to compute the full 4,019,715-parameter actor gradient of the
64-root batch mean. Neither metric includes the variability from drawing new
root worlds. Eight-replicate ratios are descriptive; they are not confidence
bounds or general variance-reduction guarantees.

Results
-------

.. list-table:: Shared divided by independent conditional variance
   :header-rows: 1

   * - Retained parent
     - Root-logit gradient
     - Full actor gradient, post-hoc
   * - 1
     - 0.298240
     - 0.620943
   * - 2
     - 0.645048
     - 0.341429

Root-logit variance is 148.059847 versus 496.445462 for parent 1 and
259.591796 versus 402.438018 for parent 2. The unweighted metric is also lower
under shared uniforms for both parents.

Full-actor variance is 29,797.228992 versus 47,987.060522 for parent 1 and
56,458.083390 versus 165,358.305713 for parent 2. Thus the observed reductions
also appear after differentiation through the actual actor. This post-hoc
calculation reuses the same samples and supplies no independent replication.
It leaves the two parent parameter digests unchanged.

At least one shared replicate distinguishes legal actions at 40/64 and 46/64 roots.
Those frequencies describe available sampled outcome differences, not accurate
action rankings or the fraction of decisions that should change.

Work and verification
---------------------

Source work totals 14,609 decisions, 985 round endings and 128 complete matches.
Branch work totals 553,186 decisions, 40,707 round endings and 9,600 completions.
Branch completions share source worlds and are not independent fresh deals.
The full-parameter analysis adds 32 backward passes and no simulator decisions.

Seven implementation tests passed, with CPU/CUDA and upstream checks executed.
They cover an exact variable-length-game gradient, independent Python-rules
trajectory replay, source-root reconstruction, private-world input invariance
and actual pretrained-actor backward computation.

The independent diagnostic audit replays every recorded source and branch
trajectory on native CPU, verifies initial deals using Python initialization,
and recomputes variances with NumPy. It does not rerun neural inference.
The post-hoc parameter-variance calculation agrees between NumPy and Torch
on all four saved gradient matrices.

Closed artifacts and commands:

* ``runs/branch_policy_gradient_diagnostic``:
  ``python -m scripts.diagnose_branch_policy_gradient`` and
  ``python -m scripts.audit_branch_policy_gradient``.
* ``runs/branch_parameter_variance``:
  ``python -m scripts.branch_parameter_variance``.

The diagnostic protocol SHA256 is
``62c90e86e2e6f0940a7ef3dd1d935dba8891fb3fdd633f478330114a3eb7217d``.
Its independent audit SHA256 is
``3b99cc11482ab3d94f05f465fe4d101e69a916bdbe69d2d3ed67e7b21b260ecf``.
Do not rerun these completed commands or overwrite their outputs.

Interpretation and next test
----------------------------

The method changes actor supervision relative to the earlier local experiments:
all-action terminal outcomes directly train the actor, without a learned
bootstrap or a choice restricted to two frozen experts. All-action policy
gradients, rollout policy improvement and common random numbers have prior
work; see ``BRANCH_POLICY_GRADIENT.rst``. The diagnostic establishes a measured
effect in this implementation, not a new general theorem.

The fixed learning screen compared shared branches, independent branches and
root-sampled REINFORCE for both parents. All final endpoints were evaluated on
fresh paired development deals. Its design and primary criterion are recorded
in ``BRANCH_POLICY_GRADIENT_TRAINING.rst``.


Completed learning screen
-------------------------

Every arm received 128 single-step Adam updates from 64 complete source matches
per update. This matches source-match counts and optimizer steps, not total
simulation cost. All six final endpoints and both parents faced original WP
and Point on 1,024 paired streams per opponent, with both seats. This GPU
batch-128 development path is distinct from the strict CPU singleton paper
comparison. The evaluation adds 32,768 matches on 2,048 distinct deal streams.

The six primary comparisons use equal WP/Point mixtures and draw-adjusted wins.
A maximum-absolute-deviation bootstrap provides simultaneous 95% intervals.
All intervals include zero; the fixed progression criterion fails. Neither a
replicated gain nor a statistically established harm is identified here.

.. list-table:: Primary win changes in percentage points
   :header-rows: 1

   * - Comparison
     - Change
     - Simultaneous 95% interval
   * - Parent 1: shared minus parent
     - -0.7080
     - [-1.8799, +0.4639]
   * - Parent 1: shared minus independent
     - -0.1343
     - [-1.3062, +1.0376]
   * - Parent 1: shared minus sampled
     - -0.4883
     - [-1.6602, +0.6836]
   * - Parent 2: shared minus parent
     - +0.0366
     - [-1.1353, +1.2085]
   * - Parent 2: shared minus independent
     - +0.4517
     - [-0.7202, +1.6235]
   * - Parent 2: shared minus sampled
     - +0.6348
     - [-0.5371, +1.8066]

.. list-table:: All secondary matchup means (descriptive)
   :header-rows: 1

   * - Model
     - Opponent
     - Win percent
     - Final point margin
   * - parent1_shared_actions
     - wp
     - 49.4141
     - +0.472656
   * - parent1_shared_actions
     - point
     - 50.7812
     - -1.906250
   * - parent1_independent_actions
     - wp
     - 49.5605
     - +0.794922
   * - parent1_independent_actions
     - point
     - 50.9033
     - -1.598633
   * - parent1_sampled_return
     - wp
     - 49.7314
     - +1.000977
   * - parent1_sampled_return
     - point
     - 51.4404
     - -1.432617
   * - parent2_shared_actions
     - wp
     - 49.8291
     - +1.420898
   * - parent2_shared_actions
     - point
     - 50.1221
     - -1.991211
   * - parent2_independent_actions
     - wp
     - 49.4141
     - +1.203125
   * - parent2_independent_actions
     - point
     - 49.6338
     - -2.227539
   * - parent2_sampled_return
     - wp
     - 49.5117
     - +1.512695
   * - parent2_sampled_return
     - point
     - 49.1699
     - -2.403320
   * - parent1
     - wp
     - 49.6338
     - +1.141602
   * - parent1
     - point
     - 51.9775
     - -0.897461
   * - parent2
     - wp
     - 49.4629
     - +1.101562
   * - parent2
     - point
     - 50.4150
     - -2.381836

.. list-table:: Training work by arm
   :header-rows: 1

   * - Arm
     - Source decisions / rounds / matches
     - Branch decisions / rounds / completions
   * - parent1_shared_actions
     - 943,284 / 64,289 / 8,192
     - 2,372,739 / 175,872 / 39,399
   * - parent1_independent_actions
     - 949,889 / 64,004 / 8,192
     - 2,391,902 / 175,613 / 39,227
   * - parent1_sampled_return
     - 946,537 / 64,204 / 8,192
     - 0 / 0 / 0
   * - parent2_shared_actions
     - 956,535 / 63,796 / 8,192
     - 2,437,789 / 176,745 / 39,361
   * - parent2_independent_actions
     - 964,105 / 63,670 / 8,192
     - 2,483,054 / 178,367 / 39,478
   * - parent2_sampled_return
     - 962,630 / 64,086 / 8,192
     - 0 / 0 / 0

Totals: 5,722,980 source decisions, 384,049 round endings and 49,152 matches;
9,685,484 branch decisions, 706,597 round endings and 157,465 branch completions.
The evaluation uses 3,787,953 decisions and 254,310 round endings.

Training and evaluation audits passed. They replay every recorded transition
on native CPU, verify initial deals and root/outcome accounting, and independently
recompute all primary simultaneous intervals. These are transition/statistics
audits, not full neural-update replay. Eighteen implementation tests passed,
including CPU/CUDA exact checkpoint continuation and Python-rules replay.

A post-hoc check compares the first four paired streams from each of the four
baseline matchup cells with the original CPU singleton adapters. All 3,843
recorded action requests agree. The check covers 32 saved match trajectories;
it does not make the complete GPU evaluation interchangeable with CPU evaluation.

Training session 64742 and completion session 15164 exited successfully. No
study job remains live. All artifacts are closed; do not restart these jobs.
The playable model, initial dealer and reserved 101M deal block remain unchanged.

Receipts under ``runs/branch_policy_gradient_training``:

* Training completion:
  ``e8a28db0e045b87054df96ebe2106873e9d6e96764a74083e8a896cc3ea1472a``.
* Training audit:
  ``6c95abeba7d2a2a447298c9402c4f72cbaa341c14501c2d07e97fb35b9e03658``.
* Evaluation summary:
  ``3b03a2f902b70d40be14e1756e3ed92b3ebcd502cb481f7432956c171297976f``.
* Evaluation audit:
  ``7d87dfd2a6a71eae875fa12f641f12092078966cd7714ab54f2d59d0dc5475f3``.
* Baseline CPU check:
  ``70b4704d7aa03603350c63f8588e0ec4eef26781d76e96476ef5308f3f45d60d``.

The present contribution is a measured reduction of conditional gradient noise
and a completed controlled learning test. Stronger play and a distinct broader
methodological contribution remain open. This small screen does not decide
convergence or the best allocation of the simulation budget. Changing the
learning rate, root sampling or training duration would require a new experiment;
none is retrospectively substituted for these fixed results.
