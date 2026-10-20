Branch-gradient signal diagnostic
=================================

Sharing continuation-action uniforms reduced the observed conditional gradient
variance for both retained parents. This supports testing the learning method.
The diagnostic did not update either model or measure a playing-strength gain.
The subsequent six-arm training study is a separate experiment.

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

The fixed learning screen compares shared branches, independent branches and
root-sampled REINFORCE for both parents. It retains all final endpoints and uses
fresh paired development deals. Its design, work distinction and simultaneous
primary criterion are in ``BRANCH_POLICY_GRADIENT_TRAINING.rst``. A claim of
stronger play requires that evaluation, beyond the signal measurements here.
