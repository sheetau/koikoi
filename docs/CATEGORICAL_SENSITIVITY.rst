Exact categorical sensitivity envelope
======================================

The running priority-root study uses a public probability proxy,
0.05 + sqrt(1 - sum(p_a^2)). This separate saved-data analysis examines how
closely that proxy describes the gradient it is intended to allocate.
It changes neither the running study nor its candidate and primary criterion.

An exact finite-action bound
---------------------------

Let p = softmax(z), let each action's sampled terminal outcome q_a belong to
[-1, 1], and differentiate only through p. The normalized-logit gradient of
the expected outcome is::

    g_a = p_a (q_a - sum_b p_b q_b).

For a subset S of actions, write s = sum_{a in S} p_a. Its exact maximum norm
over the outcome box is::

    B(p)^2 = 4 max_S [
        (1-s)^2 sum_{a in S} p_a^2
        + s^2 sum_{a not in S} p_a^2
    ].

To see this, g is linear in q and its squared norm is convex. Every point
of the box is a convex combination of vertices, so a maximizing vertex exists.
Choose q_a = +1 on S and -1 elsewhere; its mean is 2s-1. Substitution gives
the displayed expression. Complementary subsets give opposite gradients with
the same norm. Thus at most 2^(k-1) partitions suffice for k legal actions;
the local game has at most eight. A one-action policy has zero gradient.

For two actions the bound is 2 sqrt(2) p(1-p). For an even number k of equally
likely actions it is 1/sqrt(k). Cauchy--Schwarz also gives the looser envelope
sqrt(1 - sum(p_a^2)): write g as the covariance of the one-hot action and q,
use Var(q) <= 1, and sum the coordinate bounds. Near a deterministic binary
policy, the exact envelope vanishes linearly in the minority probability;
the looser envelope vanishes only as its square root.

This is a bound for normalized logits, not for all network parameters.
Temperature contributes a common 1/T factor when differentiating raw scores;
the network Jacobian can change relative gradient norms between observations.
The exact envelope also does not predict whether sampled outcomes distinguish
actions. Equal outcomes produce zero gradient even when B(p) is large.

Allocation interpretation and fixed diagnostic
----------------------------------------------

For a fixed list of roots with selection probabilities rho_t, an
importance-corrected sum estimator has second moment
sum_t E[||g_t||^2] / rho_t. Replacing each unknown moment by B(p_t)^2 gives an
upper bound minimized by rho_t proportional to B(p_t), on positive-bound roots.
This minimizes that algebraic bound; it is not an actual variance-optimality
claim for a game or a neural actor. The diagnostic retains the same positive
0.05 floor as the current sampler, so it does not use that ideal allocation
exactly either. No invention of norm-based importance sampling is claimed.

Before reading its result, the diagnostic fixes exactly three pool proposals:
uniform, 0.05 + the Gini envelope, and 0.05 + the exact envelope. It reuses
both 8,192-root pools from the closed coverage analysis, with no new simulator
decisions, optimizer steps or independent samples. Stored float32 probabilities
are renormalized in float64 on legal actions; the largest correction is reported.
The sampled root-logit gradients include their original eligible-decision K
weights. The ratio to uniform second moment is::

    mean(w) * mean(||K*g||^2 / w) / mean(||K*g||^2).

This is hypothetical resampling across each saved pool. It is not a measurement
of within-match reservoir variance, future learning efficiency or playing
strength. The exact-envelope comparison is exploratory; its outcome cannot
change the already-running priority study.

Implementation: ``scripts/diagnose_categorical_sensitivity.py``.
Tests include binary/uniform closed forms and a separate finite-difference
softmax calculation over all ternary outcome assignments for two three-action
policies. Existing artifacts are never overwritten.

Completed saved-pool result
--------------------------

Both pools contain 8,192 roots and at most eight legal actions. Relative second
moments (uniform = 1) are:

.. list-table:: Finite-pool root-logit second moments
   :header-rows: 1

   * - Parent
     - Gini proxy
     - Exact-envelope proxy
   * - 1
     - 0.6455107821
     - 0.6045907253
   * - 2
     - 0.6654026332
     - 0.6380281976

The exact-envelope proposal lowers this particular pool quantity by about
6.34% and 4.11% relative to the Gini proposal. The normalization corrections
are at most 2.069e-7 and 1.785e-7. They explain the small difference between
the Gini numbers here and the original calculation on unrenormalized stored
probabilities. Both analyses compute the summaries in float64.
All observed sampled gradients satisfy the exact bound, which also lies below
the Gini bound. Nonconstant sampled outcomes occur at 2,550 and 2,648 roots.

The diagnostic is complete. Its three tests pass with zero skips; no new
training, simulation or independent sample was added. A modest improvement
in this reused-pool quantity alone does not justify another full training grid.
First complete and interpret the fixed priority/step-size experiment.
Artifacts: ``runs/categorical_sensitivity``.
Protocol SHA256 8a42b1d1feddc5785fc17cadeb0ce5124f38c726562fa70587f3f906502180e4.
Summary SHA256 3043b0c4735cb489448ce518cabea19bca2b9165c9405c781f2b5e9cc8eb1617.
