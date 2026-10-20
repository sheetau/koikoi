Gradient blocks and the next missing control
============================================

The completed priority/step-size study fails its strength criterion. Increasing
step size produces a significant parent-relative loss in one primary comparison.
Before choosing another parameter restriction, this analysis reuses the saved
full-actor gradients to distinguish output weights from encoder parameters.

Both retained actors contain 4,019,715 parameters: 768 output weights, three
output biases and 4,018,944 encoder parameters. All eight old gradient replicates
are retained for both parents and both continuation-coupling conditions. Source
hashes, parameter order and parent identities are checked. The three blocks sum
back to each previously reported full-gradient variance and squared mean norm.
No simulator call, forward/backward pass or optimizer update is added; loading
models only establishes parameter order and identity.

Results
-------

Variance is the trace of the raw-parameter gradient covariance conditional on
the same 64 source roots. It excludes variability from new roots/worlds and is
not the variance of an Adam update. The last column estimates the squared norm
of the conditional expected gradient by subtracting variance/8 from the squared
replicate mean. Negative estimates indicate sampling uncertainty; actual squared
norms cannot be negative. Eight replicates do not provide precise signal bounds.

.. list-table:: All fixed parameter blocks
   :header-rows: 1

   * - Parent
     - Coupling
     - Block
     - Variance trace
     - Squared mean norm
     - Noise-corrected estimate
   * - 1
     - shared_actions
     - output_weights
     - 178.410476
     - 9.72682057
     - -12.574489
   * - 1
     - shared_actions
     - output_bias
     - 5.55035383e-12
     - 1.7408297e-13
     - -5.19711258e-13
   * - 1
     - shared_actions
     - encoder
     - 29618.8185
     - 2557.05148
     - -1145.30083
   * - 1
     - independent_actions
     - output_weights
     - 201.052034
     - 46.0448152
     - 20.9133109
   * - 1
     - independent_actions
     - output_bias
     - 1.47752287e-11
     - 5.40367751e-12
     - 3.55677392e-12
   * - 1
     - independent_actions
     - encoder
     - 47786.0085
     - 8477.85822
     - 2504.60715
   * - 2
     - shared_actions
     - output_weights
     - 1502.94993
     - 469.338397
     - 281.469655
   * - 2
     - shared_actions
     - output_bias
     - 3.89783444e-12
     - 2.27373675e-13
     - -2.59855629e-13
   * - 2
     - shared_actions
     - encoder
     - 54955.1335
     - 17031.035
     - 10161.6433
   * - 2
     - independent_actions
     - output_weights
     - 3448.06477
     - 1151.22619
     - 720.218092
   * - 2
     - independent_actions
     - output_bias
     - 7.70532844e-12
     - 8.53539461e-13
     - -1.09626594e-13
   * - 2
     - independent_actions
     - encoder
     - 161910.241
     - 45218.3066
     - 24979.5265

Output weights contain about 0.60% and 2.66% of shared-condition variance, but
removing the encoder would also remove its gradient signal. The output and
encoder corrected estimates are both negative for parent 1 and both positive
for parent 2. This small conditional diagnostic does not establish that an
output-only update is preferable. No head-only training is launched from it.

The data-use control
--------------------

A more direct missing control is apparent in the actual learner code.
``sampled_return_loss`` uses one reservoir-selected decision per source match,
with its eligible-decision count K as weight. Let g_t be that match's ordinary
Monte Carlo score-function contribution at eligible decision t, including the
terminal reward from that decision's player's viewpoint. Conditional on the
complete source trajectory tau, uniform root selection gives::

    G_root = K * g_T
    E[G_root | tau] = sum_t g_t = G_full.

The conditional-variance decomposition therefore gives::

    Cov(G_root) = Cov(G_full) + E[Cov(G_root | tau)].

Using all source decisions removes this additional root-selection covariance
without changing the expected gradient. This is the standard conditional-
expectation argument, not a new reinforcement-learning algorithm. It compares
full-source Monte Carlo with root-sampled Monte Carlo using the same source
returns. It does NOT prove that full-source Monte Carlo has lower variance than
the independently simulated all-action branch estimator, and it does not prove
a win-rate improvement after clipping or Adam.

The old complete-MC training study used MLP128x2 and PPO, as recorded in
EXPERIMENTS.md's complete-MC sections. It is not this retained-Transformer,
single-step complete-trajectory comparison. Earlier retained-agent directional
probes used complete returns for diagnostics; they were not this training study.
The next implementation should retain every eligible source state/action and
accumulate the full Monte Carlo gradient before one optimizer step. Normalize
by the number of complete matches, not the number of eligible decisions, to
retain the same expected per-match decision sum. It should
compare against the root-sampled and all-action estimators under disclosed
source, branch and optimizer budgets. A further branch-control-variate method
must earn its advantage against this full-source control.

Status and artifacts
--------------------

The block analysis is complete and closed. It introduces zero new independent
observations. The source sum checks execute on all four real gradient matrices;
Ruff passes. Full-trajectory training is not yet implemented or launched here.
Implementation: ``scripts/branch_gradient_blocks.py``.
Artifacts: ``runs/branch_gradient_blocks``.
Protocol SHA256 682f27a594436313a44a67beca95da019fe72c697b057b618208af177154c2d4.
Summary SHA256 28c18f8df7eff06854981784176b25e419da38e3b837d8e17a8f6a00cbb2d278.
