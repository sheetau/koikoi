Boundary residual correction: measured result
=============================================

The 50-cell diagnostic completed and passed independent scalar verification.
The fixed alpha=1 correction reduces variance modestly, but its additional
cost outweighs that reduction in every measured cell. It is not adopted for
the next value-learning experiment.

====================  =============  ==================  =================
Hidden-world belief   Variance ratio  Variance x seconds  Variance x steps
====================  =============  ==================  =================
Uniform               0.937784       1.188984            1.319444
Identified history    0.934917       1.221901            1.323920
====================  =============  ==================  =================

Entries are means of per-cell ratios relative to the ordinary terminal
estimator. Lower is better; there are 25 cells per row. All 50 time-cost ratios
exceed one. Both parents and both opponent groups show the same direction.
Ratios combine empirical variance with observed cost; they are not a trial
with exactly matched wall-time budgets or a population confidence guarantee.

With 16 fixed estimator blocks per cell and an independent 2,048-sample
terminal reference, mean block squared errors are 0.00351173 versus 0.00360707
for corrected versus ordinary uniform estimation, and 0.00399291 versus
0.00453186 with identified history. The corrected blocks use extra cheap
samples, so these error differences alone cannot establish greater efficiency.
The reference mean itself contains sampling noise.

Recorded high-fidelity simulation takes 101.83 seconds, cheap simulation 26.13
seconds and independent reference simulation 408.58 seconds. These timings
include the simulation/lookup calls; they exclude hidden-world construction,
initial loading, prior exact inference and report writing. They do not measure
end-to-end playing latency. The finding is unfavorable even under this narrower
timed scope.

Work and verification
---------------------

======================  ===========  =============  =================
Category                Decisions    Round endings  Terminal endings
======================  ===========  =============  =================
Coupled high/proxy      4,202,458     308,866        51,200
Independent cheap       1,529,919     204,800        2,528
Independent reference   16,823,174    1,235,907      204,800
======================  ===========  =============  =================

Cheap trajectories stop at their first round ending; only 2,528 of them also
end a match. These are hypothetical conditional paths, not fresh ordinary
evaluation matches. There are zero actual-source completions, training updates
or new evaluation deals. Source reconstruction adds 953 transitions.

The independent auditor reconstructs every proxy from public endpoint fields,
absolute seat/model axes and raw table arrays, then verifies signed terminal
differences, scalar moments, fixed-block estimates and work totals. It adds
953 source transitions and zero model calls. It does not rerun the hypothetical
neural trajectories. Four capability tests pass, including two actual CUDA/
independent-reference trajectory checks for both seats. Two scalar-auditor
tests also pass. All have zero skips.

Protocol 2095651ec1a1027e23aac7e1fe6b9a2dc5f89ec43da16620a546e8e181fd941b
binds 198 inputs. Artifacts in runs/boundary_residual_diagnostic:

* Report: 5f697a32b281ff7bc891e2980195b5c6a5b82e5752543aff3c47c4b53ee48a2c.
* Audit: e74599970f1b0d4fe69a61e5bfee086353f6f4b6d570c1922c6e7df4e51fbba2,
  binding 253 inputs.

Post-hoc allocation check
------------------------

A separately labelled calculation fits the coefficient and asymptotic sample
allocation to the observed variances, covariance and per-sample cost. It can
always choose the ordinary estimator by setting alpha=0. The resulting mean
plug-in variance-cost ratios are 0.992473 for seconds and 0.996522 for decisions;
27/50 and 19/50 cells, respectively, receive nonzero fitted coefficients.
These are optimistic estimates from the same pilot data, not measured gains
from independently tested settings. Their average suggested gains are below
one percent, giving little reason to prioritize another allocation trial.

Four tests check the allocation formula against independent dense-grid
minimization and covariance boundary cases. The analysis adds no neural calls
or simulation and adopts no parameter. Receipt
f87983c730d115b56efb27746c037da81141fd35153ce29548ca71e644256d1c binds 257 inputs.

The correction identity and use of variance reduction in game planning are
established prior work, documented in BOUNDARY_RESIDUAL_PRIOR_ART.rst. This
completed application does not supply a new strength or efficiency claim.
