Global and local hand proposals
===============================

The previous exact diagnostic found severe finite-sample error and an isolated
supported hand under one-card swaps. ``mixed_history_sampler.py`` therefore
initializes by uniform rejection and then mixes uniform global proposals with
single-card exchanges. Each proposal has Boolean historical-action likelihood
under one deterministic model selected from prior-round public evidence.

Both proposals are symmetric on the fixed-size hand space. Reject incompatible
or invalid hands, retaining the current state on every rejection. The uniform
distribution over supported hands is invariant. Positive global probability
allows movement between disconnected local components. Initialization, invalid
hands and rejections all consume the fixed attempt budget. If no supported
initial hand is found, return an explicit failure. Cache deterministic likelihood
answers within this root only. Model selection provenance remains the caller's
responsibility; the true opponent label is not evidence.

The first accepted global hand is uniform over support, conditional on finding
one within budget. Its value is independent of the initialization waiting time.
Subsequent states are stationary marginally but correlated. Report chain length
and distinct hands, never treat chain length as independent effective samples.
This is a standard invariant-kernel construction, not a new general MCMC method.

Eight tests pass,zero skips: exact rational transition probabilities for the
actual sampler on a disconnected finite fixture; rejection holding and caching;
budget failures; full historical replay; hidden-world substitution; invalid
controls rejected before sampling. The root wrapper does not choose game actions.

Fixed development comparison before execution
--------------------------------------------

Reuse all four cases from ``sampled_history_exact_check`` (three distinct
own/public roots, two old deal seeds). Use the identified-model exact support
as a cached likelihood oracle. Compare fixed 128-proposal independent rejection
with the mixed chain at global probability 0.2. Run 256 proposal-RNG replicates
per case, seed ``1701 + 1000000*parent + 1000*case_index + replicate``.
Pair the RNG initialization so both methods share the initialization prefix;
their initialization success/failure must agree. Do not retry or discard failed
replicates. The oracle adds no neural calls or future game trajectories.

Measure marginal squared error against the exact posterior, divided by the
public unknown-card count. Report failures, conditional mean error, paired
differences, distinct hands and actual unique oracle queries. Attempt budgets
are equal; neural cost and wall-clock game latency are not claimed equal.
Compute the same conditional mean-squared errors analytically from the full
finite transition kernel, averaging over the initialization waiting time and
the binomial number of accepted independent proposals. These calculations
characterize these finite cases, not new opponent or root generalization.

Freeze inputs before comparison. Retain all cases, including the duplicated
Point root and every failure. No action-strength test or deployment follows
automatically. Existing strength protocols and queued training are unchanged.
