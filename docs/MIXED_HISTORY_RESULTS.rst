Mixed-proposal hand estimation: development result
=================================================

Global/local hand proposals reduce posterior approximation error at both WP
development roots, but increase it at the Point root. The implementation is
validated as an inference component; it is not a universal replacement for
independent rejection sampling or evidence of stronger game decisions.

The design was frozen in ``MIXED_HISTORY_SAMPLER.rst``. Every method receives
128 proposal attempts per replicate, including initialization, invalid hands
and rejections. The chain uses global probability 0.2. The four cases are three
distinct own/public roots sharing two existing deal seeds; the Point root is
duplicated across parents.

Conditional expected marginal squared error against the exact posterior,
averaged over public unknown cards and all proposal randomness:

================  ===========  ===========  ==================
Case              Independent  Mixed chain  Change
================  ===========  ===========  ==================
Parent1 / WP      0.077195     0.057376     -25.67 percent
Parent2 / WP      0.074251     0.042909     -42.21 percent
Point (both)      0.012263     0.013946     +13.72 percent
================  ===========  ===========  ==================

These expectations condition on finding an initial supported hand. Both methods
have the same theoretical failure probabilities: 0.263712 for parent1/WP,
0.372472 for parent2/WP, and 0.00000459 for Point. Failed runs return no estimate;
the implementation does not silently replace them with a uniform posterior.
All failures remain in the benchmark records.

The finite-state calculation averages chain autocovariances over the random
initialization waiting time. Independent rejection uses the conditional
binomial accepted-count distribution. An exhaustive rational two-attempt test
checks both expectations against every possible path, independently of the
spectral calculation. The actual sampler also passes an exact rational
stationarity/connectivity test and public-observation privacy checks.

At 256 paired proposal-RNG replicates per case, failures are 66, 90, 0 and 0,
respectively. Empirical errors show the same direction as the finite-state
expectations. The Point chains visit about 40 distinct supported hands versus
about 12 for independent rejection, yet have greater estimation error: distinct
hand count is not an independent effective sample size. No confidence interval
or generalization claim is assigned to four selected development cases.

Actual published-model integration
----------------------------------

One fixed RNG replicate per case was repeated with actual published-policy
likelihoods. All chain states and checked likelihoods match the cached exact
oracle. Substituting the hidden world while preserving the acting observation
leaves the complete result unchanged. Full unpruned replay also agrees for
every valid queried hand. The checker shares the canonical-world constructor
and published inference adapter; it is not a second model implementation.

Nine unit tests pass, zero skips (sampler eight in 1.26 seconds; analytic
expectation one in 1.27 seconds). Ruff and format checks pass for the new code.
The cached benchmark uses 262,144 proposal attempts, 188 recorded source-prefix
replay decisions and zero neural calls. Runtime validation adds 188 source
replay decisions, 3,206 pruned historical transitions/2,104 model calls for
actual and hidden-substituted runs, plus 3,530 full historical transitions/
2,051 model calls. Its 1,536 proposals comprise actual, hidden-substituted and
cached-oracle runs. These are separate from unit fixtures. No new evaluation
deal, future rollout, training match or game action was added.

Artifacts:

* Benchmark protocol: d2cfa2a3df9f703faf3aef0dce0defcff88f617f539a7a5451356cf0a744c0e3.
* Benchmark report: 1211943d0ac82c375cec6a9f12d18d5068f4a0c828015c7bce196531dad572ac.
* Actual-model verification: eaf60c75b07a335c047db0b18f4e854ef5298215589542311f92d629a1d93cb7.

The next decision must account for both initialization failures and the
opponent-dependent error tradeoff before using this estimator for search.
The canonical playable model and ongoing strength trials remain unchanged.
