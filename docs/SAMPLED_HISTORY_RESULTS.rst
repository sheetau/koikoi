Bounded hand-inference development result
========================================

The fixed-budget sampler reaches larger hand spaces, but its belief accuracy
does not improve consistently across both parent policies. It is not ready for
deployment or a playing-strength claim. The main multi-round training and
evaluation pipeline is unchanged.

The prespecified diagnostic retained all 32 training-prefix attempts, sharing
eight source deal seeds. Twenty-five attempts supplied an eligible root; seven
did not. All selected roots had four opponent cards remaining, with 4,845--5,985
possible hands. Each received 128 uniform proposals. Of 3,200 draws, 65 were
invalid initial hands and consumed the budget. Every estimator returned positive
mass at all 25 roots. No failed attempt or draw was replaced.

Mean squared card-membership error over public unknown cards (lower is better):

================  =====  ========  ===========  =====================
Source parent     Roots  Uniform   Joint models  Publicly identified
================  =====  ========  ===========  =====================
Parent 1          12     0.157604   0.161474     0.178599
Parent 2          13     0.156065   0.139112     0.140134
Both              25     0.156804   0.149846     0.158597
================  =====  ========  ===========  =====================

These are descriptive development measurements on selected roots, not
independent strength observations. Joint conditioning lowers the pooled error
by 0.006958 but increases it for parent 1. Restricting to the publicly identified
model increases the pooled error by 0.001793 relative to uniform proposals.
Correct opponent identification alone therefore does not establish better
finite-budget hand estimates in this diagnostic.

For identified-model estimates, effective proposal counts range from 1 to 95,
with median 7; 14 of 25 roots have effective counts at most 10. Distinct supported
hands have median 7. Joint-model effective counts have median 9. These small
supports motivate testing proposal quality and uncertainty handling before using
these estimates for decisions. They do not prove the cause of the observed
error differences. Increasing the budget on these same roots would be further
development, not independent confirmation.

Inference took 33.58 seconds total, median 1.29 and maximum 2.40 seconds per root,
with two Torch CPU threads while the separate match evaluation was running.
This is not an isolated latency benchmark. The current playable policy remains
unchanged.

Validation and work
-------------------

The sampler's four tests and diagnostic's three tests pass with zero skips.
The independent replay checker verifies all 3,135 valid proposed hands against
both published models using full histories without early rejection. All
compatibility flags agree. Separate scalar weighting and error arithmetic agree
within 1e-14. The checker shares the canonical-world constructor and published
inference adapter; it does not independently reimplement ineligible-root
selection. No hidden cards enter proposals or likelihood predictions. True
hands are accessed only after predictions to score errors.

Producer work is 3,144 recorded source decisions, 216 recorded round endings
and three recorded match endings, plus 15,478 hypothetical history transitions
and 14,831 published-model calls. Verification adds 953 source-prefix replay
decisions, 53,278 full hypothetical-history transitions and 28,404 model calls.
These are replay work, with zero new evaluation deals, future rollouts or
training matches. Unit-test fixture work is separate from these diagnostic
counts.

Artifacts under ``runs/sampled_history_diagnostic``:

* Protocol: e4cf707c26b1c5f5ccead8681ba827abe652ef3d5eb4c655f736d5080dbafdd3.
* Producer report: 1ae94dc9b18c5b3b262247c6883a4ee54074fddf1fa6c3ab16ff74887970c032.
* Independent replay: 2c3b08ca3cfbf19a420576533c34a2bb7db8ce0937d8fb88bfc815580ec1eac4.

See ``SAMPLED_HISTORY_SUPPORT.rst`` for the design fixed before execution.
The result supplies an implementation capability and a concrete weakness to
address; it does not establish a new general inference algorithm or stronger
Koi-Koi play.

Exact follow-up: approximation error and support connectivity
------------------------------------------------------------

The subsequent design in ``SAMPLED_HISTORY_EXACT_CHECK.rst`` selected the first
eligible source case in each parent/opponent group before computing exact
posteriors. These four cases share two source seeds and only three distinct
own/public roots: the two Point cases are the same root. They are development
cases, not independent confirmation.

All 5,985 candidate hands per case were enumerated: 23,940 combinations, 21,657
valid hands and 2,283 invalid initial hands. Both conditioned exact distributions
lower card-membership error relative to exact uniform inference in every case:

================  ========  ===========  ================  ==================
Case              Uniform   Joint exact  Identified exact  Identified sampled
================  ========  ===========  ================  ==================
Parent1 / WP      0.154195   0.116084     0.089490          0.285714
Parent1 / Point   0.152385   0.124019     0.128393          0.165209
Parent2 / WP      0.154199   0.118226     0.090062          0.095238
Parent2 / Point   0.152385   0.124019     0.128393          0.091270
================  ========  ===========  ================  ==================

The 128-proposal approximation is inaccurate at the two WP roots: identified
estimates have mean absolute marginal errors 0.245776 and 0.221532 relative to
the exact distribution, with only one effective proposal each. Sampled joint
inference has a maximum per-card error of 0.926829. A lucky sampled hand can
also score better than the exact posterior on one realized outcome, as the
second parent's Point case illustrates. Single-hand error does not by itself
measure posterior approximation quality.

The exact computation took 172.39 seconds across four cases. It adds 188
source-prefix replay decisions, 61,727 hypothetical-history transitions and
73,389 model calls. All stored sampled compatibility entries agree with the
corresponding exact entries. Separate scalar verification reconstructs every
valid hand set and checks marginals and errors to 1e-12; it adds 188 source
replay decisions and zero model calls. Exact likelihoods outside the preceding
sample are computed by the existing tested early-rejection routine, not a
second independent full-history implementation.

An exploratory graph calculation then connects supported hands differing by
one exchanged card. The parent1/WP support splits into components of 61 and one
hand; the isolated hand has no valid local move. Parent2/WP's 46 hands and the
Point root's 548 hands are connected. Under a uniformly proposed outgoing and
incoming card, stationary local acceptance is 0.1200, 0.1432 and 0.4354,
respectively. These are exact finite-graph properties, not measured sampler
accuracy or speed. The calculation performs no simulation or neural inference.

This rules out relying solely on local swaps for a general sampler: a chain
cannot leave its initial support component. A candidate improvement should
retain global hand proposals as well as local moves, account for correlations,
and be checked against these exact distributions before any strength trial.

Exact artifacts under ``runs/sampled_history_exact_check``:

* Protocol: a3169caff74a839e6e12e0a92e7be5d48ec65f70b58e8b1824f4dcfb26e88544.
* Report: 8feae6976f43329e7816d1ec6a27119229387cc9f804ac8246ceed6266f7c0eb.
* Scalar verification: c9a9b51f4f337bfc9bf70396ca14604679f0fa920cf5593b2cafda7d205e6bf9.
