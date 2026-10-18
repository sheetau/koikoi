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
