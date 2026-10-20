Finite-sample bounds for certified expected improvement
======================================================

scripts/certified_gain_bounds.py computes a conservative lower confidence bound
on mean nonnegative conditional improvement. Together with a matching deal law,
it provides a way to measure the population effect of exact local improvements.
It is a statistical component, with no study, threshold grid or sample size yet
selected. The running symbolic comparison must finish before the next evaluation
is chosen. Existing fixed-seed results and their failed strength criterion stay
as recorded; this component has not been applied to them.

Population and observations
---------------------------

Let H be an independently sampled parent prefix under a fixed deal/opponent law.
At its first eligible intervention, let Delta(H) be the independently verified
exact conditional win-score advantage of the committed controller over its
parent. It is zero when the controller follows its parent, including sources
with no eligible root or insufficient computation. Every source contributes
one value to the denominator. Delta lies in [0, 1] because adoption requires an
exact improvement and the win score lies in {0, 1/2, 1}.

When the evaluator's full posterior is the conditional law of the source,
INCUMBENT_GUARANTEE.rst gives E[Delta] as the expected match-win improvement.
This is a conditional-expectation estimate, distinct from a count of observed
wins. UNIFORM_DEAL_SOURCE.rst supplies the deal-sampling component; integrating
it with a fully checked history/opponent posterior remains part of any future
study. Deterministic seeds alone do not supply independent draws from that law.

Bound and computation
---------------------

Before sampling, fix n, an error probability alpha, and a strictly increasing
grid 0 < t_1 < ... < t_m <= 1. Set t_0=0. For each threshold, count

  k_j = sum_i 1{Delta(H_i) >= t_j}.

These are binomial counts within an IID source population. Compute the one-sided
Clopper--Pearson lower bound L_j at error alpha/m by inverting

  P_{p=L_j}(Binomial(n,p) >= k_j) = alpha/m.

For k_j=0, use L_j=0. The implementation evaluates binomial tails with integer
arithmetic and returns the solution rounded down to a 2**(-bits) grid, with
bits=32 by default. It cannot round the lower limit upward through floating-point
error. The zero-success case remains zero rather than receiving a positive
pseudocount. This is standard exact binomial inversion, described by
`NIST's one-sided exact binomial limits
<https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbino.htm>`_.

The union bound makes all m survival lower bounds simultaneous with probability
at least 1-alpha. For every x in [0,1],

  sum_j (t_j-t_{j-1}) 1{x >= t_j} <= x.

Taking expectations therefore gives the reported bound

  E[Delta] >= sum_j (t_j-t_{j-1}) L_j

with probability at least 1-alpha. Threshold indicators can be dependent; the
union bound does not assume independence between them. Integrating the fixed
threshold staircase uses information from more than one gain magnitude. This
construction applies established confidence bounds; it is not a new general
statistical theorem or a substitute for improving the controller.

Study integration requirements
------------------------------

The function requires the declared full source count and exact rational values;
it rejects a different denominator, negative values, values above one, floating
inputs and malformed thresholds. A negative controller-minus-control difference
must not be clamped to zero to use this method. Its intended comparison is an
exactly certified controller against the parent it actually follows on fallback.

The future protocol must freeze the policy, caps, source law, threshold grid,
sample count, practical effect criterion and error allocation before sampling.
Choosing thresholds or stopping from observed gains invalidates this stated
fixed-sample guarantee. Every planned model/population claim needs an allocation
from the total error budget. For fixed strata, allocate error to each stratum
and combine its mean lower bound using the target population weights. Do not
pool different opponent/seat laws as if all observations were IID. Between-stratum
independence is unnecessary for this union-bound combination; IID observations
within each stratum are still required.

The calculator does not infer independence, verify a posterior, enforce a frozen
protocol, or certify input values. Those are responsibilities of the study's
source collector and independent audit. Shared rollout leaves are not additional
source samples. The result supports only the declared population and executable
controller, not arbitrary opponents or minimax strength.

Verification
------------

Five tests compare binomial tails with independent enumeration of all Bernoulli
sequences through n=6, bracket the rational inverse including extreme counts
through n=100, and exactly sum noncoverage probabilities for small binomial
populations. They also enumerate all 81 ordered four-sample outcomes from two
three-point gain distributions to check the mean-bound coverage. Denominator,
fallback-zero, threshold-equality and invalid-input checks exercise study-facing
contracts. These are finite statistical fixtures, with no Koi-Koi source matches,
planning transitions or neural inference.
