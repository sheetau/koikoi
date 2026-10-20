Policy-randomness coupling: target and information conditions
============================================================

Closest newly identified prior work
-----------------------------------

Hu et al., *OSCC: Certified Observation-Safe Coupling Optimization for
Gradient-Noise Control in Imperfect-Information Learning*, arXiv:2609.33543v2
(16 October 2026), studies rollout coupling under partial information.
It assesses covariance after projection through policy gradients, includes
physical sampling cost, and selects couplings on calibration data.
Its admissible class requires separate policy-action streams for each branch
(condition C3). Source: https://arxiv.org/html/2609.33543v2.

The local implementation shares continuation-action uniforms across branches,
so it does not implement that admissible class or its certified selector.
Neither projected-gradient covariance nor cost-aware coupling selection should
be claimed as a new local invention. The question here is narrower: which
conditions preserve a branch's policy law when policy draws are shared?

Related precedents include Oberst and Sontag's Gumbel-Max counterfactual models
for finite POMDPs (ICML 2019,
https://proceedings.mlr.press/v97/oberst19a.html), Fan et al.'s correlated-rollout
gradient estimators for categorical sequence generation (ICLR 2020,
https://arxiv.org/abs/1912.13151), and Dong, Mnih and Tucker's coupled categorical
gradient estimators (NeurIPS 2021, https://arxiv.org/abs/2106.08056).
These precedents rule out claiming correlated categorical rollouts generally
as a new construction.

Sufficient condition for the local construction
-----------------------------------------------

Fix the actor parameters during all continuations. A branch's policy input is
its own allowed observation; neither another branch's state or result nor
the full hidden simulator state is supplied to the actor. At relative step t,
let U_t be a fresh uniform independent of the root world, environment chance
stream and all previous policy uniforms. Every live branch uses its own
categorical inverse CDF with this same U_t.

Before drawing U_t, every branch state is a function of its root, environment
chance and past uniforms. Consequently, conditioning on a branch's complete
past leaves U_t uniform. Applying that branch's inverse CDF therefore gives
exactly its declared action probabilities. Induction over steps preserves its
full trajectory distribution. Early termination only stops that branch; it does
not reuse a past draw or change any surviving branch's next marginal.
This argument does not require independence between counterfactual branches.
It does require correct within-branch conditional draws.

For fixed-root action values Q_a, marginal preservation implies E[R_a] = Q_a.
Linearity then gives::

    E[sum_a grad pi(a | o) * R_a]
        = sum_a grad pi(a | o) * Q_a.

Thus sharing policy innovations can preserve the expected fixed-root gradient
while changing its covariance. This elementary marginal argument is not a new
policy-gradient theorem. It also is not a guarantee for Adam, clipping,
reused targets after policy updates, or overall playing strength.
The existing complete-match/root-selection calculation is a separate step.

This establishes sufficient conditions for the local construction; it does
not claim to satisfy OSCC's stricter chosen C3 contract. A shared uniform cannot
simply be recycled at later decisions within a branch: its previous actions
then convey information about that same uniform, altering the next conditional
action probability. Similarly, a jointly adaptive selector that reads other
branches' outcomes needs a separate marginal-preservation argument.

Exact variable-length check
---------------------------

``scripts/check_policy_coupling.py`` defines two counterfactual branches with
one-, two- and three-action terminal paths. Policy probabilities depend only
on their own public histories. A hidden terminal bit affects rewards but never
policy inputs. All thresholds are multiples of 1/8, so the eight uniform cells
integrate each draw exactly.

An independent recursive probability tree supplies the reference law.
The independent-policy comparator retains the same hidden world across branches;
only its policy draws are independent, matching the local experimental control.
Enumerating three fresh shared uniforms reproduces both complete marginal path
distributions and the expected root gradient exactly, using rational arithmetic.
Reusing a single uniform within each branch fails the marginal check.
Two predeclared reward cases, differing by a sign on the second branch, give
opposite signs of the shared-minus-independent variance change. This shows why
unbiasedness and variance improvement must be checked separately.

This is an exact mathematical fixture, with no Koi-Koi training or evaluation.
The active eight-arm study and its bound numerical sources remain unchanged.

Completed exact check
---------------------

Three tests pass with zero skips, including hand-derived path probabilities.
The executable check also passes. The expected first-logit gradient is 309/4096
in the base reward case and 183/2048 in the sign-inverted case, for both fresh
shared and independent policy draws. Reusing a uniform instead gives 3/32 and
3/64, respectively, and changes the trajectory marginals. Shared-minus-independent
return-contrast variances are -5965/32768 and +5965/32768 in the two cases.
No Koi-Koi decisions, optimizer updates or independent observations are added.
Receipt: ``runs/policy_coupling_conditions.json``;
SHA256 6b50cbc10b68f4ded9c10ad2d6c9ccea41a4f97627c43308ed25657d4d90b533.
The check is closed; do not rerun or overwrite the receipt.
