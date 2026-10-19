Posterior action values with boundary residual correction
=======================================================

Motivation and prior work
-------------------------

Exact public-history conditioning improves current-hand prediction on the
existing 25 development roots. Two terminal-rollout action diagnostics did not
establish stronger play. Earlier paired-value learners used realized-world
targets; the round-value search prototype used approximate boundary tables
without a terminal residual. Neither establishes the utility of the estimator
specified here for posterior-conditioned action targets.

Control variates, two-level Monte Carlo and multifidelity estimation are prior
work. See Peherstorfer, Willcox and Gunzburger, SIAM J. Scientific Computing
38(5), A3163--A3194 (2016), DOI 10.1137/15M1046472, verified through
https://dspace.mit.edu/entities/publication/f104683d-44c4-40ec-bd06-47502c5c0bfd.
Their general use of surrogate and high-fidelity evaluations is not claimed as
new here. The proposed application separates public hand inference, a cheap
round-boundary proxy and correction to a declared terminal continuation target.

Estimator
---------

For a fixed root and action pair, X is half the difference in signed terminal
win utilities. Y is half the difference in fixed approximate boundary values
at the first round ending. The proxy averages the existing independently built
tables A and B; neither table is selected using these roots. It uses only public
round, score, dealer, terminal status, parent seat and sampled opponent model.

An independent sample of M cheap trajectories gives mean(Y_c). Another sample
of N trajectories gives coupled X_h and Y_h from the same first-round paths.
With alpha fixed before sampling, estimate

    alpha * mean(Y_c) + mean(X_h - alpha * Y_h).

Conditional on the fixed table and the stated hidden-world distribution, its
expectation is E[X]. Its variance is

    alpha^2 Var(Y)/M + Var(X-alpha*Y)/N.

The table's approximation bias cancels in expectation; its quality still
determines residual variance. Do not add table-bootstrap error independently,
fit alpha using these outcome samples, clip the resulting estimate, or infer
that the terminal execution equals the strict CPU opponent. This study fixes
alpha=1 and retains alpha=0 (ordinary terminal mean) as its control. The
identities above are elementary instances of established estimation methods.

Both fidelities use the same fixed64 CUDA first-round path. All lanes stop at
their first round boundary before the high-fidelity path resumes from those
exact states. This barrier avoids a difference in prefix execution caused by
mixing already advanced lanes into ongoing first-round inference. The tail
retains the sampled RNG and new-round cards inside simulation; no hidden fields
enter policy features or boundary lookups. The target is this explicit barrier
execution, not an asserted exact reproduction of singleton CPU or prior
unbarriered batched continuations.

Fixed diagnostic
----------------

Use all 25 eligible roots from the completed exact-hand diagnostic, preserving
seven ineligible attempts in coverage. For each root use uniform belief and
identified-history belief. The control varies both hand evidence and public
identity, so a difference cannot be attributed to either alone. If the estimator
is useful, a later mechanism study must retain the four-way factorization.
The compared alternative is the first legal action other than the strict CPU
parent, in ascending card order. Do not select using earlier proposal values,
accepted actions, true hands or conditional outcomes.

For each of the 50 root/condition cells, collect exactly 512 coupled high/proxy
samples and 2,048 independent cheap samples. Use independent RNG seeds
104100001+10000*parent+case_index and 105100001+10000*parent+case_index,
respectively. Condition pairs share seeds. Then collect 2,048 independent
high-fidelity reference samples using 106100001+10000*parent+case_index.
Reference samples are never used to alter estimates or choose coefficients.
All seeds generate hypothetical worlds, not fresh actual evaluation deals.
Use batches of 512 high/reference and 2,048 cheap samples, TF32 disabled and
two CPU threads. Retain every cell, including adverse or zero-variance cases.

Report per-cell and grouped sample variances, covariance, correction magnitude,
realized decisions/rounds/terminal endpoints and wall time. Divide high samples
into 16 fixed blocks of 32 and cheap samples into 16 corresponding blocks of
128. These provide 16 independent estimator replicates per cell, compared with
the independently sampled reference mean. Report both corrected and ordinary
terminal squared errors; the finite reference has its own sampling error.
Also report variance-times-cost ratios against the terminal estimator, using
both measured decisions and elapsed time. Those ratios describe estimated
efficiency, not a matched-cost accuracy trial or a confidence guarantee.

No action is deployed, no actual-source completion is evaluated, no policy is
trained and no model is selected in this diagnostic. It tests whether corrected
posterior action targets warrant a separate value-learning comparison. A useful
estimator result is not a match-strength result. Keep reserved 101M unused.

Bind code, tests, weights, exact-table audits and boundary-table evidence before
execution. Afterward independently reconstruct proxy values, scalar statistics
and work accounting. The existing CUDA/reference check covers transition
execution; an arithmetic audit is not a second neural replay of all samples.
