Root allocation and actor step size
==================================

Evidence motivating the comparison
---------------------------------

The completed branch-gradient screen reduced conditional gradient variance
without establishing a win-rate gain. Its saved data now identify a more
specific allocation problem. In the two shared-action arms, 30.04% and 29.42%
of sampled roots assign at least 99% probability to one action. Those roots
contribute only 0.0145% and 0.0163% of the empirical squared root-logit gradient
norm, including the original decision-count weights. Sampled branch outcomes
are identical across all legal actions at 68.87% and 67.68% of roots.

The analysis reconstructs both complete optimizer histories from saved roots
and outcomes. Actor and Adam tensors agree exactly at all 16 saved checkpoints.
It adds 256 replayed optimizer steps and no new simulation or trained endpoint.
Artifacts: ``runs/branch_update_coverage``.

On the separate fixed 64-root diagnostic banks, the final shared actors change
one and three greedy decisions from their parents. Mean KL divergences are
0.005682 and 0.003636. Small per-update changes do accumulate; these measurements
do not establish that the learning rate alone caused the failed win comparison.

A fixed, public-policy priority proxy was chosen before examining this analysis::

    w(o) = 0.05 + sqrt(max(0, 1 - sum_a pi(a | o)^2))

Hypothetically reweighting each saved root pool by this proxy lowers its
empirical logit-gradient second moment to 0.6455 and 0.6654 of uniform sampling.
This is a finite-pool calculation using observed gradients. It is not a measured
variance reduction for future within-match sampling or full network gradients.
It motivates a controlled learning test rather than default adoption.

Sampling and correction
-----------------------

Each fresh on-policy source match still supplies one root. At each unforced
decision, compute w using the current actor's probabilities. A streaming weighted
reservoir selects decision t with conditional probability w_t / W, where W is
the sum over all eligible decisions in that match. Use the detached importance
weight W / w_t in the existing all-action terminal-outcome gradient. Uniform
sampling uses w_t = 1 and reduces to the previous K-weighted estimator.

Conditional on a source trajectory, summing over the reservoir choice gives::

    sum_t (w_t / W) * (W / w_t) * g_t = sum_t g_t

Thus the correction retains the original expected decision sum. Priorities
depend on public actor inputs and are computed before branch outcomes. The
sampler does not change source behavior, policy inputs or game rules. The
positive floor preserves support; no importance clipping or self-normalization
is introduced. Weights and sampling probabilities are detached during updates.
Forced decisions still have zero masked-policy gradient.

The priority is a heuristic proxy. It does not use the unknown outcome variance
or full parameter-Jacobian norm and need not be variance-optimal. All-action
branches still continue to match termination, sharing their hidden simulator
world and continuation uniforms while exposing only public/own observations
to the actors.

Prior work and contribution boundary
------------------------------------

Papini, Manganini, Metelli and Restelli's *Policy Gradient with Active Importance
Sampling* optimizes a behavior policy to reduce gradient variance
(https://arxiv.org/abs/2405.05630). Huang and Jiang derive policy-gradient
estimators from the importance-sampling family
(https://proceedings.mlr.press/v119/huang20b.html).

Wang, Liu and Wang's *Variance-Optimal State Resampling for Reinforcement
Learning with Verifiable Rewards*, submitted on 11 October 2026, directly
studies state-allocation variance. Its practical VSR method uses a recent prefix
buffer and a variance proxy for GRPO
(https://arxiv.org/html/2610.15888v1). This is a relevant recent antecedent, not
an omitted competing claim.

The local experiment uses fresh complete-match trajectories, a known conditional
reservoir probability, its explicit correction, and full legal-action terminal
branches in imperfect-information self-play. These distinguish the implementation
from the prior local uniform-root study and from prefix replay. Importance
sampling, prioritized states and reservoir sampling are not claimed as inventions.
The research question is whether this allocation improves measured learning and
play at a disclosed simulation cost.

Fixed factorial development design
----------------------------------

For each of both retained parents, train all four combinations:

* Uniform roots, Adam learning rate 1e-5.
* Priority roots, Adam learning rate 1e-5.
* Uniform roots, Adam learning rate 1e-4.
* Priority roots, Adam learning rate 1e-4.

Every arm receives exactly 128 updates of 64 complete source matches. Each
update uses one Adam step, epsilon 1e-5, no weight decay or entropy term,
global gradient clipping at 0.5 and score temperature 0.05. All branches share
continuation uniforms. Parent seeds are 114110001 and 114210001, with separate
action, reservoir and branch streams at offsets 1M, 2M and 3M. Optimizer state
starts fresh. Record every final endpoint; no early stopping or checkpoint
selection. Source/update counts match, while actual branch counts and costs can
differ because the sampled roots have different legal-action sets and lengths.

The predefined candidate is priority roots with learning rate 1e-4. Evaluate
all eight endpoints and both parents against original WP and Point on 1,024
fresh paired streams per opponent, with both seats. WP seeds begin at 118110001,
Point at 118210001. Use the existing GPU batch-128 greedy/endgame path, separately
labeled from the paper's strict CPU singleton comparison. The reserved 101M
block remains unused.

The four primary quantities are candidate-minus-parent and
candidate-minus-uniform-high-rate for each parent, measured as equal-mixture
draw-adjusted win changes. Bootstrap 10,000 compound deal blocks with seed 118310001,
retaining both seats, opponents and models together. Use simultaneous 95%
intervals from maximum absolute centered deviations across the four quantities.
The progression criterion requires all four lower bounds above zero.
Retain every result on failure. Absolute opponent-specific win rates, point
margins and the remaining factorial contrasts are descriptive secondary results.
No canonical promotion follows from this development screen alone.
