Complete-match branch gradients
==============================

Question
--------

Can direct terminal outcomes from all legal first actions improve the public
actor when the earlier auxiliary losses and learned-return changes did not?
This experiment moves the simulation budget into actor learning. Deployment
would keep the existing observation-only network and require no branching.

The existing Q-boosting implementation fits a centralized action-value critic.
The earlier paired-outcome studies fit a selector between frozen expert actions.
This proposal instead updates the full actor, enumerates every legal root
action, and samples subsequent decisions from its stochastic policy. It has no
learned value bootstrap. Those are experimental differences, not a claim of a
new general policy-gradient principle.

Estimator
---------

Finish a batch of self-play matches with a fixed actor. In each match,
reservoir-sample one of its K unforced decisions uniformly. Preserve K as the
root weight. Forced decisions have zero masked-policy gradient. For each legal
action a, copy the sampled simulator world, force a, and sample both seats'
observation-only policies to match termination. G_a is the root player's
win/draw/loss outcome. The root gradient is::

    K * sum_a pi(a | o) * (G_a - b) * grad log pi(a | o)
    b = sum_a pi(a | o) * G_a

The outcomes and baseline are detached. Average over source matches, not over
their number of decisions. Reservoir weighting recovers the expected sum of
decision contributions. It must not be dropped or normalized separately per
match. Counterfactual continuations use fresh policy randomness. Hidden hands,
stock and future deals instantiate training simulations; none is an actor input.

For a fixed opponent this is the usual player's policy-gradient construction.
With a shared actor at both seats it estimates the simultaneous self-play
update: each acting player optimizes its own return while treating the opposing
policy as fixed. It is not the derivative of the identically zero sum of both
players' rewards. Adam, gradient clipping, finite samples and multiple updates
do not acquire an improvement guarantee from the estimator identity.

All branches share the actual root world and future round-deal streams.
The proposed coupling additionally shares one inverse-CDF uniform by root and
relative simulator step. Its control keeps the same world but uses independent
continuation-action uniforms. Diverging branches can have different players,
phases and legal actions at the same relative step. Consequently covariance
can have either sign; reduced variance must be measured.

Prior work
----------

Expected Policy Gradients (Ciosek and Whiteson) and Mean Actor Critic (Allen,
Asadi, Roderick, Mohamed, Konidaris and Littman) already use expectations over
actions for actor updates. Rollout Sampling Approximate Policy Iteration
(Dimitrakakis and Lagoudakis) uses simulations to improve a policy. Yadav and
colleagues study common random numbers in rollout planning. Fan and Farina's
VRPO uses a centralized Q critic with Expected-SARSA traces. The local proposal
combines established ingredients to test a concrete learning-signal bottleneck
in this game; superiority or methodological priority is not established.

Primary sources inspected on 2026-10-20:

* https://arxiv.org/abs/1801.03326
* https://arxiv.org/abs/1709.00503
* https://arxiv.org/abs/0805.2027
* https://arxiv.org/abs/2605.04732
* https://arxiv.org/html/2605.19235v1

Bounded diagnostic before training
---------------------------------

Use both retained parents, 64 fresh source matches per parent and eight
continuation replicates per root in each coupling condition. Keep T=0.05 and
public endgame masks. Collect one weighted root per source match, without
selecting phases, disagreements, outcomes or favorable roots.

Report source decisions/rounds/matches separately from branch decisions,
rounds and completions. Record all actions for independent transition replay.
Measure the trace of conditional variance of the root-logit gradient across
continuation replicates and the fraction of roots with an action-outcome
difference. This variance conditions on the sampled world and root: it excludes
between-world/root-sampling variance and is not full parameter-gradient
variance. Record the cost and retain both coupling outcomes, whichever wins.

This is a feasibility and learning-signal diagnostic. It cannot establish a
playing-strength gain. A subsequent fixed-budget actor-training comparison,
both training seeds and fresh full-match evaluation remain necessary.
