Directional return diagnostic
=============================

The temperature comparison is complete and does not establish additional
playing strength. The next diagnostic measures return-estimator differences
in actual policy-score contributions. The earlier MLP return diagnostic
compared returns with realized outcomes; those errors alone cannot establish
a policy-gradient error. The finite observer example establishes a possible
failure, not its prevalence in the retained Koi-Koi agents.

Policy family and estimand
--------------------------

Freeze retained parents A/B, including their critics. Both seats sample the
same frozen parent with temperature 0.05 and public endgame assistance.
For fixed public features f_d(o,a), define the local family
pi_theta(a|o) proportional to pi_0(a|o) exp(sum_d theta_d f_d(o,a)).
At theta=0 its score is f_d(o,a) minus its expectation under pi_0.
The base probabilities and features remain fixed when differentiating.

Use five bounded directions, fixed before data collection:

* Continue bias: f=1 for action49 in the koi phase, zero otherwise.
* Four phase preference directions: f=pi_0(a|o) in one of discard, hand
  capture, stock capture and koi, and zero in the other phases.

These are local policy-family derivatives, not full neural parameter gradients.
Forced decisions have zero score. No private opponent hand, unseen stock or
future event is an input to a probability, feature or value prediction.

For one seat, sum score times (terminal seat payoff minus current value) over
its decisions. The expectation is the derivative of that seat's terminal
payoff against the frozen opponent. Average the two seat contributions within
each complete match. The target is the average of two unilateral responses,
each with the other policy fixed; it is not a derivative of the identically
zero total payoff when both sides change together.

This is the standard likelihood-ratio identity with an action-independent
baseline. Schulman et al., *High-Dimensional Continuous Control Using
Generalized Advantage Estimation*, Section2 and Proposition1, distinguish
return prediction from its score-weighted expectation.
https://arxiv.org/html/1506.02438
No new gradient-estimation theorem is claimed.

Return comparisons
------------------

Compare complete Monte Carlo with signed GAE at lambda0.95 through termination,
lambda0.95 with128-step cutoffs, and lambda1 with the same cutoffs. Gamma is1.
At a nonterminal cutoff, retain the next-state value but reset the eligibility
trace. At termination, use zero bootstrap. Round endings do not stop returns.
Complete lambda1 must equal Monte Carlo and serves as an arithmetic control.

Assign each match an independent uniform cutoff offset in0..127 before its
trajectory is observed. This defines a randomized rollout alignment; it does
not reproduce the distribution of offsets in steady-state training. Use the
same offset for corresponding deals of both parents. All estimators receive
the same actual trajectory, frozen critic values and score features.

Report per-match seat-averaged contributions, their paired differences from
Monte Carlo, and uncertainty over complete matches. Preserve both seat values;
decisions within a match are not independent observations. These statistics
exclude PPO clipping, advantage normalization, entropy losses and optimizer
updates. A difference cannot by itself identify observer mismatch as its cause:
critic approximation error also contributes.

Planned collection and verification
-----------------------------------

First require a capability run with both retained models at the intended
CUDA batch size64, complete termination, unchanged parameters and CPU replay.
Capability source starts at98100001; it is distinct from the main corpus.
Do not launch the main collection until the collector and scalar auditor pass.

The planned main corpus contains4,096 full sampled-self-play matches per
parent in64 batches of64. Both parents share source seeds98110001..98114096:
8,192 actual matches on4,096 shared initial deal streams. Sampling RNG is
separate,with seed98210001 +10000*parent_index +batch_index (parent_index0/1).
Cutoff-offset RNG is98300001+batch_index,shared across parents. A1,024-step
safety limit raises on incomplete matches rather than treating them as terminal.
No training, model selection, reserved101M deals or strength evaluation occurs.

Store actions, probabilities, legal masks, values, active seats, phases,
valid/terminal flags, round/work outputs and initial/final simulator states.
States are replay artifacts and never policy inputs. Bind source, runtime,
model and raw-output identities. An independent CPU action replay must verify
transitions, public masks and work. Separate scalar calculations must verify
probe derivatives, returns and match contributions; they need not rerun neural
inference, and that distinction must be stated.

The main analysis will use10,000 whole-deal bootstrap replicates with seed
98410001,retaining both seats and both parents at each index. Report every
probe/estimator,not only significant coordinates. Define and test simultaneous
uncertainty calculations before launch. A reversal for the primary continue
direction requires opposite nonzero Monte Carlo and128-step GAE0.95 signs with
simultaneous intervals at both parents. Other findings remain descriptive.

The immediate implementation is the score/return/contribution arithmetic.
Its tests use finite differences, an exactly enumerable reversal and separate
per-root TD expansion. Actual Koi-Koi collection, uncertainty implementation
and independent replay remain to be completed. This diagnostic can inform a
subsequent learning experiment; it does not qualify a stronger deployed agent.
The policy is sampled self-play, not greedy deployment against WP/Point.
Any proposed learning repair still needs a separate complete-match strength
comparison under the deployment protocol.
