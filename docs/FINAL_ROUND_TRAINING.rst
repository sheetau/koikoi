Fixed-prefix terminal-outcome adaptation
=======================================

Purpose
-------

The completed final-round search pilot recovered one parent-2 loss against
RL-Point, but did not establish replicated strength. Its accepted choices cost
about five to seven seconds each. This follow-up learns a deployable final-round
actor, using the original parent unchanged in rounds one through seven.

The learner starts at naturally reached round-eight boundaries and completes
every sampled tail. Each learner decision receives the terminal match reward
from its own seat's perspective, including when the opponent makes the terminal
decision. There is no truncated-tail bootstrap or opponent-observation critic.
The existing clipped PPO optimizer uses these Monte Carlo targets and its
public observation value baseline. The actor architecture, input features and
public koi assistance remain unchanged. All heads may adapt in round eight.

Relationship to the match objective
----------------------------------

Fix a distribution of deals, seats and opponents. Let A mean that the match
reaches the round-eight boundary, H its complete simulator history there, and
G the terminal match reward. For a policy using the fixed parent before that
boundary and tail policy pi afterwards, the law of total expectation gives::

    J(pi) = E[1(not A) G] + P(A) E[V_pi(H) | A].

Both the early-terminal contribution and the distribution of (A,H) are
independent of pi. Thus improving the population conditional tail expectation
improves the match expectation for that same fixed distribution whenever
P(A)>0. This is an elementary fixed-prefix decomposition, not a new general
policy-improvement theorem. It does not say a particular PPO update improves J.

Uniformly sampling all surviving roots from a uniformly collected source bank
retains empirical survival weighting across seats/opponents. Rebalancing the
surviving cells independently would change that objective. A finite bank is
only an empirical approximation; repeated tails do not add independent deals.
Actors receive public/own observations, not H or the true hidden hands. Full
states are simulator restart data. Opponent labels route frozen opponents and
never enter the trainable network.

Training opponents use declared fixed64 GPU inference. Source prefixes and
actual evaluation use strict CPU policies. Near-tie arithmetic can change
actions between those modes; consequently the training tail target is an
approximation to the strict-opponent evaluation objective, not an exact identity
between those two numerical policies. The mathematical decomposition applies
separately to each consistently specified opponent policy.

Complete rewards remove the earlier cross-observer bootstrap issue from these
targets. They do not eliminate finite-sample error, root-bank overfitting,
greedy-versus-sampled policy differences, PPO clipping/normalization effects
or the sampled opponent distribution. Strength is decided by fresh full matches.

Prior work and contribution boundary
------------------------------------

Start-state curricula and guide-policy prefixes have substantial prior art:

* Resnick et al., Backplay (2018; revised 2022), construct a curriculum by
  starting near a demonstration's end and moving backwards:
  https://arxiv.org/abs/1807.06919
* Florensa et al., Reverse Curriculum Generation (2017), grow a start-state
  curriculum from a supplied goal state:
  https://arxiv.org/abs/1707.05300
* Uchendu et al., Jump-Start Reinforcement Learning, ICML 2023, use a guide
  policy to produce starting states for an exploration policy:
  https://proceedings.mlr.press/v202/uchendu23a.html

The current method keeps a permanent, public round boundary at deployment,
uses natural hidden-information game roots and optimizes the remaining match
outcome. It does not move that boundary backward, invent a curriculum method
or inherit the cited algorithms' sample-complexity guarantees. The research
question is whether this localized adaptation produces replicated improvements
at practical inference cost, and what the fixed-prefix/terminal-target analysis
explains about those results. A successful outcome would be an empirical and
domain-specific contribution; the implementation alone is not that evidence.

Implementation and accounting
-----------------------------

scripts/final_round_training.py is isolated from production trainers and the
playable model. Natural-prefix collection checks every transition against the
independent Python engine. Early bankruptcies are retained. Tail collection
freezes completed lanes and refuses a batch that fails to terminate within its
safety bound. Terminal returns are mapped by lane, never by the final actor.

Checkpoints retain both the frozen prefix actor and adapted tail actor, root
bank and its independent sampling RNG, frozen opponents, optimizer and global
RNG states. Inference routes using only the public round number. Prefix matches,
prefix decisions, conditional-tail decisions, conditional round endings and
tail completions must remain separate. The trainer's ordinary full-match count
stays zero. An input shape check alone does not prove a root is natural: source
records and independent replay supply that evidence.

Capability tests cover natural and early-ending prefixes, CPU/CUDA complete
trajectories against the Python engine, both reward perspectives, opponent
terminal actions, legal masks, frozen opponents, critic-only warmup, real actor
updates, exact optimizer/RNG restart and unchanged earlier-round inference.
These are correctness tests, not strength observations. A separate prospective
study must fix source deals, training budgets and evaluation before launching.
