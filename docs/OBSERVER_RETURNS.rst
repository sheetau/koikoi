Bootstrapping across player information sets
===========================================

Finding
-------

Correctly changing value signs when the acting player changes does not make an
observation critic a sufficient statistic for another player's learning target.
The finite example below reverses one policy-gradient component even though both
critics give exact conditional expectations for their respective observations.
The current Python, native CPU and CUDA return implementations reproduce it.
This diagnoses an estimator limitation, not a sign error or simulator mismatch.

Critic conditioning under partial observability is an established issue.
Baisero and Amato, `Unbiased Asymmetric Reinforcement Learning under Partial
Observability <https://arxiv.org/html/2105.11674v2>`_ (AAMAS 2022), analyze bias
from substituting state critics and develop a history-state formulation. Lyu et
al., `A Deeper Understanding of State-Based Critics in Multi-Agent Reinforcement
Learning <https://ojs.aaai.org/index.php/AAAI/article/view/21171>`_ (AAAI 2022),
analyze both bias and variance of centralized state critics. Our example is a
direct derivation for the alternating-observer recurrence in this repository;
it is not a claim to have discovered the general critic-conditioning problem.

An exactly enumerable game
--------------------------

A uniform private signal X in {0,1} is observed by the learner. The learner
chooses a public action A in {0,1}, with pi(A=1|X=x)=sigmoid(theta_x). Initially
theta_0=theta_1=0. The opponent sees A but not X, makes an exposed decision with a fixed dummy move, and
the game ends with learner payoff R in {-1,+1}. The opponent payoff is -R.
The reward distribution has the following conditional means:

=========== ======== ========
Signal X    A=0      A=1
=========== ======== ========
0           0.20     0.21
1           1.00     -1.00
=========== ======== ========

For each cell, P(R=+1|X,A)=(1+E[R|X,A])/2. The learner's exact observation values
are v_0=0.205 and v_1=0. The opponent's exact value, expressed in learner payoff
units, is m_0=0.6 or m_1=-0.395 according to the observed action. Its own-payoff
critic therefore outputs -m_A. These are exact under the current uniform policy;
there is no value-fitting error in the example.

The acting-player reward sequence is (0,-R), and the value sequence is
(v_X,-m_A). With discount 1 and a terminal second transition, signed GAE gives::

    Ahat_learner = lambda * R + (1-lambda) * m_A - v_X.

At X=0, action 1 is better by 0.01. For lambda=0.95, however, the expected target
for action 0 is 0.22, while that for action 1 is 0.17975. The surrogate favors action 0.
After weighting the private signal probability 1/2 and sigmoid derivative 1/4::

    true dJ/dtheta_0                 =  1/800     =  0.00125
    expected signed-GAE component    = -161/32000 = -0.00503125

This coordinate has the correct positive sign only when lambda>199/201, about
0.99005. The statement concerns theta_0; it does not say the entire two-coordinate
gradient is opposite to the true gradient. These are expected score-function
updates before PPO clipping and empirical advantage normalization.

Three controls restore the correct expectation in this example:

* Set lambda=1, removing the intermediate value prediction from the return.
* Treat the opponent move and terminal reveal as part of one learner transition.
  Its advantage is R-v_X, even with lambda=0.95.
* Give the intermediate critic joint information (X,A), so its learner-payoff
  expectation is E[R|X,A] rather than E[R|A]. This is a training-only control.

The second control needs no opponent-private feature. Its single-sample variance
for the theta_0 gradient is 0.1197453, versus 0.1080973 for the biased recurrence.
Removing bias does not automatically improve finite-budget learning.

Executable evidence
-------------------

``scripts/diagnose_observer_bootstrap.py`` enumerates all eight signal/action/
reward combinations with their exact probabilities. It compares the analytic
derivation with a direct objective gradient and the production return functions.
It checks the opponent critic's conditional means explicitly. All four controls
pass on CPU and CUDA, with floating-point tolerances stated in the script.
The calculation samples no matches and performs no policy fitting.

Final reports:

* ``runs/observer_bootstrap_verified_cpu.json``; SHA256
  4e3e05b73b4e019bf9e68cf2b90afe0d61a2829b0e821dc303d1224b6a13aeab.
* ``runs/observer_bootstrap_verified_cuda.json``; SHA256
  50ecb0247856ed6ddd19b91fb1db841fc52f0783682ea6fc59780fbb4306de7b.

Each binds the diagnostic and full production source identity. No existing
production file, rule, checkpoint or live response-study input was changed.
The response study continues with its original estimator and full protocol.

Next capability to test
-----------------------

For a fixed opponent, collect returns between successive learner decisions.
Opponent actions remain part of the simulated environment; their outcomes and
costs still count. Bootstrap only at the next learner observation, or at terminal
zero. This is a decision-time reduction, not a new general policy-gradient theorem.

A fixed primitive-step rollout needs explicit tail handling. If its final learner
action has neither a following learner observation nor a terminal outcome inside
the buffer, omit that unfinished transition from actor/value losses. Its starting
observation can still bootstrap the preceding complete learner transition. Report
the omitted samples. An opponent-caused terminal outcome must be attributed to
the preceding learner action, and a reset must prevent cross-match reward leakage.
Multiple same-player capture decisions remain separate learner decisions.

To preserve the original per-step eligibility decay, a gap of k primitive steps
uses lambda**k on the next learner advantage (discount remains1). Removing an
opponent critic also changes optimization and clipping, so a comparison must
identify these choices and actual learner sample counts. It cannot be called an
isolated information effect merely because simulator budgets match.

This capability has not yet been implemented or evaluated in Koi-Koi. The example
motivates it; it does not establish that observer switching caused earlier weak
results, that finite current observations suffice for unbiased PPO, or that the
new method will produce a stronger player. Those questions require a new bounded
comparison after implementation and validation, preserving the current study.
