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

Learner-decision implementation
-------------------------------

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

``scripts/learner_returns.py`` implements the return scan at discount1. It accepts
primitive [time,lane] arrays, learner-relative rewards and explicit learner-role
indicators. It returns advantages, value targets and a valid-loss mask in the
same layout. Invalid rows have zero targets as well as zero advantages. No input
is mutated and no RNG is consumed; opponent value entries are never used.

``scripts/learner_response.py`` connects it to pretrained published-actor PPO.
All lanes face the fixed whole-match WP/Point mixture. Rewards use the learner
seat before the environment reset. The actor and public value inputs remain
unchanged, and inference needs only the learner actor. The exported format is
``learner_response_v1``; the human-play model is not replaced. Checkpoints include
frozen opponents, exact optimizer/environment/RNG state and implementation hashes.

All primitive actions remain in the collection and permutation budget. PPO
advantage normalization uses the valid-loss mask. Each minibatch then removes
opponent and unfinished learner rows before both policy and value losses. An
empty minibatch skips Adam entirely, including momentum and step counters.
Diagnostics retain separate simulator, learner, valid learner and omitted-tail
counts, plus actual per-opponent work. Minibatch metrics use the inherited mean
across attempted minibatches; an empty attempted minibatch contributes zero.

This full intervention removes the opponent critic and normalizes value loss
over valid learner rows instead of all primitive rows. It consequently changes
value-loss scale, joint clipping, effective minibatch sizes and potentially Adam
step counts. Equal simulator work is not equal learner samples or an isolated
critic-information experiment. Tail exclusion also depends on which role is
active at the boundary; it is not a general claim of unbiased sampling.

Validation
----------

Nine return tests pass on CPU/CUDA, zero skipped. An independent forward expansion
of learner-time TD residuals checks randomized mixed lanes, terminals, lambda0,
0.37,0.95,1 and both tail roles. Opponent values may be replaced by NaNs without
changing an output. Fixtures cover opponent termination, reset isolation,
consecutive learner phases, primitive-gap eligibility and an unfinished last
learner action whose starting value still bootstraps its predecessor. All-learner
traces agree with ordinary GAE. Actual native game traces agree with the Python
reference before return comparison; gradients are detached and inputs unchanged.

Six integration tests pass, zero skipped. CPU/CUDA collection agrees with the
reference transitions, actual pinned opponent actions, pre-reset reward signs
and separate work counts. CPU tests verify warmup, actual actor learning, exact
restart, frozen opponent parameters, invalid-row NaN exclusion and no Adam drift
on an empty minibatch with existing momentum. Initial deployment actions match
the retained parent in all four phases. Source tampering on restore is rejected.
The later guarded CUDA run also passes the exact optimizer/restart test (below).
Commands::

    .venv/Scripts/python.exe -m pytest tests/test_learner_returns.py -q
    .venv/Scripts/python.exe -m pytest tests/test_learner_response.py -q

The fixed full-match strength comparison is now running; it has no completed result yet. The example and
implementation motivate a bounded comparison; they do not establish that observer
switching caused the earlier weak results or that current finite observations
suffice for unbiased PPO. The existing response study keeps its original sources,
return estimator and protocol throughout.

Production-size CPU capability
------------------------------

Session94836 exited0. The retained first parent initialized LearnerConfig with
CPU2 threads,128 environments,128-step horizon,512 primitive rows per minibatch,
3 epochs,critic width256,seed20295001 and1 warmup update. Three updates (one value
warmup plus two policy updates) finish49,152 simulator decisions,3,219 round ends
and375 match ends. These are capability-training counts, not evaluation matches.
There are24,521 learner actions,24,315 valid loss rows,206 omitted tails and24,631
opponent actions. No attempted minibatch is empty. All96 parameter Adam states
are finite:90 actor tensors have192 steps and6 value tensors have288 steps.
The actor stays unchanged during warmup and changes during both policy updates.

WP accounts for12,403 opponent decisions,1,644 round ends and188 match ends;
Point accounts for12,228 opponent decisions,1,575 round ends and187 match ends.
Round/match endings count entire assigned matches, not only opponent actions.

* Audit: ``runs/learner_response_capability_cpu/audit.json``;
  SHA216cc2ef25fead9f58b22ccb0ed66d54169fa6c958ee6cf664b2fdb361b3a1de.
* Checkpoint: ``runs/learner_response_capability_cpu/final.pt``;
  SHA94c19d41adada8b421d11685086a3bf64bfffbc86c8a139f457b219a1eeedcde.
* Per-update metrics SHA
  d5341d1a7e8f9e2e081a811b895b3c94e372bc3bb393bdac5f8642f832f3f910.

The audit contains the complete config, parent/source/runtime identities, actual
optimizer steps and counts. To reproduce the learning computation, construct
``LearnerTrainer.from_parent`` with that config and the bound parent, and call
``update()`` three times. Source checks verify all29 live-study inputs both before
and after this CPU capability run. No second GPU trainer was started. The later CUDA gate and fixed research launch are recorded below.


CUDA capability and fixed research launch
----------------------------------------

Guarded launcher23277 waits for all four original training workers to exit0,
then actually runs the CUDA exact-restart test:1 passed,0 skipped,2 upstream
warnings in6.14s. Three production-size CUDA updates finish49,152 decisions,
3,240 round endings and386 matches. Learner24,599,valid24,406,omitted193,
opponent24,553; no empty minibatches. All96 actual Adam states are finite and
match their executed budgets:90 actor tensors192steps,6 value tensors288steps.
WP12,176 opponent decisions /1,631 rounds /193 matches;
Point12,377 /1,609 /193. These are capability-training counts, not win evidence.

Audit runs/learner_response_capability_cuda/audit.json SHA
c6f9d428bba1d253ea5344994c7ffcab446dc38b99648e120b770ce91060fb17;
checkpoint SHA5f11e298ac807c65fa52d2dee2734774e0e32fdfb30ea71621dabf72f7e17750;
metrics SHA371b5b5ab371525203cc75fb51ce0166ed93a3035b8bc5d82f405aa7af0570b7.
Its source bindings and the separate exact-restart log are verified before the
research protocol is created. Old GPU training, capability validation and new
GPU training have non-overlapping recorded intervals.

The fixed learner-return study now runs under protocol SHA
675638eb8561f52e2d023c7f8fedc3006eac59a051d32e3b54e8e7cb78e10674,
with51 frozen inputs. The first new initialization has exact actor/value/environment
equality with its original response control and fresh optimizer state. CUDA RNG
is equal;CPU RNG differs. This comparison does not imply identical later samples.
No outcome-based selection or promotion has occurred.
