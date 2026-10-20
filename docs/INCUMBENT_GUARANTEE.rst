From root certificates to expected match improvement
===================================================

Exact root certificates give an expected-match guarantee when the evaluated
beliefs equal the conditional laws of the declared generative model. Observed
win rates and robustness to a different model require separate evidence.

Statement
---------

Fix a parent policy pi and a model law nu for opponent policies and deals.
Let tau be the first eligible final-round learner decision, determined by the
learner's own/public history H. Until tau, the controller follows pi exactly.
At tau it either retains pi or commits to a complete history-contingent policy
sigma_H. No policy is adopted without an exact parent reference.

Assume the belief used for evaluation is nu's conditional law given H; opponent
actors and game rules match that law; and all retained policies respect the
learner's information. Search limits and candidate selection depend only on H,
the declared models and fixed computational rules. For every adopted candidate,
the controller checks

  Delta(H) = E_nu[W(sigma_H) | H] - E_nu[W(pi) | H] > 0,

where W is the final draw-adjusted match win score in {0,1/2,1}. Set Delta=0
when no eligible root exists, computation cannot value the parent, or no
strictly better completed candidate is available. Then

  E_nu[W(controller)] - E_nu[W(pi)] = E_nu[Delta(H)] >= 0.

Proof
-----

Both policies have the same pre-intervention history law because they use the
same parent until tau. Conditional on each reached H, the committed policy has
the exact evaluated continuation value. On fallback histories it is the parent
itself. Partitioning by H and applying the law of total expectation gives the
displayed identity. Every term is nonnegative by the adoption check. A positive
probability set of strict-gain histories makes the overall inequality strict.
The proof requires no globally optimal candidate and no completed comparison
of every root action.

Adaptive computation does not introduce a statistical selection problem here:
the completed candidate values are exact under the same conditional model.
An unfinished action contributes only an upper bound and never supplies the
executed candidate's value. A learned or sampled value estimate would require
a different argument. Executing only the root action, changing the opponent
after evaluation, or dropping the committed continuation would also invalidate
the identity.

What this means in the current system
------------------------------------

The current belief law uses uniformly weighted valid deals and fixed deterministic
opponent models. It is explicitly not the posterior over the simulator's finite
PRNG seeds. Earlier-round public identification restricts the model class;
complete current-round likelihood filtering supplies the hand posterior.
Uniform remaining draw-prefix multiplicities then give exact continuation Qs.
The above guarantee applies only when that conditional belief is the evaluation
law. It does not cover arbitrary real opponents, drifting policies, omitted
likelihood information, a different deal prior, or execution outside support.

A reachable strict-gain witness can establish strict improvement under a model
that gives its history positive probability. It supplies no useful population
effect-size lower bound without that history's probability. The saved cohort's
mean conditional gains are not estimates from a fresh strength evaluation.
The current actual continuations still need to be distinguished from independent
matches. Neither this statement nor the optimum-gap certificate proves minimax
optimality, world-best play, or absolute superiority over RL-Point.

Prior work and contribution boundary
-----------------------------------

`Smith and Simmons' HSVI (UAI 2004)
<https://publications.ri.cmu.edu/heuristic-search-value-iteration-for-pomdps>`_
already returns an anytime POMDP policy with a provable regret bound. Thus
retaining policies with lower/upper value information is established planning.

`Laroche, Trichelair and Tachet des Combes' SPIBB (ICML 2019)
<https://proceedings.mlr.press/v97/laroche19a.html>`_ studies baseline improvement
from a fixed dataset without direct access to the true environment, using
baseline behavior where uncertainty is high. `Castellini et al.'s MCTS-SPIBB
(ICML 2023) <https://proceedings.mlr.press/v202/castellini23a.html>`_ develops
online local search for that framework. Their settings should not be conflated
with this known-rule, finite-belief exact comparison.

The statement above follows from conditional expectation and does not introduce
a new general safe-improvement theorem. The project-specific work is the
verified Koi-Koi posterior, executable policies with consistent information,
and auditable evaluation under fixed computation limits. Its strict reachable
improvements, runtime costs and prospective match strength are distinct results.

Finite check
------------

``tests/test_incumbent_guarantee.py`` independently sums a four-world example.
The controller raises conditional value from 1/2 to 3/4; a gate probability 3/10
gives total expected improvement 3/40. The candidate still loses on one world
where the parent wins. Changing the true hidden-world weights while retaining
the declared uniform evaluator reverses the improvement to -6/11. This verifies
the expectation identity and illustrates its limits, without simulating any
new Koi-Koi source matches. One test passes without skips in 1.40 seconds.
