Observer changes and finite-rollout bias
=======================================

Purpose and prior work
----------------------

The fixed-observer comparison tests a particular return-construction change.
The identity below explains its target and separates internal bootstrap error
from rollout-boundary error. It is an application of the usual GAE/TD(lambda)
telescoping identity,not a new general policy-gradient theorem. The relevant
foundations are Schulman et al., High-Dimensional Continuous Control Using
Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438 , and the
history-versus-state/individual-versus-joint critic analysis of Lyu et al.,
On Centralized Critics in Multi-Agent Reinforcement Learning, JAIR77 (2023),
295-354, https://arxiv.org/html/2408.14597v1 .

The repository-specific result is the exact failure of an alternating-observer
signed return,including a shared-parameter descent example,and a controlled
implementation that can test its practical importance. A change of observer
can matter even when both observation critics are conditionally exact. The
fixed-observer implementation is not asserted to retain a sufficient history.

A pathwise identity
------------------

Consider a root decision at time0 by seat i and a rollout of n>=1 primitive
decisions,with gamma=1 and constant lambda in[0,1]. Terminal states are absorbing
with zero subsequent rewards and values; a reset belongs to a new segment.
Let p_k denote the actor at time k. Express all rewards and predictions in the
root seat's payoff units:

* r_k is the root seat's reward on transition k.
* W_k=(-1)^(i+p_k) V_k^(p_k) is the signed active-observer value.
* G_k is the actual root-seat return from k through the match end,including
  the unobserved suffix beyond the rollout. Thus G_k=r_k+G_(k+1).

The product of consecutive player-change signs telescopes to(-1)^(i+p_k).
Consequently,the signed recurrence at the root is exactly

  A_lambda = sum[k=0..n-1] lambda^k (r_k+W_(k+1)-W_k).

Substitute r_k=G_k-G_(k+1) and collect coefficients. For e_k=W_k-G_k,

  A_lambda = G_0-W_0
             +(1-lambda) sum[k=1..n-1] lambda^(k-1) e_k
             +lambda^(n-1) e_n.                                      (1)

This is a pathwise algebraic equality for arbitrary values,rewards and player
sequences. Consecutive decisions by the same player introduce no extra sign.
The last coefficient is lambda^(n-1),not lambda^n. The convention0^0=1 applies
when n=1 or lambda=0. At a terminal boundary,W_n=G_n=0,so the last term vanishes.
At an ordinary rollout cutoff,it generally remains,including at lambda=1.

Expected score-function error
-----------------------------

Fix the behavior policy and treat value targets as constants for the policy
update. Let S_0=grad log pi_i(a_0|H_0^i). Assume W_0 is action-independent given
the root policy history H_0^i and the score has conditional mean zero. Subtract
the root likelihood-ratio contribution g_0=E[S_0 G_0] from(1):

  E[S_0 A_lambda]-g_0
    =(1-lambda) sum[k=1..n-1] lambda^(k-1) E[S_0 e_k]
      +lambda^(n-1) E[S_0 e_n].                                    (2)

An exact conditional prediction for the FUTURE actor does not in general make
E[S_0 e_k] zero: that actor's information may omit the root's private signal.
Unconditional zero-mean prediction error is insufficient; its correlation with
the root score matters. Equation(2) isolates precisely the cross-observer issue
in the two-step counterexample.

For comparison,let H_k^i be the root observer's full history at k,including
its root action,and let U_k=E[G_k|H_k^i]. If S_0 is H_k^i-measurable,then

  E[S_0(U_k-G_k)]
    =E[S_0 E[U_k-G_k|H_k^i]]=0.

This tower-property argument applies also at the nonterminal cutoff. Replacing
W_k by exact root-history values therefore removes all terms in(2). It does not
require observing the opponent's hand. A finite observation encoding can forget
information needed to make the root score measurable; exactness conditional on
that compressed encoding alone is not enough. The currently tested561 features
are not proved sufficient,so the argument motivates rather than guarantees the
Koi-Koi intervention. PPO clipping,empirical advantage normalization and fitted
critics are further differences from this population calculation.

A useful bound follows if ||S_0|| is integrable and |W_k-U_k|<=epsilon_k
pathwise. The norm of(2) is at most E[||S_0||] times

  (1-lambda) sum[k=1..n-1] lambda^(k-1) epsilon_k
    +lambda^(n-1) epsilon_n.

For a terminal rollout and uniform internal error epsilon,this reduces to
epsilon(1-lambda^(n-1))E[||S_0||]. At a nonterminal cutoff,uniform epsilon across
all predictions gives epsilon E[||S_0||],independent of lambda. This is an error
bound,not a guarantee that increasing lambda improves realized learning.

What the shared parameter does and does not mean
------------------------------------------------

Equations(1)--(2) concern a root score contribution to one player's objective.
Summing that player's score terms gives its policy-gradient expression while
the opponent policy is held fixed. If both seats share parameters and both are
updated toward their own payoffs,the combined update is not thereby the gradient
of the sum of their expected rewards: that sum is identically zero in a zero-sum
match. No monotone global self-play objective or exploitability theorem follows.
The existing scalar descent example has one learned actor and a forced opponent;
it makes a valid local expected-payoff statement for that specified objective.

Exact numerical specialization
------------------------------

Use the existing private-bit game,mu=((1/5,21/100),(1,-1)),uniform root actions,
and shared logits(theta,-theta/100) at theta=0. The true gradient is3/800.
The intermediate opponent's root-payoff prediction is m_A=E[R|A],whereas the
root-history prediction is mu_XA=E[R|X,A]. Exact enumeration gives

  E[S_0(m_A-mu_XA)] = -20301/160000.

For the complete two-step rollout with lambda19/20,the only extra term is
(-20301/160000)/20=-20301/3200000. Adding3/800 gives the already verified
signed expectation-8301/3200000. Root-history bootstrapping gives3/800.

If the rollout stops after the root action,before the opponent's terminal
transition,lambda1 still bootstraps at that opponent observation. Equation(2)
now has only the boundary term,and the signed expectation is-19701/160000.
Root-history bootstrapping again gives3/800. Thus a finite-rollout lambda1 arm
is not a Monte Carlo or generally unbiased control. This is a new calculation
on the same finite game,not a new Koi-Koi strength result.

Executable scope
----------------

scripts/diagnose_observer_trace_identity.py checks(1) with exact rational
arithmetic across arbitrary repeated/changing player sequences,terminal and
nonterminal cutoffs,and lambda endpoints. It enumerates the existing game's
eight outcome atoms to check(2) and the boundary example,and compares both
return modes through the actual CPU implementation. No sampled matches,
training,search branches or CUDA work are performed. The live strength-study
protocol and all its bound sources remain unchanged.
