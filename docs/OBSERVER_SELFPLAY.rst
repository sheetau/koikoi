Fixed-observer self-play: capability and comparison design
========================================================

Question
--------

The completed learner-decision experiment has positive mean win changes in all
eight primary comparisons, but passes only one confidence-bound requirement.
It also changes value fitting and omits unfinished learner tails. A new test
will keep an observer's information convention throughout each primitive-step
return, including opponent turns, in the self-play setting where the original
published-policy adaptation succeeded. This is a new capability stage, not a
registered strength study or a replacement for the failed learner primary gate.

Information and returns
-----------------------

For each live state, construct two SEPARATE value inputs: the 560 ordinary
features relative to that observer and a bit indicating whether the observer
is the actual actor. Public phase, history and before-turn yaku points describe
the actual turn. Do not reinterpret them as an inactive observer's turn.
The implementation changes PLAYER only in a copied observation scratch buffer.
It never steps those copies, uses their legal masks, or forwards them to the
actor. The actor retains its original packed published observation and mask.

A shared row-wise MLP evaluates each input independently. No concatenation of
the two private hands, cross-row attention or batch normalization is allowed.
Opponent hand identities, stock order, RNG state and future events are absent
from each observer's own features. Parameter sharing during training is allowed;
it does not assert that fitted parameters are independent of training data.

Let r_t be seat-zero match reward and v_t^i the value from observer i's input.
With gamma=1, define delta_t^i = (-1)^i r_t + live_t v_(t+1)^i - v_t^i and
A_t^i = delta_t^i + live_t lambda A_(t+1)^i. A match end sets live to zero;
a round end does not. Fit targets v_t^i + A_t^i for both observers and use
A_t^(actor_t) for the policy. The tail has both observers' values, so an
opponent turn at the boundary does not require dropping a learner action.
The two conditional values need not sum to zero despite zero-sum payoffs.

Matched control
---------------

Both new arms will use the same architecture, initialization, per-observer
value targets, value-loss normalization, optimizer, clipping, minibatches and
simulation budget. The signed control will select active-observer values and
use the historical signed-GAE policy advantage. The treatment will select the
fixed-observer advantage. Critic-only warmup must be identical under matched
seeds. Subsequent trajectories need not remain identical. This control is
distinct from ordinary historical self-play, which must also remain visible
in any strength comparison along with unchanged parents.

The intended initial critic pads the parent's first-layer weight with a zero
role column, preserving its original value on active observations. The actor
is unchanged initially. Trainer integration, exact restart and a production-size
capability check are required before defining a fixed strength protocol.
Concrete budgets and fresh seeds must be recorded before running the strength
comparison. Reserved 101M tests remain unused.

Prior work and contribution boundary
-----------------------------------

Independent per-agent critics and parameter sharing are established methods.
Lyu et al., On Centralized Critics in Multi-Agent Reinforcement Learning,
JAIR 77 (2023), 295-354, distinguish individual-history, joint-history, state
and history-state critics and analyze bias/variance and approximation tradeoffs:
https://arxiv.org/html/2408.14597v1 . Its predecessor papers are already cited
in the manuscript. This implementation is not claimed as a first per-agent
critic or a general unbiased PPO algorithm. Ordinary finite Koi-Koi features
are not proved sufficient histories; critic approximation and PPO optimization
remain additional sources of error.

The focused contribution being tested is the effect of switching conditional
information across turns in a signed shared-policy return, its explicit
counterexample, and a matched remedy in pretrained imperfect-information
self-play. On the existing two-step example, retaining the learner's private
signal at the intervening public opponent turn should restore the correct
population gradient. A Koi-Koi strength benefit still needs measurement.

Validation gates
----------------

* Forward-expanded return oracle, match/reset and round-boundary cases,
  lambda endpoints and terminal-tail masking on CPU and CUDA.
* Exact shared-parameter counterexample; opponent-only value perturbations
  cannot change the fixed-observer advantage for the other seat.
* Native observations equal independent Python reference projections for
  both viewers and all phases; hidden-world and RNG changes leave each own
  input unchanged. Live simulator state must remain byte-identical.
* Trainer initialization, matched critic-only warmup, actual action/reward
  routing, CPU/CUDA exact restart and finite optimizer/counter checks.
* Fixed capability budgets count decisions, round endings, completed matches
  and value rows separately. They provide no strength estimate.
