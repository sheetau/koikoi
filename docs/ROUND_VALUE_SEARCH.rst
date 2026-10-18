One-round search with shared value-table uncertainty
===================================================

Implementation and information
------------------------------

scripts/round_value_search.py adds an experimental adapter over the existing
bounded HistoryConditionedSearch. Strict CPU parent fallback, public endgame
assistance, the hand-space cap, equal initial WP/Point prior, strict historical
compatibility, and uniform control are unchanged. It is not the playable default.

At an eligible root, sample observation-consistent paired worlds and fixed
opponent model labels. Simulate each candidate and parent action to the FIRST
round ending with the existing frozen batched continuation policies. True match
termination returns signed win/draw utility. Otherwise look up the next-round
predeal value using only round index,public points,absolute dealer,parent seat
and the sampled opponent label. Do not inspect the newly dealt private cards.
Neither actual opponent identity nor actual hidden cards enter the inference.

Tables are tied to the exact retained-parent checkpoint. The loader requires a
completed ROUND_VALUE_ACCURACY study and verifies its bound inputs and artifacts.
Completion means that the estimator comparison has been performed; it does not
certify accuracy or playing strength. Synthetic tables are only test fixtures.

Uncertainty and selection
-------------------------

Let D_i(V) be half the alternative-minus-parent signed value for paired world i.
The point estimate is the average D_i for the fitted table. A bootstrap replicate
uses one complete resampled value table and one independently resampled set of
world indices. All alternatives share the same worlds,model labels and bootstrap
world weights. Shared table error is retained across every particle and public
context; it is not divided by the number of worlds as though each lookup had an
independent fitted error.

The decision margin is mean(D)-confidence*std(joint_bootstrap_means),with default
confidence2.5. Also record conditional particle and table standard errors as
diagnostics. The combined bootstrap is approximate; it is not a distribution-free
confidence guarantee and does not correct a misspecified opponent/chance model.

Use table A for all proposals. Select the largest mean improvement,with legal
action order breaking exact ties. Only a positive-margin selected candidate is
confirmed. Confirmation uses table B,fresh worlds and a separate bootstrap RNG;
it never reranks candidates. Require a positive confirmation margin to change
the parent action. Confirmation cannot be disabled for this adapter. This is a
specified use of independent estimates,not a new general selection theorem.

Record public endpoint fields,paired values,model labels,bootstrap seed,all joint
replicate means and actual work. Work counts decisions,conditional round endings
and true terminal endpoints. A round ending is not a completed rollout match.
Retain cumulative confirmation work separately. reset_seed resets proposal,
confirmation and the two uncertainty streams,so duplicate deals are reproducible.

Correctness and remaining evidence
---------------------------------

Five focused tests pass on CPU/CUDA. Independently replayed Python trajectories
from both parent seats match all first-round endpoints; raw public fields rebuild
each fitted/bootstrapped value and the joint margin. Tests cover exact terminal
draw/bankruptcy precedence,absolute axes,read-only arrays,and invalid live fields.
The history fixture confirms A-to-B routing,fresh samples,all legal alternatives,
unchanged roots and invariance to actual hidden-hand/stock reassignment. That
test forces acceptance solely to exercise confirmation,not to claim improvement.

The synthetic tests add no strength evidence. The real estimator study is still
separate. Before promotion,a new fixed comparison must measure paired-action
estimation error and cost against full terminal rollout,then replicated complete
match outcomes against the retained parents and unchanged WP/Point comparators.
No policy training,real-table search study or deployment promotion is specified
by this implementation document. Keep all existing completed diagnostics fixed.
