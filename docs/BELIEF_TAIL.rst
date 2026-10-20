Finite belief-tail planning
==========================

The completed full-source comparison does not establish a retained-parent win
gain. The next intervention changes the continuation decision problem. Existing
retained-agent searches compare first actions and subsequently use fixed actor
policies. Here future learner decisions are optimized jointly across worlds
that share the same visible history.

This is finite-horizon belief-state dynamic programming, an established POMDP
construction, not a new general planning algorithm. Its application here needs
an exact current-round hand distribution, an executable contingent policy and
a bounded computation. The primary reference is Kaelbling, Littman and Cassandra,
*Planning and Acting in Partially Observable Stochastic Domains*, Artificial
Intelligence 101 (1998), 99--134:
https://www.cassandra.org/arc/papers/aij98.pdf.
The accessible author technical-report version, CS-96-08, Section 4.1,
explicitly defines observation-contingent policy trees and their belief-weighted
values: https://cs.brown.edu/research/pubs/techreports/reports/96/cs96-08.pdf.
The absence of an identical method in a limited Koi-Koi search is not evidence
for a priority claim.

Finite worlds and information
----------------------------

Restrict the implementation to the final round of paper8. Let d be the maximum
number of future draws before the round ends. For a discard or hand-capture
decision, d=16-turn; after the current draw, d=15-turn. If n stock cards remain,
then n-d=8: eight cards will remain unused even when all sixteen turns finish.

Enumerate each ordered d-card prefix of the remaining stock. Every such prefix
has exactly (n-d)! compatible stock suffixes, so those suffixes cancel from
normalized expectation. Thus the maximum three-draw case needs 11*10*9=990
prefixes per compatible hand/model pair instead of 11! complete stock orders.
Earlier termination is handled by retaining the multiplicity of prefixes that
lead to the same terminal observation. It does not resample unused draws.

Reuse the complete current-round hand/model likelihood table. All valid hands
have the same prior permutation multiplicity under the declared uniform valid
deal model; each compatible hand/model pair receives its model's prior mass.
Do not normalize hands separately within each model. Integer prior weights give
exact integer world masses and rational final action values. Zero prior mass
may encode a model already eliminated using an earlier round's public evidence.
This is not the posterior over the implementation's finite PRNG seeds.

Only hypothetical worlds enter the solver. The actual opponent hand, stock
order and RNG are overwritten. Worlds retain allowed own/public inputs for
every policy call. A learner information key contains its current and initial
hand, public locations, initial field, match/turn/phase context, hand counts and
complete within-round action history. Opponent hand identities, stock order,
seed and episode are absent. Absolute seat/dealer identity is retained because
the published policies use it.

The dynamic program groups worlds by this key before every learner choice.
One action is applied to the whole group. Opponent policies may use their own
private hands, but their actions are produced by the existing observation-only
adapters. Different observed opponent actions and draws split subsequent
learner beliefs. Maximizing independently inside hidden worlds is prohibited.
Policy caches use the acting player's own/public key and assume fixed,
deterministic, stateless models.

Value recurrence and executable plans
------------------------------------

Keep unnormalized integer masses w throughout the tree. A terminal group has
value sum(w*R), where R is the learner-relative signed match outcome. At an
opponent node, execute its declared policy separately for each hypothesis, then
sum values over the resulting visible-history groups. At a learner node, take
the maximum of that sum over legal common actions. Final normalization converts
signed values to draw-adjusted win scores.

Induction on the remaining finite tree proves optimality within the stated
belief and opponent model. The fixed-continuation control replaces future
learner maxima with its retained parent policy, while still comparing every
first action. Consequently every optimized first-action value is at least its
fixed-continuation counterpart; that control's best first action is at least
its unchanged parent's first action. These are model-conditional comparisons,
not evidence of observed full-match superiority or minimax optimality.

The solver returns a complete visible-history-to-action policy for every first
action, as well as the best policy. Deploying only its first action and then
discarding the optimized continuation would invalidate its reported value.
Ties retain the parent action when it is optimal. A public world cap or tree
budget failure raises an explicit refusal; an unfinished tree never supplies
an allegedly exact value.

Implementation verification
---------------------------

``scripts/belief_tail.py`` contains the integer-mass planner, exact prefix
enumeration and reusable native CPU transitions. No production engine or
existing study source is changed.

Six tests pass, with no skips. Four finite-game cases independently enumerate
all information-contingent pure strategies, with/without hidden-bit revelation
and with balanced/unbalanced masses. They check optimality, world-order
invariance, fixed-continuation controls and budget refusal. The unrevealed case
explicitly rejects the value that a hidden-world maximum would produce.

Two actual Koi-Koi fixtures, one per seat, enumerate all 4,950 worlds at the
same three-draw position. Every native branch transition agrees with the
independent Python rules. Every first action and its entire fixed/optimized
policy is then executed in all worlds with that oracle; all rational values
agree. Tests also check unchanged root states, complete draw-prefix coverage,
actual-hidden-hand/stock/RNG invariance and computation caps. These fixtures
are correctness checks, not natural-root opportunity or strength estimates.
Final command: ``python -m pytest tests/test_belief_tail.py -q``;
6 passed in 11.70 seconds.

Fixed natural-root capability check
-----------------------------------

Before inspecting new outcomes, fix 64 complete unchanged-parent sources:
two retained parents x assisted original WP/Point x both seats x eight
replicates. Source seed is 121110001+1000*opponent_index+replicate, with WP=0,
Point=1. Parents and seats share the sixteen distinct deal streams.
All actors use strict singleton CPU execution with two threads and public
endgame assistance. No training or reserved101M data is used.

Observe opponent actions through the existing public snapshot/revelation
detector. Within each source, retain its first learner decision in the final
round with at most three future draws, provided a unique WP/Point identity
was already established in an earlier round. The solver receives that public
identity, not the scheduler's opponent label. Complete all sources using the
unchanged parent, regardless of whether an eligible root exists.

At each retained root, enumerate the identified model's complete current-round
hand support. Caps are 100 candidate hands, 100,000 worlds, 500,000 observed
nodes and depth 32. Keep missing roots and refusals; never replace them or
extend source collection. Compare fixed and optimized continuation in that
order with a shared policy cache. Report call counts and times, explicitly
acknowledging warm-cache order; they are not isolated throughput comparisons.

Replay every first-action contingent policy in every enumerated world with
the independent Python engine. This uses the recorded observation-keyed
opponent actions and checks exact outcome sums, but does not repeat every
neural prediction. Freeze and save all plans before actual-world assessment.
Then complete each original root using unchanged parent, best fixed-continuation
policy and best optimized policy against fresh strict opponent calls. Retain
all actual continuation actions, outcomes and deviations. These are conditional
completions of the 64 source matches, not additional independent matches.

Count source matches/decisions/rounds, native source verification, historical
support replay, planning branches, independent policy replay and actual tail
assessment separately. Exact model value gains and a few realized changed
outcomes cannot establish a prospective replicated match-win gain. A successful
capability check would support a separately fixed online-policy comparison,
including the retained parent and fixed-continuation control.

Command after committing the verified implementation and this design::

    .venv/Scripts/python.exe -m scripts.check_belief_tail

The default directory is ``runs/belief_tail_capability``. Preserve it on failure
or completion; the runner refuses an existing directory.
