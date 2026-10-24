Public round-boundary continuation values: engineering specification
===================================================================

V2's fixed 32-decision search is still being evaluated. The control's early
results motivate a separate diagnostic of its learned leaf values, not a
change to that running protocol. An eight-match engineering run at horizon
700 completes every rollout match: the declared eight-round rules permit at
most 624 decisions. Its score is not evidence for selecting a rule or method.

Repeatedly simulating every later round is expensive. For a fixed adapted
policy on both sides, the state before a fresh deal is described by the round
index, absolute next dealer, and player-0 points (the total stays 60). Past
round histories are absent from these policies' inputs. A reusable value
table can therefore average later-round policy play before the fresh deal
is observed. It is an approximate Monte Carlo value table, not an exact
certificate and not a neural critic trained against a different action law.

The table has shape [8, 2, 61] and stores expected terminal utility for player
0. Absolute dealer must remain in the key: the released feature encoder
contains that public seat identity, and an explicit test rejects the tempting
seat-symmetry reduction. Live states have 1--59 points. Boundary values at 0
and 60 are -1 and +1. Each cell samples independent valid initial deals, runs
the fixed observation-only policies to the first round settlement, and uses
terminal utility or the already computed next-round table. Work proceeds
backward from round index 7 to 0. New-deal cards generated during settlement
are not used to condition the cached boundary value.

At a root, sample hidden worlds consistent with own/public cards, evaluate
every legal action on common worlds, and finish the current round. Use the
table at its boundary and retain the same paired-advantage guard. An
unfinished round is an error; no learned-value substitution is allowed.
Bind every table to its exact checkpoint, rule, deterministic policy and
public endgame assistance. It cannot be reused for a different opponent,
checkpoint or rule without a new value calculation.

The initial engineering table uses four samples per cell and is not a
strength endpoint. Its conditional standard errors omit uncertainty
propagated from later tables; no safety guarantee follows. A production
comparison must separately freeze sample counts, guards, evaluation streams,
all offline preparation work and online planning work. Training, conditional
round rollouts and complete-match evaluation must remain separate counters.
