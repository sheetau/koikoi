Direct response learning: fixed development protocol
===================================================

Question
--------

Does updating the pretrained actor on trajectories against the fixed public
WP/Point mixture improve match wins over both its unchanged parent and an
equal-simulator-budget self-play continuation? Earlier one-action target studies
did not establish a replicated benefit. This study changes the actor through
on-policy interaction instead of fitting a selector once on a fixed source.
Mixture response training and PPO are established methods; their integration
alone is not an algorithmic novelty claim.

Conditions
----------

Use both retained warm parents, seeds6710001 and6720001, update520. Each has two
fresh continuations: response against the fixed WP/Point mixture, and ordinary
current-policy self-play. Training seeds are19910001 and20010001 respectively,
shared between the two conditions within each parent. Retain every final endpoint;
no intermediate ranking, early stopping or budget expansion follows results.

Both conditions initialize actor and learner critic from the same parent and
start new Adam/environment/RNG states. They use the existing T0.05, actorLR3e-6,
valueLR1e-4,128 environments,128-step horizon,512 minibatches,3 epochs,20 critic-only
updates and500 policy updates. Exactly8,519,680 simulator decisions per condition;
34,078,720 across four continuations. The response condition has a fresh separate
opponent-role critic and frozen public opponents. Its actual learner decisions
are fewer than self-play's; report them separately. Extra critic gradients also
participate in joint clipping, so the comparison is the full training intervention,
not an isolated effect of opponent identity at equal actor samples.

Within each response lane, learner seat is(lane+episode) modulo2; opponent is
(floor(lane/2)+floor(episode/2)) modulo2, ordered WP,Point. Both remain fixed for
the whole match. Across four episodes a lane visits all four combinations. Match
lengths determine actual source proportions, which are reported rather than
assumed equal. Opponent policies are deterministic, public and assisted. Learner
actions are sampled during training. Frozen turns contribute critic/return data
but no actor loss. The complete checkpoint preserves both critics, opponents,
optimizer, environment, RNG, identities and exact accounting.

Evaluation and decision
-----------------------

Each of the four endpoints and both unchanged parents faces strict CPU2 greedy
assisted RL-WP and RL-Point. Use exactly5,000 duplicate pairs per matchup, sharing
development deal seeds23,000,000--23,004,999 across all12 matchups. These are5,000
unique deal streams,120,000 matches, not60,000 independent deals. Training seeds
and this evaluation range have no matches in the retained protocol inventory.
The previously consumed100M final range and reserved101M range remain separate.

Recompute win score, mean final point difference and paired-change intervals
from raw seat-swapped results. Bootstrap5,000 resamples of complete deal pairs
with seed99. The fixed qualification criterion requires positive lower95 bounds
for response-minus-parent AND response-minus-selfplay, for each of both parent
models against each public opponent. Point changes are secondary. All models,
both objectives and all contrasts remain in the report. This development decision
does not authorize deployment or establish superiority to other implementations,
humans or an equilibrium solution. Pair intervals do not quantify training-seed
uncertainty.

Execution and validation
------------------------

Run python -m scripts.run_response_study. A fresh directory is required. Before
work, bind production source, runner/helpers, parents, upstream Python sources
and all six comparator weights. Recheck before worker launch and at training
checkpoints. At most one GPU trainer and three CPU2 evaluators run concurrently;
endpoint evaluation waits for its successful training and accounting audit.
On failure stop only owned subprocess trees and retain failure/cleanup records.

The final audit checks complete scheduling, source/parent identities, exact
initial weights, fixed budgets, finite tensors/metrics, actual Adam steps, frozen
opponents, per-opponent work, checkpoint receipts and recomputed raw match
statistics/contrasts. It does not independently replay every training trajectory.
The reduced --smoke path uses one parent,two updates,two pairs and a separate seed
range. It reports no strength qualification and is used only for pipeline tests.
