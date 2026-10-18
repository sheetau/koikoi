Prospective final-round card-search pilot
========================================

Motivation
----------

The completed first-round terminal pilot improves conditional-reference
estimation with history conditioning but changes no actual final point result.
Two earlier card/koi diagnostic corpora contain no final-round roots. This new
pilot tests that missing setting: a final-round continuation has no subsequent
rounds, and its terminal result directly measures the remaining match objective.
Neither observation predicts a positive effect by itself.

Frozen cases and policies
------------------------

Use both retained parents, opponents WP and Point, seats0/1, and16 replicates
per cell:128 source matches on32 distinct deal streams. Case order is parent,
opponent,seat,replicate. Source seed is87110001+100*opponent_index+replicate.
Both parents and both seats share each opponent's deals; bootstrap units are
whole duplicate-deal pairs. Reserved101M streams are not used.

Play every source match to termination with strict CPU parent/opponent policies
and their existing public koi assistance. Record all naturally reached unforced
card decisions in round8. For each of phases0/1/2 retain only the first such
learner decision. Forced actions, earlier rounds, opponent decisions and koi
decisions are not roots. Retain missing opportunities and early bankruptcies.
No root is selected using its score estimate, realized outcome or hidden hand.

Define three separate candidate policies, one per card phase. A candidate
changes at most its first eligible action in that phase of the final round;
all other actual choices retain the strict parent. These are not a combined
three-intervention policy. Cached root choices can be replayed exactly because
the match prefix is unchanged. Complete all128 matches for each candidate,
including unchanged or missing opportunities:384 candidate matches.

Search and references
---------------------

Use FinalRoundSearch with1024 proposal particles,2048 fresh selected-action
confirmation particles and the unchanged2.5-standard-error margin. Compare
every legal alternative against the strict parent's base action. Select by
largest proposal mean, breaking ties in legal-action order; confirmation
cannot re-rank alternatives. The root seed is87210001+3*case_index+phase;
confirmation uses SeedSequence([root_seed,1]).

Worlds use the existing observation-consistent uniform hidden proposal and
equal WP/Point model mixture. The label is fixed across a particle's branches.
All simulated policies use fixed64 inference. Actual opponent identity is
evaluation metadata and is not supplied to search. The proposal ignores
opponent-action likelihoods within this round and earlier rounds: it is an
approximation, not a posterior or exact conditional value.

For every root and every alternative, collect4096 independent reference
particles in eight512-particle chunks, continuing RNG seed87310001 plus the
same root index. Reference worlds use the same approximate distribution;
their results are independent Monte Carlo checks of that target, not an
assessment of posterior misspecification. All branches must terminate in the
current round. Preserve every chunk, action, sampled-world hash, opponent label,
paired point result and common-baseline equality check.

Analysis and stopping
---------------------

Report opportunity counts, proposals, confirmations, action changes, estimator
errors against the noisy reference, costs and all actual win/point differences.
For each parent/opponent/phase cell report candidate minus parent on16 paired
deals:12 contrasts. Use10000 whole-pair bootstrap draws, seed87410001, shared
across all cells. Intervals are descriptive pilot intervals without a
multiple-comparison or population-strength certificate. Do not discard null
or adverse cells, extend the deal pool, lower the margin or choose a new
threshold from this pilot. A positive pilot would motivate a separately
declared independent strength study, not automatic deployment.

Count128 source matches,384 candidate matches, simulated branch completions,
round endings and decisions separately. The512 actual matches still share
only32 independent deal streams. The independent audit replays source and
candidate choices, root selection, hidden-world sampling and result arithmetic;
it does not claim to rerun every neural future. Existing CPU/CUDA trajectory
tests provide separate execution evidence. Preserve any failed execution and
its source identities before correcting it.

Rollout policy improvement is established methodology. This pilot tests a
specific remaining opportunity; it does not make final-round search itself a
new algorithm. Canonical human play, dealer choice and the manuscript remain
unchanged until there is additional evidence.
