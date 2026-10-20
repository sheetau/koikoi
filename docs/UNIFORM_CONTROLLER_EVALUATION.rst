Prospective evaluation under the controller's conditional deal law
================================================================

Design selected on 21 October 2026, before generating any new source matches.
The runner and complete audit still need integration tests and a hash-bound
protocol freeze. This document and uniform_controller_design.py specify the
planned evaluation; they do not launch it or report a new strength result.

Question and population
-----------------------

Does the executable symbolic controller improve expected draw-adjusted match
win score over its unchanged parent under independent uniform valid round deals?
Evaluate each of the two pinned PPO parents against each pinned, deterministic
WP/Point opponent, with the same public endgame assistance as before. These are
four separate populations. Randomize the learner seat independently and uniformly
for every source, before its first sampled round deal. This makes each population
an IID equal-seat mixture rather than pooling two fixed-seat samples as IID.

Use paper8 and ReferenceEnv seed1/episode0, whose existing initial-dealer rule
selects seat0. Every initial and subsequent seeded auto-deal is discarded and
replaced by sample_valid_deck before an actor observes the new round. The seed
therefore fixes the initial public dealer context; it does not supply the source
deck distribution. Later dealers follow the unchanged rules. A randomly assigned
learner seat gives the learner each initial-dealer role with probability1/2.

The logged source uses independent secrets.randbelow choices for the seat and
Fisher--Yates/rejection draws. Its independence/uniformity are explicit entropy
assumptions, not properties proved by observing a finite sample. Decks, seat,
attempts, actions and work counts are logged for exact replay. Source identifiers
are public counters and do not encode the evaluator's opponent assignment for
the controller. Opponent identity still comes from delayed public revelation.
LOGGED_POSTERIOR_COMPOSITION.rst states the source/posterior factorization.

Policy and budget
-----------------

Use the completed SymbolicFinalTurnPlanner controller without tuning it on the
new sources. It follows its pinned parent until the first eligible learner
decision in round8, with at most five remaining draws and opponent identification
from an earlier round. Enumerate all valid history-compatible hands, capped at
1,000 candidates. Use capacity4,096, depth32, and separate fixed/optimized phases
of at most2,000,000 physical decisions and500,000 nodes each. Existing skip and
strict exact-adoption rules remain. There is no budget transfer or root-only
search arm in this study. The earlier computational comparison already measured
that control; this study targets the deployed candidate's expected improvement
over the parent it follows on fallback.

Independently verify each adopted parent/program value by full Python continuation
replay. Preserve the symbolic executable continuation rule in saved artifacts.
Save the selection before executing parent/controller conditional tails. Report
the realized tail outcomes descriptively and separately from the primary expected
improvement estimate. Replay logs and recheck all source actor decisions, public
identification, eligible-root selection, hand support, program bindings and budget
counts in the complete audit. No source is accepted solely because its stored
gain is positive. Actual execution is also a correctness check of the saved
program, not an extra independent source sample.

Fixed sample and inference
-------------------------

Collect exactly2,048 sources in each of the four populations:8,192 total.
Interleave populations by replicate with one source per population. Each source
gets its own independent seat/deal draws, with no paired deck reuse across
parents, opponents or seats. Use16-source worker chunks and four workers. The
small bounded processes avoid the native loader's constructor accumulation.

Every source contributes one exact Delta in[0,1]. Missing roots, public hand-cap
refusals, unavailable parent values and all other legitimate parent fallbacks
contribute zero. Negative contrasts must never be clamped. A source/audit failure
halts inference; do not replace a failed or interrupted source with new randomness.
Technical recovery must preserve already sampled records and the fixed protocol.
No stopping for significance, outcome-selected extension or threshold revision.

The seven fixed thresholds are

  1/1000, 1/250, 1/100, 1/50, 1/25, 2/25, 4/25.

Use certified_gain_bounds.py with32-bit downward dyadic rounding. Allocate total
error1/20 equally to the four populations:1/80 each, and1/560 per threshold.
The28 survival bounds are therefore simultaneous at least95% confidence under
the stated IID laws. For each parent, average its WP/Point lower bounds with
fixed weights1/2,1/2. This derives simultaneous mixture bounds from the same
event and requires no additional error allocation.

The study succeeds only if all four population lower bounds are strictly positive
AND both parent mixture lower bounds exceed1/1000 (0.1 percentage point).
The latter corresponds to more than one expected draw-adjusted win per1,000
matches in the equal-opponent mixture. It is a prespecified minimum practical
effect for this evaluation, not a universal competitive threshold. Exact fractions
decide the criterion; display rounding must not determine success.

The claim is incremental improvement over each parent, with replication across
the two training runs and positivity against each declared opponent. It is not
an absolute win rate above50%, an observed-win superiority test, an improvement
against arbitrary opponents, or a minimax/world-best claim. All four estimates
and both mixture bounds must be reported even if the joint criterion fails.

Design rationale and separation from old evidence
------------------------------------------------

Before sampling, the closed512-source symbolic results were used to inspect
gain magnitudes and rough budget requirements. Projections scaled their empirical
threshold frequencies to fixed future sample counts and applied the planned
bound calculation. The eight-fixed-seat-strata alternative was more conservative;
independent seat randomization preserves each desired equal-seat population and
avoids the unnecessary separate seat error allocations.

At2,048 sources per randomized-seat population, the closed-data projection gives
lower bounds of approximately0.155/0.096pp for parent1 against WP/Point and
0.211/0.130pp for parent2. Their mixture projections are approximately0.126/0.170pp.
The old and new source laws differ. These numbers are design projections only:
they are neither power calculations, valid new-population confidence bounds,
nor promised results. They explain the chosen fixed effort and threshold grid.
No new sources were used to select the design. The prior actual-win criterion
remains failed; none of these projections changes it.

Prerequisites before launch
---------------------------

Bind the complete runner/auditor dependency graph, checkpoints, native source,
design module, tests and this document in a new immutable protocol. Validate
fixed-deck pipeline fixtures, missing-root/fallback accounting, malformed and
incomplete-cohort rejection, source replay, posterior reconstruction, both tail
executions and statistical export. The complete audit must reproduce the full
8,192-source denominator and all four population counts before calculating a
study result. Pending work and freeze status belong in STATUS.md. Human play,
dealer logic, previous cohorts and the reserved101M cohort remain unchanged.
