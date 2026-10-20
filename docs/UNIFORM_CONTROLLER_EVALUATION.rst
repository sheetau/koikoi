Prospective evaluation under the controller's conditional deal law
================================================================

Design selected on 21 October 2026, before generating any new source matches.
The runner, complete audit and statistical export have passed fixed-fixture
validation. This document and uniform_controller_design.py specify the design;
the hash-bound protocol and execution state are recorded separately in STATUS.md.
The design and capability checks alone report no new strength result.

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


Validated implementation
------------------------

uniform_controller_study.py persists each seat and accepted round deck before
using it, and refuses an existing source directory even after a partial failure.
Actors receive only the unchanged environment. Every complete source is saved
before planning. Each program/selection is saved before actual tail execution.
Sixteen-source collection/planning workers and separate audit workers run in
bounded processes. The audit recomputes actor actions, public inference, root
selection and the complete hand posterior; it compares reconstructed worlds,
phase bindings/caps, strict selection and saved executable tail behavior.
Complete chunk receipts bind every source/deck/program/selection file. The final
reduction requires every declared case in order and its matching audit, then
exports exact rational bounds, a readable report and a binding receipt.

uniform_controller_assessment.py independently evaluates the unchanged parent
by invoking its actor directly. It does not read the native planner's parent
map or action caches. It evaluates the candidate's explicit map and recognized
parent-default rule with separate reference-built caches, expanding every stock
chance and reaching terminal states without outcome shortcuts. This independently
checks both Qs supporting adoption. The historical implementation remains intact.

On21October2026,36 sampler/collector/statistical/design/runner/assessment tests
passed in4.36s. Eight additional pinned neural pipeline fixtures passed in126.06s,
with only the upstream Transformer performance warning. The eight fixtures use
the mechanically chosen indices in LOGGED_POSTERIOR_COMPOSITION.rst. They retain
their historical public dealer contexts and fixed tapes, not fresh entropy.
Their computed selections exactly match the closed symbolic comparison: two
strict adoptions, five no-gain fallbacks and one unvalued-parent fallback.
The synthetic assessment tests separately exercise the own-win parent-default
rule and both viewer seats. No fixture is a new strength observation.

For the eight neural fixtures, source/native/source-audit work is970decisions,
64round endings and8matches each, separately counted. Planning uses3,549,448
physical decisions and3,119,002completions; represented work is31,925,565decisions
and6,833,106completions. The independent two-value oracle uses398,803physical
decisions/353,500completions and2,875,140represented decisions/617,760completions,
with1,209new parent and101new opponent calls. No native action cache is imported.
Actual parent/controller tails use52decisions/16completions, with a separate
identical trace replay. Each support pass and audit reconstruction uses10,101
decisions/5,117opponent calls. These are verification costs, not independent
sample counts or a latency benchmark.
