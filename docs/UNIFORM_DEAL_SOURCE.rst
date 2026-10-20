Sampling the deal law used by the conditional guarantee
======================================================

The current conditional evaluator uses a uniform distribution over valid deals
and fixed deterministic opponents. The completed and running512-source studies
instead reproduce deals from small fixed seeds. Uniform remaining-card weights
are not the posterior over those finite seeds. Their actual outcomes and exact
conditional values must therefore remain distinct evidence. None of the existing
protocols, outcomes or strength criteria changes here.

scripts/uniform_deal_source.py prepares a separate source generator for a future
evaluation under the guarantee's stated deal law. No evaluation is scheduled or
launched by this module. The existing symbolic comparison must complete before
choosing the next study, its population, estimator, sample size and criterion.

Sampling and replay
-------------------

Fisher--Yates uses independent uniform draws from0..i at each positioni. Each
permutation corresponds to exactly one sequence of choices and has probability
1/48!. Rejecting deals that violate the existing initial-month rule gives the
uniform law conditional on validity. Fresh bounded draws are required between
rounds and sources. The default uses secrets.randbelow; the uniformity statement
assumes independent uniform entropy and does not certify the operating system's
entropy. Injected deterministic test tapes establish algorithm/transition
correctness, not statistical strength or entropy quality.

Accepted decks, round indices, attempts and bounded-draw counts are logged
outside the actor-facing ReferenceEnv. Actors receive that unchanged environment;
they do not receive the sampler, round log or future deck list. Existing feature
construction retains its information restrictions. Initial dealer selection is
unchanged; later dealers and scores come from the same game rules. The harness
replaces only a newly started round's deal, before the next actor observation.
ReferenceEnv's internal seeded auto-deal is discarded at that boundary. Sampling
counts refer to the separate valid-deal generator, not that internal overhead.

Replay injects the logged accepted decks at the same boundaries and executes
all recorded actions. It requires one terminal match and exactly its used round
deals, and rejects invalid/missing/out-of-order/extra deals. Chance sampling work
is separate from decisions, round endings and match completions. Reaching the
attempt cap raises an error rather than substituting a seeded deal; a failed
source cannot continue through the harness. A future study must stop and inspect
such failures under its declared protocol.

What this enables
-----------------

Under a matching deal/opponent law and exact conditional beliefs, averaging the
adopted policy's conditional advantage over independently sampled parent prefixes
targets the expected match-win difference in INCUMBENT_GUARANTEE.rst. This is a
conditional-expectation evaluation, not a realized win count. The law, complete
prefix denominator, fixed search/selection procedure, opponent models and
independent value verification must all be specified before a new study.

This module alone establishes neither the correctness of a future study's full
posterior nor stronger play. It does not retroactively turn existing conditional
gains into empirical win gains, combine results across different sampling laws,
or change the failed prospective strength gate.

Validation
----------

Tests enumerate all24 bounded-choice sequences for four cards and obtain every
permutation once, check invalid-deal rejection and separate sampling counts,
and reject out-of-range draws and silent fallback after sampling failure. Two
deterministic fixture matches use paper8/research12: every transition/result
matches the native simulator after injecting the same logged deal at each new
round. Full Python replay reproduces each terminal state and work count. These
are capability fixtures, not naturally sampled benchmark matches.
