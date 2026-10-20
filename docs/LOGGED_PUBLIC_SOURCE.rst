Public-history collection from logged round deals
================================================

scripts/logged_public_source.py connects LoggedDealSource to the existing
delayed public opponent reconstruction. It prepares the source-collection side
of an evaluation under independently sampled valid deals. It defines no study,
sample size, strength criterion or experiment launch. The symbolic comparison
must complete before the next evaluation is chosen.

Collection
----------

collect accepts a fresh paper8 LoggedDealSource, a public source identifier,
learner seat, parent/opponent actors, named public predictors and one native
environment. The public identifier need not contain the opponent's evaluation
label. Actors receive the ordinary ReferenceEnv, through their existing
observation-only adapters; they receive neither the external deal log nor the
sampler. The parent plays its complete match regardless of whether a root is
selected. Missing-root matches remain complete source records.

The collector saves the first learner decision in the final round whose public
remaining-draw count satisfies the supplied cap and whose opponent model was
identified from an earlier round. Identification must be available before the
decision. Gate selection receives the public model-reconstruction object, not
an evaluator-supplied opponent identity. Its default draw cap is five, matching
the previous collector; a future protocol must explicitly bind its chosen cap.

Every transition and result is checked against the native engine. At a new-round
boundary, both engines first execute the unchanged game rules. Their metadata
must agree before the native auto-deal is replaced with the same logged valid
deck. The full resulting states must then agree. Initial dealer selection,
later dealer changes, scoring and termination remain unchanged.

Records contain actions, round decks, the selected root, public identification
events/calls, terminal state and separate source/native work counts. Sampling
attempts and external bounded-draw counts are separate from decisions, round
endings and match completions. The full selected state is evaluator evidence;
planning must still construct its own history-consistent hypothetical worlds.

Replay and policy verification
------------------------------

audit reconstructs the match from saved deals and actions, recomputes public
identification, and checks the first eligible root and its complete state. It
also calls the supplied parent/opponent actor at every decision and requires
the recorded action to agree. Trajectory replay, policy-verification adapter
calls and public-prediction calls are reported separately. It draws no fresh
external chance; ReferenceEnv's discarded internal seeded deals still occur,
as explained in UNIFORM_DEAL_SOURCE.rst.

The audit rejects mismatched actions, work counts, identification events, root
addresses/states, terminal states and missing/unused/out-of-order round deals.
It requires exactly one complete match and uses a finite decision bound.
Both collection and replay require a fresh successful source. A failed sampler
therefore cannot silently fall back to the simulator's seeded auto-deal.

A future study must bind these actor/predictor implementations, their weights,
the source identifiers and sampling procedure, then verify the complete cohort.
The collector's logged_round_decks label deliberately makes no IID claim:
injected deterministic fixture tapes also work. Independent uniform draws and
valid-deal rejection must come from the declared sampler. Full posterior-law
agreement, exact continuation values, comparison selection and population
confidence bounds remain separate requirements before claiming mean match
improvement. This module changes none of the existing fixed-seed studies.

Verification
------------

Six tests cover both seats, native transition/result parity and complete logged
replay, unchanged initial dealer selection, ambiguous model sets, and real
delayed public revelation distinguishing two deterministic rule actors. Separate
synthetic identification fixtures isolate the first-eligible-root address and
state. These use a 16-draw test gate to exercise the first final-round decision;
they neither change the production default nor select a study gate. Invalid
caps and previously started sources are rejected before actor calls. Tampered
actions, work counts, model-identification events and root addresses fail audit.

The tests use deterministic complete-match fixtures and their native/Python
replays, including deliberately corrupted replay attempts. They are simulation
work for capability verification, not newly sampled benchmark matches, planning
rollouts or neural inference. No published model is loaded in these tests.
