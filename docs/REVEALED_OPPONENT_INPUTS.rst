Retrospective opponent inputs from public revelation
===================================================

Capability
----------

scripts/revealed_opponent_inputs.py reconstructs an earlier actor input after
that actor's eight original hand cards have become public through hand discards.
It combines that revealed set with a historical public snapshot. It neither
classifies the opponent nor chooses a response policy.

The snapshot erases both private hand membership and its unknown-card
complement. It retains public board/captures, pending card, initial field,
played-card history, scores, counts, phase, round, seat and dealer. The decoder
subtracts the actor's earlier public hand discards from the revealed initial
hand, fills the remaining hand and unknown complement, then applies the existing
public-feature decoder. Legal and publicly assisted legal masks are reconstructed
separately. No simulator state, stock order, RNG or true private hand is supplied
to this decoder.

Revelation records must be bound to a caller-provided public match identity and
actor seat. The decoder rejects different matches/rounds/actors, incomplete or
duplicate hand sets, conflicting public discards, and use before revelation.
Its current scope is a historical decision preceding the eighth public hand
discard; it is not an online predictor of unrevealed cards.

Verification
------------

Two tests pass, zero skips, in2.85s. The final run checks744 retrospective
decision addresses from12 deterministic toy-source matches, covering both seats
and all four phases. Decoded300x50 actor features equal the independent native
builder exactly. The corresponding48/50-column inputs also equal the strict
upstream feature builder exactly at every checked address. Both legal masks
equal the engine's masks, including public assistance.

Seat/phase coverage (discard, hand capture, draw capture, koi):

* Seat0:312,16,14,31.
* Seat1:320,11,14,26.

Changing either unobserved hand assignment, while preserving public history,
does not change the stored public snapshot or its reconstruction from the same
public revelation. Early use and cross-match/round/actor substitutions are
rejected. The tests generate their toy source trajectories more than once;
these repeated fixtures are not independent playing-strength observations.
There is no neural policy inference, fitting or held-out evaluation in this
capability test. The short CPU checks overlap the ongoing tail evaluation,
whose timings are not isolated deployment measurements.

Research interpretation and prior work
-------------------------------------

The preceding PUBLIC_HAND_REVELATION.rst diagnostic finds many such opportunities
in the retained training prefixes. Exact reconstruction on these fixtures makes
the next test concrete: compare the fixed WP/Point models' predictions on the
retrospective inputs, using each revelation only after it occurs. Contradicting
a deterministic model can eliminate it from a declared finite candidate set;
agreement alone does not identify it. An empty candidate set must be treated as
model mismatch, not silently assigned to one known opponent.

Bayesian opponent inference and response selection are established ideas.
Southey et al.'s Bayes' Bluff (UAI2005) separates card uncertainty from opponent
strategy uncertainty and computes responses to opponent beliefs:
https://webdocs.cs.ualberta.ca/~mbowling/papers/05uai.pdf
Rosman, Hawasly and Ramamoorthy's Bayesian Policy Reuse (Machine Learning2016)
uses observed signals to choose policies from a learned library:
https://www.research.ed.ac.uk/en/publications/bayesian-policy-reuse/

The present contribution at this stage is a tested Koi-Koi input-reconstruction
capability. It does not establish a new general opponent-modelling algorithm,
identification accuracy, calibrated posterior, useful specialised responses or
stronger play. Any identification or response experiment is separate from the
running fixed-prefix study. That study's policies and inputs remain unchanged.

Evidence: tests/test_revealed_opponent_inputs.py,
runs/revealed_opponent_input_tests.xml and
runs/revealed_opponent_input_capability.json.
