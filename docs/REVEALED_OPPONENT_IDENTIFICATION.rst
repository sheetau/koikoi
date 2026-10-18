Delayed public revelation and finite opponent identification
==========================================================

Method
------

RevealedModelSet starts each match with both fixed assisted WP and Point
policies as candidates. It stores only redacted public snapshots of opponent
decisions and their subsequently observed actions. Until eight distinct
opponent hand discards have been seen in a round, it makes no model predictions.
After the eighth discard, it reconstructs all buffered earlier inputs using
REVEALED_OPPONENT_INPUTS.rst and compares observed actions with each surviving
candidate's strict CPU singleton prediction. A mismatch eliminates that fixed
deterministic candidate. Pending within-round history resets at a new round;
candidate exclusions persist until the match ends.

A singleton identifies a member only under the declared finite model-class
assumption. Multiple survivors remain ambiguous. An empty set is explicit
model mismatch, not a reason to guess or reset the candidates. The method does
not assert that a human, stochastic or changing opponent belongs to this class.
Agreement also does not establish a calibrated Bayesian posterior. The source
opponent label is used only to score the diagnostic, never as predictor input.

Three capability tests pass, zero skips, in2.08s. Toy policies check delayed
use, persistent exclusions, ambiguous/empty support and match/order validation.
Both real published variants also match their original assisted CPU actions
at sixteen seat/phase fixtures. This is execution coverage, not identification
accuracy on the intended source corpus or evidence of stronger responses.

Fixed diagnostic scope
----------------------

After the ongoing tail study completes its strict inference evaluation, inspect
ALL2,048 retained training prefixes from its two parents, on512 shared deal
streams. This does not collect new deals or read held-out evaluation records.
Use the existing natural public-revelation events, including early-ending
matches and cases with no revelation. Both parents' source records are already
complete and immutable. Do not select particular histories based on whether
they distinguish the models.

Replay each recorded action. The tracker receives only the public snapshot
and observed opponent action after the action occurs. A separate oracle compares
every reconstructed-input hash with the strict upstream feature builder using
the archived simulator state; that state is never passed to the tracker or
predictors. Verify assisted masks independently as well. Preserve every event,
model prediction, elimination, first identification time and unresolved case.

Report, for each parent/opponent cell, correct and wrong singleton identities,
ambiguous and empty sets, true-model exclusions, first-identification rounds,
final-round survivors with correct identity, prediction counts and replay work.
These are descriptive counts on reused training histories, not held-out accuracy
or a policy-strength study. Replayed prefixes, neural calls and original deal
count remain separate. No classifier is added to the current tail evaluation.

The next decision depends on this diagnostic: identify numerical/model failures
if they occur; otherwise assess whether an identifiable type has a useful
specialised response. Existing opponent-modelling and policy-reuse prior art
is cited in REVEALED_OPPONENT_INPUTS.rst. Neither this finite-model elimination
rule nor a successful diagnostic alone establishes new general methodology or
stronger play.

Implementation: scripts/revealed_opponent_models.py.
Runner: scripts/check_revealed_opponent_identification.py.
The runner refuses to start before the tail study's evaluation is complete.
