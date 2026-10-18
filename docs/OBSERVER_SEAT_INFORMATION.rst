Public seat information in the actor and critic
==============================================

Finding
-------

The retained published actors distinguish absolute dealer seats through rows
50 and 51 of their 300-row input. The original pinned Python implementation
constructs this as ``feature_one_hot(round_state.dealer-1, 2)``. The ordinary
560-dimensional critic instead knows whether its viewer is the dealer. The
561-dimensional observer critic adds a viewer-is-current-actor bit, but still
does not know the viewer's absolute seat.

Consequently, simultaneously renaming seats 0 and 1 leaves the corresponding
critic observations equal while exchanging the actor's dealer rows. This is
a concrete mismatch between information supplied to the policy and value
models. It does not require either model to receive the other player's hand.

Observation probes and measurements
----------------------------------

``scripts/diagnose_observer_seat_information.py`` records 256 consecutive
own-greedy-self-play decisions for each retained parent, using source seeds
26080001 and 26081001. Inference uses singleton CPU2, temperature 0.05 and the
same public endgame mask as the retained actor. No test-deal results are used.

Each probe swaps hand/capture ownership, scores, koi counts, player and dealer
labels, plus the saved initial-hand blocks. Cards, field, stock and public
action history are preserved. The transformed state is never stepped: it is
an observation-equivalence probe, not an independently seeded played match.
Only the original trajectory advances. For all 512 probes:

* Corresponding viewers have exactly equal 561-dimensional value inputs.
* Active ordinary features, phase and action masks are equal.
* Only actor input rows 50 and 51 differ.
* Reading observations does not mutate the live state.

Parent 1 changes its greedy action at 4/256 probes; parent 2 at 5/256. All
changes are discard decisions. Mean total variation between the two action
distributions is 0.017142 and 0.012221; maxima are 0.453385 and 0.354074.
These are paired input-perturbation descriptives on two short trajectories,
not independent samples for a population confidence interval.

``scripts/audit_observer_seat_information.py`` reconstructs all original and
relabeled positions using the pinned upstream Python game. Its 1,024
within-round history reconstructions verify exact native/upstream feature
parity and the same two differing rows. The audit also checks saved raw-row
aggregates, artifact identities and all 57 live-study dependencies.

Successful diagnostic source work is 512 decisions, 33 round endings and
4 completed matches. A first attempt completed parent 1's 256 decisions,
17 rounds and 2 matches before JSON serialization rejected a NumPy counter;
its raw rows and states are byte-identical to the successful parent 1 trace.
Both attempts count: 768 executed source decisions, 50 rounds and 6 completed
matches. Upstream audit reconstructions are separate history replay, not new
source games or future rollout branches. No training or CUDA work is added.

Interpretation and next test
----------------------------

Different policies can have different conditional continuation values, so
merging their public seat contexts is a plausible approximation problem.
The measured action differences alone do not prove unequal true values,
biased Koi-Koi gradients, or a playing-strength cost. Exact equality of values
could still occur. The previous observer-return theorem also requires a
sufficient history; adding one seat bit would not establish that condition.

A separately specified follow-up can append the public viewer-seat bit to
the observer critic. Together with the existing viewer-is-dealer bit, it
recovers absolute dealer identity. Zero-initializing the new input column
preserves starting value predictions and parent actions. A matched comparison
must hold return handling, initial weights, seeds, fitting budget and opponent
evaluation fixed. Additional inputs or seat symmetry are established ideas;
the contribution would be identifying and measuring this particular policy/
critic information mismatch, with any strength benefit reported separately.

The current observer/signed comparison remains unchanged. Its fixed 561-input
arms must finish and be reported as designed. This diagnostic does not change
dealer selection, the playable model, evaluation gates or reserved test seeds.

Evidence
--------

* Diagnostic: ``runs/observer_seat_information_v2/audit.json``, SHA256
  ac30ff58a6b921716ee2b8f5b59c93de5d7e3a818f5c4626e7fc626191bd1684.
* Independent audit: corresponding ``independent_audit.json``, SHA256
  1d8d9687f2f3e9fba3f1f5e0b3ec699803fecd17dc9e1e20311c0bf2467e2255.
* Upstream revision: 89d789565c7c36c84b74662e43001d960bf80901.
