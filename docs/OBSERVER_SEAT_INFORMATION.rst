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

CPU implementation and validation
---------------------------------

``scripts/seat_observer_selfplay.py`` now implements the candidate in an
isolated trainer. Both public and masked conditions use a 562-input critic;
the masked condition always supplies zero in the additional column. Existing
observer/signed return code, collection, value targets and PPO updates are
reused unchanged. Each row remains one observer's own information. The active
policy/value forward reconstructs the public active-seat bit from the existing
packed dealer and relative-dealer features, so it needs no simulator access.
Raw 560-input critic calls are rejected instead of silently guessing a seat.

Both new columns (role and seat) are zero-padded when importing a retained
560-input parent. Actor tensors are copied unchanged. The checkpoint format
``seat_observer_selfplay_v1`` records the context and source identity, and
supports exact CPU training restart. ``load_policy`` loads the corresponding
CPU actor/value model. The historical formats are not silently reinterpreted.

Five CPU tests pass, zero skipped. They cover all four decision phases,
independent reference features for both seats, hidden-card/stock/RNG
noninterference, relabeling, row independence, parent action equality,
initial-value agreement within1e-7 absolute/1e-6 relative tolerance, active-seat
reconstruction, exact resumed model/Adam/environment/counter/RNG state,
matched observer/signed critic-only warmup within each context, and rejection
of invalid formats/source identities. Public and masked initial tensors and
RNG agree; their warmup updates may diverge because the information differs.

The first test run exposed an unsuitable zero-value fixture: before the first
match end, a zero critic has zero targets, so the new column cannot yet update.
The final test gives both return modes an identical nonzero output-layer
fixture and verifies real learning of the public column. No production code
change was needed for that correction.

``scripts/check_seat_observer_selfplay.py`` additionally imports both actual
retained parents into both contexts and executes two critic-only updates plus
one policy update per case,CPU2,8 environments,64 steps,256-row minibatches,
one epoch,width256. All four runs complete, with6,144 decisions,406 round
endings,32 completed matches and12,288 observer value rows. Each checkpoint
has96 finite optimizer parameter states,6 value Adam steps and2 actor steps
per phase. Public-seat-column L1 norms reach0.082305/0.097818; masked columns
stay zero. Initial actor logits agree exactly with each parent and saved-model
inference agrees exactly with the trained model. Report:
``runs/seat_observer_capability_cpu/audit.json``, SHA256
 d6f00fccd056731590d5bb7e9a9a30eb0f55fcf48b1460329b2e482527ba0d59.

These short runs establish CPU integration and a functioning learning path.
They do not establish strength,CUDA capability or a sufficient value history.
The original four-arm observer study continues unchanged. A full comparison
protocol and GPU validation remain pending; no checkpoint is promoted.

CPU match-evaluation adapter
---------------------------

``scripts/seat_observer_agent.py`` adds ``SeatObserverAgent(checkpoint)``.
It loads the explicitly tagged checkpoint and reuses ``NeuralAgent.act`` for
singleton greedy public observations and the existing endgame guard. It can
be passed directly to ``koikoi.evaluate.duel``. The general evaluator factory
and the running study's adapters remain unchanged; that factory does not yet
recognize this isolated checkpoint format automatically.

Eight CPU inference tests pass,zero skipped. All four actual-parent initial
checkpoints match their retained parent's actions on256 source decisions each,
covering every phase and both public/masked contexts. Perturbing the current
opponent hand,stock and RNG leaves decisions unchanged,and acting leaves the
reference state unchanged. All four small trained capability endpoints retain
the saved actor tensors and agree with direct published-actor logits or the
public endgame solution at each visited candidate decision. Each completes
a duplicate pair against the greedy fixture using the standard duel path.
These are adapter correctness checks,not comparative strength estimates.

Initial traces execute1,024 decisions,68 round endings and8 completed matches.
The trained endpoint checks execute another870 decisions across8 complete
matches on2 shared deal streams;the standard duel helper does not report round
endings for those matches. No new training or reserved test use is involved.

Guarded device capability
-------------------------

``scripts/check_seat_observer_device.py`` provides a production-size CUDA
capability and a reduced CPU validation mode. It checks256 saved source
positions per context against independent Python observer features,including
active-seat reconstruction. It requires identical public/masked initial states
and actor/environment/counter/CPU+CUDA RNG warmup contents;the critics may
differ. One final policy update is replayed from a saved checkpoint,with exact
metrics,model,Adam,environment,counter and RNG comparison. Replay simulation is
included separately in physical work.

Three tests pass,zero skipped,in13.25s. They reject missing,live,failed or
duplicate current GPU workers and verify that a pending guard creates neither
a model nor an output directory. The actual reduced path runs both contexts
with4 environments,64 steps,128-row minibatches,one epoch,one warmup and two
policy updates. Including replay,it executes2,048 decisions,132 round endings,
8 matches and4,096 value rows. All96 optimizer states per endpoint are finite
and agree with recorded phase counters. The public critic changes its seat
column;the masked column stays zero. Initial identities and exact restart pass.

Final reduced evidence is preserved in ``runs/seat_observer_device_smoke``;
audit SHA4df957a460039d3171390224baff6a716d6ed98b0952654252b4d53b61adcc58.
A preceding successful test invocation before the protocol-identity guard was
added executed the same amount of work and is counted separately.

The CUDA invocation is queued with::

    .venv/Scripts/python.exe -m scripts.check_seat_observer_device --output runs/seat_observer_capability_cuda --wait

Session20636 is waiting for all four existing observer-study GPU training jobs
to terminate successfully. It checks their fixed protocol,training audits and
checkpoint identities before allocating the new CUDA trainer. It will run
public/masked conditions from parent1,128 environments,128 steps,512-row
minibatches,three epochs,20 warmup plus2 policy updates,seed26095001. Logical
endpoint work is720,896 decisions;two extra restart updates add32,768,for
753,664 planned physical decisions and1,507,328 value rows. Match/round counts
will be measured. Bound protocol SHA
e3679b894abae603fd268b1a1865d34a01b88f3879b0e8df0d73d765091ad894.

GPU validation is pending;this queue is not a strength run. The64 bound inputs,
including its checker,test,trainer and proposed study design,must not change
while the queued/running capability owns them. Do not restart its directory.
