Score-auxiliary training audit
==============================

The independent audit implementation is ready. Full six-arm main training is
still running; a full training_audit.json has not been issued. The closed first
arm prefix through update25 passes. This verifies data and stored state, not a
playing improvement.

Scope
-----

scripts/audit_score_auxiliary_collection.py replays recorded actions with the
CPU native environment, checks legal/public-endgame actions and complete match
termination, and independently reconstructs the per-match/player/phase reservoir.
All selected public feature tensors must match their stored digest. Final
points, observer-relative margins, clipped65-class labels, inclusion probabilities,
visit counts and actual decision/round/match totals must agree. Stored reference
log probabilities are checked for legal support and normalization, not recomputed
from the actor. No hidden state is passed to a policy or used as an input feature.

scripts/audit_score_auxiliary_training.py imports no learner or collector. It
checks original Point identities, source bindings, runtime settings, parameter
shapes/digests, PPO accounting, both optimizers' per-phase call counts and finite
moments, and unchanged auxiliary heads/empty auxiliary Adam state for none and
warmup. Local shuffle reconstruction gives auxiliary minibatch/phase counts.
Every main checkpoint, collection and metric row is checked. Initial/warmup PPO
states and first collections must agree across the three conditions within each
seed. The none arm's collection actor digest must match its checkpoint when the
checkpoint is at an auxiliary boundary.

PPO trajectories, gradients and neural action choices are not rerun. Mean training
losses are checked for finite valid values, not independently recomputed. The
existing PPO accounting auditor is reused explicitly. These limits are retained
in the final audit artifact. The audit is not evidence of algorithmic novelty.

The full command refuses incomplete or failed studies and existing audit output::

    .venv/Scripts/python.exe -m scripts.audit_score_auxiliary_training --root runs/score_auxiliary_study

It binds all107 frozen protocol inputs, all saved model/data/log artifacts, the
auditors and tests, then checks the bindings again before writing an exclusive
training_audit.json. Main evaluation preparation already requires this artifact.
The required test receipt must report10passes with zero failures/errors/skips.

Completed capability
--------------------

Command::

    .venv/Scripts/python.exe -m pytest tests/test_audit_score_auxiliary_training.py -q --junitxml=runs/score_auxiliary_training_auditor_tests.xml

Ten tests pass,zero skips,4.87s. They use existing capability artifacts without
new learning or policy inference. Twelve saved checkpoints across none/win/score
pass. Tests reject changed selected steps/features/margins/work, wrong PPO or
auxiliary Adam counts, negative/shapeless moments, wrong source identity, and an
incomplete study. Four selected observations,one per phase, also match the
separate Python reference and upstream feature reconstruction exactly.

One valid replay and three late-corruption replays execute60,680recorded
decisions/3,892round endings/512match terminations. The invalid-step case fails
before replay. Reference prefixes add155recorded decisions. An earlier one-off
valid replay adds15,170decisions/973round endings/128terminations. These are
replays of the same128closed deals,not new policy matches or training work.

Main closed-prefix check:initial/update20/update25,409,600recorded PPO decisions;
one collection of15,095decisions/980round endings/128matches,968retained rows,
zero auxiliary optimization in none. The saved PPO accounting is inspected,
not replayed. Only the immutable25-row metric prefix is bound while its full
log grows. Receipt runs/score_auxiliary_training_prefix_audit.json:
31ede13ab447d643419a3b24779eaa1c43904bf4487390476821bfef9b9a8213.
All107 main protocol bindings remain unchanged. No strength evaluation has run.

Milestone digests
-----------------

* Training auditor:82e38fc2c81d56ef357386a51268c86ab2fb9ab75aa07ebca55a8d72edafec81.
* Collection auditor:de15d23fa2fba5771783b81b17539b447f1d513698ac639594b68b89fa474653.
* Tests:ba16c53cfedf949b0eb93daf09b77b71fe5ae41fe5a57669ecd0a3658c556cf7.
* Test XML:c8cd4f0398068b4f47502ff058588a96069434cf9137fc09a7deb9d545e07862.

Next connect the completion queue and implement the fixed mechanism diagnostic.
The canonical player,dealer,prior experiment outputs and verified manuscript
remain unchanged. Stronger play and substantive novelty remain open.
