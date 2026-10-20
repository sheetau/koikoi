Separating outcome and within-outcome score supervision
======================================================

Status: complete-outcome collection, auxiliary training and exact resume are
implemented and checked at production batch sizes. The fixed main learning
study in SCORE_AUXILIARY_MAIN.rst is now running; strength evaluation and
promotion have not occurred. The current manuscript reports completed studies and does not include
this proposal as a result. Initial dealer selection is unchanged.

Motivation and distinction
--------------------------

The completed Point-initialization comparison improves point margins across both
seeds and both sampling settings, but does not establish additional match wins.
The published PPO actor receives policy/entropy gradients while its separate
560-input critic learns returns. Consequently, value supervision does not train
the inherited Transformer encoders. This motivates testing whether score-outcome
supervision can improve those representations while retaining match-win rewards.
The prior results do not demonstrate that representation learning is the cause
of the win plateau.

This is distinct from the completed score-distribution selector: that experiment
froze the experts and trained a separate scorer on paired proposed-action
continuations. Here the intended intervention trains the actor's own encoders
from complete self-play outcomes; deployment keeps its original action heads.
It is also distinct from changing the actor's reward to expected points.

Established foundations
-----------------------

Phasic Policy Gradient explicitly separates policy optimization from auxiliary
representation learning, with a policy KL penalty during the auxiliary phase.
It allows general auxiliary objectives; adding such a phase is not a new
algorithmic principle:
https://proceedings.mlr.press/v139/cobbe21a.html (Algorithm 1 and Section 2).

UNREAL studies auxiliary tasks for representations in reinforcement learning:
https://arxiv.org/abs/1611.05397.
Distributional RL models return distributions rather than only expectations:
https://proceedings.mlr.press/v70/bellemare17a.html.
The present head predicts Monte Carlo terminal margins; it is not a C51 Bellman
update. The likelihood decomposition below is the ordinary probability chain
rule, not a new theorem. These primary sources were inspected on20 October2026.

The contribution sought is an experimentally isolated role for information
within a match-outcome class, and a reproducible playing improvement. Neither
has yet been established. General priority or world superiority is not claimed.

Matched supervision
-------------------

Let d be terminal own-minus-opponent points. Under paper8 it is an even integer.
The label is (clip(d,-64,64)+64)/2, giving65 categories. Compression preserves
loss/draw/win exactly; point values beyond the support are not predicted exactly.
For65 logits z, let G(y) contain all categories with the same sign as label y.

* Outcome loss: logsumexp(z) - logsumexp(z[G(y)]).
* Within-outcome loss: logsumexp(z[G(y)]) - z[y].
* Full score loss: the sum, exactly ordinary categorical cross entropy.

Both auxiliary arms use identical head architecture and initialization. The
win arm uses only outcome loss; the score arm adds the within-outcome term.
The draw term is zero because zero margin has a single category. Initial logits
give equal loss/draw/win mass, uniform within each sign group. The same output
dimension avoids confounding the comparison with different parameter counts.
The extra loss necessarily changes gradient magnitudes and directions; the
comparison does not isolate information independently of that optimization effect.

The within-outcome loss has zero summed logit gradient within its sign group
and no gradients in other groups. This does NOT preserve group probability
after a parameter update, or guarantee nondecreasing match wins. A numerical
counterexample is included in the tests. A KL penalty on observed inputs is
also not a global performance guarantee.

Implemented boundary
--------------------

scripts/score_auxiliary.py provides label compression, both likelihood terms,
masked forward KL at divisor0.05, and ScoreAuxiliary. The wrapper reuses each
inherited encoder and evaluates its original action head unchanged. It pools
legal action-token embeddings into one auxiliary prediction; hand/draw capture
share the same inherited encoder/head. Three auxiliary linear heads add50,115
parameters (3 x (256 x65 +65)); deployment needs none of them.

Forward accepts only native public/own-card features, public phases and legal
masks. Terminal scores enter only the training loss as labels. Attaching zero
initialized heads preserves the CPU sampling RNG. Dropout remains disabled as
in PublishedActor. No production actor, trainer, native rule or inference loader
is modified. The component loss uses auxiliary coefficient1 and clone
coefficient1; this is not yet a frozen full-study configuration.

Required next implementation
----------------------------

The collection/training/resume requirements below are now checked. The fixed
three-arm,two-seed main study is documented in SCORE_AUXILIARY_MAIN.rst and
running at runs/score_auxiliary_study. Its auxiliary interval5 replaces the
provisional component default25 for this experiment, based on bounded sample
volume rather than strength outcomes. Main initialization, budgets, mechanism
diagnostics and paired evaluation are fixed before results. Next implement
separate evaluation/auditing while preserving all learning endpoints.

Use equal PPO decisions and equal requested complete auxiliary matches across
arms. The resulting auxiliary decisions, retained rows, minibatches and runtime
can differ as policies change; do not call these identical total compute.
Keep every final endpoint. Do not consume the reserved101M confirmation deals
or promote from capability outcomes. The broader research objective is active.

Completed capability
--------------------

Command::

    .venv/Scripts/python.exe -m pytest tests/test_score_auxiliary.py -q -o junit_family=xunit1 --junitxml=runs/score_auxiliary_capability_final_tests.xml

Five tests pass, zero skips,4.91s. CPU and CUDA each replay two complete matches
against the independent Python transition reference:251decisions,16round
endings,2matches per device, on the same two105010001/105010002deal streams.
These are4 executions of2distinct capability deals, not strength evidence.
Each device performs4 auxiliary optimizer steps/16example presentations.
All four phases are exercised. Action scores match the inherited actor exactly
with and without autograd; finite auxiliary gradients reach its encoders.
Scalar fixtures check likelihoods, masking, forced actions, outcome-preserving
tail compression and seat symmetry.

An earlier five-test execution used an incorrect reporting expression for
round counts (terminal positive rewards); that accounting expression was fixed
before the final run. Its separately executed work is another502decisions,
4matches on the same deals and8auxiliary optimizer steps; its round count was
not correctly recorded and is not added to verified totals. The obsolete XML
remains preserved. No trained endpoint was saved or evaluated.

Complete-outcome collection and training integration
---------------------------------------------------

scripts/collect_score_auxiliary.py now finishes one frozen-current-policy match
per lane atT0.05 with public endgame masks. Local action and reservoir generators
leave PPO RNG streams untouched. Uniform reservoir sampling retains one public
observation per visited match/player/phase stratum. Every eligible observation
within that stratum has inclusion probability1/n; strata, not all individual
decisions, receive equal sample weight. Forced decisions remain eligible.
Completed public scores label each observer in its own perspective. A hard
step-bound failure returns no partial training labels.

The in-memory features are used for learning; compact checkpoint evidence saves
their hash, selected decision indices, all executed actions, public masks,
reference probabilities and terminal points. Independent replay can reconstruct
the large input tensor without storing another copy at every cycle.

scripts/score_auxiliary_trainer.py composes the existing PublishedTrainer with
an auxiliary phase. Ordinary PPO rewards, critic and optimizer are unchanged.
A separate persistent Adam uses the PPO actor learning rate for the encoders
and action heads, and the value learning rate for auxiliary heads; epsilon1e-5,
global auxiliary gradient clipping0.5, clone coefficient1, auxiliary coefficient1.
Outcome and score arms use the same6epochs and128-row minibatches by default.
The none arm collects the same requested number of full matches but performs
no auxiliary optimization. It has no auxiliary Adam steps.

Checkpoints are immutable directories containing standard ppo.pt, auxiliary.pt
and a final manifest. The manifest binds source, state-file digests, configuration
and counters. The auxiliary file retains heads, optimizer and last collection.
A partial failed cycle cannot be saved or continued. Restore checks digests and
runtime and reproduces the next full cycle exactly. The ordinary ppo.pt actor
can use the existing inference path; auxiliary heads are not inference inputs.

Three collector tests pass with0skips(4.20s). CPU/CUDA native action traces,
sampled inputs reconstructed through upstream public features, masks, labels,
stratum coverage and counts agree with the Python reference. Repeating with
different global RNG states gives the same within-device result. CPU and CUDA
multinomial samplers can generate different trajectories; cross-device action
identity is not claimed. Final XML: runs/score_auxiliary_collection_fixed_tests.xml.
Work including repeat calls:876decisions/54round endings/8complete matches,
including2deliberately incomplete decisions, on2distinct capability deal streams.
An earlier import error executed no collection or training.

Six integration tests pass with0skips(14.84s), including ordinary-PPO identity,
CPU/CUDA exact next-cycle model/optimizer/RNG/environment/data resume for both
auxiliary conditions, phase-specific Adam counters, changed-digest rejection
and failed-cycle handling. XML: runs/score_auxiliary_trainer_tests.xml. Actual
test PPO work:736decisions/48round endings/6matches; auxiliary work including
repeated resumes:2,916decisions/187round endings/28matches, including2unfinished
failure-fixture decisions. These small tests are not a strength experiment.

Production-shape capability
--------------------------

Command::

    .venv/Scripts/python.exe -m scripts.check_score_auxiliary_training --output runs/score_auxiliary_training_capability

All three conditions complete one critic-only and two policy updates at
128environments x128horizon,512PPO minibatch,3PPO epochs. The capability uses
interval1 to exercise both auxiliary cycles, each128complete matches and
6auxiliary epochs. This intentionally differs from a main-study schedule.
Initial/warmup PPO states and first auxiliary collections match across arms.
Each second cycle is repeated by restoring the first; all learning states and
collection evidence match exactly. The none arm also matches an ordinary-PPO
update without the wrapper at production shape.

Unique per-arm learning work summed across three arms:
147,456PPO decisions/9,301round endings/1,107matches and91,178auxiliary
decisions/5,870round endings/768matches. Resume verification additionally
executes49,152PPO decisions/3,124rounds/405matches and45,533auxiliary
decisions/2,924rounds/384matches. The ordinary-PPO control executes another
16,384decisions/1,042rounds/135matches. Auxiliary collection uses256distinct
deal streams shared across arms; repeats do not add independent deals.
Unique/repeated auxiliary optimizer steps192/96; example presentations
23,028/11,496. PPO optimizer accounting remains in the separate saved states.

Runner elapsed62.8509s includes saving and verification. Peak allocated CUDA
memory is2,279,875,072/2,313,819,136/2,313,032,704bytes for none/win/score.
This is capability timing, not an inference-latency benchmark or strength claim.
All49saved file digests verify. Report runs/score_auxiliary_training_capability/report.json,
SHA256 49e24dc6f33247b69b8922d71d2716fcdaf676220b75c2630785ea829b833f9e.
The runner session12969 exits0. No full study is live and no candidate is promoted.
