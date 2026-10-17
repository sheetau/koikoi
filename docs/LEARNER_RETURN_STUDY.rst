Learner-decision returns: fixed development comparison
====================================================

Question and conditions
-----------------------

Does bootstrapping only at learner observations improve fixed-opponent PPO over
the original alternating-observer estimator? The exact counterexample and the
implemented return/loss semantics are in OBSERVER_RETURNS.rst. This comparison
tests playing strength; the counterexample alone cannot answer that question.

Use the two retained warm parents and their already completed response endpoints
from the original response study. Add two new continuations per retained parent:

* learner: LearnerTrainer, lambda0.95, no opponent critic, valid learner losses;
* lambda1: ResponseTrainer, lambda1, original separate opponent-role critic.

The original signed lambda0.95 response endpoints are the primary trained
controls. The unchanged parents measure improvement over the starting player.
Every new condition uses the original response training seed for that parent
(19910001,20010001), fresh Adam/environment, the same parent actor and learner
value weights, T0.05, actorLR3e-6,valueLR1e-4,width256,128 environments,128-step
horizon,512 primitive rows per minibatch,3 epochs,20 critic-only and500 policy
updates. Each performs8,519,680 simulator actions; four new conditions perform
34,078,720. The reused original response training is a separate17,039,360-action
dependency, not fresh work. No endpoint is selected from intermediate results.

All conditions use the same fixed whole-match WP/Point mixture and seat schedule
as RESPONSE_STUDY.rst. Match reward, rules, assisted comparison actors and actor
observations stay fixed. Same training seeds do not promise equal whole training
trajectories or equal CPU RNG consumption; verify actual initial model, value and
environment equality and report RNG differences. Report all simulated actions,
learner actions, valid loss rows, omitted tails, per-opponent work and Adam steps.

The learner intervention also removes the opponent value loss, changes joint
gradient clipping, normalizes value loss over valid rows, filters minibatches and
skips empty optimizer steps. This study compares the full implementation; it does
not isolate critic information from these optimization changes. The lambda1 arm
removes interior GAE bootstraps, but still uses the original finite-rollout tail
bootstrap, which can be an opponent observation. It is not a fully observed or
unbiased-return control. Report it regardless of whether it helps.

Evaluation and decision
-----------------------

Evaluate all eight players (two parents,two archived response endpoints,four new
endpoints) against assisted WP and Point. Use strict CPU2 greedy inference and
5,000 duplicate pairs per matchup on new development seeds24,000,000--24,004,999.
This is16 matchups,160,000 matches and5,000 shared unique deal streams. Do not
reuse the23M response-study evaluation outcomes for this comparison. The100M
consumed final range and101M reserved range remain separate. This development
comparison does not consume the independent final confirmation.

Compute complete-pair win score and point margin from raw seat-swapped records.
Resample the common deal index jointly across compared players, with5,000
bootstrap replicates and seed99. The primary qualification requires positive
paired win-change lower95 bounds for learner minus unchanged parent AND learner
minus signed-lambda0.95 response, for both parent models against both opponents
(eight required bounds). Report every endpoint, all per-opponent win/point
intervals and all learner/lambda1/response versus parent and pairwise trained-arm
contrasts. Lambda1 and point margins are secondary; no post-result substitution
of the primary method, criterion, seeds or budget. Pair uncertainty does not
measure variation across all possible training seeds.

Keep each endpoint and every completed cell, without intermediate ranking,
sample extension, checkpoint selection or deployment promotion. A failure of the
primary gate remains a result. Any additional run requires a separately stated
question and protocol. The method is a decision-time reduction; its name and a
positive comparison would not by themselves prove a new general PG theorem.

Execution gates
---------------

Bind production, runner/helpers, method files, parent and archived response
checkpoints, upstream sources and all comparator weights before research work.
Archived response endpoints must pass the original training audit and independent
optimizer-budget audit. Their original response study remains unchanged.

Before a research GPU run, validate a CUDA optimizer update and exact restart
for LearnerTrainer plus a reduced real scheduler/training/evaluation pipeline.
Do not start a new GPU trainer while any original response-study training job
is active. Do not start new research CPU evaluations while that study still owns
its evaluation slots. Globally allow one research GPU trainer and three CPU2
evaluation workers; endpoint evaluation requires successful training and audit.
On failure stop only owned worker trees and retain the failed state and logs.

The runner is ``scripts/run_learner_return_study.py``. Its reduced ``--smoke``
path uses CPU training, one parent,two updates,two pairs and seed20298000; it
never reports strength qualification. A real smoke completes ten subprocess
jobs,two training conditions and eight evaluation cells (32 matches). Tests also
check source/protocol/receipt tampering, upstream resource gates and launch-failure
records. Smoke validation does not consume the research GPU slot.

``scripts/check_learner_gpu.py --wait --launch-study`` waits for successful exit
receipts for all four original training jobs. It freezes its verification inputs,
then runs the actual CUDA exact-restart test and a three-update,production-size
CUDA capability. A fresh capability directory is required. Any failure stops the
sequence; there is no automatic retry or waiver. Only after success does it start
the fixed research runner, which also checks the CUDA proof and its identities.
The new research GPU training can overlap remaining original CPU evaluation;
new research CPU evaluation waits for the original study's final audited exit.
Final schedule auditing combines both studies' job intervals to verify the
global one-GPU/three-evaluator limits. The startup command is::

    .venv/Scripts/python.exe -m scripts.check_learner_gpu --wait --launch-study

At protocol preparation no new research training or match evaluation has run.
Progress and final measured outcomes belong in STATUS.md and EXPERIMENTS.md;
this protocol remains bound and unchanged during the study.
