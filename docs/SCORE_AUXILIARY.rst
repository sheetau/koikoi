Separating outcome and within-outcome score supervision
======================================================

Status: experimental component implemented and checked; no strength trial or
promotion. The current manuscript reports completed studies and does not include
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

Connect complete-outcome collection, a separate auxiliary phase and exact
checkpoint/resume to the existing published PPO trainer. Keep policy rewards
and the original critic unchanged. Use complete matches under a frozen current
policy to label public observations, with a bounded sample per player/phase;
do not silently label unfinished matches or select only short completed ones.
Keep auxiliary sampling on a separate reproducible RNG stream. Count its
additional decisions, rounds and matches separately from PPO collection.

Before a strength run, verify label signs across player changes and match
termination, auxiliary head/optimizer persistence, identical no-auxiliary
control behavior, and actual production-shape execution. Then freeze a matched
three-arm, two-seed comparison: PPO alone, outcome auxiliary, full-score
auxiliary. Include a learning-curve mechanism diagnostic, equal auxiliary
collection work, all endpoint results and paired fresh-deal evaluation. Freeze
budgets and success criteria before inspecting any strength outcomes. Do not
consume the reserved101M confirmation deals or promote a candidate from a
capability check. The broader research objective remains active.

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
