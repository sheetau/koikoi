Public opponent memory for final-round adaptation
================================================

Implementation
--------------

The delayed-revelation tracker supplies a persistent candidate set from public
actions in rounds 1--7. At round-eight entry it is frozen. Singleton WP/Point
sets encode as (1,0)/(0,1); ambiguous or empty support encodes as (0,0).
Opponent routing labels are never substituted for missing public predictions.
The frozen prefix parent continues to act in rounds 1--7.

The isolated actor uses previously unused published feature rows 2 and 3 as
constant public bits over cards. Their first-convolution weights initialize
to zero in both the public-memory arm and the masked-memory control. Native
feature construction, original pretrained actors and deployed agents retain
their original semantics. The two arms have identical architecture, initial
weights, learning rates, sampled root RNG and update budget. Only the bit inputs
differ. The ordinary 560-input observation critic is unchanged and receives no
candidate bits. Input rows for fixed opponents remain zero in both arms.

Natural final-round roots and public candidate codes are joined by audited
case/root addresses. The fixed-prefix tail objective and complete terminal
returns are inherited from FINAL_ROUND_TRAINING.rst. Same-parent public/masked
arms use the same root bank and sampling seed. After policy updates, different
actions can produce different trajectory lengths and optimizer minibatch counts;
these are recorded rather than described as equal decision budgets.

Evaluation requires explicit public events after opponent actions. It captures
pre-action public snapshots, submits them after the action executes, and keeps
the final action of a round before resetting the history. Match starts reset
all memory. Inference refuses double-conditioned features and misbound events.
Predictions use the pinned strict CPU opponents; training continuation still
uses fixed64 GPU opponents, a disclosed numerical-policy difference.

Capability evidence
-------------------

Four trainer/information tests pass with no skips (14.50s), including CPU and
CUDA exact restart, original initial scores, matched critic-only warmup,
independent reference transitions and opponent-terminal reward signs. One
complete-match lifecycle test passes (5.44s), checking all actions against the
initial parent in both seats against both published opponents.

The first test attempt had one CPU failure because constructing the independent
comparison model consumed the test's training RNG. The fixture now preserves
that RNG around comparison construction. The implementation was unchanged;
the original failed XML is retained. The repeated suite passes without relaxed
numeric tolerances.

The production-shape CUDA check runs two three-update arms, each with 128 lanes,
three epochs and minibatch512: 10,952 decisions, 768 conditional tails and zero
full training matches in total. It reuses four toy-source natural roots. Both
warmups agree exactly; every optimizer state is finite, frozen prefix/opponents
remain unchanged, and only the public arm learns nonzero memory-column weights.
Audit ``runs/public_memory_tail_capability/audit.json`` SHA256
``35072349596c1bb731388c0bc3d3afae13e0b8594a890c3ac3040a5168a45acb``.
These are capability checks, not evidence of stronger play.

Research relationship
---------------------

Policy conditioning on opponents is established methodology; DRON (He et al.,
ICML 2016, https://proceedings.mlr.press/v48/he16.html) explicitly integrates
opponent observations into a learned policy. Bayes' Bluff and Bayesian Policy
Reuse are further precedents for inferring/reusing responses to opponent/task
types. This implementation claims no new general opponent-modelling algorithm.

The proposed domain contribution is a verified delayed public reconstruction,
its persistence across an otherwise erased round boundary, and a controlled
test of whether that information improves match decisions. The first two have
capability evidence; the third requires the prospective strength comparison.
