Numerical batching: prior work and the Koi-Koi observation
========================================================

Sources inspected on 18 October 2026
---------------------------------

PyTorch 2.12 documents that mathematically equivalent batched and individual
computations need not agree bit for bit. It also distinguishes numerical
differences across devices and versions from random-number control [1]_.
The official page records its last update as 29 January 2026.

Horace He and collaborators at Thinking Machines describe batch-size-dependent
reduction strategies in deterministic model inference and implement invariant
reductions for matrix multiplication, normalization and attention [2]_. Their
10 September 2025 article also distinguishes size invariance from position
invariance: some reduction strategies can vary with an element's position within
the batch. Thus fixed size alone is not a general proof of invariant inference.

Relationship to this project
----------------------------

Our measured finding is a concrete failure of paired Koi-Koi rollout coupling.
Changing the alternative branch changes the number of rows sent to a WP discard
head; an otherwise identical baseline observation then chooses a different card.
The saved score inspection and trajectory reproduction are bound in
``runs/midround_batch_divergence``. They identify this particular manifestation,
not a newly discovered floating-point phenomenon.

The fixed64 implementation controls input shape by padding ordinary existing
phase-head calls. It does not implement the external invariant kernels or claim
their guarantees. Its evidence consists of order/padding tests on both devices
and 1,101,312 shared baseline state/action checks in the archived case 9 trajectory
audit. It chooses 18 on the isolated tie, where strict CPU chooses 32; the two
execution policies remain explicitly distinct.

These sources support treating fixed64 as an execution repair. A future paper
section about this work should credit the established numerical issue and
describe the specific paired-rollout failure and its verification. The repair
itself is not a new learning algorithm or evidence of stronger play. The current
terminal-search pilot must establish its empirical outcomes separately.

No running study input or verified manuscript was changed for this source review.
No external kernel package was installed or used.

.. [1] PyTorch 2.12. Numerical accuracy, especially "Batched computations or slice
   computations". https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html

.. [2] Horace He, with collaborators at Thinking Machines. Defeating Nondeterminism
   in LLM Inference. 10 September 2025.
   https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
