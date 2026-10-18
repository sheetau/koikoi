History-conditioned terminal search: adapter and fixed capability protocol
========================================================================

Implementation
--------------

scripts/history_conditioned_search.py connects the completed bounded joint
hand/opponent sampler to paired complete-match rollouts. The frozen parent is
the strict CPU2 singleton fallback. Public endgame solutions and forced actions
return immediately. Other roots with more than100 public hand combinations
retain the parent action before historical model inference. Empty compatible
support also retains the parent; it never becomes a uniform posterior.

The history mode assumes an equal WP/Point prior before the current round and
conditions on that round's observed opponent decisions. It receives no actual
opponent identity. The uniform control uses uniform valid hands and an equal
independent WP/Point model label, with the same public eligibility cap.
Neither mode updates a cross-round model belief or conditions simulated future
root decisions again. These are explicit opponent-model approximations.

A particle's model label persists across candidate branches, player changes
and round boundaries. At our fixed seat, a frozen batched copy of the retained
actor continues; at the other seat, the particle's frozen WP or Point actor
continues. Every actor receives only its own legal observation and the public
endgame mask. Both modes use identical continuation construction. Historical
support is certified by strict CPU models; floating-point batched continuations
are separately identified approximations, not exact CPU-policy certificates.

All alternatives share their initial worlds, opponent labels and future deal
streams. Baseline terminal points must agree exactly across repeated comparisons.
Every repeated baseline branch is counted. Select the largest mean improvement
in draw-adjusted win score, breaking ties in legal-action order. A positive
2.5-standard-error margin proposes a change. Fresh independent particles then
compare only that selected action against the baseline; no confirmation reranking.
The margin is a heuristic uncertainty guard, not a distribution-free safety bound.

Verification
------------

The combined sampler/adapter/all-action suite passes16 tests with zero skips.
CPU and CUDA complete actual paired branches from both root seats, including
multiple round boundaries, all three model routes and terminal accounting.
Every transition/state is checked against the Python reference; routed actor
features are independently rebuilt by the pinned upstream reconstruction.
The observed batched actions agree on these probes, not universally.

The suite also checks actual-hidden-world invariance of complete conditioned
decisions, independent proposal/confirmation worlds, selected-alternative
routing, public budget and empty-support fallbacks, and a uniform control that
never scores historical action likelihoods. The confirmation path is forced
only inside a correctness test; its action changes are not strength evidence.
Saved old diagnostic roots are correctness fixtures, not new evaluation data.

Fixed capability and cost check
-------------------------------

Before inspecting new root outcomes, fix this diagnostic:

* Source: eight complete paper8 matches, two retained parents x WP/Point x both
  parent seats. Strict CPU2 singleton policies and public endgame assistance.
  Parent i (1 or2), opponent j (WP=0,Point=1), seed31210001+100*i+j; the two
  seats share a seed. Thus four distinct source deal streams, not eight.
* In each match, retain only the first unforced, unsolved parent decision with
  <=100 public hand combinations. Selection occurs before its action or outcome.
  Complete every source match. A missing eligible root stays missing; no
  replacement deal or source-budget extension. Independently compare every
  source transition/state with native CPU.
* At every retained root run both uniform and history modes,64 proposal
  particles and256 conditional fresh confirmation particles, confidence2.5,
  CUDA float32, TF32 disabled, two CPU threads. Search seed31220001+case_index
  resets independently for each mode. No search action enters source collection.
* Keep every legal alternative, raw paired terminal points, model indices,
  sampled-world hashes, history counts, work and elapsed time. Check the actual
  opponent hand against its support only AFTER action selection as a validation
  assertion; it must never influence sampling, posterior weights or actions.
* Report source decisions/rounds/matches, historical replay decisions and
  future branch decisions/rounds/completions separately. Historical replay does
  not finish a round. Count repeated baselines and confirmations fully.
* Independently recompute action statistics and selections from the raw terminal
  points. Preserve input hashes and every output. No training, final-test use,
  checkpoint selection or match-win inference. Changes at eight source roots
  cannot establish a stronger deployed policy.

Command after committing this protocol and verified implementation:

  .venv/Scripts/python.exe -m scripts.check_history_conditioned_search --output runs/history_conditioned_search_capability

Do not extend this diagnostic or tune its threshold from partial or final
outcomes. Its purpose is feasibility, cost and coverage before a separately
specified fresh match comparison. A subsequent strength study must include
unchanged parents and the uniform opponent-mixture control; no automatic
promotion or use of the reserved101M confirmation range.
