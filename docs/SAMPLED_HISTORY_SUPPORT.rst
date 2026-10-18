Bounded historical-action inference
==================================

``scripts/sampled_history_support.py`` extends the existing exact-history
component to larger hidden-hand spaces. It draws a fixed number of uniform
hand proposals, retains duplicate occurrences, and scores each proposal against
the observed actions of each declared deterministic, stateless opponent model.
Invalid initial hands consume the budget. Rejected proposals are not retried.
Small spaces that fit the same budget are enumerated exactly.

Model priors and the candidate model set must precede the current round's
evidence. Joint hand/model normalization is required. A public identification
from an earlier round can restrict the candidate set; routing labels cannot.
The actual hidden hand, stock order and future RNG are not likelihood inputs.
Zero sampled likelihood is an explicit estimation failure, not proof that the
exact posterior is empty. The component does not choose game actions.

Four tests pass with zero skips (1.461 seconds): comparison with independent
full enumeration, full historical replay of sampled proposals, hidden-world
invariance, multiplicity, invalid-hand accounting and zero-mass rejection.
These fixtures perform correctness simulation and historical replay; they are
not fresh evaluation matches. Ruff check and format check pass.

Development diagnostic fixed before execution
--------------------------------------------

Use only the existing 88M training histories in
``runs/final_round_training_study/parent{1,2}/source.jsonl`` and the independently
audited public identification records. Retain replicates 0--3 for both parents,
both opponents and both seats: 32 attempts, including ineligible attempts.
Select the first recorded learner discard with more than one legal action,
three or four opponent cards remaining, and an identification available in an
earlier round. Selection does not use hidden cards or future outcomes.

Each eligible root gets 128 proposals, using seed
96010001 + 10000 * parent + source_case_index (proposal RNG only, not new deals).
Both published policies are evaluated on the same proposal histories using
strict CPU inference with two Torch threads. Compare three belief estimates:
uniform valid proposals, equal-prior joint hand/model conditioning, and
conditioning on the prior-round public identification. The last two reuse the
same likelihood matrix. Do not retry failures or tune the budget on results.

Record coverage, zero-mass failures, distinct supported hands, effective
proposal count, historical transitions, model calls and elapsed inference time.
After all predictions for a root have been computed, use its true opponent
hand only to score mean squared card-membership error over the public unknown
card set. Report individual records and paired descriptive differences where
both estimates exist, with their denominator. No strength or significance
claim follows from this diagnostic. It adds no future rollouts or training.
All source prefixes are replayed to their recorded endpoints for integrity;
those transitions are counted separately from hypothetical history replay.

The runner freezes its inputs and design before inference and refuses to
overwrite an existing output. The concurrent strength protocols remain intact.
General Bayesian opponent modeling and importance sampling are prior methods;
the experiment tests whether this particular public-history integration is
useful enough to justify a later playing-strength comparison.
