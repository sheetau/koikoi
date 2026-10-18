Exact comparison for the bounded inference diagnostic
====================================================

Purpose: separate finite-proposal estimation error from the finite-model
posterior's error on the actual hand. This is a follow-up development diagnostic
on already inspected training histories, not independent confirmation.

Before computing any exact posterior, select the first eligible source case
in each parent/opponent group of the completed 32-attempt diagnostic. This gives
parent1/WP case0, parent1/Point case512, parent2/WP case0 and parent2/Point
case512. All four have 5,985 candidate hands. The selection uses source order
and public eligibility, not their errors or support sizes.

Enumerate every candidate once with the existing deterministic likelihood
checker, capped at 6,000 candidates per root. Invalid hands retain their count;
there are no extra draws or retries. Use strict CPU inference with two Torch
threads. Compare exact and sampled uniform, joint-model and publicly identified
model card marginals, effective support and true-hand membership errors.
Compute all estimates before accessing the hidden hand for scoring. Check every
stored sampled candidate's likelihood against its exact-table entry. Report all
four cases separately; these share two old deal seeds and cannot establish
replicated playing strength or generalization.

Inputs and this design are frozen before enumeration. Preserve the prior
diagnostic and concurrent strength studies. Count source-prefix replay,
hypothetical-history transitions and neural calls separately. No future
rollout, training, new evaluation deal or policy deployment is involved.
