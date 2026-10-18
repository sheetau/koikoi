Midround history-conditioned search
==================================

Question and implementation
---------------------------

The completed opening diagnostic reduces estimation cost but changes no action.
The next intervention uses opponent actions already observed in round one.
At each seat's fifth discard (zero-based turns 8 and 9), four own cards remain.
The pre-validity opponent-hand spaces contain 4,845 and 816 combinations.
All other decisions retain the strict parent/public guard. Each seat thus has
at most one intervention opportunity in a complete match. Earlier endings are
retained as missing opportunities, not replaced by later rounds.

Pruned history replay shares observed transitions across named deterministic,
stateless models. On a model's first disagreement its likelihood is zero, so
later factors cannot change its posterior mass. When all models disagree, the
remaining replay is omitted. Surviving pairs reconstruct the complete root.
This is exact short-circuit evaluation of the existing finite-model likelihood,
not a new Bayesian method. Inputs must be valid recorded histories; an omitted
suffix is not independently validated by the pruned evaluator. CPU singleton
published policies remain the likelihood models; no batched approximation is
substituted. Shared historical transitions and individual model calls are
counted separately. Future stock order is freshly sampled, never canonical.

Fixed diagnostic, before collection
----------------------------------

Both retained parents, WP/Point source opponents, both candidate seats and two
replicates produce 16 complete original-policy source matches on four shared
deal streams. Seed = 83110001 + 100 * opponent_index + replicate, with WP index0
and Point index1. Actual opponent identity is used only to generate sources;
the search receives an equal WP/Point prior and public history. Restricting to
round one avoids a claim of carrying opponent evidence across round resets.

Compare uniform and history-conditioned round-value search on every eligible
source root, all three alternative cards. Both use 1,024 proposal particles,
2,048 fresh confirmation particles only for the proposal's highest-mean card,
and the existing 2.5 joint-standard-error margin. Table A proposes and table B
confirms. Each mode resets to 83210001 + case_index. Their distributions differ,
so these seeds do not assert identical worlds. Do not select tables, tune the
margin, extend this root sample or promote a model from this diagnostic.

For all three alternatives, collect 4,096 independent history-conditioned
terminal reference particles in eight continuing 512-particle chunks, seeded
83310001 + case_index. Reuse the exact computed support but neither proposal
worlds nor value tables. The model label stays fixed within each future match.
Full batched continuation remains the previously identified approximation to
strict CPU actor execution. Report all means and paired standard errors, not
only accepted actions. Uniform estimates target a different distribution; their
error against conditional references measures the cost of omitting history,
not unbiased-estimator variance. References remain noisy, model-conditional
action-value measurements, not actual full-match strength evidence.

Save full sources, support membership, all proposals, optional confirmations,
reference endpoints, timings, work counts and pinned inputs. Check source
transitions against the native CPU implementation. Independently replay source
selection and recompute action selection, margins and all reference statistics.
Validation uses the fixed old roots only for implementation equivalence and
does not count those fixtures as fresh strength evidence. The reserved 101M
test streams, dealer rules, playable checkpoint and manuscript stay unchanged.
