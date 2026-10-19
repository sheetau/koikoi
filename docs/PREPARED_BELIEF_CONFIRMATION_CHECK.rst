Independent confirmation of positive-mean candidates
===================================================

This is a follow-up development diagnostic, not a revision of the preceding
four-condition experiment. That experiment uses a positive 2.5-standard-error
margin twice: first to screen a proposed action and then to confirm it. Its
initial results show that the first screen rejects even the positive-mean
candidates. Freeze the complete preceding result and its independent audit
before beginning this follow-up.

Retain all 25 roots and all four conditions. Reuse each previously selected
alternative, whose selection used 128 proposal particles. If its recorded mean
win difference is positive, evaluate that one alternative against the unchanged
parent on 512 fresh paired particles. Otherwise retain the parent without a
new rollout. Accept a changed action only when the fresh mean minus 2.5 standard
errors is positive. Never change the selected alternative after confirmation,
reuse proposal samples for confirmation, or rank conditions by actual outcomes.

The fresh simulation seed is 103100001+10000*parent+case_index, shared across
conditions. It is distinct from both streams in the preceding experiment.
Reuse the exact public-history tables and fixed64 CUDA continuation execution;
the same numerical-policy limitation and two CPU threads apply. There is no
new hidden-hand inference, training or fresh actual deal.

Freeze all new choices before their actual-hidden-world assessment. Reuse the
preceding unchanged-parent completions; evaluate only distinct new selected
first actions with strict CPU continuation. Record every new action and final
state for independent replay. Report all conditions, skipped confirmations,
actual simulation work and reused outcomes. No confidence claim about population
strength or model promotion is based on these old development roots.

This check separates a screening bottleneck from failure of independently
confirmed candidate actions. It is ordinary sample-splitting policy search,
not a claim of a new general algorithm. Even favorable conditional outcomes
would still require a prospective full-match comparison to establish strength.
