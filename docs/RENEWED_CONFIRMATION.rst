Prospective confirmation after the renewed-chance comparison
============================================================

This specification is recorded while the renewed-chance study is training,
before any of its 95M evaluation outcomes. It does not change that study or
start a confirmation run. The reserved 101M deals remain unused.

Qualification
-------------

Require the completed 40,960-match study, its passed independent audit and
completed summary, with unchanged inputs and matching result hashes. Only the
predeclared replicated strength criterion qualifies: renewed-public must beat
both original and fixed-public in the balanced opponent mixture at both
parents, with every corresponding lower 95% win bound positive. A secondary
memory, interaction, point, opponent-specific or one-parent result cannot
substitute. If this criterion fails, do not consume the confirmation range.

Freeze both renewed-public update520/final endpoints, their original parents
and the corresponding fixed-public endpoints. No alternate endpoint, masked
model or favorable parent is selected. The first parent remains the canonical
candidate designation; results cannot rerank the two seeds.

Confirmation allocation and claims
----------------------------------

Compare exactly three policies per parent: original, fixed-public and
renewed-public. Use 4,096 paired deals per opponent, both seats and both parents.
Deal seed is 101100001 + 10000*opponent_index + replicate, giving 8,192 shared
distinct deals and 98,304 actual full matches. This is one fixed allocation;
no early outcome inspection, budget extension or pooling with 95M results.
Reserve bootstrap seed101300001 with10,000 stratified paired resamples.

Use the unchanged paper8 rules, public-event activation/fallback, strict
singleton CPU policies, two Torch threads per worker and four CPU workers.
Process count affects scheduling only; policy numerical settings remain fixed.
Run only after preceding training/evaluation/audits have completed. Preserve
all no-identification, no-later-opportunity and early-bankruptcy cases.

Confirm strength with the same conjunction of four lower 95% mixture-win
bounds: renewed-public minus original and renewed-public minus fixed-public
for both parents. Report all three pairwise contrasts in each opponent stratum
and mixture, with win and point intervals, plus each candidate's absolute
performance against WP and Point. Per-opponent and point findings are secondary;
the supported strength claim concerns this balanced frozen-opponent benchmark.

This three-policy confirmation does not retest the memory ablation or the
chance-by-memory interaction. Those mechanism claims retain their separately
reported development evidence; strength confirmation cannot upgrade them.
It also does not establish superiority against arbitrary humans or opponents.

Completion and deployment
-------------------------

Before executing any 101M match, implement and verify the evaluator, independent
replay/statistics checker and complete-grid reporting. Freeze source/model
hashes, ensure fresh output paths and check all seed allocations. Correctness
fixtures must use already consumed development deals, not confirmation deals.

After all98,304 matches, independently replay every trajectory and verify
historical public inputs, unchanged frozen prefixes, model identities and all
statistics. Publish negative and positive outcomes alike. No automatic model
promotion follows from the qualification gate. A candidate adopted for human
play must also pass its public-history adapter, unknown-opponent fallback and
latency checks; dealer selection remains unchanged. Reconcile the manuscript
and artifact package with the completed evidence before declaring the research
extension finished.

``scripts/renewed_confirmation_gate.py`` implements qualification checks only.
It does not load policies, train, simulate, write a confirmation protocol or
consume final deals. The evaluator remains conditional future work.
