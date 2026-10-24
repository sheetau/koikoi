Public-stock rule exploration: measured results
==============================================

This report covers the new eight-profile factorial in PUBLIC_STOCK_RULES.rst.
It does not revise the closed nine-rule frozen-weight transfer studies.

V1 qualification failure
------------------------

Two control policies trained from scratch for 33,554,432 decisions each.
They completed 4,382,014 rounds and 558,001 matches in total. Training counts
are not strength measurements. The planned fourteen noncontrol jobs were
not launched after the control-quality failure; the already running second
control was allowed to finish. No rule screen or confirmation was consumed.

On 256 shared new duplicate streams, with both seats, the qualification
results are:

================= ============ =====================
Training seed     Greedy       Released RL-WP
================= ============ =====================
911               63.7695%     31.9336%
912               65.8203%     28.8086%
================= ============ =====================

RL-WP here uses unassisted GPU batching; these are not the historical assisted
strict-CPU comparator endpoints. Both controls fail the predeclared 45% WP
quality gate. Seed 911 also fails the 65% greedy gate. The four endpoints
contain 2,048 matches, 16,171 rounds and 229,470 decisions. Every decision
was replayed in Python and all final states matched. These results justify
changing policy initialization; they say nothing about which new rule is best.

Source implementation is retained at Git revision 0a1c70f; protocols and
checkpoints are under runs/public_stock_training_v1 and qualification traces
under runs/public_stock_qualification_v1. The closed.json record explicitly
marks the unlaunched jobs. The separate eight-update engineering pilots and
eight-match search smoke checks are not added to qualification evidence.

V2 frozen-parent adaptation (in progress)
-----------------------------------------

PUBLIC_STOCK_STUDY_V2.rst fixes a separate protocol starting from the two
existing stronger parent policies. It freezes their actors and learns
public-stock corrections and rule-specific values. Initial control logits
preserve the parent; five focused tests verify that behavior, absence of
teacher gradients, bounded encoding batches, and hidden-state invariance of encoding and root search.
The 45 engine/sampling/search checks from the first milestone remain relevant.

Eight rules times two parents receive 4,194,304 additional decisions each,
conditional on the two control quality gates passing first. Final-model rule
screening uses 0/2/8/32 particles per action, 32 decisions per rollout and
new duplicate streams. No result or preferred rule is asserted while these
runs are in progress.

Interpretation limits fixed before results
-----------------------------------------

Particle counts give a 1:4:16 nominal budget ratio within a rule. They do not
equalize actual transitions across rules: legal-action counts, extra stock
choices, terminal timing and rollout lengths differ. A 32-decision horizon
also spans fewer ordinary turns when stock choice is enabled. Report actual
work and the full curves. A superior win-score increment alone does not prove
superiority at equal wall time or equal transition budget. Such a claim needs
a separately specified matched-cost comparison.

Each screen opponent is that rule's own adapted no-search actor. This measures
the additional value of computation, not an absolute ranking of strength
between different games. Winning by more points under doubled stakes is not
strength evidence. Two parent identities, a finite rule family and one fixed
rollout algorithm cannot establish globally maximal AI strength or that all
further improvement has been exhausted.
