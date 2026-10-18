Round-value accuracy study
=========================

Purpose
-------

The two-sample full-grid capability verified execution and backups. It did not
estimate continuation values accurately enough to use in a policy. Before any
search adoption, measure sampling uncertainty and agreement with full terminal
rollouts. Keep the retained parents, initial dealer rule and production evaluator.

Stage A: independent tables
---------------------------

For EACH of the two retained parents, build two tables, A and B, with exactly128
fresh-deal samples per public context. Use the existing round kernel builder,
batch256, and all3,776 contexts. Base seeds are42,110,001 for A and52,110,001 for B;
round r uses base+1,000,000*r+[0,127]. Both parents share deal inputs within each
table, while A and B use disjoint streams. These streams are separate from the
two-sample capability and reserved101M confirmation deals.

Fixed order:parent1/A,parent1/B,parent2/A,parent2/B. Never extend a cell or choose
a table after viewing values. Each cell has483,328 conditional round endings;
all four total1,933,312. Count actual decisions and true terminal endpoints
separately. There are2,048 distinct (round,seed) inputs across the whole study;
they are reused across contexts and parents. These are zero independent ordinary
complete matches and zero policy updates. Preserve raw public transitions and
all logs, including failures. Independently audit every table before proceeding.

Report the table-A/table-B signed-value difference over all contexts, stratified
by round, parent, seat and opponent; do not select only favorable states. Also
report differences of adjacent-point values, preserving each dealer axis.
For uncertainty, use256 bootstrap replicates with seed62,000,001. Resample the
128 deal indices WITHIN each round and table; the same index weights must apply
to all point/dealer/seat/model contexts and both parents. Use independent weights
for different rounds and for A versus B. Back up each resampled kernel anew.
This is a finite-sample bootstrap diagnostic, not a distribution-free guarantee.

Stage B: terminal reference
--------------------------

Use new data and the same frozen batched policies. For each parent, use all72
combinations of starting round in{0,3,7},points0 in{24,30,36},absolute dealer0/1,
parent seat0/1 and opponent Point/WP. Fix512 replicates per context. No context
replacement or value-dependent selection. Base seed62,110,001; a fresh round r
uses base+1,000,000*r+[0,511]. Keep scores/dealers/roles/models/parents coupled by
sample index. These seeds are disjoint from table fitting.

Run both chance constructions: (i) the native multi-round simulator, which keeps
the initial seed across subsequent rounds; (ii) independently reseeded rounds,
which replace only a nonterminal new deal with its fresh round-r seed and keep
the resulting public points/dealer. Both start from exactly the same initial
deals. They coincide in the final round. Each construction contributes36,864
conditional terminal paths per parent; the full reference contains147,456 paths.
These artificial public-context starts are not ordinary initial-state evaluation
matches. Report decisions, rounds and conditional terminal paths separately.

Compare EACH table with mean signed terminal utility in EACH chance construction.
Report every context, aggregate bias/RMSE, and empirical uncertainty from the
shared-deal sample blocks. Report native-minus-independent differences as a
chance-model diagnostic. Do not treat correlated contexts as independent samples
or finite-sample agreement as proof of exact model equivalence. Keep the full
reference out of table fitting and table selection.

Validate the reference rollout's reseeding, policy routing, terminal utility and
accounting against Python/native CPU trajectories before its full CUDA run.
Freeze its implementation identities before collecting reference outcomes.

Decision and scope
------------------

This is an estimator study. It has no strength acceptance rule, no training,
no model promotion and no extension based on intermediate values. Search using
the tables requires separate implementation checks and a new fixed comparison
against full terminal rollout, including both rollout and shared-table error.
Independent A/B tables may later support proposal/confirmation separation, but
must not be presented as removing table estimation error. A strong result would
be reduced estimation error/cost and then replicated match improvement; neither
is assumed by this protocol.
