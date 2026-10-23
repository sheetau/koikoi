Public-certificate pruning for optimized proposals
=================================================

The original optimized proposal planner uses public score bounds within its
continuation tree. It can nevertheless expand root actions before reaching
those leaves. At a complete frontier belief whose every world already has a
policy-independent outcome certificate, every legal continuation has exactly
the same value. Skip the entire proposal/reference comparison there. This
extends the preparation shortcut in CERTIFIED_FRONTIER_SKIP.rst to the
original optimized candidate family, preserving that family at every other
node. The existing bounds, transitions and original frozen modules are intact.

Unlike parent-default preparation, the optimized proposal shortcut can save
physical transitions. The saved allowance remains available for subsequent
uncertified nodes in the same two-million-transition global budget. Therefore
the final budget-limited comparisons can receive larger allowances. Do not
claim identical programs in general: measure both the exact gain and work.
Skipped nodes have zero possible improvement, but any consequences of budget
reallocation must be evaluated rather than inferred from the skip alone.

An isolated eight-source development study uses all original five-draw
fixtures. Its baseline is the imported fixed arm of reusable_frontier_dev_v2.
Both use the same parent prefix, node order, two-million global allowance,
250,000 prefix cap and 50,000 per proposal/reference phase. Save each new
program and the skipped public keys/masses before independent Python prefix,
adopted-pair and actual-tail verification. Count all refusals and zero results.

The predeclared development efficiency criterion is no smaller total exact
gain, no fewer positive sources and strictly fewer total planning transitions.
Only then consider a separately frozen fresh comparison; this reused screen
cannot establish population strength or latency. Two workers are used after
the already queued serial benchmarks finish. No default promotion or change
to any closed cohort follows from this protocol alone.
