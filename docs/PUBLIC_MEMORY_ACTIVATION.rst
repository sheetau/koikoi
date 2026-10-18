Earlier public-memory activation: training-history diagnostic
============================================================

The current frozen comparison adapts only round8. A separate read-only replay
asks when the already identified opponent could first condition a subsequent
fresh round. It uses all2,048 retained training prefixes and the independently
audited public candidate history. No held-out evaluation trajectory, new neural
prediction or candidate continuation is read or executed.

Eligibility requires a singleton identification to have become public, a later
round than the identifying round, and public turn0/discard phase. Terminal
states and same-round decisions are excluded. This counts an opportunity; it
does not train or evaluate a policy taking it.

================  ==========  =============  ==============================
Group             Identified  Next boundary  Later recorded learner actions
================  ==========  =============  ==============================
Parent1 / WP      428         425            12,075
Parent1 / Point   435         430            13,217
Parent2 / WP      438         434            12,590
Parent2 / Point   445         442            13,216
================  ==========  =============  ==============================

The1,731 eligible prefixes activate in rounds2--8 with counts
463,387,309,199,138,146,89. Thus1,642 have a fresh boundary before round8;
1,159 could activate in rounds2--4. Of1,746 identified prefixes,15 terminate
before any subsequent fresh round. The302 unresolved prefixes stay unresolved.
All cases are retained in the accounting.

There are51,098 actual learner decisions after these boundaries in the retained
round1--7 histories. Final-round decisions are not present in this corpus and
are not inferred. Earlier adaptive actions could change all later trajectories;
these counts describe the unchanged parent prefixes only.

The replay repeats212,212 decisions,14,101 round endings and192 early complete
matches on512 shared deals. It adds zero independent deals, neural predictions
or training. Two public-timing tests pass with0 skips in1.45s. The short CPU
replay overlaps the current comparison; no isolated latency claim is made.

Artifact: ``runs/public_memory_activation.json``, SHA256
``b9086bbf6fbb82e8f1eef888bff715d868435e88ab16da82c7cdbb4241b2f625``.
The artifact binds the successful identification audit, all its inputs, this
diagnostic and its tests. The currently running public-memory study is unchanged.

This motivates a possible future stopping rule: use the frozen parent until
the first publicly identified subsequent fresh round, then adapt all remaining
rounds. Such a policy would require complete multi-round continuation training,
a matched masked-information control, and a new prospective full-match study.
No such policy or strength result is claimed here. First finish the current
round8 comparison and its independent audit.
