Frontier implementation timings with explicit policy checks
==========================================================

Both serial timing studies are complete. They use frozen CPU actors with two
PyTorch threads and fresh dynamics/action caches for each arm. Each paired
case runs old/new then new/old. Loading, serialization and independent audits
are outside timed planning. The studies run after the continuation experiment,
without another research worker. Timings are local descriptive measurements;
two repetitions do not provide a general hardware or population latency claim.

Reusing unchanged interval receipts
----------------------------------

The four cases are the first original fixture in each parent/five-or-six-draw
stratum: 0, 2, 8 and 9. All sixteen saved policies pass exact comparison of
receipts, selections, prefix worlds/masses and physical work. Each arm uses
3,371,324 planning transitions over its eight runs. No source match is new.

Total elapsed planning time is 110.613346 seconds for the original comparator
and 112.426840 for the receipt-reuse implementation: ratio 1.016395, or 1.64%
longer. This does not establish a speedup. The comparison measures the net
implementation, including the new actor registry's defensive identity checks;
it does not isolate the cost of caching from that adapter overhead. The
candidate registry is empty, so the policy family remains unchanged.

Public certificates before candidate construction
------------------------------------------------

The second study selects original fixture 2 because all 1,304 frontier nodes
already have policy-independent public outcome certificates. This structural
selection is explicit: it is a special-case benchmark, not a representative
sample of search roots. The shortcut omits candidate construction at those
nodes and keeps the parent policy. All remaining comparisons and all physical
work match; each run uses 4,974 prefix transitions and makes no local change.

.. csv-table:: Mean seconds per run, two repetitions
   :header: Candidate family,Before,Pruned,Time reduction

   Parent-default,5.362988,2.215854,58.6825%
   Plus other parent,8.486422,2.227916,73.7473%
   Plus WP and Point,16.313990,2.241812,86.2583%

All twelve saved programs pass the comparison check. All six pruned runs
also pass independent prefix verification and actual-suffix replay; actions,
terminal states and work match the unchanged parent. Continuation-actor
queries fall to zero. The original named-family queries are 1,304 for the
other parent and 1,304 each for the other parent, WP and Point in the larger
family. These are planning-time savings with no strength change.

The same public-certificate idea can precede the original optimized proposal
search. Unlike preparation-only pruning, that placement can avoid physical
transitions and redirect the saved allowance to later unresolved nodes.
It is a separate development question specified in
``docs/PRUNED_OPTIMIZED_FRONTIER.rst``; this timing result alone does not
establish its budget-limited gain or population performance.

Evidence and limits
-------------------

Frozen studies: ``runs/frontier_receipt_benchmark_v1`` and
``runs/certified_frontier_benchmark_v1``. Independent reduction reloads saved
programs, checks exact equality/pruning, verifies all result/audit bindings
and recomputes timing sums and ratios:
``docs/paper/generated/frontier_timing_evidence.json``, produced by
``scripts/export_frontier_timings.py``. This reduction reuses measured wall
times; it is not a new timing experiment. Original protocols, programs and
results remain unchanged. All runs reuse development sources and add zero
independent matches.
