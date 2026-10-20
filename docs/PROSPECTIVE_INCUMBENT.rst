Prospective comparison of bounded contingent play
================================================

Question and fixed cohort
-------------------------

Does optimizing a complete contingent endgame policy improve actual match
outcomes beyond the retained parent and a root-only search with the same maximum
computation? The old 64-source cohort is closed. This comparison uses 512 new
paper8 source matches: two fixed parents, strict assisted WP and Point, both
seats, and 64 replicates per combination. There are 128 distinct deal streams.
Seeds are 123110001 + 1000 * opponent_index + replicate, replicate 0--63.
The 101M final block remains reserved. No model is trained or promoted here.

The implementation freezes all cases and source hashes before collection.
The gate remains the first learner decision in round8 with at most five draws
left, after unique opponent identification from a preceding round's public
revelation. Missing opportunities and support refusals remain in the denominator
as unchanged-parent play. The true opposing hand and future stock are available
only to the simulator/evaluator. Choices use the declared history-conditioned
uniform valid-deal belief and public identification, never the evaluation label.

Arms and costs
--------------

The parent follows its unchanged strict CPU policy. Root-only search changes the
root action and follows that parent thereafter. Contingent search may change
later own/public decisions, committing to its complete selected policy.

Each search arm starts with empty action caches. The contingent arm receives
500000 live nodes and 2000000 physical transitions for fixed continuation, then
the same separate allowance for optimized continuation. Existing skip rules
apply; unused allowance is not transferred. Root-only receives 1000000 live
nodes and 4000000 transitions in one fixed-continuation phase. Both have depth32,
frontier4096 and candidate-hand1000 limits. Equality keeps the parent.

Thus each arm has at most4000000 native planning transitions per eligible source;
the complete study's worst-case planning allowance is4096000000. This is equality
of maximum allowances, not of realized work or wall time. History enumeration,
public identification, independent Python verification, source matches and actual
conditional completions are additional work, counted separately. Order alternates
by seat. Four workers share the machine, so timings are operational costs, not
isolated latency measurements. Source/history construction is shared by arms and
its cost is disclosed. No previously filled planning cache is loaded.

Execution and verification
--------------------------

The two arms must have identical belief bindings. Their exact parent values must
agree whenever both are available. Every adoption independently replays its
unchanged-parent and selected contingent policy through Python rules over all
weighted worlds. This verifies the two values needed for adoption; it does not
replay every rejected candidate or repeat neural inference in hypothetical worlds.

Both selections are saved before either actual candidate tail executes. Parent,
root-only and contingent tails then run on the original source continuation.
Every tail is replayed separately; the parent's trace must reproduce its source
suffix and a fallback must reproduce the parent exactly. Native/reference source
transitions are also compared throughout collection. Candidate tails are
conditional completions, not additional independent deal samples.

Workers handle16 sources and terminate, avoiding long-process native library
registration growth. Compressed local policy artifacts preserve executable maps,
beliefs, phase results and opponent action caches. A worker failure stops the
queue from scheduling pending work; any already running workers finish. Failed
work is not silently retried or omitted. New chunks require20GiB available.

Analysis fixed before collection
--------------------------------

The primary score is match win=1, draw=1/2, loss=0. Both seats are averaged within
each deal stream. Four paired contrasts are reported: contingent minus parent
and contingent minus root-only, separately for the two parents. A stratified
cluster bootstrap resamples64 deals within each opponent, with shared indices
for both parents and all contrasts:20000 resamples, seed123990001. Percentile
interval endpoints .00625 and .99375 apply Bonferroni adjustment to the four
two-sided intervals (nominal simultaneous95%, bootstrap approximation).

The development strength gate requires all four lower endpoints strictly above
zero. It does not authorize promotion or final confirmation automatically. Zero
observed differences produce zero intervals and fail the gate; such intervals
are not proof of exact population equality. The cohort is budget limited and is
not claimed to be powered for sub-percentage-point effects.

Exact model-conditional gains, adoption counts, failures, point margins and
realized compute are secondary. The same contrasts/cluster calculation on
conditional gains describe opportunity under the declared belief; they cannot
replace the primary actual-outcome gate. No outcome-based arm selection, seed
extension, search-cap increase or early success stopping is permitted.

Commands
--------

::

   .venv/Scripts/python.exe -m pytest tests/test_prospective_incumbent.py -q
   .venv/Scripts/python.exe -m scripts.prospective_incumbent --freeze
   .venv/Scripts/python.exe -m scripts.prospective_incumbent --launch

The queue produces a complete-only summary after checking all chunk hashes and
the fixed cohort. A separately reviewed report follows; a running queue is not
a strength result.
