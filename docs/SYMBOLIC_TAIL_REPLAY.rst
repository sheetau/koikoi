Complete-cohort comparison of executable endgame programs
=======================================================

This prepared computational comparison evaluates SymbolicFinalTurnPlanner
against the completed opponent-only certificate comparison. It reuses every
one of the same512 sources, including missing roots, under the same maximum
allowances. It has not been frozen or launched. The existing comparison must
finish, pass its audit and be exported before this protocol can be frozen.

The new program adds a public own-final-turn win certificate and an executable
unchanged-parent continuation. An explicit first action remains mandatory;
other omitted actions are permitted only inside the checked certificate region.
The opponent-only loss certificate remains available. The original policy,
dealer selection, source deals, belief reconstruction and opponent models stay
fixed. No outcome-selected subset, new source seeds or phase-budget transfer
is introduced.

Controller phases each allow2M physical transitions and500k live nodes;
root-only allows4M transitions and1M live nodes. Both use fresh actor caches,
the original arm order, depth32 and capacity4096. The32 workers each process
16 sources, with at most four workers concurrent and a20GiB free-space guard.
Source records and prior artifacts are bound by their completed audit and
export hashes. Implementation, tests and this protocol are frozen together.

For every root action completed by both versions, exact values and retained
explicit actions must match. Every omitted explicit history must satisfy the
public own-turn win predicate. Each adopted parent/candidate program is also
independently replayed to terminal scoring with Python rules, including all
omitted decisions. This replay does not use the certificate's outcome as a
substitute for executing the remaining rules. Newly completed programs must
carry the recognized rule even when no previous complete root is available.

Selections are saved before executing the three actual conditional tails.
The unchanged parent must reproduce the saved source suffix; fallback must
reproduce that parent. All actual traces are independently replayed. The audit
requires all512 source identities in their original order and all32 completed
chunks. It reports physical and represented planning work, certified mass,
parent-value coverage, adoptions, independent verification work and descriptive
actual/conditional contrasts. Symbolic parent calls count within all parent
adapter calls and are separately identifiable. Program entries are distinct
from simulated transitions and represented probability mass.

The comparison uses saved computational controls. Operational times are not an
isolated latency benchmark. No fresh confidence intervals or strength gate are
defined on these reused sources; the original prospective gate remains failed.
Completing more parent values or adopting additional exact improvements is a
coverage result, not by itself evidence of greater sampled match strength.

Commands after reviewing the completed predecessor
-------------------------------------------------

::

    python -m scripts.symbolic_tail_replay --freeze
    python -m scripts.symbolic_tail_replay --launch
    python -m scripts.symbolic_tail_replay --audit
    python -m scripts.export_symbolic_tail_replay

Freeze and launch refuse existing output/queue paths. A failed process must be
inspected rather than silently overwritten. No command is scheduled by this
document.

The complete-only exporter checks audit/chunk/artifact bindings and recomputes
all source-denominator outcome means with exact fractions. It exports coverage,
planning work, certified mass, program structure, independent replay and actual
adapter calls. Its receipt retains exact contrasts as well as report hashes.
Existing output paths, incomplete queues, changed records and inconsistent
aggregate means are rejected. Export adds no simulation or inference.
