Complete endgame programs with a certified parent continuation
=============================================================

The isolated SymbolicFinalTurnPlanner extends the opponent-only loss shortcut
with an own-turn win region. It returns an explicit action map plus one declared
rule for executing omitted decisions. This extension is not used by either the
running prospective study or the scheduled opponent-only certificate comparison.

Let D be the learner's point margin. If only the learner's last turn remains in
the final round, the learner may score or the dealer receives the exhausted-round
one-point award. No opponent scoring turn remains. The final margin is at least
D-2. Therefore D>2 guarantees a win for every legal remaining action and draw.
The strict boundary excludes a possible draw. The previously verified loss
region (opponent alone acting on turn15, D<-2) remains available.

The program has these execution semantics:

* Its first decision must match the bound root history.
* An explicit own/public-history entry supplies the action when present.
* An absent entry may use the unchanged parent only in the declared own final-
  turn win region. The executor checks that public predicate at each such
  decision. Every other absent entry, unknown rule or illegal action fails.

Thus the omitted own-turn subtree has a defined executable continuation. The
win certificate is independent of which legal parent actions it chooses.
The planner adds its positive integer mass to the value and lowers the remaining
cutoff by that mass. An opponent-only loss subtracts its mass and raises the
cutoff. Completed root values are recorded after these contributions are
recombined. Symbolic resolutions and simulated terminal completions are distinct.

The independent Python interpreter checks the permitted default region but does
not use its predicted outcome. It executes the parent action and enumerates
every remaining draw through terminal scoring. It rejects missing actions outside
the region. Actor calls, physical verification work and represented completions
are counted separately. This verifies both values and reachability of executable
defaults; a partial explicit map alone is never treated as a complete policy.

Capability evidence
-------------------

Synthetic three-draw fixture seed31010001 has its round and scores changed to a
final-round six-point learner lead. Fixed continuation uses9034 native decisions
without the certificate and1258 with it; optimized continuation uses9082 and1306.
Both exact root values and the chosen root action are preserved. The selected
explicit map shrinks462to6 entries in fixed mode and251to6 in optimized mode.
The program additionally contains the parent-continuation rule.

At a1258-transition cap per phase, the original controller cannot finish its
parent reference. The new fixed phase completes that reference at204/275.
The optimized phase reaches its cap but retains a complete program valued
461/550, improving by53/550 while keeping root action29. Independent Python
replay verifies all retained programs. This is a synthetic capability witness,
not naturally sampled match-strength evidence.

A separate seed31010004 fixture reaches a final-turn two-way capture. Both
extremal legal parent choices are replayed fully and preserve the certified win.
Other tests cover both seats, nonunit weights, strict boundaries, execution root
and rule identity, missing-key rejection, opponent-only preservation and pickle
round trips. No new natural evaluation cohort or production deployment is part
of this milestone.
