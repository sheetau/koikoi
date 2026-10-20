Certified incumbent selection and execution
==========================================

The controller compares exact values under one evaluation binding: hypothetical
worlds, masses, model indices, viewer, draw horizon, parent identity, opponent
identities and root observation. Different bindings are rejected. A fixed-phase
parent-first continuation supplies the unchanged parent's value. An optimized
parent-root continuation cannot substitute for that reference.

Accept a complete candidate only if its exact draw-adjusted win value is strictly
higher than the reference. On equality keep the original parent. Equal candidate
values prefer the fixed phase. A positive optimum gap does not prevent adopting
a verified improvement, but remains explicit; a fixed-phase zero gap does not
prove optimality over future policies.

The fresh search schedule runs fixed first, then optimized, each with500k live
groups/2M physical decisions/depth32 and shared root policy caches. Unused phase
budget is not transferred. Skip optimized search if the parent value is unavailable
or the best completed fixed value is1. Thus the maximum is4M physical decisions
per available root, with all partial work retained.

Execution starts at the certified own/public root and follows the selected
contingent map. It reads no actual hidden hand, deck order or future event. A
missing adopted-policy key, wrong root/seat/parent identity or illegal action is
an error. Parent fallback is explicit when no strict certified improvement exists.

Saved-plan execution audit
--------------------------

Before fresh search or new source collection, test selection and execution using
the closed64-source /16-stream cohort. All33 available roots are included;31
missing roots retain the parent. Use the paired bounded comparison's verified
plans for28 complete roots, and every completed plan saved by bounded recovery
for its5 refused roots. Do not reconstruct unsaved partial policies or rerun search.
If the fixed candidate is1, ignore the saved optimized phase to match the skip rule.

Bind and freeze all selections before any actual continuation. Then run parent
and controller from every available root against fresh strict opponents. Require
the parent to reproduce its source suffix and final state. Require identical
controller and parent traces wherever no policy was adopted. Save every action
and separately replay all traces in the exporter audit. Report conditional gains,
realized changes and all work with the original64-source denominator.

This checks available saved-plan adoption, not the full fresh search schedule:
the prior studies did not save interrupted root incumbents. Three available roots
may therefore lack a parent value even though a future incumbent search could
recover it. Reuse existing independent policy-value evidence only through unchanged
maps, values and artifact bindings. There are no new search transitions, history
replays, source matches, training updates or independent strength samples.
Actual tails are conditional continuations and remain exploratory.

Four core tests pass without skips in1.64s. They cover adoption after physical
caps, same-belief/identity rejection, fixed-versus-optimized reference semantics,
strict ties, phase skipping, both-seat native execution, missing-map/root/seat
errors and explicit parent fallback. The deployment model and dealer are unchanged.
