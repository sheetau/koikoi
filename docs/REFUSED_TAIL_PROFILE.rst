Passive profile of the five remaining tail refusals
==================================================

The coarse public outcome bound did not certify any refused root. The existing
refusal records do not identify which root action exhausted the transition
budget. This diagnostic replays exactly sources 0, 4, 20, 36 and 52 with their
original parent, support and opponent. Source 36 completes the fixed control
before failing in optimized continuation; that order and shared cache are kept.

Limits remain 500,000 live groups, 2,000,000 physical transitions per plan,
4,096 frontier rows, depth 32, five remaining draws and 1,000 candidate hands.
All five roots are included. There are no new source matches, strength samples,
actual-tail interventions, training updates or automatic budget increases.

ProfiledPlanner delegates choices and transitions to the frozen ceiling solver.
It records completed/partial root actions and successful physical transitions
by turn and phase. Completed exact values are distinguished from partial work;
no Q is assigned to an incomplete root action. All old work counters, call
counts, failure reasons and completed fixed values must reproduce exactly.
Timing includes instrumentation and is not compared as a speed measurement.

The diagnostic adds computation: repeated current-round support inference and
planning, including partial branches, must be counted separately from sources.
It saves verified support worlds, partial action caches and any complete plans
for a subsequent implementation check. No partial tree is a completed solution.
The new artifact and input hashes are bound at completion.
