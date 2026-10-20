Exact action comparison with unfinished-continuation bounds
==========================================================

The passive refusal profile shows that parent-first root action evaluation can
complete while later action comparisons exhaust the physical budget. The new
planner can stop an inferior action before its full continuation is evaluated.

At an action's continuation, let S be the exact signed utility already summed
over resolved terminal rows and fully solved observable groups. Let R be the
remaining nonnegative integer mass. Since each terminal utility is at most 1,
the action value is at most S+R. If S+R is no greater than the exact incumbent
action value B, the candidate cannot win the prescribed tie rule and is discarded.
For an opponent group of mass m, passing cutoff B-S-(R-m) into its continuation
propagates the same valid inequality. Learner groups are still optimized exactly
with one common action per own/public information set.

The parent action is evaluated first without a cutoff. Remaining actions retain
their original legal order. Induction over the finite tree shows that discarding
an action with upper bound <= B preserves the exact selected action, value and
entire selected contingent policy, including parent-favoring ties. A computed
value of total mass also retains the previous ceiling shortcut.

BoundedPlan.values contains only exact values, and root_policies only complete
policies for those values. action_upper_bounds contains the disjoint omitted
actions, including partially evaluated candidates. Together the two action sets
equal the legal root actions. Every upper bound is <= the chosen exact value.
A partial policy is never returned as an executable selected policy. This is
standard bounded chance-tree evaluation, not a general algorithm priority claim.

Verification and fixed comparison
---------------------------------

Ten core tests pass without skips. Hidden/revealed toy games and nonunit masses
check information constraints. Delayed-loss games check a real partial cutoff
below the global ceiling, both between groups and propagated through an opponent
group. Native fixtures at both seats compare all computed exact values, bounds,
selected policies and action sets with full enumeration, then independently replay
every computed policy with Python rules. Physical and live-node caps are checked.

Stage 1 compares ceiling and bounded planning on all 28 completed roots from the
closed five-draw and recovery studies. Method order alternates across source
indices (14 first each), with fresh separate policy caches and fixed then optimized
within each method. Both use 500k live groups, 2M physical decisions, 4096 frontier
rows and depth32. No source matches are added.

Require exact chosen action/value/policy equality. Every newly computed action
value and policy must equal the control and the saved independently verified
policy. Each partial-action upper bound must contain the control's exact value;
if the control omitted that action at the ceiling, both certify upper bound1.
Physical transitions and adapter calls must not increase. Report all roots and
all additional planning work. Reuse the existing Python oracle evidence through
whole-policy/value equality; do not claim new full Python or neural replay.

Only after that comparison and the passive five-refusal profile complete with
valid bindings, a separate recovery runner may attempt all five refusals at the
same limits. It can reuse the saved history-support worlds; source collection
and gates remain unchanged. Independently verify newly completed policies and
freeze all attempts before actual-tail assessment. Retain refusals and missing
roots in the original denominator. No automatic increase or final-test use.
