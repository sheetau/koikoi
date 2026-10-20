Retaining completed policies at a search limit
============================================

A complete root action can have an exact value and executable contingent policy
even if comparing a later root action exhausts the search budget. Discarding the
entire search then loses a feasible policy. IncumbentBeliefPlanner records only
root continuations that returned an exact value and complete policy before the
exception. It never exposes the currently interrupted partial policy.

The wrapper does not change action order, transitions, cutoffs or limits. Complete
searches return the same policies, values, upper bounds and work as bounded
planning. On a budget refusal, completed candidates retain their exact rational
values. Finished cutoff certificates retain their upper bounds; an interrupted
or unvisited action receives only the universal upper bound1. The selected
completed policy uses the existing parent-first/legal-order tie rule.

The difference between the largest action upper bound and the best completed
value is a nonnegative optimality gap. A positive gap explicitly leaves the
global optimum unresolved. If no root continuation completed, no plan is returned.
All partial simulator work remains counted.

Safe use requires a separate comparison
--------------------------------------

An optimized continuation of the parent root action is not the unchanged parent
policy. Its value alone is insufficient as a parent-policy reference. Before an
execution controller accepts any retained candidate, it must have the unchanged
parent policy's exact value under the same root belief, model and rules, and
compare the candidate's exact value against that reference. The fixed phase's
completed parent-first continuation can supply this value. A strict improvement
can justify a model-conditional intervention even when global optimality is open.
It cannot establish actual match-win improvement on its own.

Three tests pass without skips in2.18s. A toy game hits its physical cap after
finding an executable policy with Q=3/4 versus the parent's1/4, retains it with
unresolved optimality gap1/4, and exposes no value/policy for the interrupted
candidate, with interruption both before and during that candidate. Its completed maps and values match full enumeration. Native Koi-Koi
fixtures check complete-search identity in both modes, independent Python policy
replay and empty-incumbent refusal accounting.

This is a separate experimental component, not yet an execution controller or
a new empirical study. It leaves the running bounded recovery protocol intact:
that study continues to assess only fully completed fixed+optimized comparisons.
Any future incumbent controller needs a fixed resource schedule, verified
same-belief parent comparison, frozen policies before assessment, and all missing
or refused roots retained. General anytime incumbent retention is established
search practice; no algorithmic priority is claimed.
