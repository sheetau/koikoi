Terminal aggregation in exact belief tails
=========================================

The five-draw capability check completed 18 roots and refused 15 at its 500,000-node
limit. Refused trees spent most physical transitions on terminal successors.
Terminal rows need a weighted outcome sum, not a new information-set decision.
``TerminalReducedPlanner`` therefore evaluates them before grouping live rows.

For a weighted state set W, write T for its terminal rows and L for its live
rows. Its continuation value is sum(w*R over T) plus the ordinary continuation
value on L. This follows directly from the terminal boundary condition. Terminal
observations contain no later action choice, so their grouping cannot change
an executable policy. The live recursion, chance expansion, opponent adapters,
maximization and parent-favoring tie rule remain the existing implementation.

This is an exact computational simplification of standard dynamic programming,
not a new optimality theorem. It reduces information-key construction and Python
group traversal. It does not reduce the number of physical transitions needed
for the same complete policy search.

Budgets
-------

The reduced planner counts live information groups. Its 500,000-live-node default
is not the previous 500,000-all-node cap. A separate 2,000,000 physical-transition cap
per solve bounds the actual simulation work. Frontier and depth caps remain
4,096 rows and 32 decisions. A refusal never yields a partial alleged optimum.
Future coverage comparisons must report these changed budgets explicitly.

Verification and fixed paired cost check
---------------------------------------

Both-seat tests with nonunit prior mass preserve every rational action value,
root action, complete root-policy map and physical/represented transition count.
Independent weighted Python replay confirms the resulting policies. Physical-
transition and live-node refusals are tested. Two tests pass, no skips, in 2.99 seconds.

Before examining runtime results, fix a paired comparison on all 18 completed
roots of the closed five-draw corpus. This excludes its 15 refusals and 31 missing
roots by completion status, not gain. It provides no new strength evidence.

Run both deferred and reduced methods at every root, each with separate fresh
policy caches. Alternate their execution order by completed-root ordinal;
each method runs fixed then optimized with its own shared caches. Both use
strict singleton CPU policies with two threads and the same parent/opponent.
Both receive a 2,000,000 physical-transition cap. The control's all-node cap is also
2,000,000; the reduced method's live-node cap is 500,000. These caps are nonbinding
on the saved completed solutions; a refusal fails this equivalence check.

Require exact equality with every saved rational value, action and complete
policy map, and equality of physical work and policy-adapter call counts.
The saved independent policy replay already verified those same maps and values.
This comparison reuses that bound evidence; it does not repeat every Python
transition. Report all root timings, method totals and maxima. One execution
per method/root is a paired implementation measurement, not a precise deployment
latency distribution.

Command after committing the verified implementation and protocol::

    .venv/Scripts/python.exe -m scripts.check_terminal_reduced_tail

The fresh output is ``runs/terminal_reduced_tail_equivalence``. No previously
refused root is solved by this check, and no canonical actor or manuscript changes.
