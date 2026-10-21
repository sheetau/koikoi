Frontier patch development screen
================================

Retain all eight previously selected five-draw parent-refusal fixtures:
16,168,246,387,393,433,867,1054. Their mechanical selection predates this method.
The concurrent root-proposal screen is a separate fixed experiment; its outputs
are not inputs to this screen and do not select or replace any fixtures.

Cut the parent prefix at the first learner observation with at most three draws.
Allow 250,000 prefix transitions and 65,536 stored/pending rows. A shared total
of two million physical transitions covers this prefix and all local attempts.
Each local attempt has at most 50,000 proposal and 50,000 reference transitions,
allocated equally from the remaining total before it begins. Keep 25,000 live
nodes/expanded tasks per local phase, native capacity 4,096 and depth 32. One
bounded worker processes each fixture in a fresh process. No parameter grid.

Reconstruct each root support against the original source and saved worlds.
Save the patch before any value replay. For every completed prefix, independently
reconstruct all frontier posteriors using the Python engine, and replay both
continuations of every adopted local patch to terminals. The reach-weighted
sum provides the exact root increment under the fixed model; root absolute
values are not computed in this screen. Small synthetic tests separately check
the decomposition against full-root values. Non-adopted candidate values are
not reported as independently verified evidence.

Play parent and patched continuations on the retained actual source, replay all
actions and require the parent to reproduce its recorded suffix. If no patch is
adopted, both actual outcomes and traces must be identical. Record source/support,
prefix construction, each local planning phase, independent prefix replay,
local value oracles and actual/replay work separately. No fresh chance draws.

Import the old failed fixed-phase work separately; nominal integrated work is
that historical work plus the new shared allowance, at most four million.
This run does not reexecute the old phase or implement a production hybrid.
Any improvement here establishes targeted feasibility only. The closed 8,192-
source cohort and its failed practical-gain criterion remain unchanged; a fresh
prospective evaluation is required for a population strength claim.
