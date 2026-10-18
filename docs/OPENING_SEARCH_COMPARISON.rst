Fixed opening selected-action comparison
=======================================

Source and scope
----------------

Use the two retained parents,strict assisted WP/Point opponents,both candidate
seats,and two replicates:16 original-policy paper8 source matches. Seed is
81,110,001+100*opponent_index+replicate,index0 for WP/index1 for Point,replicate0/1.
Four shared deal streams are reused across parents/seats. Initial dealer remains
the simulator's ordinary dealer. Retain the initial state only when the candidate
is that dealer;otherwise record no eligible opening. Exactly eight opening roots
are expected by the seat-pair construction,not by outcome-based replacement.
Finish every source match with original policies and check native CPU equality.

No opponent has acted at the retained roots. Use the declared uniform valid-deal
prior and equal WP/Point model prior from OPENING_SEARCH.rst. Do not supply actual
opponent identity or score hidden hands by later source outcomes. The continuation
policies are the existing frozen batched approximations. After the opening,each
proposed policy uses only its retained parent and public endgame guard.

Candidate comparison
--------------------

Run OpeningTerminalSearch and OpeningRoundSearch on every eligible root. Both
use256 proposal particles,1,024 conditional fresh confirmation particles,and
confidence2.5. Reset search seed to81,210,001+case_index for both. Evaluate all
seven alternatives to the strict parent action on common worlds/model labels;
verify proposal world hashes and labels match between methods. When both confirm,
their confirmation world hashes/labels must match too. Keep candidate selection,
tie order,positive proposal/confirmation gates,and fixed A/B table roles unchanged.
Record every candidate estimate,margin,paired outcomes,work and online act time.
Offline table cost and construction/loading remain separate.

Independent selected-action reference
-------------------------------------

Take the UNION of the two methods' highest-mean proposal candidates,whether or
not either passed its gate. It contains one or two card actions. For every root,
compare every action in that union with the parent on4,096 NEW worlds,using seed
81,310,001+case_index and exactly8 sequential512-world chunks with one continuing
RNG. All reference alternatives share worlds/model labels within each chunk.
Run to actual native match termination with the same frozen continuations.

Reference sampling does not change either method's selected/accepted action.
No candidate reranking,additional samples or replacements follow the reference.
Save all paired terminal point margins and raw statistics. Report both methods'
estimates of EACH union action against the same independent reference,alongside
their selected and accepted choices. This is a selected-action diagnostic;it
does not estimate error over the six or seven unreferenced alternatives or over
the population of all opening hands. Each reference itself has sampling error.

Work and closure
----------------

Fixed proposal work per method is eight roots x seven alternatives x256 worlds
x two actions:28,672 branches. Round-search branches stop at their first round
ending;terminal-search branches run to match end. Confirmation adds at most
16,384 branches per method. Reference work is65,536--131,072 terminal branches,
depending only on whether the two prescribed selections agree. Count decisions,
rounds,terminal branches,source matches and distinct source streams separately.

Retain the entire fixed experiment,including absent nondealer cases,negative
estimates,refused changes and failures. All16 source matches are completed before
search. No actor training,table selection or extension of earlier diagnostics.
Source roots share four deals and are not eight independent strength trials.
No full-match strength inference,promotion,manuscript result claim or101M final
evaluation follows from this diagnostic alone. Independently replay source
selection and recompute all candidate/confirmation/reference arithmetic after it
finishes. A positive result would motivate a separately fixed match comparison.
