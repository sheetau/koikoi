Training-root and chance diversity
==================================

Measured scope
--------------

The read-only diagnostic ``scripts/check_continuation_root_reuse.py`` examines
the two frozen continuation banks and the completed parent1-public endpoint.
It does not read evaluation outcomes,run a policy,step the simulator,or change
the ongoing comparison. Output: ``runs/public_continuation_root_reuse.json``,
SHA256 fcb28bcb12cd28809daa7f563972bab962984122712e33b4a98cf9e8f421d7fc.

=============================== ======== ========
Bank property                   Parent1  Parent2
=============================== ======== ========
Natural root records                 855      876
Distinct packed states               820      834
Distinct seed/episode streams        489      501
Distinct seed/episode/round keys      717      756
Distinct initial deck permutations   717      756
Public match/seat/memory contexts     548      557
=============================== ======== ========

The last row uses round,dealer,both point totals,learner seat and public memory
code. A repeated packed state need not be a repeated training observation:
learner seat or public memory may differ. Do not deduplicate natural roots;
their multiplicity also reflects the retained source-prefix distribution.

Parent1-public samples all855 roots in its66,560 conditional continuations.
Each root appears47--111 times,mean77.84795. The diagnostic replays only the
CPU root-index generator from the initial checkpoint for520 batches of128;
its ending RNG state equals the completed checkpoint exactly. Torch and NumPy
count the same index frequencies independently. This is not simulator replay.

Why repeated continuations do not imply fresh deals
--------------------------------------------------

``ContinuationTrainer.collect`` copies the sampled packed states without
changing their seed or episode. Both reference and native dealing construct
each round's random stream from seed,episode and round. For any given key,the
deck therefore repeats;the diagnostic also verifies identical stored initial
permutations for repeated keys. Different actions can change scores,dealer,
round duration and whether later rounds occur. Thus the result concerns
chance diversity,not identical complete trajectories or independent samples.

Possible next comparison
-------------------------

A separate prospective comparison could retain the public stopping context
(points,dealer,round,learner seat and publicly inferred opponent code),then
draw a new legal round deal and a new future chance stream per continuation.
The public identity comes from the earlier completed round and must remain
fixed;new hidden cards or a true opponent label must never supply that identity.
Redealing must follow the existing rejection rule and preserve the match
context. It must not alter the dealer-selection rule.

Such a comparison would test chance renewal against the present fixed-root
training. It would require reference/native transition and observation checks,
separate sampled-deal accounting,matched public/masked controls,and fresh full
match evaluation including failures to identify. It would not prove that
chance renewal is a new general learning method.

The diagnostic establishes repeated use of a finite chance bank. It does not
establish overfitting,explain any unobserved evaluation result,or predict that
renewal will improve wins. Complete the running fixed comparison first;do not
modify its bank,seed schedule,budget,endpoint rule or primary criterion.
