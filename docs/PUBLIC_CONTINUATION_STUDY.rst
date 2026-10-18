Prospective public-memory continuation comparison
=================================================

Question and fixed design
-------------------------

Does adapting all remaining rounds after public opponent identification improve
match wins over matched masked-memory training and the original parent? The
completed final-round-only comparison did not establish a gain. This study
tests earlier adaptation using the already materialized public stopping roots;
it does not extend or reinterpret the prior comparison.

Both original selected parents initialize new public/masked arms. Training uses
all855/876 natural roots from the audited88M source prefixes. A root is the first
fresh-round boundary strictly after the round in which the public deterministic
WP/Point candidate set becomes a singleton. Its public timestamp must already
be available. No true opponent label supplies the actor code. The code is frozen
at activation;the separate opponent label selects the training simulator policy.
Unidentified or inconsistent histories retain the original parent throughout
evaluation. All source attempts remain accounted,including317 without a root.

Both arms have identical architecture and zero-initialized public input columns,
with the masked arm receiving zero code bits. They share their parent's root
sampling/training seed and public stopping rule. Only the complete remaining
match reward trains the learner;there is no opponent-observation bootstrap or
round-reward substitution. Frozen opponents use fixed64 GPU inference in training;
evaluation uses the pinned strict CPU implementations. The two numerical paths
remain distinct policies,as documented in the preceding capabilities.

Each arm executes exactly520 updates:20 critic-only warmup and500 policy updates,
128 environments,448-step completion safety bound,three epochs,minibatch512,
actor learning rate3e-6,critic1e-4,and actor temperature0.05. Training seeds are
92110001 and92120001,one per original parent and shared between treatments.
Each arm executes66,560 conditional match continuations;all continue to actual
match termination and completed lanes remain frozen. Trajectory lengths and
actual learner/optimizer counts can diverge after policy learning,so equal
updates/completions do not mean equal decisions. Report all actual counters.

Save initial,warmup20,every100 updates,and520/final. Only the final520 endpoint
enters evaluation. No endpoint selection,interim win inspection,extra budget or
extension based on results. Independently verify frozen ancestry,bank/code joins,
finite Adam states and phase counts. Both matched initial and warmup states,
including optimizer and RNGs,must agree exactly before evaluation.

Evaluation and primary criterion
--------------------------------

After all four trainers complete,compare parent/public/masked against strict
assisted WP and Point on1,024 paired deals per opponent with both seats. Deal
seed93110001 +10000*opponent_index +replicate. These2,048 distinct fresh deals are
shared across parents/arms,total24,576 actual full matches. They are separate
from the previous91M comparison and reserved101M confirmation range. Use three
CPU workers,two threads each,with no concurrent GPU training.

Original-parent play remains exact before the public activation boundary. That
boundary is announced on every decision,including an opponent-dealer opening.
Include early bankruptcy,no identification and no later learner opportunities;
do not evaluate only the identified subset. The dealer mechanism is unchanged.

The primary criterion is the same balanced equal-WP/Point mixture conjunction
as the preceding memory comparison. For each parent,the lower95% bounds of both
public-minus-masked and public-minus-original paired win changes must be positive:
four conditions in total. Seat-average within each deal,independently bootstrap
the1,024 deals within each opponent stratum,and equally average the two strata.
Use10,000 bootstrap replicates with seed93310001. This tests the balanced mixture;
report all opponent-specific win/point and masked-minus-parent contrasts as
secondary results. Do not substitute a favorable secondary cell for the primary.

Independent evidence and implementation
--------------------------------------

``run_public_continuation_study.py`` freezes source,model,root and test identities
before training. ``audit_public_continuation_study.py`` independently reconstructs
all roots,checks checkpoint ancestry/optimizer/matched warmup,replays every actual
match,verifies exact pre-activation prefixes and public event timing,reconstructs
historical upstream inputs/masks,and recomputes all paired/bootstrap statistics.
It does not rerun every recorded neural prediction. Record original simulations,
conditional continuations and verification replay separately.

Pre-launch evidence:CPU/reference/resume tests2 passed;CUDA/reference/resume1
passed;online adapter tests2 passed;capability guards2 passed;production-shaped
two-arm check passes with96 finite Adam states and exact restarted updates.
Capability audit1ba645ade180878ebf5e8803211abe46e1b1cc8161c160100b733b5fb0647bee
and all214 inputs verify. New protocol tests3 passed,zero skips,9.24s,including
independent reconstruction of all1,731 roots from2,048 source histories and
rejection of changed activation/identification/event times,input hashes and
terminal scores. The source replay repeats212,212 decisions/14,101 rounds/192
early full completions and creates no independent deals. One full-match fixture
and its validation replays are correctness work,not strength observations.

No checkpoint is automatically promoted. The canonical human-play model and
manuscript's existing confirmed results remain separate. Any future confirmation
or deployment decision requires its own completed evidence. Capability and prior
negative-study artifacts remain preserved.
