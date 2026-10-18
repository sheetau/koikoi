Prospective public-memory tail comparison
=======================================

Question: does information inferred from public past rounds improve final-round
play beyond an otherwise matched additional-training control?

Population and arms
-------------------

Use both frozen canonical research parents. Reuse all natural 88M training
prefixes and their audited public identification results: 938/918 final-round
roots, with 797/801 identified cases. Do not exclude unresolved, early-terminal
or incorrectly classified cases. No evaluation data enters training.

For each parent, train public-memory and masked-memory arms from the original
parent. Both zero the first-convolution weights for feature rows2/3, retaining
every other parameter and the original critic. Both use the same architecture,
memory/root bank, root-sampling RNG and optimizer configuration. Masked inputs
are zero; public inputs use only the candidate code inferred before round8.
The old unconditioned-tail checkpoint is not used as the matched control.

Use CUDA float32,128 environments,64-step safety bound,3 epochs,minibatch512,
actor learning rate3e-6,critic rate1e-4,temperature0.05. Run exactly520 updates:
20 critic-only and500 policy updates,66,560 completed conditional tails per arm.
Seeds are90110001 and90120001 by parent,shared across its two arms. Save initial,
warmup20,every100 and520 endpoints; only520 is evaluated. No early selection or
budget extension. Retain actual decision/minibatch counts,not an equal-decision
claim. Verify exact initial and warmup model/optimizer/root-RNG equality.

Evaluation
----------

After all four training runs, evaluate parent/public/masked policies against
strict CPU assisted WP and Point,1,024 paired deals per opponent with both seats.
Deal seed91110001 +10000*opponent_index +replicate. Parents/arms share all2,048
distinct deals;24,576 actual full matches total. Three CPU workers,two threads
each; no concurrent GPU training. The initial dealer mechanism is unchanged.
Prefix play stays exactly equal to its parent in rounds1--7. Candidate memory
comes from live public events and is frozen on final-round entry. Retain early
bankruptcy and no-learner-tail cases. Record both policies' terminal points,
every action,public exclusions and all prediction/transition work separately.

Primary estimand and decision
----------------------------

The training task has a balanced fixed WP/Point opponent mixture. The primary
estimand is therefore the equal-weight mixture of the two paired match-win
changes,computed separately for each parent. Wins score1,draws0.5,losses0.
For each opponent average the two seats within each deal. Bootstrap the1,024
deals independently within each opponent stratum; average the two stratum
means. Use10,000 resamples,seed91310001,and percentile95% intervals.

Declare replicated benefit only if all four primary lower bounds are positive:
public-minus-masked and public-minus-original for BOTH parents. This is a
conjunction criterion;isolated nominal positive cells do not establish it.
Report masked-minus-original,all per-opponent win contrasts,and point-margin
contrasts,including negative ones. This estimand is fixed before any new
training/evaluation result,not retroactively applied to earlier failed studies.
It supports a balanced-mixture claim,not superiority against every opponent.

Independently replay all raw matches,verify identical prefixes,public revelation
timing and candidate exclusions,checkpoint ancestry/frozen parameters/Adam state,
and reproduce stratified bootstrap arithmetic. No model promotion,101M seed
use or paper strength claim before complete results and audit.
