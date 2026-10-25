Public-stock adaptation maturity: conditional protocol v5A
=========================================================

Purpose and launch condition
----------------------------

Large search gains over a fixed actor can partly repair errors that more
policy learning would remove. This study measures stronger adaptation before
any claim that a search advantage survives stronger baseline play. It does
not infer strength from update counts, training loss or checkpoint age.

Launch only after the complete independent V4 reduction reports that at least
one finalist passes all six score comparisons. A failed V4 work-admissibility
check remains unresolved: passing this adaptation study cannot repair it.
If neither finalist passes the V4 score criterion, first diagnose that result;
this protocol does not authorize an automatic fallback training run.

Retain all six V4 policy/rule systems: control, preview plus choice, and all
three factors, each with parents 0 and 1. Do not select between the finalists
using V4 ranking or intermediate training results. Preserve all prior models,
sources, tables, protocols and consumed cohorts. GPU jobs run serially after
V4 has finished.

Additional policy learning
--------------------------

Start from each retained V2 policy_00128.pt, verified against the completed
V3 evidence receipt. Continue the correction network and critic for exactly
384 additional updates, with 512 environments and 64 decisions per update:
12,582,912 additional decisions per system and 75,497,472 across six systems.
Including the retained adaptation, each system has 512 updates and 16,777,216
adaptation decisions. Original parent training is still reused and separate.

Use scripts/continue_public_stock_residual.py. The original teacher remains
frozen. Use fresh Adam with learning rate 0.0001 and epsilon 0.00001; this is
a weight warm start, not restoration of an earlier optimizer, RNG or live
environment. Keep the original residual architecture and PPO settings:
three epochs, minibatch 4096, clip 0.2, value-loss weight 0.5, entropy 0.01,
teacher KL weight 0.02, gamma 1, signed GAE lambda 0.99 and gradient norm 0.5.
The reward remains terminal match-win utility, without score shaping.

Use sampling seeds 1101/1102 and initial training seed blocks
1,220,000,000/1,230,000,000 for parents 0/1, shared across the three rules.
Save every 128 additional updates for diagnosis. Only the final 384-update
checkpoint is eligible; no early checkpoint selection or training extension.
Record additional decisions, rounds, matches and updates separately from
cumulative adaptation counts. Verify finite tensors, original teacher identity,
initial checkpoint identity and unchanged source bindings.

Independent strength qualification
----------------------------------

Use 1,024 new duplicate streams beginning at 1,240,000,000, both seats,
batch size 64, with deterministic greedy policy actions and no online search.
Use the existing recorded residual evaluator with particles zero, horizon 32,
confidence 1 and search seed 1103 (inactive without search). For each rule and
parent evaluate the new final policy against both retained old policies from
that same rule. Also evaluate each retained old policy against the other old
parent. This is twelve new-policy endpoints plus six old-policy baselines,
36,864 actual matches on 1,024 shared independent streams. It is not 36,864
independent deals. Save and Python-replay every scored action; independently
reduce scores, margins, match/round/decision counts and opponent identities.

For each of the six systems compute two contrasts on paired per-stream
win scores (ties count one half):

* Against its own retained policy: new-minus-0.5. The old policy against
  itself has exactly 0.5 duplicate-seat score by deterministic seat symmetry.
* Against the other retained parent: new score minus its own retained
  policy's score against that same opponent on the same streams.

Keep all twelve contrasts in one prespecified family. Use 20,000 paired
duplicate-stream bootstrap replicates, seed 1109, with one-sided lower
quantile 0.05 / 12. Report all means and lower bounds. A rule qualifies only
if both contrasts have positive lower bounds for both parents. This supports
improvement against the two retained opponents, not universal policy dominance
or equilibrium play. Do not pool parents, extend deals, select checkpoints,
change the threshold, or reuse qualification outcomes as search-test results.

Consequences and remaining search question
-----------------------------------------

If control and at least one finalist qualify, the qualified finalist(s) and
control can enter a separately fixed search study using the new policies,
new policy-specific continuation tables and new calibration/evaluation
streams. The old self-policy tables must not be attached to new policies.
Resolve any V4 cost-calibration failure in that new design before claiming a
matched-work replication. Qualification itself is not evidence of retained
search gain; that question requires its own reserved evaluation.

If qualification fails, report all failures and distinguish lack of measured
policy improvement from loss of search value. Inspect learning and opponent
effects before defining any further adaptation method. This closed cohort
cannot be extended or used to select an intermediate checkpoint. No claim
that stronger-policy robustness has passed follows from a failed gate.
