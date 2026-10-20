Fixed score-auxiliary comparison
===============================

This protocol is fixed before main learning and strength outcomes. It tests
whether learning point distinctions within a win/draw/loss class improves the
inherited actor, beyond an outcome-only auxiliary task and ordinary PPO.
SCORE_AUXILIARY.rst records established foundations, implementation and capability.
No new general auxiliary-learning algorithm is presumed.

Learning conditions
-------------------

Two fresh PPO seeds106110001/106210001, each with none/win/score conditions.
All start from original published RL-Point, not a selected adapted endpoint.
Within each seed, initial actor/critic/environment/optimizer/RNG are identical.
The three conditions also have identical first auxiliary collections. Across
conditions only the auxiliary objective differs; subsequent trajectories can
diverge. The none condition collects data but takes no auxiliary gradient steps.

Each arm uses20critic-only+500policy updates,128env x128horizon,512PPO minibatch,
3epochs,float32CUDA,actor LR3e-6,value LR1e-4,entropy0.02,lambda0.95,T0.05,
paper8 rules and public endgame masks. PPO decisions8,519,680per arm,51,118,080
total,including warmup once. PPO rewards remain terminal match wins/draws/losses.

An auxiliary phase follows every5policy updates:100phases per arm. Each phase
collects128complete current-policy self-play matches and retains one uniform
observation per visited match/player/phase stratum. It trains6epochs with
128-row minibatches. Auxiliary Adam uses actor LR3e-6,head LR1e-4,epsilon1e-5;
global gradient clip0.5,clone KL coefficient1,auxiliary coefficient1. Heads and
auxiliary optimizer persist across phases. The outcome-only and full-score
conditions use identical65-logit heads/initialization. Full score likelihood
equals outcome likelihood plus within-outcome likelihood.

The cadence5 is fixed from data-volume accounting, not strength outcomes.
Reservoir sampling yields at most1,024rows per collection, so a phase uses at
most6,144example presentations and48auxiliary optimizer steps. Five PPO updates
use245,760presentations and480optimizer steps. Thus the auxiliary schedule is
at most2.5% of PPO sample presentations and10% of its optimizer steps, versus
0.5%/2% at the provisional component interval25. This is a bounded practical
intervention, not a claim of sufficient representation learning or parity with
PPG's full observation buffer. Actual rows/steps are reported, not replaced by
these maxima. The small phase-specific sample is not uniform over all decisions.

Each arm requests12,800complete auxiliary matches; total76,800. Auxiliary
decisions/rounds/rows/runtime may differ across policies. Report these separately
from PPO; equal requested matches do not imply equal total compute. No extra
learning, endpoint extension or checkpoint selection follows interim outcomes.

Auxiliary deal ranges107110001--107122800 and107210001--107222800 correspond
to the two seeds. Local action seeds start108110001/108210001; reservoir seeds
108310001/108410001; order seeds108510001/108610001. Cycle index0--99 is added
to each local RNG seed; deal seed increments128per cycle. All three conditions
within a seed share these schedules. Prior recorded seed-bearing JSON is checked
before preparation; that scan cannot prove anything about unrecorded work.

Evidence and mechanisms
-----------------------

Save initial,warmup,first auxiliary and every25policy-update checkpoint through
the fixed endpoint. Preserve every auxiliary collection's executed actions,
selected public row identities,feature hash,legal mask,reference probabilities,
final public margins and counters. Native/reference replay validates samples
and accounting; it is not a neural-policy rerun or new independent match set.
Model/optimizer/counter audits remain distinct from strength evaluation.

Record all100auxiliary phases' mean outcome NLL,within-outcome NLL and clone KL.
These are training diagnostics, not held-out prediction accuracy or evidence
of causal mediation. Compare each final actor and original Point on the old
25validation groups of runs/published_batch_validation.npz: per-phase greedy
disagreement and normalized entropy at commonT0.05,with forced choices counted
separately. Use the already defined source-ID modulo4==3 split; no outcome-based
subset choice. That corpus is reused public input, not deployment visitation.
No additional mechanism-driven parameter search changes this fixed comparison.

Full-match evaluation
---------------------

Keep all six endpoints and both retained WP-initialized parents. Each seed group
compares parent/none/win/score against strict assisted published RL-WP and
RL-Point. Use4,096shared paired deal streams per opponent,each in both seats:
WP109110001--109114096;Point109210001--109214096. All policies and seed groups
share these deals. Total131,072actual matches on8,192distinct initial deal
streams. Use unchanged greedy CPU2 inference and public endgame assistance.
No auxiliary head, search or opponent-private input is used in deployment.

Primary criterion: full-score auxiliary must improve equal-WP/Point-mixture
draw-adjusted win score over none,outcome-only and the retained parent at BOTH
training seeds. One simultaneous95% family covers all six paired differences;
all lower bounds must be strictly positive. Bootstrap10,000replicates with
seed109310001 resamples4,096compound blocks,each holding the same indexed WP
and Point deal pair,all policies,both learner seeds and both seats. This retains
the shared-deal covariance. Reuse the established max-standardized bootstrap
construction with an independent scalar reproduction of the final coordinates.

Report all120secondary absolute/paired win and point coordinates over mixture,
WP and Point scopes with descriptive marginal95% intervals. Do not substitute
a marginal opponent win,point-only gain or one-seed result for the primary
criterion. Report all six primary results even if one fails.

Seed1_score is the prospectively named candidate if the complete criterion
passes; do not pick the better seed. Passing this development study permits
preparing prospective confirmation, not automatic deployment. Reserved101M
confirmation deals remain unused in this study. Dealer and canonical player
remain unchanged. The broader stronger-play/novelty objective is not achieved
merely by completing this experiment.
