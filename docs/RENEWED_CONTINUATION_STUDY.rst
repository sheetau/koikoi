Prospective chance-renewal comparison
=====================================

Question
--------

Does renewing the cards and future chance at a public adaptation boundary
improve full-match wins at the same conditional-training budget? The fixed
bank contains855/876 roots but only717/756 initial deck permutations. That
diagnostic motivates a generalization test;it does not establish overfitting.

This design is written before inspection of any current multi-round evaluation
contrast. It is a prospective specification,not an executed or frozen protocol.
Execution requires the existing comparison's completed independent audit,the
queued renewed learner's real CUDA and production capability,and verified
runner/auditor implementations. Neither running protocol is changed here.

Intervention and controls
-------------------------

For each original parent,use the following five policies:

* Original frozen parent.
* Fixed-bank public-memory endpoint from the current comparison.
* Fixed-bank masked-memory endpoint from the current comparison.
* Newly trained renewed-deal public-memory endpoint.
* Newly trained renewed-deal masked-memory endpoint.

The latter four form a2x2 comparison:fixed versus renewed chance,and public
versus masked opponent identity. Reuse the two fixed-bank endpoints exactly;
do not retrain or select them according to the current comparison's outcomes.
All five policies receive fresh full-match evaluation on the same deal grid.
The original parent is an additional absolute-strength control.

Renewal samples the original bank records with their existing multiplicities.
Keep round,dealer,points,learner seat and the earlier publicly inferred identity.
Draw a fresh legal deal using the existing rejection rule and a fresh future
chance key. Simulator routing retains its separately recorded opponent label;
this label must not replace public identity in the actor's observation.
Changing cards at a fresh-round boundary preserves the scoring/dealer rules.
It does not justify resampling a partially played hand.

Under independently shuffled rounds,the new round's cards are independent of
earlier revealed hands conditional on the retained match context. This motivates
training over fresh future chance while retaining the stopping-context sample.
The concrete simulator uses deterministic seeded pseudorandom streams;the
implementation checks agreement with its existing dealing semantics. Neither
independence of finite seed outputs nor a new general learning theorem is claimed.

Training allocation
-------------------

Each new arm starts from its original parent,not a previously adapted endpoint.
Retain520 updates,20 critic-only then500 policy,128 environments,horizon448,
three epochs,minibatch512,actor learning rate3e-6,critic1e-4,temperature0.05 and
complete remaining-match returns. There are66,560 conditional completions per
arm. Fixed and renewed arms can have different trajectory lengths and optimizer
steps;report their actual counters and wall time as well as equal completions.

Use the same training seeds92110001/92120001 as the corresponding fixed arms.
Root-index sampling uses each seed+1,independently of policy sampling. The saved
root-index schedule must match the fixed arm at each saved endpoint;renewal must
not change which public contexts are sampled. Public/masked renewed arms share
the fresh chance allocation:

==================== ============== ============== ============
Parent               First key      Last key       Draws/arm
==================== ============== ============== ============
1                    94110001       94176560        66,560
2                    94210001       94276560        66,560
==================== ============== ============== ============

The four new arms produce266,240 conditional completions using133,120 distinct
chance keys shared between public/masked arms. Do not call those completions
full matches or four independent sets of deals. Correctness replay is separate.
These ranges avoid the queued94990001 capability allocation and all current
evaluation ranges. The runner must reject allocation overlap before training.

Initial and warmup20 actor/value/optimizer/RNG states must match exactly within
each renewed public/masked pair. Warmup states need not equal fixed-bank warmup:
the training cards have intentionally changed. Preserve frozen prefix/opponents,
public metadata provenance and every checkpoint's sampler cursor/RNG identity.
Save initial,warmup20,every100 updates,and520/final. Evaluate only520. No interim
win inspection,checkpoint selection,budget extension or seed replacement.

Fresh evaluation and claims
---------------------------

After training and endpoint checks,run all five policies from the beginning of
each match against strict assisted WP and Point. Use1,024 paired deals per
opponent with both seats and both original parents. Deal seed is95110001 plus
10000 times opponent index plus replicate. Thus2,048 shared distinct deals
produce40,960 actual full matches. Prior93M outcomes are not pooled with these
outcomes. The reserved101M confirmation range remains untouched.

Use the existing public-event activation rule,including opponent-dealer first
decisions. Include no-identification,early bankruptcy and no-later-opportunity
matches. Before activation,use the original parent exactly. Renewal is a
training intervention;the online adapter never redeals or reads simulator
chance keys. Evaluate with three CPU workers,two threads each and no concurrent
GPU training,using the pinned strict singleton numerical policies.

The primary strength criterion is a conjunction of four positive lower95%
paired win bounds:for each parent,renewed-public minus original and
renewed-public minus fixed-public. Seat-average within each deal,bootstrap
1,024 deals independently within each opponent stratum,then equally average
WP and Point. Use10,000 bootstrap draws with seed95310001. Report all four
conditions even when one fails;none can be replaced by a favorable point margin.

Separately test the proposed information mechanism for each parent:

* Renewed-public minus renewed-masked measures the public signal's benefit
  under renewed chance.
* The interaction (renewed-public minus renewed-masked) minus (fixed-public
  minus fixed-masked) measures whether renewal changes the signal's benefit.

Apply the same paired stratified bootstrap to these prespecified contrasts.
Both-parent positive bounds on the former support a replicated public-memory
benefit;both-parent positive interaction bounds support the narrower interaction
claim. Passing the strength criterion alone does not establish either claim.
These are secondary mechanism analyses,not substitutes for primary success.
Retain all opponent-specific win/point results and all ten pairwise contrasts
among the five policies,including fixed/renewed masked controls. Label those
exploratory comparisons and do not select a new canonical model from them.

Independent verification and interpretation
------------------------------------------

The future protocol must pin the completed prerequisite audits,new source/test
identities,all four old endpoints/banks and both original parents. Independently
check renewed endpoint ancestry,finite optimizers,root schedules,fresh-key
accounting and paired warmup. Replay all actual full matches,public event
availability and reconstructed historical inputs;recompute every statistic.
Report the verifier's neural-replay scope explicitly rather than equating action
trajectory replay with a rerun of all network predictions.

A successful study would establish improved adaptation from these sampled
public stopping contexts against the two pinned opponents. Chance renewal,
opponent modeling and guided starting states have prior precedents. The research
claim would rest on the measured Koi-Koi gain and the controlled mechanism test,
not on presenting those ingredients as new principles. A failed primary result
closes this fixed comparison without additional budget or retrospective claims.
