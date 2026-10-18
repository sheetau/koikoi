Fixed one-round versus terminal koi-search comparison
====================================================

Question
--------

Do the round tables change the error/cost of an actionable stop/continue
comparison? Complete the ROUND_VALUE_ACCURACY study first. Compare both search
estimators and retain all outcomes,including absent roots and rejected changes.
This is an action-value diagnostic,not a full-match playing-strength test.

Fixed source and inference
--------------------------

Use the two retained parents,assisted strict CPU WP/Point opponents,both parent
seats,and two source replicates per matchup:16 completed source matches.
Source seed is72,110,001+100*opponent_index+replicate,with opponent_index0 for WP
and1 for Point,and replicate0/1. Four seed streams are deliberately shared across
parents/seats;do not count sixteen independently sampled deals.

Within each complete source match,retain its FIRST parent koi decision with both
actions legal,not solved by the public guard,and at most100 public current-hand
combinations. Select before the parent acts and without inspecting values. Keep
missing roots without replacement. Source actions remain the original parent
and fixed comparator throughout;search never changes source trajectories.
Check every source transition against native CPU.

At each retained root,compare HistoryConditionedSearch and RoundValueSearch.
Both use the same strict historical WP/Point posterior,equal start-of-round model
priors,512 proposal particles,1,024 fresh conditional confirmation particles,
and confidence2.5. Round search includes joint table/particle uncertainty and
uses the fixed A/B tables;terminal search uses its paired-terminal standard
error. Both have only one alternative at a koi root. Reset search seed to
72,210,001+case_index identically in both methods. Verify proposal world hashes
and model labels agree. When both methods confirm,verify their confirmation
world hashes/model labels agree as well. Keep every candidate estimate,margin,
raw endpoint record,change/refusal,counter and time. Root timers cover act(),
including history enumeration and inference,but exclude agent construction and
table loading. Record offline table construction separately. Timing is
descriptive,not an isolated latency guarantee.

Independent terminal reference
------------------------------

For EVERY retained,nonempty-support root,simulate the parent and its sole
alternative on4,096 new history-conditioned worlds. Use base seed
72,310,001+case_index,processed as exactly8 sequential512-world chunks with one
continuing RNG. Reuse the verified support;do not rescore history or supply the
actual opponent identity/hand. The reference uses the same frozen batched
continuation family and ordinary native future-round deals. This is a model-based
reference with sampling error,not an oracle for arbitrary human opponents.

The reference does not select a candidate or trigger additional samples. Save
all paired terminal point margins and model labels. Compute the signed-to-win
conversion from terminal points independently,then mean/standard error and the
descriptive2.5-standard-error margin. For both methods report proposed/accepted
changes and reference change values for ALL roots,not only accepted ones.
Compare proposal error against the common reference,paired-difference variance
or estimated uncertainty,and work/time. Reference error remains in observed
estimator errors;correlated source roots are not independent strength trials.

Closure
-------

Fix all source/action budgets and streams before execution. No reranking,threshold
tuning,root replacement,extra reference samples,training or deployment promotion.
An empty/negative diagnostic is retained. Any later complete-match strength
comparison needs a new fixed protocol and must retain both parent replications,
unchanged opponents and an appropriate search control. The reserved101M final
evaluation remains untouched.
