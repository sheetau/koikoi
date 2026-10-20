Direct terminal-outcome learning: fixed development screen
=========================================================

The completed diagnostic measured lower conditional root-logit gradient
variance with shared continuation uniforms for both retained parents. Its
shared/independent ratios were 0.29824 and 0.64505. That motivates an actual
learning comparison; it does not predict a match-win gain.

Six final endpoints are fixed before training: both retained parents, each
with shared-action branches, independent-action branches, and root-sampled
REINFORCE. The latter uses the recorded source action and that match's terminal
root-player return, without a critic or fitted baseline. All conditions retain
the same uniform reservoir sampling and decision-count weighting. The two
branch conditions enumerate all legal actions and continue to actual match
termination with a frozen stochastic actor. Public endgame assistance remains.

Every arm runs exactly 128 updates of 64 complete source matches. Each update
uses one Adam step, learning rate 1e-5, epsilon 1e-5, no weight decay or entropy
term, global gradient clipping at 0.5, and score temperature 0.05. There are no
reused-target epochs, warmup steps, intermediate model selection or early
stopping. Seeds are 111110001 and 111210001; separate policy, reservoir and
branch RNG streams use offsets 1M, 2M and 3M. Initial weights are the retained
parents, with fresh optimizer state.

This matches source-match counts and optimizer steps. It does not match total
simulation work: sampled REINFORCE uses no branches. Actual source and branch
decisions, round endings and completions must be reported separately.
The experiment is a small learning screen, not a convergence comparison.

Retain every final endpoint and every update's source/branch action transcript.
Checkpoints include model, optimizer, source environment, all three sampling
streams and work counters. Completed diagnostic artifacts remain unchanged.

Fixed evaluation
----------------

Use 1,024 fresh paired deal streams against each of original WP and Point,
with both seats, for all six endpoints and both parents. WP seeds begin at
113110001 and Point at 113210001. Use GPU batch128, greedy action selection and
the existing public endgame mask for all policies. This is a separately labeled
GPU development screen; it is not interchangeable with the strict CPU singleton
comparison in the paper.

The six primary quantities are shared-minus-parent, shared-minus-independent
and shared-minus-sampled for each parent. Each is an equal-mixture draw-adjusted
win change against WP and Point. Resample compound deal blocks, retaining both
seats, models and opponents together, with 10,000 bootstrap replicates and seed
113310001. Use simultaneous 95% intervals from the maximum absolute centered
deviation across the six coordinates. The progression criterion requires all
six lower bounds above zero. Retain all results if it fails. Point margins
and opponent-specific values are secondary. No canonical model promotion or
reserved final-test use follows from this screen.
