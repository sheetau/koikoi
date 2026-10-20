Complete-trajectory Monte Carlo control
=======================================

Question and relation to earlier work
-------------------------------------

The completed root-allocation/step-size screen did not improve playing strength.
Its source matches contain many decisions, but each optimizer update uses only
one selected decision per match. A direct control can use every eligible source
decision without another simulation.

For a completed trajectory tau with K eligible decisions, write g_t for its
player-relative Monte Carlo score-function contribution. Uniform root selection
satisfies E[K*g_T | tau] = sum_t g_t. Thus full-source Monte Carlo removes the
conditional root-selection covariance from the sampled-return estimator.
This standard conditional-expectation argument neither orders the variance of
full-source and all-action branch estimates nor guarantees stronger play.
The new learner is a complete-trajectory REINFORCE control, not a new algorithm.
A future branch method must demonstrate its benefit against this control.

The earlier complete-MC training study used MLP128x2 with PPO. Retained-agent
complete-return direction probes measured gradients but did not train this
single-step Transformer control. Neither substitutes for the comparison here.

Learners
--------

At each retained parent and each learning rate, compare:

* Full-source Monte Carlo: all eligible decisions from all completed matches.
* Root-sampled Monte Carlo: one uniform reservoir root per match, weighted by K.
* All-action branches: the same uniform reservoir root, all legal first actions,
  complete-match continuations with shared policy uniforms, weighted by K.

Full-source collection preserves the existing source actions, independent
reservoir stream and root selection. It additionally records every eligible
pre-action state, action, lane and step. Each label is the final signed outcome
from that state's acting player's viewpoint. Forced and terminal-padding rows
have no policy loss. Raw simulator states are internal training records; actor
inputs remain the existing own/public projection.

Accumulate negative log-probability times detached terminal return in fixed
chunks of 256 rows, divide by the number of complete matches, and take one Adam
step only after all chunks. This is not an average over decisions or an optimizer
step per chunk. Apply global gradient clipping once after accumulation.
There is no critic, return bootstrapping, entropy term, PPO ratio or replay of
old training targets. The three phase networks retain all 4,019,715 parameters.
As in the earlier self-play learner, each decision uses its current player's
payoff; this is not differentiation of the sum of both zero-sum utilities.

Fixed development comparison
---------------------------

Both retained parents receive all six combinations of three estimators and
learning rates 1e-5/1e-4: twelve arms. Every arm receives 128 updates of 64 fresh,
complete source matches, with T=0.05, Adam epsilon 1e-5, global clipping 0.5,
fresh optimizer state and the existing public endgame masks. Parent seeds are
119110001 and 119210001. No earlier trained endpoint is substituted for a new
control. The first source collection must agree exactly across all six
conditions for each parent.

The total source budget is 98,304 matches and 1,536 optimizer steps. Report
actual source decisions/rounds, loss rows, branch work and elapsed time separately.
The four branch arms spend additional simulation work. Gradient computation and
branch costs are not equalized by matching source and optimizer counts.
Keep every final endpoint; no intermediate evaluation or checkpoint selection.
Save every update's source/branch records and checkpoints every sixteen updates.

The predefined candidate is full-source Monte Carlo at 1e-4. Compare it with
its retained parent, root-sampled/high and all-action/high at both parent seeds.
These are six primary equal-WP/Point-mixture win-score contrasts. A single
simultaneous 95% family uses the maximum absolute centered bootstrap deviation.
All six lower bounds must exceed zero to pass progression. The low-rate
contrasts and all absolute matchup means are descriptive secondary outcomes.
This choice tests whether using all source decisions makes a larger update
useful; it is not a prior claim that the high rate will succeed.

Evaluate all twelve endpoints and both parents against assisted original WP
and Point on 1,024 paired streams per opponent, with both seats: 57,344 matches
and 2,048 distinct streams. Use the GPU batch-128 greedy/endgame path, separately
labeled from the paper's strict CPU singleton comparison. WP seeds start at
120110001, Point at 120210001. Bootstrap 10,000 compound deal blocks with
seed 120310001, retaining seats, opponents and models together. No canonical
promotion or use of the reserved 101M final-test block follows this screen.

Verification and execution
--------------------------

The analytic tests compare variable-length trajectory gradients with exact
objective differentiation and check chunk normalization. Actual CPU/CUDA source
records replay through the independent Python rules, including both seats,
all phases and round endings. Actual updates have exact checkpoint continuation.
Root controls retain the old update implementation.

The study's training audit replays every transition, every full-source training
state/action/return, selected roots, branch labels, work counts and first-source
agreement. The evaluation audit verifies all transitions, deals, margins and an
independently constructed six-coordinate simultaneous bootstrap. These audits
do not repeat every neural forward or optimizer calculation.

Run the separate reduced end-to-end capability fixture before launch. It uses
two updates, four matches per arm and two evaluation pairs per opponent, with
separate 11904M--11908M streams; these counts are capability work, not evidence
of strength. The main command is::

    .venv/Scripts/python.exe -m scripts.full_source_study

The runner trains, audits, evaluates and audits in sequence. It refuses an
existing output directory. Preserve failed or completed artifacts rather than
automatically restarting them.
