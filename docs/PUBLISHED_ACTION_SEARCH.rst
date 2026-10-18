Retained Transformer: all-action terminal search diagnostic
=========================================================

The previous Transformer search adapter explored only stop/continue and found
no changed action in its fixed root diagnostic. Card-action search has not yet
been measured with these actors. The earlier full-action MLP experiments do
not resolve this separate combination.

The experimental PublishedActionSearch inherits strict CPU2 parent inference,
public endgame assistance and observation-consistent hidden-world proposals.
At every unforced decision, it evaluates every legal alternative against the
parent action. All alternatives share the same sampled worlds and fresh
future-deal seeds. PairedWorkspace executes each two-action comparison to the
terminal match, using the same frozen observation-only parent at both seats.
Repeated baseline branches are counted as executed work; they are not new
independent worlds. A discrepancy between repeated baseline outcomes fails.

The alternative with largest mean draw-adjusted win gain is selected; legal
action order resolves ties. Its gain must exceed2.5 standard errors. Exactly
that alternative is then compared with the baseline on fresh independent
worlds and must pass the same margin. Confirmation cannot choose another
alternative. This is established Monte Carlo rollout improvement, not a new
algorithm. Uniform hidden proposals and self-play continuations approximate
the actual opponent. The standard-error margin is a heuristic, not a safety
guarantee; finite samples can have zero empirical variance.

Fixed diagnostic before observing search outputs
-----------------------------------------------

Use the saved256-position corpus for each retained parent from
runs/observer_seat_information_v2. Select the first two unforced roots of each
card phase and each acting seat:12 roots per parent,24 total. Selection uses
only order,phase,seat and legal-action count. There is no new source play,
training or use of final-test seeds. Existing states can share trajectories;
the roots are not independent match-strength observations.

Use32 common proposal worlds per root and256 fresh confirmation worlds only
when a proposal passes,confidence2.5. RNG seed is29130001+1000*parent_index+
saved_root_index. Both retained parents and all selected roots must finish;
do not extend the sample or change thresholds from these outcomes. Retain
every branch's terminal point outcomes, all decisions and round/branch
endings, selected action and wall time. Recompute all recorded statistics
and decisions from terminal points with Python's standard library. Check
source/input hashes and unchanged actual root states. Timings include cold
calls and concurrent CPU evaluation; they are not deployment benchmarks.

Command::

  .venv/Scripts/python.exe -m scripts.check_published_action_search --output runs/published_action_search_capability

Before launch, test actual CPU/CUDA branch trajectories against the independent
Python reference at both seats in all three card phases; verify unchanged
observations and search outputs under changes to actual hidden cards; verify
independent confirmation can only accept or reject the selected candidate.
The running seat-critic study, its bound sources and completion queue remain
unchanged. This diagnostic only determines whether the broader search can
change actions at a bounded cost. A match comparison would require a separate
fixed protocol and both parent-relative controls.
