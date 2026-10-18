Pretrained terminal-rollout search capability
============================================

The retained Transformer parents previously had no deployment search adapter.
The older MLP search results do not measure this combination. The experimental
scripts/published_koi_search.py now compares stop and continue using sampled
hidden worlds and complete-match continuations. It changes no production file,
rule, checkpoint, playable default or ongoing learner-return condition.

Information and inference contract
----------------------------------

The fallback is the original NeuralAgent checkpoint on CPU with two threads,
greedy actions and the existing public endgame guard. Its parameters retain
their original requires-grad flags; freezing this instance would change the
near-tie inference path. All card decisions keep this fallback. Solved koi
decisions also bypass search. The adapter currently accepts paper8 only.

At an unsolved koi root, sample_hidden_batch generates observation-consistent
opponent hands and remaining stock, with fresh independent future-deal seeds.
Actual hidden assignments and future events are not rollout particles. This
uniform proposal respects initial-deal constraints but does not weight the
likelihood of observed opponent actions; it is not a posterior belief model.

Each sampled world has two branches: the fallback and the other legal koi
action. Both seats then use a separate frozen copy of the adapted actor with
acting-player features, legal actions and public endgame assistance. Both
branches retain the same sampled future deals. The reusable PairedWorkspace
runs to terminal match outcomes, failing rather than applying a critic
bootstrap if the existing512-step completion bound is exceeded. These are
branch completions, not new independent deals or training matches.

For each world, convert signed terminal payoff to draw-adjusted win score and
subtract the fallback outcome. Search proposes the alternative only if mean
paired gain minus confidence times its standard error is positive. Optional
fresh particles confirm that one alternative using an independent RNG stream;
a failed or tied confirmation retains the fallback. The margin is a heuristic,
not a distribution-free nonregression guarantee. Continuations use the parent
at both seats and approximate later play without repeated search.

GPU/batched frozen continuation inference can differ from strict singleton
CPU inference. The fallback path remains strict, but rollout values describe
this approximate continuation and hidden-world proposal. Neither complete
terminal returns nor paired randomness remove those modeling limitations.
The production evaluator does not recognize this adapter's RNG reset method;
any experimental evaluator must call reset_seed explicitly per duplicate deal.

Validation and bounded timing
-----------------------------

On18 October2026, tests/test_published_koi_search.py passes12 tests with zero
skips, including native CPU and CUDA. Actual search trajectories at both root
seats are replayed action by action through the independent Python reference,
including terminal states, points and exact decision/round/branch accounting.
Changing the actual hidden world while keeping the root observation and search
seed fixed preserves the action and comparison statistics. All three card
phases retain the original singleton action. Tests also cover independent
confirmation routing and invalid-budget rejection. Ruff passes.

Command::

  .venv/Scripts/python.exe -m scripts.check_published_koi_search --output runs/published_koi_search_capability

The fixed capability check uses16 unguarded koi roots from greedy play, shared
by both parent models,32 proposal and32 optional confirmation particles,
confidence2.5,and two warmup calls followed by16 timed calls per parent.
Source collection uses238 decisions,18 round endings and2 completed matches;
the final source match is unfinished. Search work is separate:

================= ========== ============== =================
Condition          Decisions Round endings Branch completions
================= ========== ============== =================
Parent1 warmup         12620            957               128
Parent1 timed          64888           5284              1024
Parent2 warmup         12589            956               128
Parent2 timed          65630           5280              1024
Total                 155727          12477              2304
================= ========== ============== =================

Mean timed root calls are0.319164s and0.330364s; medians0.341265s/0.354614s,
maxima0.528518s/0.574878s. Timing includes sampling, CPU fallback, GPU terminal
rollouts and any confirmation, but excludes model/workspace construction.
Another CPU evaluation was active. This is a bounded cost check on a small
greedy-play corpus, not isolated deployment latency or a match-speed estimate.
No timed root proposes or executes a change, so confirmation work is zero in
this corpus. This result does not establish a playing-strength improvement.

Launcher49769 exits0. Report runs/published_koi_search_capability/audit.json
SHA256e18e8fe1e963a01b71a9b48698dc76bd7ce29f4e94c319c6e82ffa9d47823749
binds source, retained checkpoints, root states, individual timings and work.
The running learner-return study's51 bound inputs remain unchanged.
No search strength protocol, full-match evaluation or promotion has started.

Monte Carlo rollout improvement is established methodology, as discussed in
the manuscript's Related Work. Closing this implementation gap is not an
algorithmic novelty claim. Any next comparison must report parent-relative
paired match outcomes, both parent seeds, exact inference settings and added
branch computation; its protocol should be chosen after the learner-return
comparison completes.
