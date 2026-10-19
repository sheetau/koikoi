Phase-dependent sampling: an exploration hypothesis
==================================================

The completed posterior-search and boundary-residual diagnostics do not justify
a stronger policy. A different potential limitation is visible in the original
published-score diagnostic: one common temperature produces very different
sampling behavior across decision heads. On its old random-policy corpus at
temperature 0.05, the median greedy-action probability is about 0.796 for discard
and 0.999999 for koi/stop. These are conditional probabilities computed on that
corpus, not measured visitation frequencies during learning.

The initial review report is runs/pretrained_policy_diagnostic/report.json,
SHA b10b32a59fbe9c8c0085ea428a7855a5e5a71354ffbe09f75512aec390930142.
Its independent audit and score hash are checked before the new calculation.

Completed score-only calibration
--------------------------------

scripts.phase_exploration_calibration uses only the stored GPU scores of the
original WP initialization. It excludes negative-source selected/modified
fixtures and forced choices. Whole source matches with ID modulo four equal
to three form validation; the remaining matches calibrate the temperature.
There are 75 calibration matches and 25 validation matches, totaling 12,025
retained public observations. No outcome label enters this calculation.

The target is mean normalized entropy H/log(number of legal actions) at
calibration discard states under temperature 0.05. It is 0.3760419898. Bisection
chooses a single fixed koi temperature of 0.21989584747027907 to match that
target on the 638 calibration koi observations. Discard and both capture phases
retain temperature 0.05. The resulting candidate is therefore a change only to
koi sampling during learning; it is not a trained policy or a deployment change.

On the 215 validation koi observations, normalized entropy changes from
0.199169 to 0.446049; validation discard entropy is 0.452271. Mean greedy-action
probability changes from 0.931125 to 0.842998, and its median from 0.999993 to
0.937786. This verifies the intended redistribution of sampling probability on
old held-out source matches. It does not show improved learning or game strength.

Two tests pass with zero skips (1.23s): the calculation matches independently
computed binary entropy, obeys shift/scale invariance, and rejects forced or
unattainable targets. The calculation adds no neural calls, simulation or
training. Receipt runs/phase_exploration_calibration.json has SHA
d191ca3bafe0c3bb1f268f612a3dfb80f2223dc97292ed87d0571c6024487e0a.

Required next comparison
------------------------

Implement this as an isolated training adapter with a matched all-0.05 control.
Keep actor architecture, learning rates, optimizer, returns, critic and training
budget identical. Greedy evaluation should use the existing strict CPU score
path, independently of the learning temperature. Verify non-koi behavior,
warmup, gradients, exact restart and checkpoint/agent identity before training.
Then fix fresh training/evaluation seeds, endpoints and paired criteria before
running a two-seed comparison against the matched control and retained players.
Do not reuse the old final evaluation or reserve 101M for this development test.

The earlier warm/cold comparison changed learning rate together with temperature;
it cannot isolate this phase-specific sampling hypothesis. Score calibration and
Boltzmann sampling are established techniques. Any contribution must be supported
by a controlled learning result and a clear mechanism, not by naming the adapter.
No training comparison has been launched yet.
