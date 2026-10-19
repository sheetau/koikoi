Exact-belief action diagnostics
==============================

The completed exact-hand calculation lowered descriptive card-membership error
on the 25 eligible development roots. The first action comparison nevertheless
keeps the unchanged parent's action at every root in all four conditions.
Better hand inference alone did not yield a stronger policy in this setting.

First comparison: two-stage margin
---------------------------------

The frozen design is PREPARED_BELIEF_ACTION_CHECK.rst. Each condition compares
all legal alternatives on 128 paired particles, then requires a positive
mean-minus-2.5-standard-error margin before a separate 512-particle confirmation.
Every first-stage margin is nonpositive, so no confirmation is triggered.

====================  ==========  ===================  ================
Condition             Roots       Positive mean         Changed actions
====================  ==========  ===================  ================
Uniform               25          16                    0
Identified uniform    25          10                    0
History               25          9                     0
Identified history    25          14                    0
====================  ==========  ===================  ================

The experiment evaluates 372 alternative comparisons, taking 428.36 seconds
of recorded search time. It executes 7,802,486 hypothetical decisions,
573,452 hypothetical round endings and 95,232 completed simulation branches.
All four conditions use the same particle caps; realized work is recorded.
There are zero confirmation branches or new historical likelihood calls.

After all 100 choices are frozen, actual-source outcome assessment completes
25 distinct first actions with strict CPU continuations: 1,777 decisions and
145 round endings. Identical choices reuse these outcomes for 100 additional
lookups. Each condition therefore has exactly zero conditional win and point
change. These are completions of old roots, not 25 new full evaluation matches.
Source reconstruction adds 1,906 prefix transitions. No new deal is generated.

Independent verification checks every alternative's signed win/point mean,
standard error, ranking, adoption decision and work count, then replays all
recorded actual-completion actions with the independent reference simulator.
It passes, adding 953 prefix transitions and the 1,777 completion transitions,
with zero new neural calls. It does not claim independent neural recomputation
of every hypothetical continuation. The CUDA capability test separately checks
the actual rollout transition path against the reference through termination.

Artifacts under runs/prepared_belief_actions:

* Protocol: 61b1db8cc8c5ae97c2f658a718c280b81b7c5b00248a00a4b912f32a6d9a0faf.
* Choices: 4411743fcbb1fb2553e5cd70cac7894994b23eba0cdd2ffdf594aca0c3073b5b.
* Outcomes: 46385cc0e325c853f8eb4b9ca6f7f27c20a11b7bfd1e4b0f4226a7134eb4fcba.
* Audit: c39cdb170eaefd936c4ee35f827cdfb5d42f7b02d7458dafa93b445d7569e6c2.

The pre-run protocol binds 130 inputs. The audit binds 143, including the
independent verifier and supplementary native-file receipt. Native sources
also match the package aggregate in the pre-run bound source-study protocol.
The supplementary per-file receipt was recorded after ten roots, rather than
represented as a pre-run binding. No native source changed during execution.

Follow-up: separate screening from confirmation
-----------------------------------------------

PREPARED_BELIEF_CONFIRMATION_CHECK.rst defines a separate follow-up. It fixes
all preceding selected alternatives and confirms the 49 positive-mean cases
on new 512-particle samples. It removes only the first-stage margin screen;
adoption still requires a positive margin on the independent confirmation.
The preceding result and its criterion remain unchanged. This sample-splitting
diagnostic can expose missed candidates, but cannot by itself establish a new
general search algorithm or population strength improvement.


The follow-up is complete and independently verified. Of the 49 confirmations,
one changes an action: parent 2 / WP, case 0, in the history-only condition.
The confirmed mean win difference is +0.0478515625, with margin +0.0000686385.
It changes card 11 to card 47. Actual-source completion changes a draw (point
difference 0) into a loss (point difference -38). The other 99 root/condition
choices remain unchanged. Thus history-only has a conditional win-score sum
change of -0.5 and point-difference sum change of -38 across 25 roots; the other
three conditions have zero change. This single old-root outcome neither proves
a population loss nor supplies a strength gain.

Confirmation takes 95.56 recorded seconds, 3,816,893 hypothetical decisions,
282,000 hypothetical round endings and 50,176 completed simulation branches.
It reuses 25 already verified actual-source completions and adds one new
completion with 18 decisions and two round endings. Source reconstruction adds
1,906 transitions. No new full evaluation match, actual deal or historical
likelihood call is performed. The independent audit repeats 953 source-prefix
transitions and the 18 new completion transitions, with zero neural calls.

Artifacts under runs/prepared_belief_confirmation:

* Protocol: 698d8c4ab83376d6e59b9a25e672976fced14c646fa17745ea62aed7c3c3435a.
* Report: 7c7430802f9a61848bcffd46158d26b8d00c5bd223de40649020fa7c05be5819.
* Audit: a65cd98b73b8d56b95a9d94a26467cb81ed66eb9af947f4f7367e75f9830586b.
* The independent audit binds 153 inputs.

Neither action diagnostic supports promotion. The evidence separates improved
hand inference from the ability of this bounded rollout search to exploit it.
The unchanged human-play policy and initial-dealer behavior remain in place.
