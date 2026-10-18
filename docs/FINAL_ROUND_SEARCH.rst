Final-round card-search capability
=================================

The coverage audit found no final-round roots in the two completed Transformer
card/koi diagnostics. FinalRoundSearch provides an experimental adapter for
that missing card-choice setting. No new strength experiment is launched.

The adapter compares all legal alternatives at discard and either capture
phase in round eight of paper8. Outside those roots it retains the strict CPU
parent and existing public koi guard. Forced card decisions use the parent
without simulation. Terminal roots and other rule profiles are rejected.

Unknown cards are sampled with the existing observation-consistent uniform
proposal. Each particle receives an equal-prior WP/Point opponent label;
the label remains fixed across its two branches. This distribution omits
opponent-action likelihoods from both the current and earlier rounds. It is
an approximation, not a full-match posterior. Simulated continuations use the
fixed64 numerical policy for the parent and both opponent models. Actual root
fallbacks retain strict CPU inference.

Candidate selection, the configured standard-error margin and fresh
selected-action confirmation reuse the existing search implementation. No
threshold is selected from final-round outcomes. Every simulated branch ends
in the current round and reaches a terminal match result. The canonical game,
dealer rule and live first-round comparison remain unchanged.

CPU capability verification
---------------------------

The tests take archived legal positions, select late examples of each card
phase and seat, and set their round index to seven. These artificial contexts
test execution, not playing strength or final-round opportunity frequency.

Three tests pass, zero skips, in13.83s:

* all alternatives, independent selected-action confirmation and unchanged
  outputs after changing the real hidden cards while retaining observations;
* every paired transition and terminal point result against the independent
  Python engine, for all three card phases and both seats;
* strict fallback before the last round, public koi assistance at both seats,
  forced-card bypass and rejection of invalid profiles/terminal states.

Confirmation is forced only in the integration test to exercise that path.
It is not evidence of a naturally accepted action. Native CPU simulation and
fixed64 CPU continuation are covered. GPU integration of this adapter remains
to be checked after the current pilot finishes; GPU tests for the shared
fixed64 continuation alone do not replace that check.

Command::

  .venv/Scripts/python.exe -m pytest tests/test_final_round_search.py -q --basetemp runs/pytest_final_round_v2 --junitxml=runs/final_round_cpu_tests_v2.xml

The two development CPU test runs took13.14s and13.83s while the fixed64 pilot
was running. Thus pilot wall times include concurrent CPU activity and are not
isolated inference benchmarks. No additional GPU workload was launched.

A future prospective study must specify source matches, missing opportunities,
intervention frequency and both retained-parent controls. Its design will use
the complete current pilot, including unsuccessful outcomes and costs.
Rollout policy improvement and uniform hidden-world search are established
methods; this capability is not a new algorithm or a demonstrated win gain.
