Public hand revelation across rounds
===================================

Question
--------

The current published actor uses within-round history and match context. A
completed round's action history is not retained in its input. Could public
events from earlier rounds provide usable information about a fixed opponent?
This diagnostic counts an elementary opportunity for such inference; it does
not implement an opponent classifier or a stronger policy.

A round begins with eight hand cards, and stock draws are never added to a
hand. Therefore, after eight distinct opponent hand discards have been observed
in that round, their set is exactly the opponent's initial hand. This statement
uses the public discard actions, not the simulator's private hand or deck.
It becomes available only after the eighth discard, not at an earlier decision.
Capture choices and the viewer's own hand discards do not count.

Completed diagnostic
--------------------

Replay the already collected training prefixes from
runs/final_round_training_study/parent{1,2}/source.jsonl. These are 2,048 source
attempts on 512 shared deal streams, with 1,024 prefixes per parent. Retain
early terminations. The diagnostic never reads held-out evaluation records,
performs neural inference or chooses actions. The detector accepts only public
(round, phase, player, action) tuples. Full source endpoints are used only to
verify that the recorded action replay is unchanged.

Counts over the first seven rounds:

================  ========  ======================  =================
Parent/opponent   Prefixes  With a full revelation  Revealed rounds
================  ========  ======================  =================
1 / WP            512       446                     922
1 / Point         512       446                     1025
2 / WP            512       451                     941
2 / Point         512       450                     1094
================  ========  ======================  =================

The 3,982 revealed rounds occur within 1,793 prefixes. Multiple rounds, seats
and parents share deals; these are not 3,982 independent matches. First-round
revelation occurs in 134/131/143/128 prefixes in the same table order. The
artifact retains first-revelation timing and every revealed hand.

Replay work is 212,212 decisions, 14,101 round endings and 192 early completed
matches; it reuses the existing source trajectories and adds zero independent
deals. Two detector tests pass, zero skips, in0.15s. The diagnostic overlaps the
ongoing strict evaluation for about seven seconds and is not a latency test.

Interpretation and next check
-----------------------------

Full public hand revelation is common enough in this source set to investigate
retrospective opponent inference. The next check is whether a historical public
snapshot plus the subsequently revealed hand reconstructs the earlier opponent
actor input exactly, and whether the fixed WP/Point models then make distinguishable
predictions. A classifier must use this information only after its revelation
and retain it across rounds. No private hand may enter online inference.

The counts alone do not establish identification accuracy, additional
information over all other public features, useful opponent-specific responses,
or any match-win gain. Opponent modelling and learning from revealed cards have
prior art; no general novelty is claimed for this elementary observation.
The running fixed-prefix training/evaluation protocol is unchanged and does not
use this diagnostic. Any classifier or response-policy study must be separate.

Artifacts: scripts/check_public_hand_revelation.py,
tests/test_public_hand_revelation.py, runs/public_hand_revelation_tests.xml,
and runs/public_hand_revelation.json (input hashes and all records).
