Bounded frontier at the next learner discard
===========================================

The fixed three-draw frontier recovered four improvements but exhausted its
prefix row allowance on sources 16 and 168. A draw-count boundary can miss a
learner decision: when draw capture and koi are forced, the next decision occurs
on a later turn. This makes some prefixes deeper than the intended split.

At the original five-draw gate, instead split at the learner's next discard.
This has an equivalent draw threshold determined solely by the root turn:
four draws for turn 10's draw-capture/koi phases, and three for turn 11's
discard/hand-capture phases. The threshold is 14 minus the root turn. The
remaining public-history grouping, local solver and executor are unchanged.

Prefix bound
------------

Assume the existing complete deferred belief with one identified deterministic
opponent. At a turn-10 entry, the opponent has three remaining cards among
sixteen unknown cards and the stock contains thirteen cards. There are at most
C(16,3) = 560 current hands. Only the opponent's next draw precedes the learner's
next discard, yielding at most 560 * 13 = 7,280 prefix rows. At a turn-11 entry,
the opponent has two cards among fifteen unknown cards, giving at most
C(15,2) = 105 hands. Two new draws precede the next learner discard, yielding
at most 105 * 13 * 12 = 16,380 rows. Terminal paths only reduce these counts.

Each player turn contains at most four exposed decisions: discard, hand
capture, draw capture and koi. Therefore the transition bound is the row
bound times the remaining decisions in the learner's current turn plus four
opponent decisions. The largest bound is 8 * 16,380 = 131,040. Both bounds
fit the existing 65,536-row and 250,000-transition prefix limits. Chance has
at most two future draws and the actors are deterministic, so each current
hand and draw prefix contributes at most one live path. Forced transitions
cannot skip the next discard unless the round terminates.

The bound assumes correctly enumerated unique current-hand hypotheses and
this exact entry scope; it does not apply to arbitrary roots or opponent
mixtures. scripts/next_discard_frontier.py checks the public scope and computes
the bound. Twenty-one tests cover all four entry phases and both seats,
hidden-card reassignment invariance, rejected scopes, and complete native
prefixes checked against independent Python posterior reconstruction.

Fixed comparison
----------------

Retain all eight previous development cases. Derive the threshold from each
root's public turn before planning. This changes the threshold for sources 16
and 168; the other six must reproduce their complete saved prefixes, local
programs, intervals and work exactly. Reexecute every case and count all new
work. Keep the shared two-million-transition budget, 250,000 prefix ceiling,
50,000 per local phase, 25,000 local nodes/tasks, capacity 4,096 and row cap
65,536. One bounded worker; no budget enlargement or parameter grid.

Require prefix completion within the derived bounds, every frontier node to
be a discard at root turn plus two, independent Python reconstruction of all
conditional beliefs, full local-pair replay for every adoption, and actual
suffix replay. Save programs before oracle work. Import original fixed work
separately, as in the preceding development screen. Neither preservation nor
new conditional gains from these reused sources establish fresh population
strength. A subsequent selected hybrid needs integration and prospective tests.
