Within-round retention of policy inputs and actions
==================================================

The public-seat observer features retain earlier policy inputs and chosen
actions on all 3,764 chronological pairs in the saved diagnostic trajectories.
This includes 1,968 pairs whose later decision belongs to the opponent. The
construction connects the feature design to the fixed-observer return argument:
the information used by an earlier policy decision remains available after
the turn changes.

Construction
------------

``reconstruct_past_actor`` receives one later 562-element observer row and the
public round, turn and phase of the requested earlier decision. It recovers
the observer's initial hand from the remaining hand and that player's recorded
hand plays. Starting from the recorded initial field, it replays the public
played cards, capture choices and continuations up to the requested address.
It then decodes the earlier 300 x 50 actor input. No simulator state, hidden
opponent hand, stock order, policy parameters or earlier observation is passed
to the decoder.

``reconstruct_past_input_action`` also recovers the earlier chosen action from
the corresponding public timestamp. A prior koi decision with a successor in
the same round must have chosen continuation. A stop ends the round and is
outside these chronological pairs.

Both endpoints must belong to the same played round. The caller checks match
and round boundaries; matching displayed round numbers alone is insufficient
because those numbers repeat between matches. The decoder covers paper8 and
does not reconstruct earlier-round history or the complete earlier critic row.

Validation
----------

The checks reuse 256 saved source positions per parent. Every ordered pair in
the same played round is compared with the native earlier actor input and the
original action recorded during collection. The source inputs already passed
the separate upstream feature-parity audit. All comparisons are exact.

============================== ======== ======== =====
Check                          Parent 1 Parent 2 Total
============================== ======== ======== =====
Earlier/later input-action pairs    1864     1900  3764
Later viewer is not acting           977      991  1968
Earlier discard decisions           1659     1600  3259
Earlier hand-capture decisions        93      193   286
Earlier draw-capture decisions        45       37    82
Earlier continuation decisions        67       70   137
============================== ======== ======== =====

The longest checked separation is 19 decisions. The action decoder rejects
512 requests for actions that have not yet occurred and 512 requests with a
different round number. The saved states and all bound inputs remain unchanged.
This work adds zero simulator decisions, round endings, matches, branches,
policy forward/backward passes and training updates.

Successful commands (each requires a fresh output path)::

   .venv/Scripts/python.exe -m scripts.check_observer_temporal_reconstruction --output runs/observer_temporal_reconstruction/audit.json
   .venv/Scripts/python.exe -m scripts.check_observer_score_reconstruction --output runs/observer_score_reconstruction/audit.json

The temporal receipt SHA256 is
``534146a5f301b9fdb617a81fee1493ba8848c5c00d740bc91c2873ff00ae649d``.
The input/action receipt SHA256 is
``be39d3c08a47e856aba08496a7ace57a93932f7cc52c5190434433c320d80851``.
Both scripts pass Ruff.

Interpretation
--------------

For a fixed differentiable policy, its score at an earlier decision is a
function of that decision's input and action. Recovering those arguments
therefore also determines that score on the checked pairs, without evaluating
a policy gradient. This is stronger than recovering only the current actor
input, which does not address what survives to a later bootstrap.

The finite probe set establishes measured within-round information retention.
It does not establish universal or across-round retention, learned value
accuracy, unbiased PPO or improved play. The fixed public/masked comparison
tests the strength effect separately. Its protocol and evaluation conditions
remain unchanged; no interim strength summaries were inspected for this work.
