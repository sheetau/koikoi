Public gate coverage after the closed uniform evaluation
=======================================================

This is a development diagnosis of the completed 8,192-source cohort, not a
new playing-strength experiment. It does not change that cohort's policy,
denominator, bounds or failed practical-gain criterion.

Replay and classification
-------------------------

``scripts/diagnose_uniform_gates.py`` replays the recorded decks and actions
without neural inference or fresh chance. It checks all final states and work
counts, and reproduces every original five-draw root's decision index and
complete state. Gate decisions use learner seat, public round/turn/phase and
the time of the first preceding-round singleton opponent identification.
Later revelations cannot qualify an earlier root.

Each replay accounts for 950,120 decisions, 63,678 round endings and 8,192
complete matches. These are replays of existing observations. Two complete
passes were performed: the initial diagnostic mistakenly counted an observation
feature as a legal-action mask; the corrected pass uses ``ReferenceEnv.legal``
and rejects a live root without a legal action. Only the corrected v2 report
is accepted. Source/root/reason counts agree across the two passes. Neither
pass changed the original study.

.. csv-table:: Disjoint source classification
   :header: Parent,Opponent,Five-draw root,Ended before final round,Final round without prior identity,Identified but ended before five-draw gate

   1,WP,936,133,303,676
   1,Point,971,221,252,604
   2,WP,981,158,282,627
   2,Point,990,224,238,596
   Total,,3878,736,1075,2503

.. csv-table:: Coverage if the first eligible gate is earlier
   :header: Parent,Opponent,Five draws,Six draws,Seven draws,Eight draws

   1,WP,936,1019,1187,1241
   1,Point,971,1035,1174,1224
   2,WP,981,1036,1186,1229
   2,Point,990,1071,1213,1260
   Total,,3878,4161,4760,4954

Every recorded root in this table has at least two legal actions. Six draws
add 283 previously missing roots; seven and eight add 882 and 1,076.
These counts do not establish recoverable gain: the parent may already win,
all alternatives may lose, or the larger tree may exceed the fixed budget.
An earlier first root also changes decisions on sources already covered at
five draws, so coverage alone cannot establish a stronger controller.

Corrected report: ``runs/uniform_gate_diagnosis_v2.json``.
SHA256: ``6d65d8bcfb73db3b19ce817ba63303558a2de6f33965afea5f4287921b19245c``.
It binds the completed result ledger and all 512 source-record files and
retains every diagnosed case. Reproduce into a new path with::

    .venv/Scripts/python.exe -m scripts.diagnose_uniform_gates --output runs/new_gate_diagnosis.json

Six-draw feasibility screen
--------------------------

Before planning, select the first previously missing source with a six-draw
root in each parent/opponent/seat group. The eight source indices are
1, 19, 88, 112, 237, 251, 322 and 738. Selection uses gate availability only,
not controller gains. All eight remain in the screen regardless of outcome.

``scripts/screen_six_draw_gate.py`` changes only the draw threshold from five
to six. It retains the two independent 2,000,000-transition / 500,000-node
phase caps, 1,000-hand limit, 4,096-row dynamics capacity, depth 32, symbolic
leaf rule, parents and opponents. Two bounded worker processes run at once.

Each fixture replays its full source with the actual actors and native
transition comparison, reproducing its actions, deals and final state. It
reconstructs the posterior at the earlier public root and performs capped
planning. An adopted value pair receives independent full Python evaluation
using the unchanged parent actor. The selection and program are saved before
actual-tail execution. Source/actor replay, support reconstruction, program
binding checks and parent/controller actual tails are then verified within
the fixture worker. This is implementation feasibility, not a new independent
population estimate or a separate-process strength audit.

Protocol: ``runs/six_draw_gate_screen/protocol.json``.
SHA256: ``a05bcb6b3497fc4ddfa185ec2ca66c8be987e4c6b1774f5235c86ceef1b3fada``.
The screen is complete; all eight fixtures passed. Two parents already had
value 1 (sources 1 and 237). Source 19 had parent and best candidate value 0.
The other five sources (88, 112, 251, 322 and 738) exhausted the physical
transition cap before valuing the parent. No policy was adopted. Every actual
parent/controller action trace and outcome agrees. No oracle value replay was
needed because there were no adoptions.

Planning used 10,248,697 physical decisions and 6,920,050 physical completions;
represented work was 1,338,963,814 decisions and 341,400,420 completions.
Source, native and source-audit replays each used 991 decisions, 64 round
endings and 8 matches. History/support audit each used 60,508 replay decisions.
Actual execution and replay each used 42 decisions across 16 tails. These are
separate work categories. There were no fresh chance draws. Concurrent worker
timings are operational measurements, not isolated latency benchmarks.

The complete artifact review rehashed every result and saved source/selection/
program binding, checked the frozen script and protocol, and compared every
actual pair. Summary SHA256:
``73223ce652b8aef306d8b1897839359c7d186d4dd8dd905abccb5824029e35a7``.
Review: ``runs/six_draw_gate_review.json``. Raising the gate alone did not
produce a feasible gain in this screen. Next address evaluation cost before
considering a broader gate. Any controller selected from these development
sources requires fresh prospective evaluation. Initial dealer and canonical
human play remain unchanged.
