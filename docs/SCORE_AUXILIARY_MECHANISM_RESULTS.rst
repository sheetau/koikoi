Score auxiliary learning: fixed-state diagnostics
================================================

All six endpoints were scored on the previously fixed 25 validation groups:
3,258 public-input rows, including 3,045 with multiple legal actions and 213 forced
choices. Phase populations are 2,578 discard, 122 hand capture, 130 draw capture and
215 continuation decisions among unforced rows. No new matches were generated.

The score and outcome-only models disagree on 331/3,045 (10.87%) states at seed 1
and 249/3,045 (8.18%) at seed 2. Score supervision therefore changes deployed greedy
actions. This observation does not determine whether those changes improve play;
the fixed 131,072-match strength comparison is running separately.

At common temperature 0.05, score models have lower mean action entropy than
outcome-only models in discard and both capture phases at both seeds. The
continuation phase has opposite entropy differences across seeds. Relative to
original Point, score models switch 16 and 12 continuation choices to stopping,
and switch 1 and 0 stopping choices to continuation. These counts describe the
fixed input population, not the models' on-policy state distribution.

Greedy action disagreements
--------------------------

Columns follow discard, hand capture, draw capture, continuation. Forced rows
have a single legal action and cannot contribute disagreements.

.. csv-table:: Paired endpoint disagreements
   :header: "Pair", "Discard", "Hand", "Draw", "Continue", "Total"

   seed1_none__seed1_win,220,10,9,10,249
   seed1_none__seed1_score,278,10,9,13,310
   seed1_win__seed1_score,288,14,14,15,331
   seed2_none__seed2_win,202,9,11,8,230
   seed2_none__seed2_score,243,9,10,4,266
   seed2_win__seed2_score,218,6,13,12,249

.. csv-table:: Changes from original Point and continuation direction
   :header: "Model", "Discard", "Hand", "Draw", "Continue", "Stop to continue", "Continue to stop"

   point,0,0,0,0,0,0
   seed1_none,395,5,3,16,5,11
   seed1_win,435,9,10,14,5,9
   seed1_score,358,11,10,17,1,16
   seed2_none,375,8,6,12,2,10
   seed2_win,394,11,9,12,4,8
   seed2_score,354,11,8,12,0,12

.. csv-table:: Mean unforced action entropy at divisor 0.05 (nats)
   :header: "Model", "Discard", "Hand", "Draw", "Continue"

   point,0.241195,0.242528,0.210400,0.019757
   seed1_none,0.370252,0.273333,0.225555,0.012094
   seed1_win,0.363575,0.249385,0.213232,0.023741
   seed1_score,0.330942,0.230121,0.187208,0.026637
   seed2_none,0.371331,0.280212,0.221532,0.019075
   seed2_win,0.373478,0.234910,0.235956,0.025874
   seed2_score,0.354702,0.232041,0.190425,0.010096

Scope and provenance
--------------------

The report retains all 100 phase-level training-loss summaries for every arm.
These are training losses on changing self-play collections, not held-out
prediction accuracy. Their comparison cannot establish causal mediation.
The diagnostics use only public/own-card inputs; terminal-margin labels were
used during training. The original Point scores come from the fixed cached
corpus. The six endpoints require 18,270 singleton neural calls on CPU with two threads. New matches,
simulator steps and training updates are all zero.

Commands executed once after the full training audit passed::

    .venv/Scripts/python.exe -m scripts.report_score_auxiliary_recovery prepare
    .venv/Scripts/python.exe -m scripts.report_score_auxiliary_recovery run
    .venv/Scripts/python.exe -m scripts.report_score_auxiliary_recovery audit

Preparation session 65413, scoring 1612 and independent audit 1333 all exited 0.
The independent audit checks scalar statistics, populations, legal masks,
all training summaries and input hashes; it does not repeat neural inference.
It binds 1,160 inputs. Strength, visitation and held-out prediction are outside
its scope. No endpoint selection, promotion or additional training follows
from these diagnostics.

Artifacts in runs/score_auxiliary_mechanism (SHA256):

* protocol.json: 889c2225555e8723040c73d7415a2ccd57979cb1091d17630b8de5eb9c05d1aa
* scores.npz: 76c454c3cf99d653921b1c2e262989422714d7398a0606ea38ac9f9739c5b76a
* report.json: a291e87f5505f26679c02927989bb8ad6bcef287a4d25a58e4c485fd9f9d8fd5
* independent_audit.json: d619258403def4da3aa71cbc3b02b45179abe964a73c6ccb70e802da2ea5ab52

Full training audit: runs/score_auxiliary_recovery/training_audit.json,
SHA256 08ae5b97cfcc885e9835dfaf33b79552110f21cda440d9d643667bfe34b28e8f.
This completed diagnostic does not complete the stronger-play objective.
