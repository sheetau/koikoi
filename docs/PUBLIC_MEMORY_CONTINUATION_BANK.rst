Public identification continuation roots
=======================================

Purpose
-------

The current public-memory strength comparison activates its response in round8.
The completed training-history diagnostic found earlier opportunities: a public
identification can become available several rounds before that response starts.
This artifact materializes those natural boundaries for a possible subsequent
continuation learner. It does not train a policy or change the current study.

Construction
------------

``scripts/build_public_memory_continuation_bank.py`` consumes the hash-checked
activation diagnostic, its independently audited public identification records,
and the original frozen-parent training prefixes. For each of all2,048 prefixes,
it copies the first fresh-round boundary strictly after the identifying round
and no earlier than the public availability timestamp. Subsequent source actions
are replayed only to verify the retained endpoint and counters. The copied root
does not depend on those later actions.

The two banks contain855 and876 roots. Each row includes an int32 simulator
restart state, learner seat, frozen-opponent routing label, and a separate public
candidate code. The code is joined from public model elimination, not from the
routing label. Packed restart states contain private simulator data and must
never be passed directly to a policy; future learners must use the established
observation builder. The ledger retains every source attempt, including cases
without a root. No unresolved case receives a guessed identity.

Counts and verification
-----------------------

================= ======== =====
Source group      Attempts Roots
================= ======== =====
Parent1 / WP      512      425
Parent1 / Point   512      430
Parent2 / WP      512      434
Parent2 / Point   512      442
================= ======== =====

The roots span rounds2--8, with counts463,387,309,199,138,146,89. The317 absent
roots comprise302 unresolved histories and15 identified histories without a
later live boundary. All512 shared training deal streams remain the same.

Two tests pass with zero skips in1.46s. They check natural boundary states,
changed-suffix independence, future/absent public identification, later-round
activation, corrupt source endpoints/counters, and post-terminal actions.
Fixtures generate three small full matches; these are correctness fixtures,
not strength observations. The materialization replays212,212 source decisions,
14,101 round endings and192 earlier full completions. The remaining prefixes
are incomplete matches. This adds zero neural predictions, training updates or
independent deals. The short CPU replay overlapped the ongoing comparison;
neither execution is an isolated latency measurement.

Artifact directory: ``runs/public_memory_continuation_bank``.
Report SHA256:
``f2e49203737b4b51e54ab39cd5f73fc669e47d24c68c10081226e224f2556120``.
All37 input and two output hashes reverify. Parent1 bank SHA256:
``3dda3876298e8967eb6b375a8e761a8fa25293aa8db5aadb484635a301be3be8``;
parent2:
``ecbc42afe640852213493617c2fa2fac18e2c5de9fe802bfa79effd817d8f0c4``.
The report is a construction receipt, not an independent strength audit.

Next use
--------

Finish the current round8 comparison and independent audit before defining a
new strength study. A possible continuation learner would keep the parent up
to this public stopping rule and train through all remaining rounds. Its masked
control must share the same stopping rule and roots. Histories with no public
identification retain the parent throughout. Such a learner needs complete
multi-round returns and separate conditional-completion accounting; the existing
final-round trainer's one-round validation and completion assertion cannot be
reused unchanged. No such learner, strength result or deployment change is
claimed by this artifact.
