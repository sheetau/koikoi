Complete continuations after public identification
=================================================

Purpose and implementation
--------------------------

``scripts/public_memory_continuation.py`` extends the isolated public-memory
training capability from one final round to all rounds remaining after a public
identification. The previously materialized banks provide natural starts in
rounds2--8; see PUBLIC_MEMORY_CONTINUATION_BANK.rst. This is capability work,
not a new strength study or a replacement for the active round8 comparison.

``ContinuationConfig`` retains the public/masked memory treatment and uses a
448-decision safety bound. ``ContinuationTrainer`` reuses the existing public
conditioning, original-parent initialization and PPO updates. Its independent
constructor accepts fresh-round paper8 roots in rounds2--8, requires a public
singleton code, and retains the original frozen prefix and opponents. Private
packed states restart the simulator only. Actor feature rows2/3 carry the public
code on learner turns; opponent inputs remain unconditioned and the critic
receives the existing observation only. Both treatments initialize those actor
columns to zero.

Every sampled root runs to actual match termination. Each learner decision
receives that lane's terminal win/draw/loss reward minus its own critic value
as the unnormalized advantage. No opponent-observation bootstrap, intermediate
round reward or unfinished trajectory is substituted. Completed lanes stay
frozen. Per-lane round-ending counts must equal terminal round minus initial
round plus one. The safety bound is an error condition, not a truncation rule.

Conditional completions and round endings are counted separately; ordinary
full-match and round counters remain zero because collection starts mid-match.
The checkpoint format ``public_memory_continuation_v1`` stores the full bank,
public codes, parent/opponent snapshots, optimizer, counters and RNG states.
Restore verifies implementation, runtime, bank and model identities.

The final-round trainer remains unchanged and retains its own format and
exactly-one-round completion assertion. Source duplication of its collection
and serialization is deliberate while that implementation is bound to a live
comparison; only the separate continuation module has the broader contract.

CPU verification
----------------

Command::

  .venv/Scripts/python.exe -m pytest tests/test_public_memory_continuation.py -k "not cuda" -q --junitxml=runs/public_memory_continuation_cpu_tests.xml

Result:2 passed,0 skipped,1 CUDA case deliberately deselected,38.63s;two expected
upstream Transformer warnings. Runner36592 terminated with exit0. The test uses
natural round2/4/6/8 fixture roots and both learner seats. Every native transition
and endpoint in a complete warmup batch matches the independent Python reference,
including multiple round boundaries and opponent-turn match termination. All
learner return targets equal independent final-score signs.

Public and masked models have identical initial parameters, sampled roots,
counters and complete critic-only warmup states. A subsequent policy update
changes public input columns, while masked columns remain zero. An exact resumed
update reproduces metrics, model and all optimizer state; frozen parent/opponent
snapshots remain unchanged. Tests also reject pre-identification rounds,
bankrupt roots, missing public identities and mutated memory banks.

JUnit SHA256:
``146137be1184f58c3e80027830468fcf5e1fffbb8897a80d2b5a96a97ef2c627``.
Implementation SHA256:
``5cf142f29bea2f8dc7ece9bb27c47145e429fd7f1f13b47ee837f24d26768b49``.
Test source SHA256:
``9bfaaa2b75cff38ba95078d7ebb716ea620cb5e7868e361882fc0cfb52008c40``.
Ruff passes. CPU checks overlapped the running strict evaluation; these are not
isolated latency measurements. All126 live-comparison input hashes still match.

Remaining work
--------------

CUDA verification and a production-shape capability run have not executed.
A full-match inference adapter must apply the same public stopping rule and
retain the parent when no identification becomes available. A new comparison
requires a separately frozen training/evaluation protocol after the current
round8 comparison and independent audit finish. There is no new strength claim,
model promotion, dealer change or use of the reserved101M confirmation deals.
