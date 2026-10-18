Complete continuations after public identification
=================================================

Purpose and implementation
--------------------------

``scripts/public_memory_continuation.py`` extends the isolated public-memory
training capability from one final round to all rounds remaining after a public
identification. The previously materialized banks provide natural starts in
rounds2--8; see PUBLIC_MEMORY_CONTINUATION_BANK.rst. This is capability work,
not a new strength study or a replacement for the completed round8 comparison.

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

Online activation verification
------------------------------

``scripts/public_memory_continuation_match.py`` adds an explicit public-counter
lifecycle and a strict full-match evaluator. The caller announces every decision,
including opponent decisions, before acting and supplies erased-hand opponent
snapshots after their actions. The first fresh round after public identification
activates the continuation, even when the opponent is its dealer. Memory is then
fixed. Unresolved histories retain the original parent throughout; no guessed
identity activates the continuation. Duplicate/missing decision addresses and
cross-match events are rejected. This adapter is not deployed in human play.

Two tests pass with zero skips in8.25s,with three expected Transformer warnings.
Four seat/opponent fixtures compare both public and masked initial policies with
the original parent:all actions,terminal scores,rounds and counts match. An
independent replay locates each first eligible boundary;both dealer alignments
are exercised and activation occurs before the final round. Identical candidate
predictors exercise persistent ambiguity and full parent fallback. Match reset
and malformed lifecycle checks also pass. These are correctness fixtures,not
new independent strength observations.

Command::

  .venv/Scripts/python.exe -m pytest tests/test_public_memory_continuation_match.py -q --junitxml=runs/public_memory_continuation_match_tests.xml

JUnit SHA256:
``66d1bf3520e6ce68d2e180c48d7dc0f0c58dc4cc18dcc9384efcf80f65e8bddb``.
Adapter SHA256:
``f296e9321d552defd660fda1c07f264f498d9d220281d7fcc66e9a9550cd0724``.
Test SHA256:
``2dadbd11f2bc52a5a3b39a0228c7cf6f0268869a3d79f4dd009952651a07358c``.
Ruff passes and all126 active-study inputs still verify. The short CPU fixture
run overlapped evaluation and is not an isolated timing benchmark.

Production-shape capability complete
-----------------------------------

``scripts/check_public_memory_continuation.py`` requires the completed round8
study,its successful independent audit,unchanged evidence and a passed real CUDA
test. It uses the855 audited parent1 training roots with128 environments,three
PPO epochs and minibatches of512. Both public and masked arms execute one
critic-only and two policy updates. Each third update is then replayed from the
second checkpoint and compared exactly,including model,optimizer,bank and RNGs.
Logical work and restart replay work are reported separately. Initial and
warmup endpoints must match between treatments. Every Adam state must be finite
and its step count must agree with the phase counters.

Two capability-guard tests pass,0 skipped,1.39s. They reject unfinished/missing
audits and changed evidence,and detect optimizer or RNG differences in endpoint
comparison. XML SHA256:
``e1b8c8e1a25c92923cf0c59bc22df9fe9e209e1fa6b05fd060d6fb06d8c2b06b``.
These tests do not execute CUDA or the production-shape run.

Queue76467 and both child commands finish with exit0. The CUDA reference/resume
test passes1,zero skips,two unrelated cases deselected,in6.55s. XML SHA256:
``fcdb9fa731551eef80cfc6cacb7735b34da4ecfe7df38c1de2a83dfbd1f71744``.

The real capability passes with96 finite Adam states per arm,exact initial and
warmup endpoint equality,and exact resumed third updates. Each arm executes
28,782 logical decisions (14,274 learner/14,508 opponent),384 conditional
completions and1,927 conditional round endings. Combined logical work is57,564
decisions/768 completions/3,854 round endings. Exact-restart replay adds17,980
decisions/256 completions/1,210 round endings separately. All ordinary full-match
and round counters remain zero. Public input-column L1 becomes0.07289130613207817;
masked columns remain exactly zero. No playing-strength evaluation is performed.

Capability audit SHA256:
``1ba645ade180878ebf5e8803211abe46e1b1cc8161c160100b733b5fb0647bee``.
All214 input/output hashes reverify. Completion receipt SHA256:
``0c3be9745cc9c0e2e74a781ab304431403bfd06056a7877d80458f9efc7ba0d5``.
Artifacts:``runs/public_memory_continuation_capability``.
Do not restart completed queue76467 or overwrite its artifacts.

Remaining work
--------------

CPU/CUDA,production-shape and online-adapter capability checks are complete.
The round8 strength comparison and independent audit are now complete with no
replicated gain (PUBLIC_MEMORY_RESULTS.rst). A new multi-round comparison
requires a separately frozen training/evaluation protocol. There is no new strength claim,
model promotion, dealer change or use of the reserved101M confirmation deals.
