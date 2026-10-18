Fixed-size rollout inference
===========================

Observed failure
----------------

The fixed midround comparison stopped rather than accepting unequal repeated
baseline endpoints. Its producer exited1 at parent2 case9, during conditional
terminal references. One of512 paired baseline point differences changed from
-16 to-22 across alternative batches. Six completed parent1 rows remain local;
there is no complete study, final audit or strength conclusion.

A replay of those archived particles isolates an earlier action divergence at
chunk4,continuation step4,particle229. Its complete baseline state and native
published features are identical, but dynamic WP discard batches contain446
versus501 rows. The first execution ties cards18/32 at6.634164810180664 and
chooses18. The other gives32 a1.430511474609375e-6 advantage and chooses32.
Strict CPU and singleton CUDA both choose32 on that state. This is measured
batch-dependent floating-point behavior, not a game-transition discrepancy.
The isolated earlier action divergence is not claimed to be the sole cause of
the original particle403 endpoint difference.

Execution change
----------------

scripts/fixed_microbatch_continuation.py invokes every phase head with exactly64
rows, padding its last partial call with zero features. It retains observation-
only inputs, existing model/seat routing, legal masks and the public guard.
Dropout stays disabled. No original sampler, strict CPU likelihood, root policy,
transition or previously bound study source changes. The CPU and CUDA tests
check all three actors, every live phase, reversed order and requested sizes
1,63,64,65,129. The saved differing-baseline fixture also agrees in both devices.

This is an explicitly different numerical continuation policy. On the isolated
near-tie both fixed64 batches choose18, whereas strict CPU chooses32. Tests do
not establish a universal floating-point guarantee or equality to strict CPU
inference. Padding and fixed shape are execution controls, not a new learning
algorithm or playing-strength contribution.

Fixed trajectory check
----------------------

Use only archived parent2 case9,all three alternative cards,and the original
eight512-particle conditional reference chunks seeded83310010. Keep every
common-baseline assertion. Record all24 comparisons and actual terminal work
under fixed64. This is a reproduction/capability diagnostic, not completion of
the failed comparison with a changed policy. No new source deal, table fitting,
threshold tuning or model promotion. Existing round tables were produced with
dynamic batching; their accuracy measurements must not be relabeled as fixed64
calibration. A subsequent strength study needs an explicit consistent inference
contract and fresh evaluation protocol.
