Renewing chance after public identification
===========================================

Implementation boundary
-----------------------

``scripts/continuation_chance.py`` provides an isolated CPU constructor for
fresh training deals at rounds2--8 of an unfinished paper8 match. Its immutable
context contains only round,dealer and the two point totals. Learner seat and
the publicly inferred opponent code remain caller-owned metadata;the constructor
neither derives nor changes them. A future trainer must retain their audited
provenance and never substitute the simulator's true opponent label.

``PublicContinuationContext.from_root`` reads those public fields from an
already audited fresh-round root. It checks profile,phase,turn,round and point
validity. It does not inspect the root's cards,stock,seed,episode or history.
The new constructor accepts explicit seed/episode keys and uses the unchanged
reference deal routine,including the existing rejection of four same-month
cards in either hand or the field. It initializes fresh private simulator state,
preserves points/dealer/round and resets within-round history. Policies must
continue to use the existing observation builders,not this packed state.

This changes no initial-dealer mechanism and no scoring or transition rule.
It is not connected to the current fixed-root trainer,online agent or UI.

Verified CPU scope
------------------

Three tests pass with zero skips in1.37s. They compare reconstructed starts to
natural C++ CPU round boundaries across all rounds2--8,verify independence from
old cards and future keys,reject invalid public contexts,and compare complete
multi-round continuations against the Python reference at every state,observation,
mask and transition. Eight full natural CPU fixture matches and four conditional
fixture continuations are correctness checks,not strength samples. The first
test run also passed before two lint-required exception types were corrected;
both test executions remain separate work. Final XML:
``runs/continuation_chance_cpu_final_tests.xml``,
SHA256 3f76bb94d3e06e59ee7b3bf2e8c0106b96022766f4924600742b54039e33cde9.

``scripts/check_continuation_chance_bank.py`` checks all855/876 stored roots.
Using each original chance key reconstructs every packed state exactly.
Using separate capability keys creates855/876 different legal permutations,
preserves every public context and leaves the source arrays unchanged. This
constructs1,731 fresh conditional deals and reconstructs1,731 old deals;it adds
zero simulator decisions,policy predictions or full matches. Receipt:
``runs/continuation_chance_bank_capability.json``,
SHA256 f18357c52be82afa71b6f5d14869d6616c7cc60bcbf41b08b67bd6c482961cfc.

Outstanding work
----------------

CUDA transition checks,learner integration,checkpointed chance-key scheduling,
matched controls and a prospective strength comparison remain unperformed.
The current study must finish unchanged. Its results may inform whether to
run a separate renewal experiment;this capability establishes no playing gain,
absence of overfitting or new general learning principle.
