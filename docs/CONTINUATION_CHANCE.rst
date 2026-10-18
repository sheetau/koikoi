Renewing chance after public identification
===========================================

Implementation boundary
-----------------------

``scripts/continuation_chance.py`` provides an isolated CPU constructor for
fresh training deals at rounds2--8 of an unfinished paper8 match. Its immutable
context contains only round,dealer and the two point totals. Learner seat and
the publicly inferred opponent code remain caller-owned metadata;the constructor
neither derives nor changes them. The separate trainer retains their audited
provenance and must never substitute the simulator's true opponent label.

``PublicContinuationContext.from_root`` reads those public fields from an
already audited fresh-round root. It checks profile,phase,turn,round and point
validity. It does not inspect the root's cards,stock,seed,episode or history.
The new constructor accepts explicit seed/episode keys and uses the unchanged
reference deal routine,including the existing rejection of four same-month
cards in either hand or the field. It initializes fresh private simulator state,
preserves points/dealer/round and resets within-round history. Policies must
continue to use the existing observation builders,not this packed state.

This changes no initial-dealer mechanism and no scoring or transition rule.
The current fixed-root trainer and human-play UI remain unchanged.

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

Checkpointable root sampling
----------------------------

``scripts/renewed_root_sampler.py`` selects original root records with their
existing multiplicity and assigns a fresh seed to every draw,including repeated
root indices within a batch. It retains learner seat,simulator routing label
and public memory as distinct metadata. An explicit finite seed allocation
prevents silent overflow or budget extension. A prospective protocol must
allocate a range separate from evaluation;the sampler does not choose that
experimental range automatically.

Snapshots contain source/bank identities,allocation,cursor,batch lengths and
the CPU root-index RNG. Restore reconstructs the index RNG from the recorded
draw lengths and rejects inconsistent cursors,RNGs,inputs or allocations.
It does not regenerate deals or replay policies during that validation.
Two CPU tests pass,0 skips,in1.51s,including exact serialized restart,paired
sampling,unchanged internal metadata after caller mutation,distinct routing
and public-memory values,unique per-draw chance keys,and atomic rejection of
invalid requests/snapshots. Final XML ``runs/renewed_root_sampler_metadata_tests.xml``
has SHA256 9af1d59cd7035cca2de4825207879ae20360bb03d41a7d8dddd5f5f9d46759d1.

Learner integration on CPU
-------------------------

``scripts/renewed_continuation.py`` adds an isolated terminal-return learner
with explicit chance-key allocation and format ``renewed_public_continuation_v1``.
The existing conditioning/PPO code and complete-continuation semantics are
retained. Its separate collection method replaces root copying with renewed
deals;no frozen study source is edited. Sampler cursor,fresh-deal count and
conditional-completion count must agree before collection or saving.

Checkpoints retain actor,critic,optimizer,frozen parent/opponents,source bank,
public memory,all RNG states and the complete chance sampler. Restore validates
source/bank identities and the chance schedule before resuming updates.

Two CPU tests pass,0 skips,one CUDA case deselected,in46.40s. Reference replay
checks every transition through multiple rounds,including opponent-turn match
endings and learner reward signs. Public/masked initial models and warmup
states/schedules agree. Resuming the next policy update reproduces metrics,
model,optimizer and next chance keys exactly;frozen actors stay unchanged.
Corrupt chance accounting and memory changes are rejected without advancing
the sampler. Final XML ``runs/renewed_continuation_cpu_final_tests.xml``:
19b7d3eccf220a3289fdfc22ed8041225b3e2b08650a2caed896a00d8e02621f.
The earlier CPU run also passed before import-order cleanup and the added
counter-mismatch assertion;both runs are correctness work,not strength data.

Online loading and queued production capability
-----------------------------------------------

``scripts/renewed_continuation_match.py`` loads the new checkpoint format into
the existing public stopping-rule adapter. It retains only actor weights,
configuration and public-event tracking. Training banks,optimizer state,chance
schedules and simulator routing labels do not enter online decisions. Ambiguous
identification falls back to the frozen parent for the entire match.

Three online tests pass,0 skips,in8.93s. Initial public/masked actors reproduce
all parent actions/outcomes in both seats against both published candidates;
independent reference replay verifies activation timing with both dealer
alignments. Ambiguous fallback,match reset,sequential lifecycle and wrong-format
or missing-candidate rejection pass. XML ``runs/renewed_continuation_match_tests.xml``:
a0009da4d30932a9cbc52ecd3d0004b1672baf40e6a494e3c234927f8e4f0d37.

``scripts/check_renewed_continuation.py`` requires the completed multi-round
comparison and its passed independent audit before executing. Its separate
capability uses all855 parent1 roots,128 environments,horizon448,three epochs,
minibatch512 and three updates per public/masked arm. The first update is
critic-only;the third is replayed from the second saved checkpoint. Full
model/optimizer/frozen actors/sampler/RNG equality,matched initial/warmup and
finite Adam states are required. The scheduled384 chance keys are shared
between arms;768 logical constructions and256 replay constructions are counted
separately from trajectory decisions and full matches.

Guard/equality tests pass two,0 skips,in1.39s. XML
``runs/renewed_continuation_capability_tests.xml``:
20b5c51f9a27d98aa150b0683c19307128a709640ffd73d0ba255af5085deaf7.
A copied bank path was corrected before execution;the launcher checks literal
input paths before waiting. It pins240 inputs,all verified,and runs the CUDA
unit test followed by production capability only after the current independent
audit succeeds. Its8h observation deadline does not kill the existing study.
See STATUS for the live PID and receipt;do not start a duplicate invocation.

CUDA validation,production capability and a prospective strength comparison
remain unperformed. The current study must finish unchanged. These correctness
checks establish no playing gain,absence of overfitting or new general learning
principle. The proposed empirical question is whether fresh chance improves
generalization at the same number of conditional training completions.


Prospective evaluation analysis
-------------------------------

The separate specification ``RENEWED_CONTINUATION_STUDY.rst`` fixes a five-policy
comparison before current multi-round evaluation begins. Its original design
hash remains unchanged. ``scripts/renewed_continuation_analysis.py`` implements
the complete40,960-record grid,all ten pairwise comparisons and the interaction
between chance renewal and public memory. Each comparison retains both parents,
the equal-opponent mixture and individual opponents,with win and point outcomes.
Wins are computed per actual match before seat averaging. Bootstrap resamples
paired deals independently by opponent,stratifying the equal-weight mixture.

Five tests pass,0 skips,in0.47s. Synthetic fixtures distinguish strength from
public-memory benefit and interaction,and reject a single-parent improvement,
positive points without wins,or an opponent-specific gain canceled in the mixture.
An independent scalar bootstrap with correlated contrasts reproduces the vector
calculation across chunk boundaries. Full-grid parsing rejects duplicate/missing
cells,wrong seeds,invalid identities and malformed point totals. These are
synthetic records,not simulator matches. The module has no implicit execution
entry point;the future runner/auditor must enforce completed provenance and
independently replay actual outcomes before reporting strength.
