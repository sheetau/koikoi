Public-certificate pruning before candidate construction
=======================================================

This isolated implementation leaves the complete parent prefix and all
unresolved local comparisons unchanged. At each complete frontier belief,
the existing paper8 public outcome certificates are checked before creating
candidate programs or querying their learned actors. A node is omitted only
when every world has a strict policy-independent win or loss certificate.
Its unchanged parent and every legal candidate then have the same weighted
terminal win score. Omission cannot discard a positive improvement.

The certificates use public captures, remaining turns and current match
points. No hidden hand, stock order or future event is an actor input. The
existing Python/C++ game transitions and score bounds are unchanged. All
uncertified nodes use the same candidate ordering, persistent evaluators and
physical-transition allowances. Zero-transition comparisons disappear from
the saved patch; the executable policy and all positive selections remain.
An explicit sidecar lists public keys and covered winning/losing masses.

Thirteen tests cover both seats, certain wins/losses, mixed unresolved roots,
ordinary and named continuations, and refused prefixes. All remaining
receipts, selections and planning transitions match the unpruned version;
independent Python verification checks full root-value decomposition.

The focused timing study uses original development fixture 2, whose 1,304
frontier nodes are all publicly certified. It was chosen by that structural
property, not a measured speedup. It is a special-case benchmark, not a
representative latency or strength estimate. Three families are measured:
ordinary parent-default, plus the other parent, plus WP and Point. For each,
old/new then new/old runs use fresh dynamics and actor caches, serially after
the earlier development and receipt benchmarks finish. Model loading,
serialization and independent audits are outside timed planning. Frozen
source/input identities and every timing remain in the study artifact.

The goal is to eliminate redundant preparation with no policy change. These
measurements do not convert reused development sources into fresh matches,
and do not claim that all search roots are publicly solved.
