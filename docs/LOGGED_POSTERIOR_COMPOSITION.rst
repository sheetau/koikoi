From independent round deals to the controller's hand posterior
==============================================================

The logged source and the existing posterior can be composed under independent,
uniform valid round deals and the pinned deterministic opponent class. This
connects exact local advantages to expected match improvement under that law.
The fixed-fixture checks below test the implementation connection; they are not
new strength samples or an empirical proof of the entropy source's independence.

Conditioning argument
--------------------

Fix the parent, opponent and learner seat. At each round start, sample a uniform
permutation of the 48 cards conditional on the existing valid_deal rule: none
of the two initial hands or the initial field contains all four cards of a
month. Fresh independent entropy supplies each round, independently of previous
scores, dealers, actions and deals. The game still determines the next dealer
and scores. No rule or human-play dealer change is needed.

The opponent is a fixed member of the declared WP/Point class, including its
public endgame assistance. Delayed reconstruction evaluates its previously
observed actions only after its eight discards reveal that round's initial hand.
With exact reconstruction and the same deterministic prediction convention,
the true member is never eliminated. A singleton identifies that member. The
controller requires identification from a preceding round; otherwise it follows
its parent. The evaluator's opponent label is not an input to this gate.

Consider an eligible final-round own/public history H. Condition on the own
initial hand, initial field and ordered public actions/draws in H. A candidate
remaining opponent hand, together with its observed discards, determines its
initial hand. Reject candidates violating the initial-month rule. For every
remaining candidate, the number of initial-slot arrangements consistent with
H is the same, and every order of the remaining stock is equally likely.
Conditioning on any stored initial-field order changes a common factor only;
one need not assume that this factor is always (8!)**3. Actors do not use
within-group deal slot order.

Replaying all current-round opponent actions assigns each valid candidate
likelihood zero or one. The pinned actors are deterministic and stateless;
the likelihood is one exactly when every observed opponent choice agrees.
The learner's past choices have the same likelihood across candidates because
its own/public inputs are identical. They therefore do not require an additional
hand-dependent weight. Earlier rounds do not bias the new deal after conditioning
on the public starting context and identified fixed opponent. The first-eligible
gate is also a function of own/public history, so it introduces no additional
hidden-state likelihood once H is fixed.

Consequently the compatible valid remaining hands have equal posterior mass.
For each such hand, if n stock cards remain and d draws can still occur, every
ordered d-prefix has the same number (n-d)! of unobserved suffixes. Here n-d=8.
Cancelling the common suffix factor leaves P(n,d)=n!/(n-d)! integer mass per
hand, exactly the mass used by deferred_worlds. An early terminal outcome retains
the mass of its unconsumed prefixes; it does not discard those worlds.

The selected complete executable policy and its unchanged parent are valued
under this posterior. Adoption requires strict exact improvement; all missing
roots, public-cap refusals and other parent fallbacks contribute zero. As shown
in INCUMBENT_GUARANTEE.rst, the common parent prefix and total expectation give

  E[W(controller)] - E[W(parent)] = E[Delta(H)].

This is an expected draw-adjusted match-win difference under the declared law.
It is not the observed win difference on the realized source decks. It also
does not transfer to an arbitrary opponent, the old finite-seed distribution,
or a different deal mechanism. Finite confidence bounds additionally require
independent identically distributed sources within each declared stratum.

Implementation checks
---------------------

tests/test_logged_source_posterior.py uses the first eligible record in each
parent/opponent/seat cell of the closed prospective cohort: indices
1,3,4,5,7,8,10,46. Eligibility alone selects these eight fixtures, without
consulting gain or outcome. The original protocol, chunk records, two parent
checkpoints and six opponent checkpoints are hash checked. Seeded reference
replay recovers fixed round-deck tapes for the test; these tapes are not fresh
IID samples.

For every fixture, the logged collector reproduces the complete original action
sequence, terminal state, selected root, public identification events and
identification call counts. Every transition also matches the native engine.
The separate source audit rechecks all actor actions and public inference using
only the logged decks, with no fresh external chance draws.

Complete history enumeration yields 1,1,1,5,5,1,43,4 compatible hands from
105,105,91,91,91,91,105,91 valid candidates, respectively. The true hand is
checked only by the auditor; it does not select or weight hypotheses. Every
deferred world has the root's own/public observation, the appropriate equal
P(n,d) mass, and the publicly identified opponent. Reversing each initial
eight-card slot group leaves the packed neural inputs and legal masks unchanged.
Two-world historical replays compare an alternative hand at all past learner
decisions, giving 52 exact packed-input/mask comparisons across the eight roots.

A separate exhaustive ten-card analogue checks the conditioning arithmetic.
Before deterministic-action filtering, three valid remaining hands each occur
48 times. The observed opponent choice retains two, each still with 48 orders;
their six possible hand/next-draw pairs each occur 16 times. This is a finite
counting check, not enumeration of every real Koi-Koi history.

Validation on 21 October 2026: nine tests pass and Ruff passes. Per test run,
the logged fixtures contain 970 decisions, 64 round endings and eight matches.
Seeded recovery, native verification and logged-source audit each separately
replay that same work. Hand-support reconstruction uses 10,101 decisions and
5,117 opponent calls; alternative-history checks use 210 decisions and no
neural forward calls. Each source pass and its separate audit make 486 parent
and 484 opponent adapter calls, plus 102 Point/105 WP identification calls.
These counts are fixture work, not independent strength observations.

Scope of the validation
-----------------------

The counting argument supplies the law composition; the fixtures test specified
implementation invariants. Neither nine passing tests nor the argument certifies
operating-system entropy. The sampler assumes independent uniform bounded draws.
Source failures must stop the declared study, not cause outcome-dependent sample
replacement. Any future evaluation must freeze actors, source law, computation
limits, sample size, population weights, thresholds and success criterion before
fresh sampling. The previously failed actual-win gate remains unchanged.
