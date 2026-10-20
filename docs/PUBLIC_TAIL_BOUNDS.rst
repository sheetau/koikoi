Public final-round outcome certificates
======================================

Purpose and scope
-----------------

Five saved roots still exceed the physical-transition budget. Four fail even
with the parent controlling future learner decisions. Before another planner
change, assess whether public score bounds can identify irrelevant continuation
work. This module is experimental and does not alter game rules or deployed play.
It is a domain-specific admissible bound, not a claim of a new general search
algorithm.

Bound
-----

Let C_p be the cards already captured by seat p, k_p its declaration count,
t the current turn (zero based), and d the dealer. These are public.
Let r_p count turns j in [t, 14) for which (d+j) mod 2 equals p.
No new declaration is possible on turns 14 or 15 because a score increase
forces stopping. Including the current turn is conservative in every phase.

The final captured set of p is a subset of A_p = all cards minus C_(1-p).
For fixed declaration count k, the local score S(A,k) is monotone in A:
set-completion bonuses and threshold counts cannot decrease. The light score
also cannot decrease: adding rain to three non-rain lights changes 5 to 7;
four non-rain lights plus rain changes 8 to 10; other additions preserve or
increase the score. Its remaining cases follow by enumerating the five lights.
The multiplier at a fixed k is nonnegative. Hence an upper bound on any final
round award to p is

  U_p = max(1, max { S(A_p,k) : k_p <= k <= k_p+r_p }).

The one-point dealer award is included. The explicit maximum over declaration
counts is necessary: the local additive/multiplicative formula is not monotone
at the change from three declarations to four when the base score is small.
The superset and count range may contain unreachable cases; this only makes
the bound looser.

For the learner's current lead D, every final-round terminal point margin lies
in [D - 2 U_opponent, D + 2 U_learner]. Exactly one final round award transfers
points between players. A strictly positive lower bound certifies a win; a
strictly negative upper bound certifies a loss. Equality does not certify a
strict outcome. Nonfinal rounds and terminal inputs return no new certificate.
No hidden card assignment, stock order, opponent model or future deal is used.

Verification and saved-data check
--------------------------------

Unit checks cover score-component monotonicity, nonmonotone declaration counts,
the exhaustion award, strict boundary cases, seat reversal and private-state
invariance. Independent ReferenceEnv continuation checks enumerate both players'
legal choices from eight late fixtures, with every possible next stock card and
three synthetic point margins. These finite checks exercise the proof; they
are not an exhaustive enumeration of all hidden decks or strength evidence.

scripts/check_public_tail_bounds.py reads all 28 completed roots from the closed
five-draw and recovery studies, and all 33 available source root states. It
checks hashes of its source inputs and each saved artifact. Coverage is counted
on unique saved keys within each root/method, separately for fixed, optimized
and opponent policies. All computed root-action maps are included. Counts are
neither probability weighted nor transition savings. Partial refused trees have
no saved interior cache, so the analysis cannot establish their coverage.
No new simulations, policy calls, source matches or training updates are made.

An eventual planner must represent certified live leaves separately from actual
terminal completions and use an explicit, verified fallback beneath them. It
must preserve exact values and parent-favoring ties on feasible regression
trees before cost claims or another recovery attempt. This coverage analysis
does not itself authorize a larger budget or a fresh strength study.
