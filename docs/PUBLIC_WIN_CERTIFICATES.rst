Public certificates for wins within a turn
=========================================

Status
------

``scripts/public_win_certificate.py`` is an isolated research wrapper. It leaves
the rules, initial dealer, deployed model and running response study unchanged.
The wrapper certifies an immediate match win from public information, including
card decisions before the existing stop/continue guard can apply. Its narrow
guarantee is useful independently of whether it improves a strong baseline.

Construction
------------

The input consists of the acting player's 560 public observation features and
50-action legal mask. The solver uses the own hand, field, captures, public
pending card, points, declaration count, turn, dealer and match rules. Unknown
cards include both the opponent's hand and the remaining stock. This superset
contains the actual next draw, so requiring success for every member needs no
posterior over hidden hands and no assumption of uniform stock probabilities.

For each discard, the quantifier order is::

    exists hand-capture choice
        for every unknown next draw
            exists draw-capture choice
                immediate terminal match win

The hand choice must work before seeing the draw. The draw choice can depend on
the now-public card. Interchanging the first two quantifiers would wrongly allow
the hand action to depend on a hidden future card. Deterministic captures omit
their choice node. A capture root starts at the appropriate position in this
tree; a koi root can certify STOP. No opponent decision is searched.

After the draw, an increased yaku total permits stopping (forced on turns 14/15).
A non-increasing score on turn 15 instead awards one point to the initial dealer.
The leaf accepts only a terminal match with strictly more own points, using the
configured final round or bankruptcy rule. A tie is not a win certificate.
Every symbolic draw case and leaf check is counted separately; these operations
are neither simulator decisions nor rollout matches.

``CertifiedWinAgent`` always calls its base policy. It preserves the base action
when no certificate exists or when that action is certified. Otherwise it takes
the lowest-index certified action. It must remain active at every subsequent
decision so that the certified continuation is executed. The search has at most
two choices at either capture node and enumerates each unknown card explicitly.

Guarantee and its scope
-----------------------

Assume valid observations, the documented rules, a correct implementation and
continuous use of the wrapper. At a certified root, the selected hand choice
works for every possible draw. The actual draw belongs to the enumerated set;
after its reveal at least one certified capture remains. Recomputing the
certificate therefore preserves a winning continuation. The turn ends in a
match win before the opponent can act again.

Couple a base policy and its wrapped version on the same deal stream and random
choices up to their first action difference, against the same opponent strategy.
If they never differ, their outcomes agree. At the first difference, the wrapper
has a win certificate and finishes with the maximal match-win utility. Thus the
wrapped outcome is never worse on this coupling, and its expected match-win
utility is weakly greater. This does not guarantee a strict gain or a larger
point margin. It also does not establish optimal play outside certified turns.

This is a mathematical argument for the algorithm, not a machine-checked proof
of the Python implementation. Independent reference-tree tests enumerate hidden
draw alternatives, preserve the order of decisions and check both profiles.
Additional complete-match checks verify nonregression and immediate completion
after a first override. A public-only facade and hidden-card reassignment test
exclude private-state dependencies from the wrapper.

Relation to prior work
----------------------

Observation-based winning strategies under imperfect information are established
in Chatterjee, Doyen, Henzinger and Raskin, `Algorithms for Omega-Regular Games
with Imperfect Information <https://lmcs.episciences.org/1094>`_ (2007).
Alshiekh et al., `Safe Reinforcement Learning via Shielding
<https://ojs.aaai.org/index.php/AAAI/article/view/11797>`_ (2018), synthesize
reactive shields that correct actions violating a temporal-logic specification.
Our finite within-turn calculation uses neither their general fixed-point
algorithm nor a general temporal-logic synthesis procedure. The specific
contribution here is a small public-information implementation and its
match-utility argument for these Koi-Koi rules; the broad shielding or
observation-based synthesis principles are not new.

Coverage diagnostic
-------------------

``scripts/diagnose_win_certificates.py`` uses only the training source-match split
of the two completed opponent-target datasets. It reconstructs legal actions
from public observations and examines the fixed expert-disagreement roots.
Among 52,522/52,687 roots, 157/149 have certificates. The parent proposal is outside
the certified set at 12 roots for each parent. All 157/149 stored parent branch
continuations nevertheless won in their sampled hidden worlds. This diagnostic
therefore identifies no recovered parent loss or draw. Root frequency is not an
estimate for ordinary gameplay, because collection selected expert disagreement.

No full-match strength study or deployment promotion is justified by this screen.
If future results motivate one, it must compare wrapped/unwrapped candidates and
also supply wrapped published-opponent controls. Applying this extra assistance
only to the candidate must not be presented as superior learned-policy quality.
