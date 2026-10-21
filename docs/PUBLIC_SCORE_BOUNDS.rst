Public capture-capacity score bounds
===================================

The experimental controller can certify a final-match win or loss before the
last turn using public captures, turn/phase, koi counts and the point margin.
This extends the final-turn rule while leaving game rules and the deployed
human-play model unchanged. It is an admissible pruning method with a
Koi-Koi-specific bound and an executable parent-default policy; this document
does not claim that admissible bounds or symbolic policies are new algorithms.

Bound and proof
---------------

For player p, let C be their current captures and A all cards except the other
player's captures. Every possible final capture set contains C and is contained
in A. Hidden hands and stock positions are not inspected, even when a planner
has hypothetical worlds available.

A discard or draw captures at most four cards. Each future turn therefore adds
at most eight. The current turn contributes at most eight at discard, six at
the two-way hand capture choice, two at the two-way draw capture choice, and
zero at koi. These are upper bounds, not predictions of achievable captures.
Call the resulting remaining capacity b.

For a count yaku with card class G, its count is at most
min(|A intersect G|, |C intersect G| + b). A set yaku S is possible only if
S is contained in A and |S minus C| <= b. The light award maximizes the five,
four and three-light possibilities, respecting rain already captured. Summing
these component maxima is an upper bound even when their maximizing subsets
cannot be attained together. Sake retains its dual tane/kasu classification.

The current koi count k can increase at most once on each remaining own turn
whose index is <= 13. Maximize the score bound over every integer in that
range. For k <= 3, add k; otherwise multiply the base by k - 2. Flower/moon
sake uses coefficient one at k = 0 and three afterward. Taking only the largest
k is unsound: a one-point base gives four at k = 3 but two at k = 4.

Let U_p be this bound, or zero if p has no turn left. A final-round margin D
for the viewer certifies a win if D > 2 max(1, U_opponent), and a loss if
-D > 2 max(1, U_viewer). The factor two is the point transfer, and one covers
the exhaustion dealer award. Strict inequalities preserve ties. Earlier rounds
and terminal states are excluded. Own scoring cannot hurt a positive margin;
opponent scoring cannot rescue a negative one. Hence these certificates hold
for every legal policy and every compatible hidden world, not just the fitted
opponent model.

The bounds cannot increase along a nonterminal legal transition: available
cards only shrink, capacity drops by at least newly captured cards, and the
possible koi interval only shrinks. Therefore a certified region remains
certified until the terminal outcome. Missing map entries in either certified
region can use the unchanged parent legally. All other missing entries remain
errors. Existing information-set partitioning and exact integer chance masses
are retained; certified signed mass shifts the incumbent cutoff by the same
amount. Physical completions and certified mass are counted separately.

Implementation and checks
-------------------------

``public_score_bounds.py`` implements the bound, ``score_bounded_tail.py`` the
versioned program/planner/executor. ``verify_score_bounded_tail.py`` and
``score_bound_assessment.py`` execute complete Python transitions and every draw
branch without substituting certified outcomes. The latter independently
evaluates the unchanged parent rather than trusting the fixed-phase map.
Frozen predecessor modules are unchanged.

Tests exhaust small capture subsets and koi ranges, exercise the nonmonotone
3-to-4 transition, hidden-state invariance, final-round/tie boundaries, legal
late-tail branches, both seats, and agreement with the previous exact planner
and full Python program replay. Synthetic point margins in tests establish
implementation properties, not natural playing strength.

Development screen
------------------

``scripts/screen_score_bounds.py`` retains all eight mechanically selected
six-draw fixtures and the same independent phase budgets (2,000,000 physical
transitions / 500,000 nodes), hand limit 1,000, capacity 4,096 and depth 32.
The previous sources and policies stay frozen. Input hashes, old result hashes
and every source/program binding are checked before planning. Each worker
reconstructs the support and compares it to the saved worlds. New selection
and program are saved before oracle/actual-tail execution.

Every adopted pair and every newly valued parent receives full independent
Python value replay. Actual tails are replayed and checked against source
suffixes. This is a paired development feasibility screen with no new chance
draws, not an independent strength estimate. No fixture may be replaced after
observing its outcome. A new controller chosen from these observations needs
a fresh prospective evaluation. Operational concurrent timing is not an
isolated latency benchmark.

Reproduce only into a new directory::

    .venv/Scripts/python.exe -m scripts.screen_score_bounds --base runs/new_score_bounds --freeze
    .venv/Scripts/python.exe -m scripts.screen_score_bounds --base runs/new_score_bounds --run
