Public-stock rule research: development specification
====================================================

Owner authorization: explore preview-two, choose-from-two, and own-koi doubling
plus winner-takes-yaku, individually and jointly. This starts a new research
cycle; previous cohorts and evidence remain closed and byte-preserved.

The eight profiles form a full 2 x 2 x 2 factorial over paper8:

* Preview: both next stock cards are public before the hand action, continuously
  replenished after each draw. Without choice, the first must be drawn.
* Choice: after the hand action and its capture, reveal the next two stock cards
  and select one. Retain the other at the top. Without preview, the replacement
  stays hidden until the next draw-choice phase; the retained card stays public.
* Stakes: add both players' current yaku and double once if the winner has
  declared koi. Preserve paper8's declaration bonuses. No opening, seven-point,
  opponent-koi, sake-five or exhaustion-rule changes are included.

All profiles retain eight rounds, 30 initial points, bankruptcy, original yaku,
and the 16-turn round limit. A stock choice is an extra decision, not an extra
turn or draw. Cards selected from the second position swap with the first
before removal. Public information belongs to both players. This is a proposed
local rule family, not a claim that these combinations are standard rules.

Implementation isolates Python and shared C++/CUDA engines from frozen code.
State slots 115--116 store ordered public card IDs plus one. Observation adds
two public-card channels and four phase/rule indicators to the original 560
features. Policy input never contains hidden hands, hidden stock order or RNG.
Deal-consistent search proposals fix known stock positions and resample unknown
hands and stock; they do not claim an exact action-conditioned posterior.

Research sequence
-----------------

First verify full Python/CPU/CUDA trajectories, observation invariance under
hidden-state replacement, and preservation of retained-card knowledge. Then
train rule-specific observation-only policies with matched decision budgets
and multiple seeds. A pilot selects engineering budgets, not winning rules.
Freeze development and independent confirmation streams before comparisons.

The target is added value of computation: within each rule, compare the same
adapted policy with increasing rollout budgets against fixed opponents, using
duplicate deals and seats. Report win rate, points, actual decisions, rollouts,
transitions and wall time separately. Larger points under amplified scoring
are not evidence of stronger play. Compare search gains against control with
paired uncertainty and account for rule selection. Final evidence must not
rest on a weak unadapted opponent or one training seed. Monotonic improvement
is a hypothesis, not an assumed property of these rules.

No new strength result exists at specification time. The 2023 comparison and
previous frozen-weight transfer findings remain unchanged.
