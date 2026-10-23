Rule-profile investigation, version 1
====================================

The owner authorized this research on 24 October 2026. These are explicitly
named modifications of paper8, not reproductions of any publisher's complete
rules. The original reference/native engines, playable model and paper8 results
remain unchanged. Both players use the same profile. Win score (win=1, draw=0.5)
is the comparison quantity; raw point totals cannot rank different rules.

Primary sources
---------------

Nintendo's published rules describe six-point opening hands (four cards of one
month or four month-pairs), seven-point doubling, opponent-koi doubling, a
scoreless exhausted hand and ending after the next yaku following a koi call:
https://www.nintendo.com/jp/others/hanafuda_kabufuda/howtoplay/koikoi/index.html
(accessed 24 October 2026). Its full match has twelve rounds.

The official Etohana game rules also document the opening hands and two
doubling conventions, but use a three-round match and allow further koi calls:
https://etohana-koikoi.com/rule/ (accessed 24 October 2026).
These differences motivate explicit profiles instead of assuming one universal
meaning of a local-rule name. Own-koi doubling and taking both players' yaku
are specified exploratory modifications below, without an authenticity claim.

Fixed semantics
---------------

All profiles keep eight rounds, 30 initial points, zero-sum transfers and
bankruptcy, card identities, capture rules, forced stopping on the last turn,
and the other paper8 yaku. Changes are:

* ``paper_control``: exact paper8 behavior, with additional accounting fields.
* ``opening_six``: allow a hand containing all four cards of a month; either
  that hand or four distinct month-pairs wins six immediately. Field four-of-a-
  month still causes a redeal. Simultaneous opening hands favor the dealer.
  Opening awards are six, without multipliers, and consume a round.
* ``seven_double``: double a settlement of at least seven, once.
* ``koi_cover``: double when the losing player has declared koi, once.
* ``own_koi_double``: double when the winning player has declared koi, once,
  regardless of declaration count. This is the selected meaning of koi-bai.
* ``winner_takes_yaku``: the winner receives the sum of both players' current
  yaku scores, including the existing declaration bonuses.
* ``common_mix``: opening six, seven doubling, koi-cover, five-point sake yaku,
  no declaration-count bonuses, and scoreless exhaustion with dealer reversal.
* ``single_koi_mix``: common_mix plus automatic settlement at the next yaku
  increase after either player's first koi declaration.
* ``combined_mix``: common_mix plus own-koi doubling and both-yaku settlement.

For combinations, sum eligible yaku first, then test seven, then apply the
opponent and own declaration multipliers. No multiplier affects an opening
award or scoreless exhaustion. The opening tie convention is a research choice.
Automatic opening rounds, ordinary rounds, matches and decisions are counted
separately. The native header is shared by C++ and CUDA and compared stepwise
against an independently implemented Python reference.

Interpretation
--------------

The initial screen compares existing frozen neural weights with rule-correct
public endgame assistance. Their original public feature encoder is retained
symmetrically, so these are transferred agents, not variant-trained agents or
reproductions of the original publishers' AI. A strong transferred win rate
does not establish an optimal rule or optimal policy. Development selection
requires untouched confirmation deals, paired seats and both parents/opponents.
The search certificates for paper8 do not transfer automatically to new scoring
rules; their score bounds and belief likelihoods would need separate proofs.
