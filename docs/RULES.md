# Rules: version 1

The canonical rules are those of guansanghai/KoiKoi-AI, not a claim of universal or Nintendo-standard Koi-Koi. Card ID is `(month - 1) * 4 + (index - 1)` with upstream card indices.

| Setting | paper8 | research12 |
| --- | --- | --- |
| Maximum rounds | 8 | 12 |
| Initial points per player | 30 | 30 |
| Stop if a player's points <= 0 | yes | no |
| Other round rules | upstream | identical |

research12 deliberately changes only match length and bankruptcy. Alternative scoring rules require a new versioned profile and dedicated tests.

- Deal 8 cards per hand, 8 field cards, 24 stock cards. Redeal if either hand or field has all four cards of any month. No initial-hand automatic scoring.
- Dealer plays first; round winner is next dealer.
- Each turn: discard, resolve capture, reveal stock top, resolve capture, then check increased yaku points. With 0 matching field cards place the card; with 1 or 3 take all; with 2 choose one.
- Stock is consumed from the end of the supplied permutation, matching upstream.
- Offer stop/continue only if the player's scored yaku total increased this turn. On that player's eighth turn an increase forces stopping. Otherwise, after 16 turns the initial dealer wins one point and remains dealer.
- Lights: five=10; four without rain=8; four with rain=7; three without rain=5.
- Boar/deer/butterfly=5. Tane >=5 scores count-4. Ribbons >=5 scores count-4. Kasu >=10 scores count-9. Sake counts as both tane and kasu.
- Red ribbons=5; blue ribbons=5; all six gives an additional 10 (20 combined). Preserve this upstream behavior.
- Flower/sake and moon/sake each score 1 before the player's first koi declaration, 3 afterward.
- For <=3 declarations add declaration count to base yaku score; for >3 multiply base by count-2. No seven-point doubling or opponent-koi doubling.
- Points transfer from loser to winner. Match winner has the larger final point total; ties are draws.

Actions: cards 0..47, stop=48, continue=49. Exposed decision phases: discard=0, discard capture=1, draw capture=2, koi=3, match terminal=4. Forced transitions are automatic. Illegal actions are rejected without changing state.
