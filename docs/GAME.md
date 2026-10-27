# Floating-table desktop game

The game uses **Godot 4.7.2 and GDScript** for a native Windows 3D client,
with the existing **Python 3.12** environment providing authoritative rules
and frozen CPU AI inference. Godot supplies 3D picking, two-sided objects,
rigid-body collision, internationalized UI and a portable editor/export path.
No browser, JavaScript build chain or paid engine is needed.

## Play and develop

Double-click `play.cmd`. From a console:

```powershell
./.venv/Scripts/python.exe -X utf8 -m koikoi.game_server
```

First-time engine setup (the Python research environment and restored models
are still required for the learned opponents):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_game.ps1
```

Open `game/project.godot` in the pinned Godot editor to edit scenes and scripts.
Use the Python launcher to play: it starts a private loopback service, passes
an ephemeral bearer token to its child process, and shuts down when the game
closes. There is no external listener, telemetry, game upload or online play.
The default renderer is OpenGL Compatibility with unlit materials and no shadows.

Click START, then click a hand card. There are no legal-card highlights.
If two field cards match, click the one to capture. Hover to inspect a card.
The central board panel offers koi/stop and next-round actions. Esc opens
settings; F11 toggles full screen. Japanese/English, animation speed, audio,
opponent and rule preferences persist in Godot's per-user settings directory.
AI/rule changes take effect on the next match. The match itself is ephemeral.

## Product rules and inference

Twelve rounds, 30 points per player, no bankruptcy termination. The smaller
month wins the opening dealer draw, with conventional card values 20/10/5/1
breaking same-month ties. Equal month and value draws another pair. Opening
cards return to the deck before the deal. The round winner deals next.

Base yaku/capture rules come from `docs/RULES.md`. Settings independently enable
winner-takes-both-yaku and own-koi doubling, reusing the existing variant payment
implementation. These flags do not start research or alter any frozen profile.
As in the original rules, a score increase on a player's last turn ends the
round automatically; a scoreless exhausted round pays the dealer one point.

The selected learned checkpoint remains identity-checked and unchanged:
`runs/cold_ppo_screen/seed6710001_warm/update_00000520.pt`.
It and the optional original RL-WP/RL-Point opponents use eight-round encoders.
The game-only adapter projects match context onto their existing input format:
eight-round/bankruptcy feature flags and `max(0, round_index - 4)`.
Actual legality, settlement, twelve-round termination and payoff-correct public
endgame assistance use the unprojected game. No new training, policy weights,
strength evaluation, or research-search rollout is performed. Twelve-round
feature transfer is a deployment choice, not evidence of twelve-round strength.
Practice AI is a separately selectable existing observation-only greedy agent.

## Presentation and information boundary

`koikoi/game.py` extends the existing variant environment to publish semantic
events: play, draw, capture/place, koi and round settlement. Capture events
include intermediate public snapshots before the engine auto-deals the next
round, so final hands/captures remain visible until Next round is selected.
The response never includes the opponent's card IDs, unseen stock permutation,
RNG state, future deals or packed native state. Opponent hands use anonymous
slot indexes. A card's identity is sent only when it is legally revealed.
The renderer cannot submit AI actions: the service runs the frozen adapter.
Revision checks and serialized requests reject stale and out-of-turn input.

`game/table.gd` consumes these events. A pool of exactly 48 physical objects
persists through reveals, four batches of pairwise dealing, fixed hand slots,
captures, yaku and return flights. Guided, minimum-jerk flight trajectories
keep position/orientation continuous; stronger yaku landings use real rigid-body
gravity, velocity and tabletop collision. The stock top's immediate flip is
the explicit exception. Total-yaku settlement follows semicircular overhead
paths. No infinite ground plane or dynamic shadows are present.

A physical card belonging to several yaku appears once, in the first applicable
row; all applicable yaku still receive their full points and result entries.
The player's yaku cards face the opponent, as specified. Opponent cards remain
uninspectable while face down. Settings and reference help use a readable 2D
overlay; game decisions and announcements use the central vertical 3D plane.

## Assets

All 48 SVGs are bundled from Louie Mantia's hanafuda, recolored by dotty-dev,
at commit `a4ac60e4bd60f95ab8c4e5176b50bfd2e3772992`. Originals remain unmodified,
under CC BY-SA 4.0; `game/assets/cards/SOURCE.md` and `LICENSE.txt` retain source
links and attribution. Godot imports at full vector resolution with mipmaps
for stable downsampling. Noto Sans JP is bundled under SIL OFL 1.1. In-game
credits and `THIRD_PARTY_NOTICES.md` identify these sources.

## Focused checks

```powershell
./.venv/Scripts/python.exe -X utf8 -m pytest tests/test_game.py -q
./.venv/Scripts/python.exe -X utf8 -m ruff check koikoi/game.py koikoi/game_server.py koikoi/game_opponent.py tests/test_game.py
./.venv/Scripts/python.exe -X utf8 -m koikoi.game_server --ui-test
./.venv/Scripts/python.exe -X utf8 -m koikoi.game_server --autoplay --qa
```

UI smoke checks inject Godot input events into the real rendered game and local
HTTP bridge, including a full twelve-round learned-opponent match with both
rule options enabled. Test mode does not write player preferences. Autoplay runs a complete twelve-round presentation match with the
practice opponent, conserving 48 objects. Screenshots go to ignored
`output/game-qa/`. These are functional checks, never strength evidence.

`game/tests/layout_check.gd` checks a synthetic dense public settlement in both
languages using `game/tests/dense_result.json`. It is a layout fixture, not a
reachable research match or game result.
