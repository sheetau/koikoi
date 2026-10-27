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

Click START, then a hand card. Cards that can capture are raised; discards remain down. When several different field
cards match, select the hand first and then its target; click elsewhere to
cancel. Two identical chaff cards are selected automatically. Hover does not
enlarge cards. A pending hand remains in its slot and a pending draw stays on
top of the stock until the target is chosen. New yaku are announced individually in order, followed by a short koi question
and Yes/No buttons. All buttons have hover feedback.
Drag vertically to orbit the table; use the wheel to zoom. Esc cancels selection
or opens settings; F11 toggles full screen. Japanese/English, animation speed,
audio, opponent and rule preferences persist in Godot's per-user directory.
AI/rule changes take effect on the next match. The match itself is ephemeral.

## Product rules and inference

Choose 1, 3, 6 or 12 rounds (default 12), with 30 points per player and no bankruptcy termination. Settings apply to the next match. Restart in Settings confirms ending the current match and begins a fresh dealer draw with the selected conditions; it is disabled while an event/request is in flight. The smaller
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
Actual legality, settlement, selected-length termination and payoff-correct public
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
persists through reveals, four batches of single-card dealing, fixed hand slots,
captures, yaku and return flights. Guided smoothstep flight trajectories
keep position/orientation continuous; stronger yaku landings use real rigid-body
gravity, velocity and tabletop collision. The stock top's immediate flip is
the explicit exception. Total-yaku settlement follows semicircular overhead
paths. Winner cards retain their exact positions and row/slot order; opponent
roles append below them without mixing. No infinite ground plane or dynamic shadows are present.

A physical card belonging to several yaku appears once, in the first applicable
row; all applicable yaku still receive their full points and result entries.
The player's yaku cards face the opponent, as specified. Opponent cards remain
uninspectable while face down. Settings, reference help, scores, decisions and
announcements occupy vertical planes in the 3D world. Only the small settings
button is screen-space UI. A perspective camera sees both fixed hand racks and
the floating plum board. Cards, the board and UI are flat surfaces; the stock
is one six-face cuboid whose height follows the remaining card count.
The background stays solid gold, without a floor, grid or shadows.

Dealer cards and the ordinary deal move one card at a time. The original
field/dealer/opponent order and two-card batches remain; each card completes
its flight before the next leaves the stock. Field cards flip instantly at
the stock; hand cards remain face down during the deal. An opposite-facing hand reveals
with a stationary half turn; an already aligned hand skips that turn. The subsequent flight combines translation and
alignment rotation. A field-play flight takes 0.30 seconds at normal speed
(previously 0.62). The field retains exactly two rows and
adds columns as needed. Empty slots are reused; only column expansion slides
existing cards horizontally. Captures form half-overlapping horizontal groups
for chaff, animals, ribbons and lights, with reserved space for their maximum
unscored counts. Capture groups sit just outside the board, beneath the hands.
Named roles that share cards, including transitive overlaps, occupy one unwrapped row with all role names and their scores. Count roles (animals, ribbons, chaff) retain separate rows; their counting membership never joins unrelated named roles. The sake cup still counts for chaff under the existing rules, while its physical card appears only once. Independent roles keep separate rows; light upgrades replace their lower role without adding its points. Both field cards and captures promoted to
yaku travel sideways beyond the board, round a U-turn, then arrive from right
to left. A role shares its turn position and nominal travel speed; leftmost
cards arrive first at evenly spaced times. The turn extends beyond long rows.
Returns first gather each group into an existing outer-corner card, without
adding stack height. These intentional visual merges run together. Each merged
packet then flies to the stock. All capture categories at the same seat gather together. Horizontal packets retain their facing and slide beneath the lifted stock without rising; upright role packets align while moving and return to its top.
Collision clearance is conservative, not a globally shortest-path solver.
The camera fills windows of any aspect ratio without letterboxing.

## Assets

All 48 images are the Commons traditional-color set by Louie Mantia and すけじょ.
The official 960-pixel-wide PNG renderings are retained byte-for-byte. A
geometric shader masks only the black outer
frame with #b44519; it leaves illustration ink unchanged. Source URLs, hashes
and CC BY-SA 4.0 attribution are bundled in `game/assets/cards/`. Godot imports
with mipmaps for stable downsampling. Noto Serif JP Black is bundled under SIL
OFL 1.1. The user-provided Antique Six Extra Bold is copied into the project and
used in the English UI. The Japanese UI uses Noto Serif JP Black for all
characters, including Latin letters and digits; no external font path is
required. Six user-supplied MP3 clips are copied unchanged into `game/assets/audio/`.
Dealer/hand landings use `tefuda-place`; each yaku card plays `yaku` at arrival.
New role movement and total-yaku transfers play `fuda-take` once per moving role row at departure; transfers have no yaku arrival sounds. Draws, ordinary captures and return packets use `fuda-turn`. Stationary opponent/dealer reveals are silent. Four original synthesized retro
cues cover button presses, messages, wins and losses. Their playback gains
are trimmed by 9.4–11.1 dB to match the hand/draw clips. No external audio path
is needed. In-game credits and `THIRD_PARTY_NOTICES.md` identify the sources.

## Focused checks

```powershell
./.venv/Scripts/python.exe -X utf8 -m pytest tests/test_game.py -q
./.venv/Scripts/python.exe -X utf8 -m ruff check koikoi/game.py koikoi/game_server.py koikoi/game_opponent.py tests/test_game.py
./.venv/Scripts/python.exe -X utf8 -m scripts.game_ui_check
./.venv/Scripts/python.exe -X utf8 -m koikoi.game_server --autoplay --qa
```

UI smoke checks inject Godot input events into the real rendered game and local
HTTP bridge with a repeatable seed, including a full twelve-round learned-opponent match with both
rule options enabled. Test mode does not write player preferences. Autoplay runs a complete twelve-round presentation match with the
practice opponent, conserving 48 objects. Screenshots go to ignored
`output/game-qa/`. These are functional checks, never strength evidence.

`game/tests/layout_check.gd` checks a synthetic dense public settlement in both
languages using `game/tests/dense_result.json`. It is a layout fixture, not a
reachable research match or game result.

`game/tests/concept_check.gd` checks the public opening fixture,
single-card dealing and return flights; it also captures the normal view, question
and settings. `game/tests/motion_audit.gd` checks moving card intersections with
oriented-box separating axes and checks board clearance. These checks exercise
presentation geometry only; intentional same-group return merges are excluded
from intersection counts; no research or model evaluation is involved.

Geometry and dense layout checks (using the pinned console executable):

```powershell
external/godot/Godot_v4.7.2-stable_win64_console.exe --path game --script res://tests/concept_check.gd
external/godot/Godot_v4.7.2-stable_win64_console.exe --path game --script res://tests/layout_check.gd
```

`game/tests/addendum_check.gd` exercises selection/cancellation, stationary
reveal, stock choice, two-row expansion, equivalent targets, corner-gathered return,
contact-driven rebound, camera orbit/zoom, UI depth, minimum announcement
duration, simultaneous front/back win/lose text, hover feedback and wide/tall rendering.

`game/tests/role_motion_check.gd`, `capture_motion_check.gd` and
`impact_check.gd` isolate arrival order, half-overlapping capture groups,
returns and collision-driven rebound. Strong landings temporarily use 480 Hz
physics and a contact joint between the touching cards so tabletop restitution
can rebound both thin proxies. No upward rebound velocity is injected. Cards
settle under gravity before collection. Announcements remain visible for at
least one second; AI choices wait at least 0.8 seconds (1.2 for a koi decision).
Human clicks are handled immediately. Win and lose messages occupy opposing,
single-sided faces simultaneously, so neither bleeds through the other.
`game/tests/audio_return_check.gd` checks all ten audio streams, per-card
arrival sounds, per-role transfer sounds, corner gathering and no-flip horizontal returns.

The UI smoke harness accelerates announcement/AI waits through the testing-only
`KOIKOI_QA_FAST` flag. Minimum display durations are checked independently in
`addendum_check.gd`. `scripts.game_ui_check --headless` routes input into the
viewport directly and avoids waiting for visible-window screenshots. Rendered
screenshots explicitly draw a frame so minimizing a QA window cannot stall capture.

## Development task management and fast checks

Open `dashboard.cmd` for the private task/question ledger. Agent work starts with
`.venv/Scripts/python.exe -X utf8 .dashboard/app.py brief`, then relevant IDs;
the former plan.txt is archived, not a live queue. Management code is separate
from the game runtime. See `.dashboard/README.md` for recovery and review flow.

Use `python -m scripts.check_game --suite quick` with the project virtual
environment. Suites can be combined: `management`, `quick`, `motion`, `visual`,
`full`. Visual checks use the actual renderer; other geometry checks are headless.
`--list` prints commands. `--reuse` explicitly reuses only passing results with
identical source/assets/engine/interpreter inputs, and labels them REUSED, not
newly measured. Full learned-opponent matches always run. Logs/timings/cache
remain private in `.dashboard/checks/`. Timing tests retain real waits.
The motion auditor rejects disjoint enclosing boxes before its unchanged OBB
separating-axis test. `motion_math_check.gd` compares it against the original
calculation on 12,000 deterministic pairs and reports measured runtime.

`game/tests/backlog_check.gd` checks single-card dealing/batch order, fixed winner
rows during total-yaku transfers for both seats, one cue per role, sequential
readable yaku notices, the following minimal koi prompt and faster field play.

`migration_check.gd` verifies unified capture gathering, bottom insertion, no upward
card movement, deck clearance and the revised audio routing. `grouped_roles_check.gd`
checks shared/transitive role membership, bilingual labels, stable winner rows,
conservation on return, match-length settings and restart protection. Game-service
tests verify all four match lengths and exclusive light upgrades against the existing
scoring implementation; these are functional checks, not strength evidence.

Dealer-reveal cards bypass packet merging and return one at a time, nearest
stock first. `dealer_return_check.gd` audits that path without excluding the pair
as an intentional gathering group.

Review refinements use 17–20 point text for HUD, notices and result details, with
shared translucent backgrounds; the settings typography is unchanged. Settings
and generic notice/result surfaces render from both sides. The distinct win/lose
faces remain single-sided so each seat sees its own result. Result details and
messages use bounded SubViewport scroll containers, grow upward from a fixed
bottom clearance, and route pointer/wheel input to the panel beneath the cursor.
An overflowing line scrolls horizontally instead of forcing premature wrapping.

New role titles appear at the first physical card arrival. Already settled rows
remain visible. Total-yaku transfer keeps each seat's rows separate and approaches
from the same side as normal role formation, with winner transforms unchanged.
AI koi choices visibly hover for at least 0.4 seconds before their click cue;
wall-clock enforcement prevents a costly render frame shortening that pause.
`review_check.gd` exercises these requirements, bounded result scrolling, narrow
layout and translucent settings from both sides. The game scoring is unchanged.
