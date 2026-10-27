# Koi-Koi

Play Koi-Koi against trained AI opponents, or explore self-play learning and verified endgame search. The project includes a local desktop game, Python/C++/CUDA simulators, and reproducible comparisons with the published RL-WP and RL-Point models.

## Play

Double-click **`play.cmd`** to open the new floating-table 3D game, or run:

```powershell
./.venv/Scripts/python.exe -X utf8 -m koikoi.game_server
```

The native Windows client uses **Godot 4.7.2 + GDScript**, with the existing
**Python 3.12** rules and frozen AI running locally on CPU. The gold background,
plum board, physical card animations and high-resolution traditional-color deck follow the
game plan. Japanese and English are available in Settings.

Click START, then click a hand card. If two field cards match its month, choose
one. Stock draws are automatic; use the central panel for koi/stop and the next
round. Esc opens settings; F11 toggles full screen.
There are twelve rounds, starting at 30 points each, without bankruptcy ending
the match early. Settings include winner-takes-both-yaku and own-koi doubling.

Install the pinned engine and import the bundled card/font assets once:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_game.ps1
```

See [the game guide](docs/GAME.md) for architecture, controls, licenses and focused
checks. The original eight-round Tkinter client remains available through
`python -m koikoi.play` for reference.

## Setup

Use the existing project-local Python 3.12 environment. For a fresh research
checkout, the original `scripts/bootstrap.ps1` restores dependencies and the
pinned upstream source. The game launcher itself starts no training or evaluation.
The Godot renderer uses OpenGL 3.3 Compatibility; AI inference runs on CPU.

**The learned checkpoint is not included in Git.** Restore it using the
[artifact instructions](docs/ARTIFACTS.rst):

```text
runs/cold_ppo_screen/seed6710001_warm/update_00000520.pt
```

Settings also offer original RL-WP, RL-Point, and a practice AI. Missing models
produce an error rather than silently changing the opponent. The eight-round
learned/published encoders use an explicit game-only twelve-round feature
projection; this is a deployment choice, not a new strength result. Research
models, engines, protocols and evidence remain unchanged. The new card art is
bundled and no longer depends on the external checkout. Consoleless launcher
output is written to ignored `.koikoi-work/game.log`.

## Research results

The main method verifies local improvements from public play even when a computation limit prevents completing the parent policy's root evaluation. Complete future beliefs and executable local programs let their certified gains combine into an expected match improvement.

| Comparison | Completed evidence |
| --- | --- |
| Self-play adaptation vs. assisted RL-WP | Both trained seeds exceed 50% on held-out duplicate deals: **51.480%** and **51.445%**. The separate RL-Point criterion does not replicate. |
| Base search vs. its parent | On 8,192 sources, expected gains are positive in all four parent/opponent populations. Mean opponent-mixture gains are **0.2819 / 0.2143 percentage points**; the joint 0.1-point lower-bound target is unmet. |
| Frontier extension vs. base search | On 4,096 sources, both opponent mixtures have positive simultaneous lower bounds. Mean added gains are **0.00128 / 0.000784 percentage points**. It improves 26 source beliefs; realized match scores are unchanged and computation increases. |

The new [256-source proposal comparison](docs/PROSPECTIVE_PROPOSAL_RESULTS.rst) finds 22.00% fewer planning transitions with 93.72% of the exact gain; its superiority criterion fails. A [nine-rule transfer investigation](docs/RULE_TRANSFER_RESULTS.rst) completes 182,272 matches, with both separate confirmation cohorts failing the joint criterion. Neither result promotes a new default.

The [public-root pruning comparison](docs/PRUNED_OPTIMIZED_FRONTIER_RESULTS.rst) preserves all 92 adopted programs and exact gains on eight development sources while saving **1.27%** of planning transitions. A [serial implementation benchmark](docs/FRONTIER_TIMING_RESULTS.rst) finds **58.68?86.26%** less preparation time on one fully certified root; this is not a population latency claim. [Named learned continuations](docs/CONTINUATION_PORTFOLIO_RESULTS.rst) expand development coverage but reduce total gain, so neither candidate family is promoted.

These are distinct comparisons under fixed rules and opponents. The search results measure expected win-score changes under the declared hidden-card distribution. They do not establish strongest-AI or equilibrium performance.

See the [manuscript and build instructions](docs/paper/README.rst), [work log and evidence entry points](docs/WORKFLOW.md), and [complete incremental results](docs/FRONTIER_INCREMENTAL_RESULTS.rst). The locally built PDF is `output/pdf/continuation_and_pruning_revision/manuscript.pdf`; PDFs, checkpoints and raw runs are ignored build or research artifacts.

## Development

Read the compact current handoff before resuming project work:

```powershell
./.venv/Scripts/python.exe -X utf8 -m scripts.project_work brief
```

Work logs and history search use a local SQLite store with a tracked recovery journal. See the [workflow guide](docs/WORKFLOW.md). The old STATUS and EXPERIMENTS files are frozen history; new work is recorded through the CLI.

Run the test suite or a short training example after setup:

```powershell
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe -m koikoi.train --config configs/smoke.json --run runs/smoke --updates 10
```

The smoke run checks execution, not playing strength. CUDA tests skip on CPU-only hosts.

- [Training and evaluation commands](docs/USAGE.md)
- [Architecture](docs/DESIGN.md) and [game rules](docs/RULES.md)
- [Experiment records](docs/EXPERIMENTS.md) and [artifact restoration](docs/ARTIFACTS.rst)
- [Third-party sources and licenses](THIRD_PARTY_NOTICES.md)
