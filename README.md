# Koi-Koi

Play Koi-Koi against trained AI opponents, or explore self-play learning and verified endgame search. The project includes a local desktop game, Python/C++/CUDA simulators, and reproducible comparisons with the published RL-WP and RL-Point models.

## Play

On a configured Windows machine, double-click **`play.cmd`**, or run this command from the repository folder:

```powershell
./.venv/Scripts/python.exe -m koikoi.play
```

The default opponent is the selected trained model. Gameplay runs locally on CPU and does not require a GPU. The interface is in Japanese.

1. Click a gold-bordered hand card to play it. When two field cards match its month, click the one to capture.
2. Stock cards are drawn automatically. When offered a choice, select **Stop** or **Koi-Koi**.
3. Select **Next round** to continue. Use **Restart** after changing the opponent in the top menu.

Matches use the benchmark's eight-round rules: 30 starting points each, with early termination at zero points. Some scoring conventions differ from other Koi-Koi variants; see the in-game help and [rules](docs/RULES.md).

You can also choose an opponent from the command line:

```powershell
./.venv/Scripts/python.exe -m koikoi.play --opponent rl-wp
./.venv/Scripts/python.exe -m koikoi.play --opponent rl-point
./.venv/Scripts/python.exe -m koikoi.play --opponent greedy
```

The trained opponent uses greedy inference with public endgame assistance. The expensive research search is separate from the playable game.

## Setup

The tested development setup is Windows with Python 3.12, Git, Visual Studio C++ build tools, and an NVIDIA GPU with a recent driver. The bootstrap creates a project-local environment, installs pinned dependencies, fetches the upstream models and card assets, and runs the tests:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
```

A system CUDA toolkit is unnecessary. Gameplay uses CPU after setup; Linux has not been validated here.

**The additional trained checkpoint is not included in Git.** Restore it using the [artifact instructions](docs/ARTIFACTS.rst) to use the default opponent:

```text
runs/cold_ppo_screen/seed6710001_warm/update_00000520.pt
```

If that checkpoint is unavailable, launch with `--opponent rl-wp`, `rl-point`, or `greedy`. A missing model produces an error rather than silently selecting another opponent. If `play.cmd` fails, use the PowerShell command above to see the error. All opponents use card assets from the pinned upstream checkout.

## Research results

The main method verifies local improvements from public play even when a computation limit prevents completing the parent policy's root evaluation. Complete future beliefs and executable local programs let their certified gains combine into an expected match improvement.

| Comparison | Completed evidence |
| --- | --- |
| Self-play adaptation vs. assisted RL-WP | Both trained seeds exceed 50% on held-out duplicate deals: **51.480%** and **51.445%**. The separate RL-Point criterion does not replicate. |
| Base search vs. its parent | On 8,192 sources, expected gains are positive in all four parent/opponent populations. Mean opponent-mixture gains are **0.2819 / 0.2143 percentage points**; the joint 0.1-point lower-bound target is unmet. |
| Frontier extension vs. base search | On 4,096 sources, both opponent mixtures have positive simultaneous lower bounds. Mean added gains are **0.00128 / 0.000784 percentage points**. It improves 26 source beliefs; realized match scores are unchanged and computation increases. |

The new [256-source proposal comparison](docs/PROSPECTIVE_PROPOSAL_RESULTS.rst) finds 22.00% fewer planning transitions with 93.72% of the exact gain; its superiority criterion fails. A [nine-rule transfer investigation](docs/RULE_TRANSFER_RESULTS.rst) completes 182,272 matches, with both separate confirmation cohorts failing the joint criterion. Neither result promotes a new default.

These are distinct comparisons under fixed rules and opponents. The search results measure expected win-score changes under the declared hidden-card distribution. They do not establish strongest-AI or equilibrium performance.

See the [manuscript and build instructions](docs/paper/README.rst), [work log and evidence entry points](docs/WORKFLOW.md), and [complete incremental results](docs/FRONTIER_INCREMENTAL_RESULTS.rst). The locally built PDF is `output/pdf/proposal_and_rule_results_revision/manuscript.pdf`; PDFs, checkpoints and raw runs are ignored build or research artifacts.

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
