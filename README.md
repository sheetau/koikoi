# Koi-Koi research

GPU-parallel Koi-Koi simulation, self-play PPO and Q-boosting, published-baseline evaluation, and belief-sampled inference rollouts. This is a research implementation; no strongest-AI or equilibrium claim has been established. See [current evidence](docs/STATUS.md), [rules](docs/RULES.md), and [architecture](docs/DESIGN.md).

## Setup (Windows / NVIDIA)

Requires Python 3.12, a recent NVIDIA driver, Git, and Visual Studio C++ build tools. The project-local environment leaves system Python unchanged. NVRTC is loaded from the CUDA PyTorch wheel; a system CUDA toolkit is unnecessary.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
```

The first native CPU build is cached under ignored `build/`; kernels compile at process startup. Linux uses `c++` and the CUDA PyTorch NVRTC library path, but has not been validated on this machine.

## Validate and measure

```powershell
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe -m koikoi.benchmark --envs 4096 --steps 1000 --policy
```

Tests compare independent Python rules with C++ and CUDA, scoring and full matches with the pinned original implementation, model inputs with replayed original inputs, hidden-information invariance, sampling constraints, signed returns, and exact CPU checkpoint continuation. CUDA tests skip explicitly on CPU-only hosts. Upstream tests skip unless fetched.

## Batched published-model inference

After fetching the pinned upstream checkout, the optional paper8 adapter runs its original weights against native300-feature inputs on CPU or CUDA:

```python
from koikoi.env import BatchEnv
from koikoi.upstream_batch import BatchPublishedAgent

env = BatchEnv(256, "cuda", profile="paper8")
agent = BatchPublishedAgent("wp", "cuda", endgame=True)
actions = agent.act(env)  # int32(B,), native card IDs or STOP48/CONTINUE49
```

The adapter supports `wp` and `point`; `env.published_observe()` exposes the padded features and legal mask. Terminal actions are-1. Batch/device arithmetic can differ from singleton CPU inference; record batch sizes and TF32 settings for experiments. The strict published opponent in `koikoi.evaluate` stays unchanged. Training integration is optional as described below; exact CPU history-support certification still uses the strict adapter. Compatibility/numerical/latency evidence is in [EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Train

```powershell
./.venv/Scripts/python.exe -m koikoi.train --config configs/smoke.json --run runs/smoke --updates 10
./.venv/Scripts/python.exe -m koikoi.train --config configs/ppo8.json --run runs/ppo8 --updates 1000
./.venv/Scripts/python.exe -m koikoi.train --resume runs/ppo8/latest.pt --run runs/ppo8 --updates 2000
```

`--updates` is an absolute cumulative update target. `--target-matches` instead trains until the cumulative completed-match counter reaches the specified number (possibly overshooting by a rollout). `--checkpoint-every` controls completed-update snapshots. Ctrl+C retains the last completed snapshot, never a partial update. Resuming modified source requires explicit `--allow-code-change` and creates a new provenance record; it is not an exact continuation experiment.

For an experimental fixed historical opponent, set `opponent_fraction` in the training config (for example, `0.2`) and pass `--frozen-opponent runs/old/update_00001000.pt` on a new run. This assigns floor(envs * fraction) environments to the frozen opponent and keeps the remaining environments in current-policy self-play. Learner seats alternate between completed matches. The opponent must use the same rule profile and endgame setting. PPO excludes opponent actions from learner policy/entropy/belief updates and uses a separate training-only critic for opponent turns. No strength benefit is assumed.

For an experimental published opponent, set `published_opponent` to `"wp"` or `"point"` with a positive `opponent_fraction`, or use `configs/ppo8_published_wp50.json`. This requires paper8 match-reward PPO and `cuda_rollout=false`; `cuda_graph=true` still captures optimizer minibatches. The teacher uses deterministic argmax and shares the learner's endgame setting. It explicitly disables TF32 and cuDNN benchmarking for this process. Checkpoints embed teacher tensors and record source, original weight hashes, parameters and runtime; resume still requires the pinned upstream source checkout. Do not also pass `--frozen-opponent`. Additional `published_*` counters separate teacher exposure from self-play. The exported learner does not require the teacher. No strength benefit is established.

```powershell
./.venv/Scripts/python.exe -m koikoi.train --config configs/ppo8_published_wp50.json --run runs/published_smoke --updates 2
./.venv/Scripts/python.exe -m koikoi.benchmark --training-config configs/ppo8_published_wp50.json --steps 20
```

To measure full learning throughput with a fixed opponent, use `python -m koikoi.benchmark --training-config configs/relational8_history20.json --frozen-opponent runs/old/update_00001000.pt --steps 20`. The benchmark includes opponent inference and optimizer updates, excludes five warmup updates, and reports both simulator and learner decisions per second. It starts a fresh learner; it is a throughput measurement, not a continuation or strength experiment.

Frozen weights and provenance are embedded in checkpoints; `--resume` needs no opponent path and refuses replacement. Reports record `learner_decisions` separately from total `decisions`. The budget auditor treats old pure-self-play decisions as learner decisions, excludes frozen-opponent pretraining from learner ancestry, and keeps profiles separate. Disclose the opponent's training budget separately using its saved provenance. CUDA capture includes both trainable networks in rollback; the frozen network is never optimized. Inference still loads only the learner actor.

For an experimental separate PPO return critic, set `ppo_value` to `observation` or `privileged`; `shared` preserves the original value-head behavior. Both separate modes use the same scalar MLP and initialization. The observation control pads 96 zero features; the privileged condition adds opponent hand membership and remaining stock ranks to the same 560 observation features. Only training GAE and its tail bootstrap use this critic. The deployed actor's observation-only value head is still trained on the return target for search; both value losses have `value_weight`. The extra critic shares Adam and gradient clipping with the actor but has separate parameters. `critic_loss` reports its loss. This currently supports pure-self-play PPO only, and uses PPO `epochs` rather than Q-boosting `critic_epochs`.

Example configs `configs/relational8_value_observation.json` and `configs/relational8_value_privileged.json` differ only in `ppo_value`. Use `--initialize` to start a fresh controlled experiment from the same actor checkpoint. Exact resume embeds the auxiliary weights and optimizer state; ordinary inference requires only the original actor. This is an empirical experiment, with no strength or unbiasedness guarantee.

Optional `complete_matches=true` collects exactly one full match per environment before each PPO update. Use `configs/ppo8_complete_mc.json` for terminal-outcome returns (`gae_lambda=1`); a matched complete-match control can retain `gae_lambda=0.95`. In this mode `horizon` is a hard collection limit: reaching it with unfinished matches raises an error instead of bootstrapping. Completed environments freeze until all finish, then every environment resets. Terminal padding is removed before losses and advantage normalization. `decisions` counts only actual active decisions; `collection_rows` also counts frozen executed rows. Batch length and optimizer-step count vary with match length. Initially this supports pure self-play match PPO with shared value, all-action policy, and both CUDA capture options disabled. Resume restores the mode and exact next update; deployment uses the ordinary actor. This is experimental and has no established strength benefit.

Optional `round_continuation` embeds eight `[score_slope, dealer_effect]` coefficient pairs and calibration source SHA256 identities in a complete-match paper8 config with `gae_lambda=1`. At each nonterminal round end it supplies a frozen public continuation value; actual match ends use the true win/draw/loss. Learning traces end at round boundaries, while game termination and match counters retain their original meaning. It requires weight initialization from the calibrated parent and checks table identity on resume. The table is training-only and may become inaccurate as the policy changes. This is an experimental learning target, not an established improvement. The fixed two-parent protocol is in `scripts/run_round_continuation_screen.py`.

`koikoi.published_actor.PublishedActor` is an experimental differentiable copy of the pinned published score networks. It consumes native public/own-only features and keeps dropout disabled; snapshots retain original initialization identities and current weights. It is currently separate from the PPO trainer and returns raw Q scores, so it supplies neither an implicit action distribution nor a demonstrated strength improvement. The original frozen comparison adapters are unchanged.

PPO uses the native C++/CUDA signed-return scan by default. Set `returns_backend` to `torch` to use the independent PyTorch recurrence. Tests verify bit-identical float32 returns and full PPO updates; `python -m koikoi.benchmark --returns --envs 2048 --horizon 128 --steps 50` measures this component separately from learning.

`cuda_graph=true` captures fixed-size PPO minibatch updates on CUDA. Warmup/recording updates are rolled back, including Adam moments and step counters, so they do not add hidden training steps. Graph checkpoints resume their exact next update; tests also compare graph steps with eager steps using the same capturable Adam settings. Capturable Adam can round differently from the older non-capturable optimizer, so switching modes is a distinct experiment, not a bit-identical continuation. `configs/ppo12_graph.json` enables this path. Dynamic minibatch sizes and CPU/Q-boosting capture are rejected.

`cuda_rollout=true` also captures the fixed-horizon PPO collection and return calculation. Preparation rolls back simulator state and RNG; Python counters advance only on real replay. The captured policy reads current model weights, including BF16 casts. Tests compare full trajectories and optimizer updates with ordinary collection, including match boundaries, and verify checkpoint continuation. This option supports CUDA PPO only. Add `--cuda-rollout` to a full-learning benchmark to measure it.

For an existing CUDA PPO run, first stop cleanly with its run-local STOP file, verify the process exited, then resume its completed checkpoint in a fresh directory with `--allow-code-change --enable-cuda-rollout`. The explicit execution change retains model, optimizer, simulator, RNG and cumulative counters; source lineage records the change. Starting an additional trainer against the live run's directory is not a continuation.

Measure full learning, excluding five warmup updates, with `python -m koikoi.benchmark --training-config configs/ppo12.json --steps 20`; add `--cuda-graph` for captured updates. Unlike simulator-only throughput, this includes rollout, return estimation and gradient updates. Configuration comes from the supplied file.

`--max-seconds` sets a wall-time limit checked between completed updates. Creating `runs/<run>/STOP` also requests a clean checkpoint and exit; remove that file before resuming. Starting a new training run in a directory with existing metrics is rejected. Checkpoints record the parent checkpoint hash when continued.

`--initialize runs/<source>/update_XXXXXXXX.pt` starts a new experiment from actor weights with the architecture specified in `--config`. It resets optimizer, simulator, RNG and counters, recording the source checkpoint hash and its training budget separately. This supports changing rewards or transferring from paper8 to research12 without counting eight-round pretraining toward twelve-round matches. `--resume` instead restores the entire experiment and cannot be combined with `--config`.

Audit an immutable local checkpoint with `python -m koikoi.budget runs/<run>/update_XXXXXXXX.pt --output budget.json`. Totals group completed learning by rule profile and add weight-initialization ancestry without double-counting resumed counters. This is one model lineage; reports with shared ancestors must not be summed.

The baseline scale run completed 100,001,204 new twelve-round matches. Its audited final snapshot is `runs/ppo12_scale_seed30001_captured/final_100m.pt`; see [STATUS.md](docs/STATUS.md) for counters, initialization lineage and strength results. Do not restart the completed run from an older directory. This meets the computation target; public-baseline superiority and final validation remain unproven.

`paper8` and `research12` are different versioned protocols. Do not compare published eight-round weights on a twelve-round match. Config `reward=round` is a separately labeled point-reward experiment, not the default win-rate objective. `kind=transformer` selects card-token attention; `belief_weight>0` trains the hidden-card auxiliary predictor. Policy inputs never receive its ground-truth labels.

Additional comparison configs: `vrpo8.json` selects a separate centralized Q critic ([implementation scope and deviations](docs/VRPO.md)); `features8.json` adds public role/month counts to the MLP; `transformer8_bf16.json` uses mixed precision with float32 loss calculations. The Q-boosting option is experimental, not a proven improvement. Compare by both decision budget and wall time, with multiple independent training seeds.

`ppo8_wide_bf16.json` screens a width-512 MLP with BF16 and CUDA Graph updates. Match its decision budget to smaller-policy checkpoints; it is not yet a demonstrated strength improvement.

`relational8.json` scores candidate cards with shared weights, their same-month hand/field aggregates and a public global context. Its BF16 training configuration is an architectural experiment; current evidence does not establish a strength advantage.

`ppo8_endgame.json` trains with public match-outcome dominance rules enabled in the policy mask before sampling. Checkpoints retain this behavior automatically during inference and search; reports show the actual assistance settings for each agent. `ppo8_control.json` is identical except for this flag, for controlled weight-initialized comparisons. This option requires match-outcome rewards; point difference is not its objective.

For experimental koi-only adaptation, set `policy_scope="koi"` with `ppo_value="observation"`, match rewards and zero belief loss. Card-network weights stay frozen; only the koi head, optional koi residual network, observation-only actor value head and separate return critic are optimized. Policy loss, entropy, KL and advantage normalization use koi rows only; values and signed returns still cover the complete match. Card actions are sampled from the fixed card policy by default. With `deterministic_cards=true`, they instead use its legal argmax during learning, while koi actions remain sampled. This training option requires `policy_scope="koi"`; it cannot optimize trainable card decisions. It does not change inference defaults. `policy_decisions` counts koi rows separately from total simulator decisions in metrics/checkpoints; historical phase counts are not inferred by the lineage budget auditor.

`koi_width>0` adds a 560-input, two-hidden-layer tanh network whose two outputs are added to the existing koi logits. Its zero-initialized output preserves the parent policy, and initialization preserves sampling RNG. `--initialize` accepts a base checkpoint without this branch or an exactly matching branch width; incompatible residual widths are rejected. Checkpoints embed the complete inference model and restore its frozen/trainable split. Defaults `policy_scope="all"` and `koi_width=0` preserve existing behavior.

The matched `ppo8_koi_control.json`, `ppo8_koi_head.json` and `ppo8_koi_residual.json` configs compare full-policy continuation, koi-head-only adaptation and koi-head plus width-128 residual adaptation. All use the same fresh observation critic and initialization checkpoint. The experiment is a hypothesis about decision specialization, not an established strength improvement. The additional `ppo8_koi_head_detcards.json` and `ppo8_koi_residual_detcards.json` configs isolate deterministic frozen-card behavior from their respective sampled-card controls.

## Evaluate

```powershell
./.venv/Scripts/python.exe -m koikoi.evaluate --agent greedy --opponent rl-wp --pairs 100
./.venv/Scripts/python.exe -m koikoi.evaluate --agent runs/ppo8/latest.pt --opponent rl-wp --pairs 1000 --output runs/ppo8_vs_wp.json
./.venv/Scripts/python.exe -m koikoi.evaluate --agent runs/ppo8/latest.pt --opponent rl-wp-assisted --pairs 1000 --endgame --deterministic --output runs/ppo8_assisted.json
./.venv/Scripts/python.exe -m koikoi.evaluate --agent runs/smoke/latest.pt --opponent greedy --pairs 100 --search belief --particles 32 --search-horizon 32
```

`--search uniform` requires no trained belief head. Search initially focuses on koi/stop decisions; both rollout players use only their own observations. It is a finite-budget policy-rollout heuristic, not PIMC or a Nash-safe subgame solver. Report search time separately. Neural policies sample by default; `--deterministic` is a separate evaluation setting.

`--endgame` applies public-information decisions that guarantee the best possible match outcome, such as stopping when that wins the final round's match. `rl-wp-assisted` and `rl-point-assisted` apply the same rules to the original models, matching upstream's `AgentForTest` assistance. Use these opponents for assisted comparisons; the plain names evaluate the raw networks. The guard optimizes match outcome, not point difference.

`--search-all` extends rollouts to card decisions; `--device cuda` runs neural rollouts on GPU. Search uses the deployment policy's deterministic/sampled action setting and paired hidden worlds. The default `--search-confidence 2.5` switches away from the base action only when the paired estimated improvement exceeds 2.5 standard errors. This controls some sampling noise, not critic bias or multiple-comparison error. It is not a calibrated improvement guarantee. Current pilot evidence does not justify enabling search by default.

`--search-chunk-size 32` optionally divides deterministic captured search into blocks and stops once every sampled original match has finished. It requires `--deterministic --search-cuda-graph`; zero (the default) retains the full-horizon path. Unfinished particles still use the original horizon and tail value. Reports add `search_executed_decisions` for actual rollout transitions, separate from the logical full-horizon budget and graph preparation. This can skip post-terminal computation without changing the candidate returns; timings depend on position and hardware.

`--search-confirm-particles 1024` optionally checks a proposed action change on fresh hidden worlds before accepting it. It requires deterministic search; zero (the default) disables confirmation. Only the proposed action and the base action are compared, using the same confidence margin. A nonpositive confirmation margin retains the base action. Confirmation has its own RNG stream, reset together with the proposal stream for each duplicate deal. Reports separate proposal and confirmation work; existing total search counters include both stages. This reduces sample reuse in selection and acceptance but does not establish playing-strength improvement or correct opponent modeling.

`--search-cuda-graph` captures the fixed-horizon policy-rollout loop when search uses `--device cuda`. Hidden-world sampling and root action comparison remain outside capture. Graphs are cached by simulation batch size and rule profile, and read new sampled states, candidate actions and root-player sign on each replay. Preparation restores simulator state and CPU/CUDA RNG. Reports separate `search_preparation_decisions` (three executed warmup rollouts per new graph) from logical `search_simulated_decisions`; both are search work, not training matches. Initial preparation time is included in decision latency. This changes execution, not the search policy or confidence rule; default remains eager search and search itself remains off.

Each pair exchanges agent seats while retaining identical seat-assigned deals and initial dealer. Reports include pair-bootstrap confidence intervals, draw-adjusted win rate, point difference, decision latency, exact seeds and code provenance. Small smoke evaluations cannot establish superiority. Development seeds start at 1,000,000; reserve 100,000,000 and above for final tests (`--split test`). Do not tune against final test results.

Evaluation protocol version 2 independently seeds each pair and repeats policy RNG seeds across its two legs. Reports hash the checkpoint bytes actually loaded and record their training counters/provenance. Earlier pilot reports without a protocol version used one continuous policy RNG stream; do not combine their pair records with version 2.

Compare two version-2 reports on their shared pairs with `python -m koikoi.compare first.json second.json --output difference.json`. This reports first-minus-second score and point differences with paired confidence intervals, checking opponent identity and checkpoint hashes. Reports also count decisions by phase and search-induced action changes.

For large checkpoint-versus-checkpoint comparisons, `batch_evaluate` runs independent paired matches on the native CPU/CUDA simulator:

```powershell
./.venv/Scripts/python.exe -m koikoi.batch_evaluate --agent runs/new/latest.pt --opponent runs/old/latest.pt --profile research12 --pairs 10000 --endgame --output runs/new_vs_old.json
```

This path is deterministic and applies `--endgame` to both agents. It freezes completed rows instead of redealing, and reports actual match/round/decision counts plus batch throughput. Throughput is not single-decision latency. Published baselines and stochastic/search policies continue to use `koikoi.evaluate`. Native batched results are checked against the independent sequential reference for both rule profiles and both devices. Floating-point near-ties can still make different devices or batch shapes choose different actions; execution device and batch size are recorded. Development evaluation rejects seeds in the reserved final-test range.

`--koi-agent <agent>` replaces only the candidate's koi/stop policy, retaining its card policy. For example, combine a checkpoint `--agent` with `--koi-agent rl-wp` to diagnose which decisions limit strength. Reports identify both components and their checkpoint metadata; endgame assistance is determined by the effective koi policy and explicit flags. Published components still require paper8. This substitution cannot be combined with search, and is a separate composite policy rather than evidence about either component alone.

In the sequential evaluator, add `--opponent-endgame` to apply the guard to a checkpoint opponent as well. Reports explicitly record whether the opponent receives assistance; comparisons reject a mismatch.

Original models retain their historical input quirks. The loader adapts an obsolete PyTorch class name and loads the original parameter tensors into the same architecture on modern PyTorch. Fetch source with `python -m koikoi.upstream --fetch`; the pin and license are in [third-party notices](THIRD_PARTY_NOTICES.md). Load only trusted local training checkpoints: PyTorch pickle deserialization is used.

## Experimental pretrained-policy PPO

`koikoi.published_train` initializes three differentiable phase networks from the pinned original WP weights and trains them with the existing match-reward, signed-GAE PPO. Legal action probabilities are explicitly `softmax(original_scores / score_temperature)`, with temperature 0.05 by default. These Q scores are not calibrated probabilities; stochastic training differs from the original greedy policy. A separate public-observation value network learns without shared actor parameters. The first 20 updates train only that critic by default, then all three actor networks are unfrozen. These are experimental settings, not a demonstrated improvement.

```powershell
./.venv/Scripts/python.exe -m koikoi.published_train --run runs/pretrained_trial --updates 100
./.venv/Scripts/python.exe -m koikoi.published_train --run runs/pretrained_trial --resume runs/pretrained_trial/latest.pt --updates 200
./.venv/Scripts/python.exe -m koikoi.evaluate --agent runs/pretrained_trial/latest.pt --opponent rl-wp-assisted --profile paper8 --pairs 1000 --deterministic --endgame --output runs/pretrained_trial/evaluation.json
```

Use `--config path.json` for overrides to `PublishedConfig`. Defaults use paper8, 128 environments, horizon128, minibatch512, three epochs, actor LR3e-6 and value LR1e-4. `critic_width` controls the new value network; inherited `width`/`layers` do not resize the fixed upstream256-wide/two-layer actors. Dropout stays disabled with gradients enabled. This mode requires eager float32 self-play; search, batched evaluation, belief learning, frozen opponents and captured execution are currently unsupported. Use the sequential evaluator shown above.

Checkpoints embed trained actor/value weights, original-weight identities, Adam, environment and RNG state, and runtime settings. Resume requires the pinned source classes but does not reread original weight files; runtime/source changes are checked. `--updates` is the total update target, including critic warmup. Counters distinguish policy updates, critic-only updates, actual terminal matches/rounds/decisions and phase-specific optimizer steps. A match may span the end of warmup; warmup match counters count terminations during that period.

Run `python -m scripts.benchmark_published_ppo --output runs/new_benchmark_directory` for the fixed feasibility measurement: three critic-only updates followed by five full actor/value updates. It records full-learning throughput and allocator memory separately from initialization, warmup and checkpoint IO. This diagnostic is not strength evidence and contributes nothing to the completed100M research12 scale claim.
