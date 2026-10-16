# Design and research decisions

## Objective

Build a reproducible Koi-Koi research system; exceed the published RL-WP baseline under its original rules, then evaluate a separately identified twelve-round variant. A minimum of 100 million complete twelve-round-format matches is the eventual scale target, not a claim about completed work. Search simulations do not count toward that target.

## Architecture

- A readable Python reference engine is the correctness oracle.
- One allocation-free C++ transition core compiles as a CPU shared library and CUDA kernels. GPU state, observations, legal masks and rewards remain on device.
- NVRTC compiles CUDA kernels using the CUDA libraries supplied by PyTorch; no system-wide CUDA toolkit is required. CUDA Driver API launches kernels on PyTorch's current stream.
- A masked policy/value network sees only acting-player observations, public within-round history and match context. Optional opponent-card prediction uses hidden cards only as supervised labels.
- PPO with signed, player-relative returns is the initial training baseline. Multiple decisions by one player must NOT flip value signs. Round boundaries do not terminate match returns.
- PPO versus Q-boosting, flat versus explicit public features versus card-token networks, current versus archived opponents, belief versus uniform sampling, and search versus no search are separate experiments. The VRPO-inspired option and its deviations are documented in docs/VRPO.md; no improvement or equilibrium guarantee is presumed.
- Search samples consistent hidden states, then calls observation-only policies for both seats. No perfect-information optimization of sampled hidden states.
- The relational policy scores each card using shared weights over its encoding, same-month hand/field aggregates, and public global context. Compare it against the flat and Transformer policies; do not assume an architecture advantage.
- Public match-outcome dominance rules are an explicit inference wrapper, also applied to the published comparison opponent. Optional `endgame=true` training restricts the policy mask before action sampling and records the same mask for PPO; no off-policy post-sampling replacement occurs. Such checkpoints automatically retain this assistance in inference and search, with actual assistance settings in reports. The restriction preserves best attainable match outcome; it is disallowed for point-reward training. Simulator rules and rewards are unchanged.
- Search compares paired particle returns against the deployment policy's base action. The standard-error margin can reduce noisy action changes but does not remove critic bias or establish equilibrium or strength guarantees. Search remains disabled unless measured improvement justifies its cost.

## Evaluation

Seeded deals are indexed by match and round, independent of actions. A pair repeats the same seat-assigned deals and initial dealer with agent identities swapped. Confidence intervals resample independent pairs, not individual games. Final test seeds are disjoint from training and development seeds. Report draw-adjusted match win rate, point difference, latency and training budget. Match rewards are +1/0/-1 for win/draw/loss; optional round-point warmup must be labeled and disabled for final win-rate training.

The original upstream model is evaluated on its eight-round, bankruptcy-ending rules. Its historical feature quirks must be preserved in the adapter, never silently repaired for published weights. Twelve-round transfer is a separate result.

Large deterministic checkpoint duels also run in native batches: duplicate the complete initial state, exchange agent seats, and freeze finished rows. No redealt episodes enter an evaluation pair. Check both devices/profiles against sequential reference duels and record batch shape/device, since floating-point near-ties can differ across execution paths. The native endgame guard reads only captured cards, scores, phase and public match context.

## Reproducibility

Checkpoint model, optimizer, RNG, simulator state, config, counters, code revision and source hashes. Atomic checkpoint replacement and JSONL metrics permit restart. Store generated artifacts under ignored runs/. Pin external sources and retain license notices. Code and experiment protocols are tracked; local agent instructions are ignored.

Exact resume preserves the experiment config and counters. Weight-only initialization creates fresh optimizer, RNG, simulator and counters and records the pretraining source separately. Paper8 pretraining never counts toward the research12 match requirement. Initialization lineage survives subsequent resumes.

Optional CUDA graphs capture only the fixed-size PPO minibatch update. Rollout, sampling and counter bookkeeping stay eager. Warmup and capture effects on parameters, optimizer tensors and RNG are restored in place before the first real replay. Static tensor addresses and optimizer hyperparameters must stay fixed after capture. Graph-mode Adam uses capturable device-side counters; comparisons against eager execution use identical optimizer settings, while the older non-capturable mode can differ through rounding. See the [PyTorch 2.12 capture guidance](https://docs.pytorch.org/docs/2.12/notes/cuda.html#cuda-graphs).

## Research gates

1. Rule fixtures and independent upstream differential tests.
2. Python/CPU/CUDA equality and observation noninterference.
3. End-to-end throughput and restartable PPO smoke run.
4. Baseline evaluation and multi-seed controlled comparisons.
5. Belief/search experiments and at least 100M complete twelve-round training matches. Once correctness and positive learning are verified, a clearly labeled baseline scale run can proceed alongside method selection; it does not waive the strength or held-out evaluation gates.
6. Held-out results, ablations and manuscript. Submission venue and actual submission are later decisions.
