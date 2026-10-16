# Q-boosting implementation scope

Source: Fan and Farina, *GAE Falls Short in Imperfect-Information Self-Play Reinforcement Learning*, arXiv:2605.19235v1, https://arxiv.org/html/2605.19235v1 (2026), equations 3.2–3.3 and section 4.

The implementation uses a separate centralized action-value critic, policy-weighted values over all legal actions, and signed Expected-SARSA traces. The policy receives only observations. The critic additionally receives opponent hand membership and remaining stock order, never simulator RNG identifiers. Nature transitions are folded into game steps, so deterministic-transition pathwise variance claims from the paper are not asserted for this implementation.

The actor uses on-policy PPO clipping and an entropy term (equivalent in gradient to KL toward uniform legal actions). Critic tensors are frozen during actor updates. Policy expectations and detached advantages are refreshed once per actor epoch. This is an explicit engineering deviation from refreshing at every minibatch. Critic regression uses only the current rollout, corresponding to the paper's no-additional-replay ablation. Bootstrapping occurs at truncated rollout boundaries; game terminations cut traces.

An observation-only value head is also fitted to the fixed Q targets for inference-time rollout truncation. It does not expose the centralized critic to search. This head is additional to the paper's actor objective and must be included in ablations.

Tests include matching pennies, where an exact centralized Q critic removes future opponent-action noise, plus same-player decisions, terminal masking, private-input separation, and CPU/CUDA checkpoint continuation. These validate the estimator implementation; they do not establish lower exploitability in full Koi-Koi.
