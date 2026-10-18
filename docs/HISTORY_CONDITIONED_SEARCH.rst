Finite opponent models and history-conditioned worlds
====================================================

Motivation and prior evidence
-----------------------------

The retained Transformer all-action adapter uses uniform observation-consistent
worlds and self-play continuations. It does not score the likelihood of observed
opponent actions. The earlier MLP diagnostics already established that this
assumption can change candidate-action values: among nine enumerated late roots,
one known-WP value changed sign under history conditioning. Those selected roots
did not establish a policy improvement and are not a new evaluation set.

The missing implementation is a deployable history-conditioned comparison with
the retained Transformer, including a declared opponent-model uncertainty. The
new module scripts/history_conditioned_worlds.py implements its world-generation
component. It does not yet select actions or replace the playable agent.

This is established Bayesian conditioning and determinization methodology, not
a new general algorithm. Information-set search and opponent inference have a
substantial literature. In particular, Goodman (2019), Re-determinizing
Information Set Monte Carlo Tree Search in Hanabi, uses a Bayesian opponent
model and discusses information leakage in simulated opponent decisions:
https://arxiv.org/abs/1902.06075 . The present finite-model sampler does not
implement that paper's re-determinizing tree search. Cowling, Powley and
Whitehouse (2012), Information Set Monte Carlo Tree Search, is already cited
in the manuscript. Any claimed contribution here must come from a validated
Koi-Koi construction and a measured improvement, not a renamed Bayes update.

Information and model contract
------------------------------

Only live paper8 roots are accepted. The own hand, public cards, observed
within-round actions, scores, dealer and seat determine candidate hands. The
actual opponent hand, unseen stock and deal RNG never provide evidence. Every
candidate substitutes the opponent's initial and current hands and canonicalizes
the unseen stock and RNG before replaying past actions. Canonical stock is used
only for past-action replay, never for evaluating future outcomes.

Callers provide named, immutable, deterministic, observation-only opponent
policies. A proposed search application uses assisted RL-WP and RL-Point with
equal prior mass; the actual opponent identity is not an input. This is a finite
model assumption, not coverage of arbitrary humans or other agents. Strict CPU
singleton inference defines historical compatibility; a future batched rollout
model must identify and validate its separate numerical behavior.

Every past action is replayed, including actions after an opponent-model
disagreement. Our own past actions are applied but are not scored as opponent
evidence. Every hypothesis reconstructs the complete current simulator state.
The initial four-of-a-month hand rejection remains the existing game rule.

Let H be the set of valid current hands and m the declared model. Under the
shared uniform prior over H and model prior p(m), a compatible (h,m) pair has
unnormalized mass p(m)/|H|. An incompatible pair has zero mass. The common
denominator cancels when normalizing all pairs jointly. Normalizing each model's
compatible hands separately would incorrectly discard the history's evidence
about the model identity. A hand compatible with two models retains both masses.

The supplied model prior must precede this round's evidence. Passing an earlier
root's posterior from the same round would count the repeated history twice.
No cross-round belief updater is implemented by this component.

Bounded execution and refusal
-----------------------------

Before policy inference, count all unknown-card combinations of the required
opponent-hand size. The default cap is100 combinations. Larger spaces return
too_many_hands without sampling or policy calls. The cap is public and is not
selected from outcomes. Eligible roots enumerate every combination, including
explicit accounting of invalid initial hands.

Support is bound to the complete root observation and public seat/dealer labels.
Sampling with stale support is rejected. Empty joint support is also rejected:
it does not silently become uniform or trigger repeated sampling until success.
A future search adapter must retain its base action on either refusal.

Accepted samples draw the hand and model together, then uniformly shuffle the
remaining stock and generate a fresh future-deal seed. The model index accompanies
each world and must persist across its paired candidate branches. Root decisions
must average these particles; they must not select an action after seeing one
particle's hidden hand. Continuation policies receive their own legal observations.

Validation and remaining work
-----------------------------

Four focused tests cover joint normalization with unequal priors and overlapping
support, empty-model refusal, a public budget refusal before inference, actual
hidden-world substitution, full-deck preservation, fresh random stock/seeds,
root immutability and stale-support refusal. A fixed retained parent1 root with
twelve candidate hands also compares parent/Point support with independent
native CPU history replay. Its actual hand is supported by its generating parent.
These are correctness tests, not a strength sample or opponent-identification
accuracy estimate. No new training or strength comparison has been launched.

Next integrate paired terminal rollouts while retaining strict fallback actions,
the public endgame guard and explicit refusal behavior. Verify all continuation
routes and full trajectories on CPU/CUDA, including model labels through seat
changes and round boundaries. Measure actual replay and branch cost before fixing
a fresh strength protocol. Compare against both the unchanged parent and the same
opponent-mixture rollout with unconditioned hands, so a change in continuation
opponent is not mistaken for an effect of history conditioning. Preserve all
completed study artifacts, the canonical playable model and reserved101M deals.
