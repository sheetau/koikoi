import pickle
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from scripts.diagnose_history_support import reference
from scripts.frontier_policy_patch import (
    FrontierExecutor,
    parent_frontier,
    solve_frontier_patch,
)
from scripts.interval_tail import IntervalExecutor
from scripts.verify_frontier_patch import verify_frontier_patch
from tests.test_symbolic_final_turn_tail import dynamics, setup


def solve(actor, worlds, **kwargs):
    return solve_frontier_patch(
        worlds,
        dynamics(actor, worlds),
        "parent",
        {"fixture": "fixture"},
        draws=2,
        max_decisions=kwargs.pop("max_decisions", 10000),
        prefix_decisions=4000,
        per_phase_decisions=2000,
        **kwargs,
    )


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("unequal", [False, True])
def test_local_gain_decomposes_full_policy_difference(swap, unequal):
    actor, _, worlds = setup(swap)
    if unequal:
        worlds = replace(
            worlds, weights=worlds.weights * np.arange(1, len(worlds.weights) + 1)
        )
    patch = pickle.loads(pickle.dumps(solve(actor, worlds)))
    assert patch.frontier.complete and len(patch.frontier.nodes) > 1
    assert patch.gain_lower > 0
    report = verify_frontier_patch(
        worlds, patch, actor, {"fixture": actor}, "parent", full_values=True
    )
    assert Fraction(report["exact_increment"]) >= patch.gain_lower
    assert patch.decisions <= 10000
    assert patch.frontier.work["decisions"] == report["prefix_physical"]["decisions"]
    assert (
        patch.frontier.terminal_mass
        + sum(sum(map(int, node.weights)) for _, node in patch.frontier.nodes)
        == patch.frontier.total_mass
    )
    if not unequal:
        assert patch.gain_lower == Fraction(53, 550)
        assert report["full_values"]["parent"]["value"] == "204/275"
        assert report["full_values"]["patched"]["value"] == "461/550"
        assert (
            patch.decisions
            < report["full_values"]["parent"]["physical_work"]["decisions"]
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_decisions": 0},
        {"max_decisions": 50},
        {"max_rows": 5},
        {"max_depth": 1},
    ],
)
def test_incomplete_prefix_never_exposes_partial_conditional_beliefs(kwargs):
    actor, _, worlds = setup()
    result = parent_frontier(
        worlds,
        dynamics(actor, worlds),
        "parent",
        {"fixture": "fixture"},
        draws=2,
        **kwargs,
    )
    assert not result.complete and result.refusal is not None
    assert result.nodes == ()
    assert (
        result.work["arrival_mass"] + result.terminal_mass + result.pending_mass
        == result.total_mass
    )
    if "max_decisions" in kwargs:
        assert result.work["decisions"] <= kwargs["max_decisions"]


@pytest.mark.parametrize("budget", [1, 400, 1000])
def test_shared_budget_counts_prefix_and_every_local_attempt(budget):
    actor, _, worlds = setup()
    patch = solve(actor, worlds, max_decisions=budget)
    assert patch.decisions <= budget
    assert patch.decisions == patch.frontier.work["decisions"] + sum(
        result["decisions"] for _, result in patch.comparisons
    )
    if patch.frontier.complete:
        report = verify_frontier_patch(
            worlds, patch, actor, {"fixture": actor}, "parent", full_values=True
        )
        assert Fraction(report["exact_increment"]) >= patch.gain_lower >= 0
    else:
        assert not patch.comparisons and patch.gain_lower == 0


def test_changed_local_belief_duplicate_or_gain_is_rejected():
    actor, _, worlds = setup()
    patch = solve(actor, worlds)
    broken = deepcopy(patch)
    _key, node = broken.frontier.nodes[0]
    node.weights[0] += 1
    with pytest.raises(ValueError, match="belief"):
        FrontierExecutor(broken, actor, "parent")
    with pytest.raises(ValueError, match="gain"):
        FrontierExecutor(
            replace(patch, gain_lower=patch.gain_lower + 1), actor, "parent"
        )
    with pytest.raises(ValueError, match="repeated"):
        FrontierExecutor(
            replace(patch, comparisons=patch.comparisons + patch.comparisons[:1]),
            actor,
            "parent",
        )
    with pytest.raises(ValueError, match="parent identity"):
        FrontierExecutor(patch, actor, "other")
    broken = deepcopy(patch)
    broken.frontier.nodes = broken.frontier.nodes[1:]
    broken.comparisons = broken.comparisons[1:]
    # This first node is unpatched; dropping it must still fail the prefix oracle.
    broken.gain_lower = sum(
        Fraction(
            sum(map(int, dict(broken.frontier.nodes)[k].weights)),
            broken.frontier.total_mass,
        )
        * r["selection"].gain_lower
        for k, r in broken.comparisons
    )
    with pytest.raises(ValueError, match="posterior or mass"):
        verify_frontier_patch(worlds, broken, actor, {"fixture": actor}, "parent")


def test_executor_is_public_root_bound_and_commits_to_one_patch():
    actor, root, worlds = setup()
    patch = solve(actor, worlds)
    executor = FrontierExecutor(patch, actor, "parent")
    changed = root.clone()
    changed.state[POINTS + worlds.viewer] += 1
    with pytest.raises(ValueError, match="root changed"):
        executor.act(changed)
    env = root.clone()
    active = None
    while env.state[PHASE] != TERMINAL:
        action = (
            executor.act(env) if env.state[PLAYER] == worlds.viewer else actor.act(env)
        )
        if active is not None:
            assert executor.active is active
        active = executor.active
        assert env.legal()[action]
        env.step(action)
    with pytest.raises(ValueError, match="live learner"):
        executor.act(env)


def test_patch_activation_matches_its_certified_local_executor():
    actor, root, worlds = setup()
    patch = solve(actor, worlds)
    key, result = next(
        (k, r) for k, r in patch.comparisons if r["selection"].candidate is not None
    )
    executor = FrontierExecutor(patch, actor, "parent")
    executor.act(root)
    local = IntervalExecutor(result["selection"], actor, "parent")
    env = reference(dict(patch.frontier.nodes)[key].states[0])
    while env.state[PHASE] != TERMINAL:
        if env.state[PLAYER] == worlds.viewer:
            action = executor.act(env)
            assert executor.entered and executor.active is not None
            assert action == local.act(env)
        else:
            action = actor.act(env)
        env.step(action)


def test_parent_continues_after_an_unpatched_frontier():
    actor, root, worlds = setup()
    patch = solve(actor, worlds)
    key, _ = next(
        (k, r) for k, r in patch.comparisons if r["selection"].candidate is None
    )
    executor = FrontierExecutor(patch, actor, "parent")
    executor.act(root)
    env = reference(dict(patch.frontier.nodes)[key].states[0])
    while env.state[PHASE] != TERMINAL:
        action = actor.act(env)
        if env.state[PLAYER] == worlds.viewer:
            assert executor.act(env) == action
            assert executor.entered and executor.active is None
        env.step(action)


def test_screen_actual_suffix_and_replay_use_the_committed_executor():
    from scripts.screen_frontier_patch import assess

    actor, root, worlds = setup()
    patch = solve(actor, worlds)
    env, actions = root.clone(), []
    while env.state[PHASE] != TERMINAL:
        action = actor.act(env)
        actions.append(action)
        env.step(action)
    source = {
        "selected_root": {"state": root.state.tolist(), "decision": 0},
        "seat": worlds.viewer,
        "actions": actions,
        "final": env.state.tolist(),
    }
    report = assess(source, patch, actor, actor, "parent")
    assert report["actual_work"] == report["replay_work"]
    assert report["actual_work"]["completions"] == 2
    assert report["outcomes"]["parent"]["actions"] == actions
