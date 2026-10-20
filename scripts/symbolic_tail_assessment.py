"""Independent adoption checks and paired execution of symbolic tail programs."""

from collections import Counter
from dataclasses import replace

from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from scripts.check_belief_tail import add_step, empty_work
from scripts.diagnose_history_support import reference
from scripts.export_belief_tail import replay_actions
from scripts.incumbent_tail_controller import select_improvement
from scripts.symbolic_final_turn_tail import SymbolicExecutor
from scripts.verify_symbolic_tail import verify_program


def verify_adoption(worlds, phases, selection, parent, opponents, cache=None):
    """Replay the unchanged parent reference and the exact program being adopted."""
    expected = select_improvement(phases["fixed"], phases.get("optimized"))
    if selection != expected:
        raise ValueError("Selection does not match the evaluated phases")
    if selection.plan is None:
        return {}
    baseline = phases["fixed"].result.plan
    checks = {}
    for label, plan, action in (
        ("parent", baseline, baseline.baseline_action),
        ("selected", selection.plan, selection.plan.action),
    ):
        # Keep the versioned continuation rule as part of the executable program.
        view = replace(
            plan,
            action=action,
            values={action: plan.values[action]},
            root_policies={action: plan.root_policies[action]},
            policy=plan.root_policies[action],
        )
        checks[label] = verify_program(worlds, view, parent, opponents, cache)
    return checks


def assess_three(source, selections, parent, opponent, parent_id):
    """Execute three conditional tails, accounting for symbolic parent calls."""
    if set(selections) != {"root_only", "controller"}:
        raise ValueError("Exactly the two planned search arms are required")
    binding = selections["controller"].binding
    if selections["root_only"].binding != binding or binding.viewer != source["seat"]:
        raise ValueError("All arms must share the same root and viewer")
    outcomes, actual, replay, calls = {}, Counter(), Counter(), Counter()
    for arm in ("parent", "root_only", "controller"):
        env = reference(source["selected_root"]["state"])
        executor = (
            None
            if arm == "parent"
            else SymbolicExecutor(selections[arm], parent, parent_id)
        )
        actions, work = [], empty_work()
        for _ in range(32):
            if env.state[PHASE] == TERMINAL:
                break
            if env.state[PLAYER] == source["seat"]:
                before = 0 if executor is None else executor.symbolic_parent_calls
                action = int(parent.act(env)) if executor is None else executor.act(env)
                calls["learner_decisions"] += 1
                if executor is None or executor.policy is None:
                    calls["parent_adapter_calls"] += 1
                else:
                    extra = executor.symbolic_parent_calls - before
                    calls["symbolic_parent_adapter_calls"] += extra
                    calls["parent_adapter_calls"] += extra
            else:
                action = int(opponent.act(env))
                calls["opponent_adapter_calls"] += 1
            if not 0 <= action < len(env.legal()) or not env.legal()[action]:
                raise ValueError("Illegal conditional execution action")
            actions.append(action)
            add_step(work, env.step(action))
        if env.state[PHASE] != TERMINAL:
            raise ValueError("Conditional execution exceeded the finite tail")
        outcome = {
            "actions": actions,
            "work": work,
            "final": env.state.tolist(),
            "margin": int(
                env.state[POINTS + source["seat"]]
                - env.state[POINTS + 1 - source["seat"]]
            ),
        }
        outcomes[arm] = outcome
        actual.update(work)
        audited = replay_actions(
            reference(source["selected_root"]["state"]), actions, outcome["final"]
        )
        assert audited == work
        replay.update(audited)
    assert outcomes["parent"]["final"] == source["final"]
    assert (
        outcomes["parent"]["actions"]
        == source["actions"][source["selected_root"]["decision"] :]
    )
    for arm, selection in selections.items():
        if selection.plan is None:
            assert outcomes[arm] == outcomes["parent"]
    return {
        "outcomes": outcomes,
        "actual_work": dict(actual),
        "replay_work": dict(replay),
        "calls": dict(calls),
    }
