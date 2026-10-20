"""Execute and separately replay a frozen parent/controller tail comparison."""

from collections import Counter

from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from scripts.check_belief_tail import add_step, empty_work
from scripts.diagnose_history_support import reference
from scripts.export_belief_tail import replay_actions
from scripts.incumbent_tail_controller import ContingentExecutor


def assess(source, selection, parent, opponent, parent_identity):
    outcomes, actual, audit, calls = {}, Counter(), Counter(), Counter()
    for arm in ("parent", "controller"):
        env = reference(source["selected_root"]["state"])
        executor = ContingentExecutor(selection, parent, parent_identity)
        actions, work, changes = [], empty_work(), 0
        for _ in range(32):
            if env.state[PHASE] == TERMINAL:
                break
            if env.state[PLAYER] == source["seat"]:
                if arm == "controller":
                    action = executor.act(env)
                    if selection.plan is not None:
                        changes += int(action != int(parent.act(env)))
                else:
                    action = int(parent.act(env))
                calls["parent_actor_calls"] += 1
            else:
                action = int(opponent.act(env))
                calls["opponent_actor_calls"] += 1
            actions.append(action)
            add_step(work, env.step(action))
        assert env.state[PHASE] == TERMINAL
        outcomes[arm] = {
            "actions": actions,
            "work": work,
            "changed_own_decisions": changes,
            "final": env.state.tolist(),
            "point_margin": int(
                env.state[POINTS + source["seat"]]
                - env.state[POINTS + 1 - source["seat"]]
            ),
        }
        actual.update(work)
        checked = replay_actions(
            reference(source["selected_root"]["state"]), actions, outcomes[arm]["final"]
        )
        assert checked == work
        audit.update(checked)
    assert (
        outcomes["parent"]["actions"]
        == source["actions"][source["selected_root"]["decision"] :]
    )
    assert outcomes["parent"]["final"] == source["final"]
    if selection.plan is None:
        assert outcomes["controller"] == outcomes["parent"]
    return outcomes, dict(actual), dict(audit), dict(calls)
