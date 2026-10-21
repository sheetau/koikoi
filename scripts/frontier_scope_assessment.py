"""Increment-only planning, structural audit, and old/hybrid actual execution."""

import time
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from types import SimpleNamespace

import numpy as np

from koikoi.layout import DISCARD, PHASE, PLAYER, POINTS, TERMINAL, TURN
from scripts.belief_tail import KoiTailDynamics, visible_keys
from scripts.check_belief_tail import add_step, empty_work
from scripts.diagnose_history_support import reference
from scripts.export_belief_tail import replay_actions
from scripts.frontier_hybrid_controller import FrontierHybridExecutor, validate_hybrid
from scripts.frontier_scope_controller import evaluate_scoped_frontier
from scripts.incumbent_tail_controller import bind_worlds
from scripts.interval_tail import select_interval
from scripts.next_discard_frontier import next_discard_frontier
from scripts.proposal_interval_tail import compile_proposal
from scripts.prospective_incumbent import describe_selection
from scripts.run_arbitration_study import PARENTS
from scripts.symbolic_final_turn_tail import RULE, SymbolicExecutor, SymbolicPlan
from scripts.uniform_controller_study import build_support
from scripts.verify_frontier_patch import verify_frontier_patch


def planning_work(hybrid):
    work = {"fixed": hybrid.fixed.result.work}
    if hybrid.optimized is not None:
        work["optimized"] = hybrid.optimized.result.work
    if hybrid.patch is not None:
        work["prefix"] = hybrid.patch.frontier.work
        for name, field in (
            ("proposal", "proposal_phase"),
            ("reference", "parent_receipt"),
        ):
            total = Counter()
            for _, local in hybrid.patch.comparisons:
                receipt = local[field]
                counters = receipt.result.work if name == "proposal" else receipt.work
                total.update(
                    {
                        k: v
                        for k, v in counters.items()
                        if "maximum" not in k and "peak" not in k
                    }
                )
            work[name] = dict(total)
    if sum(v.get("decisions", 0) for v in work.values()) != hybrid.decisions:
        raise ValueError("Planning phase accounting differs")
    return work


def plan_source(source, parents, opponents, protocol):
    if source["selected_root"] is None:
        return {"status": "missing_root", "increment_lower": "0"}, None
    started = time.perf_counter()
    support, worlds, identity = build_support(source, opponents, protocol["design"])
    row = {
        "history_counts": support.counts,
        "history_seconds": time.perf_counter() - started,
        "increment_lower": "0",
    }
    evidence = {"support": asdict(support)}
    if worlds is None:
        return {**row, "status": "support_refused"}, evidence
    parent = parents[source["parent"]]
    dyn = KoiTailDynamics(
        protocol["design"]["capacity"],
        worlds.model_names,
        {identity: opponents[identity]},
        parent,
    )
    started = time.perf_counter()
    hybrid = evaluate_scoped_frontier(
        worlds,
        dyn,
        PARENTS[source["parent"]][1],
        {identity: protocol["opponent_identities"][identity]},
    )
    row.update(
        status="evaluated",
        seconds=time.perf_counter() - started,
        old_selection=describe_selection(hybrid.original),
        increment_lower=str(0 if hybrid.patch is None else hybrid.patch.gain_lower),
        identical_programs=hybrid.patch is None,
        work=planning_work(hybrid),
        decisions=hybrid.decisions,
        parent_adapter_calls=dyn.baseline_calls,
        opponent_adapter_calls=dyn.policy_calls,
    )
    return row, {**evidence, "worlds": worlds, "hybrid": hybrid}


def validate_program(hybrid, worlds, parent_id, identities, config):
    """Check preserved phase schedule, public boundary, and every local allowance."""
    validate_hybrid(hybrid)
    FrontierHybridExecutor(hybrid, None, parent_id)
    binding = bind_worlds(
        worlds,
        SimpleNamespace(keys=visible_keys, legal=KoiTailDynamics.legal),
        parent_id,
        identities,
    )
    limits = config["hybrid"]
    if (
        hybrid.phase_decisions != limits["phase_decisions"]
        or not 0 <= worlds.draws <= 5
    ):
        raise ValueError("Selected schedule differs")
    fixed_plan = hybrid.fixed.result.plan
    optimize = fixed_plan is not None and fixed_plan.values[fixed_plan.action] < 1
    if (hybrid.optimized is not None) != optimize:
        raise ValueError("Old phase schedule differs")
    if (hybrid.patch is not None) != (
        hybrid.original.reason == "parent_unvalued" and worlds.draws == 5
    ):
        raise ValueError("Selected fallback activation differs")
    for name, phase in (("fixed", hybrid.fixed), ("optimized", hybrid.optimized)):
        if phase is None:
            continue
        if (
            phase.binding != binding
            or phase.mode != name
            or phase.result.work.get("live_observation_nodes", 0)
            > limits["phase_nodes"] + 1
        ):
            raise ValueError("Old phase binding or node allowance differs")
        plan = phase.result.plan
        if plan is not None and (
            not isinstance(plan, SymbolicPlan) or plan.leaf_rule != RULE
        ):
            raise ValueError("Unknown old executable leaf rule")
    patch = hybrid.patch
    if patch is None:
        return
    bound = next_discard_frontier(
        worlds.states[0], worlds.viewer, len(worlds.model_names)
    )
    frontier = patch.frontier
    if (
        not frontier.complete
        or frontier.draws != bound.draws
        or frontier.work["decisions"]
        > min(limits["prefix_decisions"], bound.maximum_transitions)
        or frontier.work["peak_rows"] > min(limits["max_rows"], bound.maximum_rows)
    ):
        raise ValueError("Selected public prefix bound differs")
    for _, node in frontier.nodes:
        if not np.all(node.states[:, TURN] == bound.target_turn) or not np.all(
            node.states[:, PHASE] == DISCARD
        ):
            raise ValueError("Frontier is not the next learner discard")
    used, nodes = frontier.work["decisions"], dict(frontier.nodes)
    # Solver visits the deterministic mass-priority order, without skipping nodes.
    if [key for key, _ in patch.comparisons] != [
        key for key, _ in frontier.nodes[: len(patch.comparisons)]
    ]:
        raise ValueError("Local scheduling order differs")
    for key, local in patch.comparisons:
        allowance = min(
            limits["local_phase_decisions"], (limits["phase_decisions"] - used) // 2
        )
        proposal, receipt = local["proposal_phase"], local["parent_receipt"]
        dyn = SimpleNamespace(
            keys=visible_keys,
            legal=KoiTailDynamics.legal,
            baseline_action=lambda state, action=receipt.program.parent_root_action: (
                action
            ),
        )
        if (
            compile_proposal(proposal, nodes[key], dyn, parent_id, identities)
            != local["candidate_certificate"]
        ):
            raise ValueError("Local proposal compilation differs")
        candidate = local["candidate_certificate"]
        if (
            select_interval(receipt, [] if candidate is None else [candidate])
            != local["selection"]
        ):
            raise ValueError("Compiled candidate selection differs")
        if (
            allowance <= 0
            or proposal.result.work["decisions"] > allowance
            or receipt.work["decisions"] > allowance
            or proposal.result.work["live_observation_nodes"]
            > limits["local_nodes"] + 1
            or receipt.work["expanded_tasks"] > limits["local_nodes"]
            or local["decisions"]
            != proposal.result.work["decisions"] + receipt.work["decisions"]
        ):
            raise ValueError("Local preassigned allowance differs")
        used += local["decisions"]
    if used != patch.decisions or (
        len(patch.comparisons) < len(nodes)
        and (limits["phase_decisions"] - used) // 2 > 0
    ):
        raise ValueError("Global work or incomplete local schedule differs")


def audit_plan(row, evidence, source, parent, opponents, protocol):
    support, worlds, identity = build_support(source, opponents, protocol["design"])
    if (
        asdict(support) != evidence["support"]
        or support.counts != row["history_counts"]
    ):
        raise ValueError("Reconstructed support differs")
    if row["status"] == "support_refused":
        if (
            worlds is not None
            or set(evidence) != {"support"}
            or row["increment_lower"] != "0"
        ):
            raise ValueError("Unjustified support refusal")
        return {
            "exact_increment": "0",
            "reason": "support_refused",
            "oracle": None,
        }, support.counts
    if (
        worlds is None
        or row["status"] != "evaluated"
        or set(evidence) != {"support", "worlds", "hybrid"}
    ):
        raise ValueError("Unexpected evaluation status")
    stored = evidence["worlds"]
    for name in ("states", "models", "weights"):
        np.testing.assert_array_equal(getattr(worlds, name), getattr(stored, name))
    if (worlds.model_names, worlds.viewer, worlds.draws) != (
        stored.model_names,
        stored.viewer,
        stored.draws,
    ):
        raise ValueError("Reconstructed belief metadata differs")
    hybrid, parent_id = evidence["hybrid"], PARENTS[source["parent"]][1]
    validate_program(
        hybrid,
        worlds,
        parent_id,
        {identity: protocol["opponent_identities"][identity]},
        protocol["design"],
    )
    if (
        row["old_selection"] != describe_selection(hybrid.original)
        or row["increment_lower"]
        != str(0 if hybrid.patch is None else hybrid.patch.gain_lower)
        or row["identical_programs"] != (hybrid.patch is None)
        or row["work"] != planning_work(hybrid)
        or row["decisions"] != hybrid.decisions
    ):
        raise ValueError("Saved incremental selection or work differs")
    if hybrid.patch is None:
        return {
            "exact_increment": "0",
            "reason": "identical_programs",
            "oracle": None,
        }, support.counts
    oracle = verify_frontier_patch(
        worlds, hybrid.patch, parent, {identity: opponents[identity]}, parent_id
    )
    if not 0 <= hybrid.patch.gain_lower <= Fraction(oracle["exact_increment"]) <= 1:
        raise ValueError("Verified incremental value differs")
    return {
        "exact_increment": oracle["exact_increment"],
        "reason": "independent_frontier_decomposition",
        "oracle": oracle,
    }, support.counts


def assess_pair(source, hybrid, parent, opponent, parent_id):
    root = reference(source["selected_root"]["state"])
    binding = hybrid.fixed.binding
    if (
        binding.viewer != source["seat"]
        or binding.root_key != visible_keys(root.state[None], source["seat"])[0]
    ):
        raise ValueError("Actual root differs from program")
    outcomes, actual, replay, calls = {}, Counter(), Counter(), Counter()
    for label in ("old", "hybrid"):
        env = reference(root.state)
        executor = (
            SymbolicExecutor(hybrid.original, parent, parent_id)
            if label == "old"
            else FrontierHybridExecutor(hybrid, parent, parent_id)
        )
        actions, work = [], empty_work()
        for _ in range(32):
            if env.state[PHASE] == TERMINAL:
                break
            own = env.state[PLAYER] == source["seat"]
            action = executor.act(env) if own else int(opponent.act(env))
            calls["learner_executor_calls" if own else "opponent_adapter_calls"] += 1
            if not 0 <= action < 50 or not env.legal()[action]:
                raise ValueError("Illegal incremental actual-tail action")
            actions.append(action)
            add_step(work, env.step(action))
        if env.state[PHASE] != TERMINAL:
            raise ValueError("Actual tail exceeded finite depth")
        final = env.state.tolist()
        outcomes[label] = {
            "actions": actions,
            "work": work,
            "final": final,
            "margin": int(
                final[POINTS + source["seat"]] - final[POINTS + 1 - source["seat"]]
            ),
        }
        actual.update(work)
        checked = replay_actions(reference(root.state), actions, final)
        if checked != work:
            raise ValueError("Actual replay work differs")
        replay.update(checked)
    if hybrid.original.plan is None and (
        outcomes["old"]["final"] != source["final"]
        or outcomes["old"]["actions"]
        != source["actions"][source["selected_root"]["decision"] :]
    ):
        raise ValueError("Old fallback differs from source parent suffix")
    if hybrid.patch is None and outcomes["old"] != outcomes["hybrid"]:
        raise ValueError("Identical programs produced different actual tails")
    return {
        "outcomes": outcomes,
        "actual_work": dict(actual),
        "replay_work": dict(replay),
        "calls": dict(calls),
    }
