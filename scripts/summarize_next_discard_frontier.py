"""Complete-only audit of the next-discard frontier and unchanged controls."""

import argparse
import hashlib
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from koikoi.layout import DISCARD, PHASE, TURN
from scripts.belief_tail import KoiTailDynamics, visible_keys
from scripts.check_frontier_hybrid_integration import compare_patch
from scripts.frontier_policy_patch import FrontierExecutor
from scripts.next_discard_frontier import next_discard_frontier
from scripts.proposal_interval_tail import compile_proposal
from scripts.run_arbitration_study import read, sha, write
from scripts.screen_next_discard_frontier import PARENT_SCREEN, checked
from scripts.screen_proposal_interval import describe_result
from scripts.uniform_controller_study import load


def summarize(base, output):
    if output.exists():
        raise FileExistsError(output)
    p = checked(base)
    if read(base / "queue.json") != {"status": "complete", "fixtures": 8}:
        raise ValueError("All eight fixtures must finish")
    summary = read(base / "summary.json")
    if summary["status"] != "passed" or len(summary["results"]) != 8:
        raise ValueError("Incomplete summary")
    totals = {
        name: Counter()
        for name in (
            "imported_fixed",
            "prefix",
            "proposal",
            "reference",
            "support_audit",
            "prefix_oracle_physical",
            "prefix_oracle_represented",
            "local_oracle_physical",
            "local_oracle_represented",
            "actual",
            "replay",
        )
    }

    def add(label, work):
        totals[label].update(
            {
                k: v
                for k, v in work.items()
                if k not in ("maximum_frontier_rows", "peak_frontier_rows", "peak_rows")
            }
        )

    rows, bindings = [], {}
    for i, fixture in enumerate(p["fixtures"]):
        folder = base / f"fixture_{i:02}"
        path = folder / "result.json"
        result = read(path)
        if (
            summary["results"].get(str(path)) != sha(path)
            or result["status"] != "passed"
            or result["fixture"] != i
            or result["case"] != fixture["case"]
            or result["fresh_chance_draws"] != 0
            or result["protocol_sha256"] != sha(base / "protocol.json")
            or set(result["artifacts"]) != {"selection.json", "program.pkl.gz"}
        ):
            raise ValueError("Fixture, protocol or saved result differs")
        for name, digest in result["artifacts"].items():
            if sha(folder / name) != digest:
                raise ValueError("Saved program or selection differs")
        patch = load(folder / "program.pkl.gz")
        row = read(folder / "selection.json")
        frontier = patch.frontier
        binding = frontier.parent_program.binding
        root_worlds = load(fixture["program"])["worlds"]
        bound = next_discard_frontier(
            root_worlds.states[0], root_worlds.viewer, len(root_worlds.model_names)
        )
        previous = load(PARENT_SCREEN / folder.name / "program.pkl.gz")
        if (
            row["entry_bound"] != asdict(bound)
            or row["changed_gate"] != (bound.draws != previous.frontier.draws)
            or frontier.draws != bound.draws
            or not frontier.complete
            or frontier.work["peak_rows"] > bound.maximum_rows
            or frontier.work["decisions"] > bound.maximum_transitions
        ):
            raise ValueError("Public boundary or derived prefix bound differs")
        for _, node in frontier.nodes:
            if not np.all(node.states[:, TURN] == bound.target_turn) or not np.all(
                node.states[:, PHASE] == DISCARD
            ):
                raise ValueError("Frontier is not the next learner discard")
        if not row["changed_gate"]:
            compare_patch(patch, previous)
        # Constructor validates complete local bindings, selections and weighted gain.
        FrontierExecutor(patch, None, binding.parent_identity)
        imported = fixture["original_fixed_work"]
        old_row = read(fixture["selection"])
        if (
            row != result["selection"]
            or old_row["selection"]["reason"] != "parent_unvalued"
            or imported != old_row["phases"]["fixed"]["work"]
            or row["imported_first_phase_decisions"] != imported["decisions"]
            or row["nominal_online_decisions"]
            != imported["decisions"] + patch.decisions
            or row["nominal_online_decisions"] > 4000000
            or row["decisions"] != patch.decisions
            or row["gain_lower"] != str(patch.gain_lower)
            or row["prefix_work"] != frontier.work
            or row["prefix_complete"] != frontier.complete
            or row["prefix_refusal"] != frontier.refusal
            or row["frontier_nodes"] != len(frontier.nodes)
            or row["attempted_nodes"] != len(patch.comparisons)
        ):
            raise ValueError("Saved selection, prefix or imported accounting differs")
        if (
            frontier.work["decisions"] > p["design"]["prefix_decisions"]
            or frontier.work["peak_rows"] > p["design"]["max_rows"]
            or patch.decisions > p["design"]["max_decisions"]
        ):
            raise ValueError("Shared/prefix budgets exceeded")
        used, nodes, adopted = frontier.work["decisions"], dict(frontier.nodes), []
        if len(row["local_comparisons"]) != len(patch.comparisons):
            raise ValueError("Local comparison count differs")
        for saved, (key, local) in zip(
            row["local_comparisons"], patch.comparisons, strict=True
        ):
            if saved != {
                "key_sha256": hashlib.sha256(key).hexdigest(),
                **describe_result(local),
            }:
                raise ValueError("Local saved certificate differs")
            node = nodes[key]
            reference = local["parent_receipt"]
            root_action = reference.program.parent_root_action
            dyn = SimpleNamespace(
                keys=visible_keys,
                legal=KoiTailDynamics.legal,
                baseline_action=lambda state, action=root_action: action,
            )
            compiled = compile_proposal(
                local["proposal_phase"],
                node,
                dyn,
                binding.parent_identity,
                dict(binding.opponent_identities),
            )
            if compiled != local["candidate_certificate"]:
                raise ValueError("Compiled local candidate differs")
            allowance = min(
                p["design"]["per_phase_decisions"],
                (p["design"]["max_decisions"] - used) // 2,
            )
            proposal_work, reference_work = saved["proposal_work"], reference.work
            if (
                allowance <= 0
                or proposal_work["decisions"] > allowance
                or reference_work["decisions"] > allowance
                or proposal_work["live_observation_nodes"]
                > p["design"]["max_nodes_per_phase"] + 1
                or reference_work["expanded_tasks"] > p["design"]["max_nodes_per_phase"]
                or local["decisions"]
                != proposal_work["decisions"] + reference_work["decisions"]
            ):
                raise ValueError("Local preassigned budgets or work differ")
            used += local["decisions"]
            add("proposal", proposal_work)
            add("reference", reference_work)
            if local["selection"].candidate is not None:
                adopted.append((node, local["selection"]))
        if used != patch.decisions or row["adopted_nodes"] != len(adopted):
            raise ValueError("Global work or adopted-node count differs")
        oracle = result["oracle"]
        if frontier.complete:
            if (
                oracle is None
                or oracle["nodes"] != len(nodes)
                or oracle["prefix_terminal_mass"] != frontier.terminal_mass
                or oracle["gain_lower"] != str(patch.gain_lower)
                or len(oracle["local_checks"]) != len(adopted)
                or frontier.terminal_mass
                + sum(sum(map(int, n.weights)) for n in nodes.values())
                != frontier.total_mass
            ):
                raise ValueError("Independent complete partition differs")
            exact = Fraction(0)
            for check, (node, selection) in zip(
                oracle["local_checks"], adopted, strict=True
            ):
                reach = Fraction(sum(map(int, node.weights)), frontier.total_mass)
                pair = check["oracle"]
                gain = Fraction(pair["candidate"]["value"]) - Fraction(
                    pair["parent"]["value"]
                )
                if (
                    check["reach"] != str(reach)
                    or check["gain"] != str(gain)
                    or not gain >= selection.gain_lower > 0
                ):
                    raise ValueError("Independent local gain differs")
                for name, receipt in (
                    ("parent", selection.parent),
                    ("candidate", selection.candidate),
                ):
                    value = Fraction(pair[name]["value"])
                    if (
                        not receipt.lower <= value <= receipt.upper
                        or pair[name]["represented_work"]["completions"]
                        != receipt.total_mass
                    ):
                        raise ValueError(
                            "Independent local value or terminal mass differs"
                        )
                    add("local_oracle_physical", pair[name]["physical_work"])
                    add("local_oracle_represented", pair[name]["represented_work"])
                exact += reach * gain
            if str(exact) != oracle["exact_increment"] or exact < patch.gain_lower:
                raise ValueError("Independent gain decomposition differs")
            for name in ("physical", "represented"):
                add("prefix_oracle_" + name, oracle["prefix_" + name])
        elif oracle is not None or patch.comparisons or patch.gain_lower:
            raise ValueError("Incomplete prefix used as a certified belief")
        source = read(fixture["source"])
        outcomes = result["assessment"]["outcomes"]
        if (
            outcomes["parent"]["final"] != source["final"]
            or outcomes["parent"]["actions"]
            != source["actions"][source["selected_root"]["decision"] :]
            or (not adopted and outcomes["parent"] != outcomes["controller"])
        ):
            raise ValueError("Actual parent/fallback suffix differs")
        add("prefix", frontier.work)
        add("imported_fixed", imported)
        add("support_audit", result["support_audit_work"])
        for name in ("actual", "replay"):
            add(name, result["assessment"][name + "_work"])
        rows.append(
            {
                "fixture": i,
                "case": fixture["case"],
                **{k: v for k, v in row.items() if k != "local_comparisons"},
                "exact_increment": None
                if oracle is None
                else oracle["exact_increment"],
                "oracle_seconds": result["oracle_seconds"],
            }
        )
        bindings[str(path)] = sha(path)
    for name in ("protocol.json", "queue.json", "summary.json"):
        bindings[str(base / name)] = sha(base / name)
    report = {
        "status": "passed",
        "scope": "Targeted development; exact increments conditional on declared model, no population inference",
        "script_sha256": sha(__file__),
        "bindings": bindings,
        "rows": rows,
        "work": {k: dict(v) for k, v in totals.items()},
    }
    write(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("runs/next_discard_frontier_screen")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.base, args.output)
