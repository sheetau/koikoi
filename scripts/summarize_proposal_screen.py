"""Complete-only review of proposal certificates, reference intervals and work."""

import argparse
from collections import Counter
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from scripts.belief_tail import KoiTailDynamics, visible_keys
from scripts.proposal_interval_tail import compile_proposal
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.screen_proposal_interval import PREVIOUS, checked, describe_result
from scripts.uniform_controller_study import load


def summarize(base, output):
    if output.exists():
        raise FileExistsError(output)
    p = checked(base)
    if read(base / "queue.json") != {"status": "complete", "fixtures": 8}:
        raise ValueError("All eight fixtures must finish")
    summary = read(base / "summary.json")
    if summary["status"] != "passed" or len(summary["results"]) != 8:
        raise ValueError("Incomplete result set")
    totals = {
        k: Counter()
        for k in (
            "proposal",
            "reference",
            "support_audit",
            "oracle_physical",
            "oracle_represented",
            "actual",
            "replay",
        )
    }
    rows, bindings = [], {}
    identities = read(PREVIOUS / "protocol.json")["opponent_identities"]
    for i in range(8):
        folder = base / f"fixture_{i:02}"
        path = folder / "result.json"
        if summary["results"][str(path)] != sha(path):
            raise ValueError("Result binding differs")
        result = read(path)
        if (
            result["status"] != "passed"
            or result["fixture"] != i
            or result["protocol_sha256"] != sha(base / "protocol.json")
            or result["fresh_chance_draws"] != 0
            or set(result["artifacts"]) != {"selection.json", "program.pkl.gz"}
        ):
            raise ValueError("Malformed fixture or artifact set")
        for name, digest in result["artifacts"].items():
            if sha(folder / name) != digest:
                raise ValueError("Saved artifact differs")
        stored = load(folder / "program.pkl.gz")
        row = read(folder / "selection.json")
        if row != result["selection"] or any(
            row[k] != v for k, v in describe_result(stored).items()
        ):
            raise ValueError("Saved selection differs")
        source = read(PREVIOUS / folder.name / "source.json")
        old_row = read(PREVIOUS / folder.name / "selection.json")
        old = load(PREVIOUS / folder.name / "program.pkl.gz")
        root_action = source["actions"][source["selected_root"]["decision"]]
        dyn = SimpleNamespace(
            keys=visible_keys,
            legal=KoiTailDynamics.legal,
            baseline_action=lambda state, action=root_action: action,
        )
        candidate = compile_proposal(
            stored["proposal_phase"],
            old["worlds"],
            dyn,
            PARENTS[source["parent"]][1],
            {source["opponent"]: identities[source["opponent"]]},
        )
        if candidate != stored["candidate_certificate"]:
            raise ValueError(
                "Saved compiled policy does not match its completed proposal"
            )
        reference = stored["parent_receipt"]
        mass = sum(map(int, old["worlds"].weights))
        if (
            reference.total_mass != mass
            or reference.program.parent_root_action != root_action
        ):
            raise ValueError("Parent identity or mass differs from the source")
        if (
            old_row["selection"]["parent_q"] is not None
            and not reference.lower
            <= Fraction(old_row["selection"]["parent_q"])
            <= reference.upper
        ):
            raise ValueError("Reference excludes the preceding exact parent value")
        proposal_work, reference_work = row["proposal_work"], reference.work
        if (
            proposal_work["decisions"] > p["design"]["proposal_decisions"]
            or reference_work["decisions"] > p["design"]["reference_decisions"]
            or proposal_work["live_observation_nodes"]
            > p["design"]["max_nodes_per_phase"] + 1
            or reference_work["expanded_tasks"] > p["design"]["max_nodes_per_phase"]
            or row["decisions"]
            != proposal_work["decisions"] + reference_work["decisions"]
        ):
            raise ValueError("Separate phase budgets or total accounting differ")
        if (
            reference_work["terminal_mass"] != reference_work["represented_completions"]
            or sum(
                reference_work[k]
                for k in ("terminal_mass", "certified_win_mass", "certified_loss_mass")
            )
            != reference.resolved_mass
        ):
            raise ValueError("Reference resolved-mass accounting differs")
        newly_exact = (
            old_row["selection"]["parent_q"] is None
            and reference.lower == reference.upper
        )
        expected = ({"candidate"} if candidate is not None else set()) | (
            {"parent"} if row["adopted"] or newly_exact else set()
        )
        if (
            set(result["oracle"]) != expected
            or result["newly_exact_parent"] != newly_exact
        ):
            raise ValueError("Missing required independent replay")
        for label, oracle in result["oracle"].items():
            receipt = candidate if label == "candidate" else reference
            if (
                not receipt.lower <= Fraction(oracle["value"]) <= receipt.upper
                or oracle["represented_work"]["completions"] != mass
            ):
                raise ValueError("Independent value or complete mass differs")
            totals["oracle_physical"].update(oracle["physical_work"])
            totals["oracle_represented"].update(oracle["represented_work"])
        if row["adopted"]:
            gain = Fraction(result["oracle"]["candidate"]["value"]) - Fraction(
                result["oracle"]["parent"]["value"]
            )
            if not gain >= Fraction(row["gain_lower"]) > 0:
                raise ValueError("Independent pair does not support the adopted gain")
        outcomes = result["assessment"]["outcomes"]
        if (
            outcomes["parent"]["final"] != source["final"]
            or outcomes["parent"]["actions"]
            != source["actions"][source["selected_root"]["decision"] :]
            or (not row["adopted"] and outcomes["parent"] != outcomes["controller"])
        ):
            raise ValueError("Actual parent/fallback execution differs")
        for label, work in (("proposal", proposal_work), ("reference", reference_work)):
            totals[label].update(
                {
                    k: v
                    for k, v in work.items()
                    if k not in ("maximum_frontier_rows", "peak_frontier_rows")
                }
            )
        totals["support_audit"].update(result["support_audit_work"])
        for label in ("actual", "replay"):
            totals[label].update(result["assessment"][label + "_work"])
        rows.append(
            {
                "fixture": i,
                "source_key": source["source_key"],
                **row,
                "newly_exact_parent": newly_exact,
                "oracle": result["oracle"],
                "oracle_seconds": result["oracle_seconds"],
            }
        )
        bindings[str(path)] = sha(path)
    for name in ("protocol.json", "queue.json", "summary.json"):
        bindings[str(base / name)] = sha(base / name)
    report = {
        "status": "passed",
        "scope": "Complete paired development screen; no fresh strength inference",
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
        "--base", type=Path, default=Path("runs/proposal_interval_screen")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.base, args.output)
