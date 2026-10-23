"""Reduce and bind independent evidence for optimized-frontier public pruning."""

import argparse
from collections import Counter
from fractions import Fraction
from pathlib import Path

from koikoi.layout import POINTS
from scripts.export_research_extension import read, require, score, sha
from scripts.public_score_bounds import outcome_certificates
from scripts.run_arbitration_study import write
from scripts.uniform_controller_study import load


def reduce(base):
    protocol, summary = read(base / "protocol.json"), read(base / "summary.json")
    require(
        summary["protocol_sha256"] == sha(base / "protocol.json"), "Protocol changed"
    )
    for path, digest in protocol["inputs"].items():
        require(sha(path) == digest, f"Frozen input changed: {path}")
    bindings = {
        str(base / "protocol.json"): sha(base / "protocol.json"),
        str(base / "summary.json"): sha(base / "summary.json"),
    }
    rows, totals = (
        [],
        {
            "baseline": Counter(),
            "pruned": Counter(),
            "prefix_oracle": Counter(),
            "local_oracle": Counter(),
            "actual_tails": Counter(),
            "tail_replay": Counter(),
        },
    )
    for i, fixture in enumerate(protocol["fixtures"]):
        folder = base / f"fixture_{i:02}"
        selection, audit = read(folder / "selection.json"), read(folder / "audit.json")
        require(
            selection["fixture"] == i and audit["status"] == "passed", "Invalid result"
        )
        require(
            audit["selection_sha256"] == sha(folder / "selection.json")
            and selection["program_sha256"] == sha(folder / "program.pkl.gz"),
            "Result changed",
        )
        recorded = summary["rows"][i]
        require(
            recorded["audit_sha256"] == sha(folder / "audit.json")
            and all(recorded[k] == v for k, v in selection.items()),
            "Summary differs",
        )
        result = load(folder / "program.pkl.gz")
        require(
            result.patch.decisions == selection["decisions"]
            and str(result.patch.gain_lower) == selection["gain_lower"],
            "Saved policy metadata differs",
        )
        nodes = dict(result.patch.frontier.nodes)
        require(len(result.skipped) == selection["skipped_nodes"], "Skip count differs")
        skipped_keys = {k for k, _, _ in result.skipped}
        require(len(skipped_keys) == len(result.skipped), "Repeated skipped history")
        for key, winning, losing in result.skipped:
            node = nodes[key]
            win, loss = outcome_certificates(node.states, node.viewer)
            require(
                not (win & loss).any() and (win | loss).all(), "Invalid public skip"
            )
            require(
                winning == sum(map(int, node.weights[win]))
                and losing == sum(map(int, node.weights[loss])),
                "Skip mass differs",
            )
        old = protocol["baseline"][i]
        old_audit = read(Path(fixture["baseline"]) / "audit.json")
        old_patch = load(Path(fixture["baseline"]) / "program.pkl.gz")
        require(
            old["exact_increment"] == old_audit["oracle"]["exact_increment"],
            "Imported value differs",
        )
        old_work_on_skips = sum(
            r["decisions"] for key, r in old_patch.comparisons if key in skipped_keys
        )
        require(
            all(
                r["selection"].candidate is None
                for key, r in old_patch.comparisons
                if key in skipped_keys
            ),
            "A skipped baseline branch had a positive certificate",
        )
        oracle = audit["oracle"]
        exact = Fraction(0)
        totals["prefix_oracle"].update(oracle.get("prefix_physical", {}))
        for local in oracle.get("local_checks", []):
            pair = local["oracle"]
            gain = Fraction(pair["candidate"]["value"]) - Fraction(
                pair["parent"]["value"]
            )
            require(gain == Fraction(local["gain"]) > 0, "Invalid local increment")
            exact += Fraction(local["reach"]) * gain
            for value in pair.values():
                totals["local_oracle"].update(value["physical_work"])
        require(
            exact
            == Fraction(oracle["exact_increment"])
            == Fraction(recorded["exact_increment"])
            and exact >= result.patch.gain_lower,
            "Exact gain differs",
        )
        require(
            selection["adopted_nodes"] == len(oracle.get("local_checks", [])),
            "Missing adoption audit",
        )
        require(
            selection["decisions"] <= protocol["shared_transitions"],
            "Global budget exceeded",
        )
        require(
            selection["decisions"]
            == result.patch.frontier.work["decisions"]
            + sum(r["decisions"] for _, r in result.patch.comparisons),
            "Physical ledger differs",
        )
        for name, values in (("baseline", old), ("pruned", selection)):
            totals[name].update(
                {k: values[k] for k in ("decisions", "seconds", "adopted_nodes")}
            )
        execution = audit["execution"]
        require(
            execution["actual_work"] == execution["replay_work"],
            "Actual tail/replay work differs",
        )
        totals["actual_tails"].update(execution["actual_work"])
        totals["tail_replay"].update(execution["replay_work"])
        source = read(fixture["source"])
        before, after = (
            execution["outcomes"][k]["final"] for k in ("parent", "controller")
        )
        require(before == source["final"], "Parent final differs")
        seat = source["seat"]
        rows.append(
            {
                "fixture": i,
                "baseline_gain": old["exact_increment"],
                "pruned_gain": str(exact),
                "baseline_decisions": old["decisions"],
                "pruned_decisions": selection["decisions"],
                "skipped_nodes": selection["skipped_nodes"],
                "baseline_work_on_skipped_nodes": old_work_on_skips,
                "actual_win_score_delta": str(score(after, seat) - score(before, seat)),
                "actual_point_delta": after[POINTS + seat] - before[POINTS + seat],
            }
        )
        for filename in ("selection.json", "audit.json", "program.pkl.gz"):
            bindings[str(folder / filename)] = sha(folder / filename)
    old_gain = sum((Fraction(r["baseline_gain"]) for r in rows), Fraction(0))
    new_gain = sum((Fraction(r["pruned_gain"]) for r in rows), Fraction(0))
    old_positive = sum(Fraction(r["baseline_gain"]) > 0 for r in rows)
    new_positive = sum(Fraction(r["pruned_gain"]) > 0 for r in rows)
    old_work, new_work = totals["baseline"]["decisions"], totals["pruned"]["decisions"]
    criterion = (
        new_gain >= old_gain and new_positive >= old_positive and new_work < old_work
    )
    require(criterion == summary["criterion_met"], "Criterion differs")
    for name, gain, positive, work in (
        ("baseline", old_gain, old_positive, old_work),
        ("pruned", new_gain, new_positive, new_work),
    ):
        require(
            Fraction(summary[name]["exact_gain_sum"]) == gain
            and summary[name]["positive_sources"] == positive
            and summary[name]["transitions"] == work,
            "Aggregate differs",
        )
    return {
        "scope": protocol["scope"],
        "sources": len(rows),
        "new_source_matches": 0,
        "rows": rows,
        "totals": totals,
        "baseline_gain_sum": str(old_gain),
        "pruned_gain_sum": str(new_gain),
        "baseline_positive_sources": old_positive,
        "pruned_positive_sources": new_positive,
        "gain_ratio": str(new_gain / old_gain),
        "transition_ratio": str(Fraction(new_work, old_work)),
        "gain_per_transition_ratio": str(new_gain * old_work / (old_gain * new_work)),
        "criterion_met": criterion,
        "result_bindings": bindings,
        "exporter_sha256": sha(__file__),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    report = reduce(args.base)
    write(args.destination, report)
    print({"sources": report["sources"], "criterion_met": report["criterion_met"]})
