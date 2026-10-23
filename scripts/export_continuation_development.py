"""Independently reduce saved continuation development audits and work ledgers."""

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from koikoi.layout import POINTS
from scripts.export_research_extension import read, require, score, sha
from scripts.run_arbitration_study import write


def reduce(base):
    protocol, summary = read(base / "protocol.json"), read(base / "summary.json")
    require(not protocol["smoke"], "Smoke cannot establish development performance")
    require(
        summary["protocol_sha256"] == sha(base / "protocol.json"), "Protocol changed"
    )
    for path, digest in protocol["inputs"].items():
        require(sha(path) == digest, f"Frozen input changed: {path}")
    bindings = {
        str(base / "protocol.json"): sha(base / "protocol.json"),
        str(base / "summary.json"): sha(base / "summary.json"),
    }
    gains = defaultdict(list)
    totals = defaultdict(Counter)
    actual = defaultdict(list)
    defaults = defaultdict(Counter)
    by_source = []
    for i, fixture in enumerate(protocol["fixtures"]):
        source, baseline = read(fixture["source"]), protocol["baseline"][i]
        old = read(fixture["baseline"])
        oracle = (
            old["oracle"]
            if i < 16
            else old["arms"]["parent_default_proposal"]["oracle"]
        )
        require(
            Fraction(baseline["exact_increment"])
            == Fraction(oracle["exact_increment"]),
            "Imported baseline differs",
        )
        gains["parent_default"].append(Fraction(baseline["exact_increment"]))
        totals["parent_default:planning"].update(
            {k: baseline[k] for k in ("decisions", "seconds")}
        )
        row_values = {
            "fixture": i,
            "draws": fixture["draws"],
            "source": fixture["source"],
            "parent_default": baseline["exact_increment"],
        }
        for family in protocol["families"]:
            folder = base / f"fixture_{i:02}" / family
            row, audit = read(folder / "selection.json"), read(folder / "audit.json")
            require(row["fixture"] == i and row["family"] == family, "Case differs")
            require(audit["status"] == "passed", "Failed audit")
            require(
                audit["selection_sha256"] == sha(folder / "selection.json"),
                "Selection changed",
            )
            require(
                row["program_sha256"] == sha(folder / "program.pkl.gz"),
                "Program changed",
            )
            recorded = [
                r
                for r in summary["rows"]
                if r["fixture"] == i and r["family"] == family
            ]
            require(
                len(recorded) == 1
                and recorded[0]["audit_sha256"] == sha(folder / "audit.json"),
                "Missing or changed audit binding",
            )
            require(
                all(recorded[0][k] == v for k, v in row.items()),
                "Summary selection differs",
            )
            for name in ("selection.json", "audit.json", "program.pkl.gz"):
                bindings[str(folder / name)] = sha(folder / name)
            oracle = audit["oracle"]
            increment = sum(
                (
                    Fraction(local["reach"])
                    * (
                        Fraction(local["oracle"]["candidate"]["value"])
                        - Fraction(local["oracle"]["parent"]["value"])
                    )
                    for local in oracle.get("local_checks", [])
                ),
                Fraction(0),
            )
            require(
                increment
                == Fraction(oracle["exact_increment"])
                == Fraction(recorded[0]["exact_increment"]),
                "Weighted exact gain differs",
            )
            require(increment >= Fraction(row["gain_lower"]) >= 0, "Invalid gain bound")
            require(
                row["adopted_nodes"] == len(oracle.get("local_checks", [])),
                "Missing local checks",
            )
            require(
                row["decisions"] <= protocol["shared_transitions"],
                "Planning budget exceeded",
            )
            defaults[family].update(row["adopted_continuations"])
            require(
                sum(row["adopted_continuations"].values()) == row["adopted_nodes"],
                "Default counts differ",
            )
            gains[family].append(increment)
            row_values[family] = str(increment)
            totals[family + ":planning"].update(
                {k: row[k] for k in ("decisions", "seconds", "adopted_nodes")}
            )
            totals[family + ":prefix_oracle"].update(oracle.get("prefix_physical", {}))
            for local in oracle.get("local_checks", []):
                require(
                    Fraction(local["gain"])
                    == Fraction(local["oracle"]["candidate"]["value"])
                    - Fraction(local["oracle"]["parent"]["value"])
                    > 0,
                    "Local gain differs",
                )
                for value in local["oracle"].values():
                    totals[family + ":local_oracle"].update(value["physical_work"])
            execution = audit["execution"]
            require(
                execution["actual_work"] == execution["replay_work"],
                "Execution/replay differs",
            )
            totals[family + ":actual_tails"].update(execution["actual_work"])
            totals[family + ":tail_replay"].update(execution["replay_work"])
            outcomes = execution["outcomes"]
            require(
                outcomes["parent"]["final"] == source["final"],
                "Parent trajectory differs",
            )
            seat = source["seat"]
            delta = score(outcomes["controller"]["final"], seat) - score(
                outcomes["parent"]["final"], seat
            )
            point_delta = (
                outcomes["controller"]["final"][POINTS + seat]
                - outcomes["parent"]["final"][POINTS + seat]
            )
            actual[family].append(
                {"win_score_delta": str(delta), "own_point_delta": point_delta}
            )
        by_source.append(row_values)
    aggregated = {}
    base_sum = sum(gains["parent_default"], Fraction(0))
    base_work = totals["parent_default:planning"]["decisions"]
    base_positive = {i for i, g in enumerate(gains["parent_default"]) if g > 0}
    for family, values in gains.items():
        positive = {i for i, g in enumerate(values) if g > 0}
        gain, work = sum(values, Fraction(0)), totals[family + ":planning"]["decisions"]
        eligible = (
            family != "parent_default"
            and bool(positive - base_positive)
            and base_positive <= positive
            and gain >= base_sum
            and gain * base_work >= base_sum * work
        )
        if family != "parent_default":
            comparison = summary["comparison"][family]
            require(
                eligible == comparison["eligible"]
                and gain == Fraction(comparison["exact_gain_sum"])
                and work == comparison["transitions"]
                and sorted(positive) == comparison["positives"],
                "Development decision differs",
            )
        aggregated[family] = {
            "exact_gain_sum": str(gain),
            "positive_sources": sorted(positive),
            "new_positives": sorted(positive - base_positive),
            "lost_positives": sorted(base_positive - positive),
            "gain_ratio": str(gain / base_sum),
            "transition_ratio": str(Fraction(work, base_work)),
            "gain_per_transition_ratio": str(gain * base_work / (base_sum * work)),
            "eligible": eligible,
        }
    eligible = [f for f in protocol["families"] if aggregated[f]["eligible"]]
    selected = (
        max(eligible, key=lambda f: sum(gains[f], Fraction(0))) if eligible else None
    )
    require(
        selected == summary["selected_for_fresh_confirmation"],
        "Selected family differs",
    )
    return {
        "scope": protocol["scope"],
        "sources": len(by_source),
        "new_source_matches": 0,
        "aggregated": aggregated,
        "totals": dict(totals),
        "adopted_continuations": dict(defaults),
        "actual": dict(actual),
        "by_source": by_source,
        "selected_for_fresh_confirmation": selected,
        "result_bindings": bindings,
        "exporter_sha256": sha(__file__),
    }


def table(report):
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Candidate default & Positive sources & Transitions & Gain ratio \\",
        r"\midrule",
    ]
    for family, label in (
        ("parent_default", "Original parent"),
        ("other_parent", "+ Other parent"),
        ("policy_pool", "+ WP and Point"),
    ):
        row = report["aggregated"][family]
        work = report["totals"][family + ":planning"]["decisions"]
        lines.append(
            f"{label} & {len(row['positive_sources'])}/{report['sources']} & {work:,} & {float(Fraction(row['gain_ratio'])):.3f}"
            + r" \\"
        )
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}", ""])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists() or args.table.exists():
        raise FileExistsError("Refusing to replace a saved export")
    result = reduce(args.base)
    write(args.destination, result)
    args.table.write_text(table(result), encoding="utf-8", newline="\n")
    print(
        {
            "sources": result["sources"],
            "selected": result["selected_for_fresh_confirmation"],
        }
    )
