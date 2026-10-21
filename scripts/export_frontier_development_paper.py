"""Re-audit the closed development evidence before exporting manuscript numbers."""

import argparse
from fractions import Fraction
from pathlib import Path

from scripts.run_arbitration_study import read, sha, write
from scripts.summarize_frontier_hybrid_integration import summarize as integrated
from scripts.summarize_next_discard_frontier import summarize as boundary
from scripts.summarize_selected_frontier_integration import summarize as selected


def export(review_folder, output):
    review_folder.mkdir(parents=True)
    checks = (
        ("frontier_hybrid_integration", integrated),
        ("next_discard_frontier_screen", boundary),
        ("selected_frontier_integration", selected),
    )
    reports, bindings = {}, {}
    for name, summarize in checks:
        path = Path("runs") / f"{name}_review.json"
        report = summarize(Path("runs") / name, review_folder / path.name)
        if report != read(path):
            raise ValueError("Closed development review no longer reproduces")
        reports[name] = report
        bindings[str(path)] = sha(path)
    rows = reports["next_discard_frontier_screen"]["rows"]
    if len(rows) != 8:
        raise ValueError("Complete eight-root development screen required")
    positive = sum(Fraction(r["exact_increment"]) > 0 for r in rows)
    adopted = sum(r["adopted_nodes"] for r in rows)
    constants = (
        "% Generated from complete, re-audited development evidence.\n"
        f"\\newcommand{{\\FrontierRootCount}}{{{len(rows)}}}\n"
        f"\\newcommand{{\\FrontierPositiveRoots}}{{{positive}}}\n"
        f"\\newcommand{{\\FrontierAdoptedNodes}}{{{adopted}}}\n"
    )
    lines = [
        "% Targeted development roots; not fresh population observations.",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Source & Cutoff & Groups & Patches & Lower gain (pp) & Exact gain (pp) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['case']['index']} & {row['entry_bound']['draws']} & "
            f"{row['frontier_nodes']:,} & {row['adopted_nodes']} & "
            f"{float(Fraction(row['gain_lower']) * 100):.6f} & "
            f"{float(Fraction(row['exact_increment']) * 100):.6f} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "frontier_development_constants.tex").write_bytes(
        constants.encode("utf-8")
    )
    (output / "frontier_development_table.tex").write_bytes(
        ("\n".join(lines) + "\n").encode("utf-8")
    )
    evidence = {
        "scope": "Eight fixed development roots; complete reviewed evidence; no population-strength claim",
        "script_sha256": sha(__file__),
        "bindings": bindings,
        "roots": len(rows),
        "positive_roots": positive,
        "adopted_nodes": adopted,
        "rows": rows,
        "work": {k: v.get("work", v.get("new_work")) for k, v in reports.items()},
        "selected_new_planning_decisions": reports["selected_frontier_integration"][
            "new_planning_decisions"
        ],
        "outputs": {
            name: sha(output / name)
            for name in (
                "frontier_development_constants.tex",
                "frontier_development_table.tex",
            )
        },
    }
    write(output / "frontier_development_evidence.json", evidence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/paper/generated"))
    args = parser.parse_args()
    export(args.review_folder, args.output)
