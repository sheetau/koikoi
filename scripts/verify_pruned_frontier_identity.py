"""Verify complete retained comparison identity for the closed pruning screen."""

import argparse
from dataclasses import replace
from pathlib import Path

from scripts.benchmark_frontier_receipts import equivalent
from scripts.export_research_extension import read, require, sha
from scripts.run_arbitration_study import write
from scripts.uniform_controller_study import load


def verify(base):
    protocol = read(base / "protocol.json")
    rows = []
    for index, fixture in enumerate(protocol["fixtures"]):
        old_path = Path(fixture["baseline"]) / "program.pkl.gz"
        new_path = base / f"fixture_{index:02}" / "program.pkl.gz"
        old, result = load(old_path), load(new_path)
        new = result.patch
        omitted = {key for key, _, _ in result.skipped}
        saved_work = sum(r["decisions"] for key, r in old.comparisons if key in omitted)
        retained = tuple((key, r) for key, r in old.comparisons if key not in omitted)
        require(old.decisions - saved_work == new.decisions, "Work is not pure elision")
        equivalent(replace(old, comparisons=retained, decisions=new.decisions), new)
        for field in ("terminal_mass", "pending_mass", "draws"):
            require(
                getattr(old.frontier, field) == getattr(new.frontier, field),
                "Prefix mass or boundary changed",
            )
        rows.append(
            {
                "fixture": index,
                "retained_comparisons": len(retained),
                "preserved_adoptions": sum(
                    r["selection"].candidate is not None for _, r in retained
                ),
                "skipped_nodes": len(result.skipped),
                "saved_transitions": saved_work,
                "old_program": str(old_path),
                "old_program_sha256": sha(old_path),
                "new_program": str(new_path),
                "new_program_sha256": sha(new_path),
            }
        )
    return {
        "scope": "Eight reused development sources; identity of all retained comparisons, executable policies and prefix worlds; not a universal budget-reallocation theorem",
        "rows": rows,
        "preserved_adoptions": sum(r["preserved_adoptions"] for r in rows),
        "saved_transitions": sum(r["saved_transitions"] for r in rows),
        "all_retained_comparisons_identical": True,
        "protocol_sha256": sha(base / "protocol.json"),
        "verifier_sha256": sha(__file__),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    report = verify(args.base)
    write(args.destination, report)
    print(
        {
            "sources": len(report["rows"]),
            "preserved_adoptions": report["preserved_adoptions"],
            "saved_transitions": report["saved_transitions"],
        }
    )
