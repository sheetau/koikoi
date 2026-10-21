"""Recheck the complete incremental cohort before exporting effect sizes."""

import argparse
import json
from fractions import Fraction
from pathlib import Path

from scripts.frontier_incremental_study import reduce_study
from scripts.run_arbitration_study import read, sha, write


def report_text(summary):
    inference = summary["inference"]
    lines = [
        "Fresh incremental frontier-controller evaluation",
        "===============================================",
        "",
        f"Complete independently audited source matches: {summary['sources']}.",
        "Contrast: selected frontier hybrid minus existing symbolic controller.",
        "The primary values are exact conditional expected win-score increments.",
        "Every missing-root, refused-support and unchanged-policy source is included.",
        "The four population lower bounds form one simultaneous 95% family.",
        "",
        ".. csv-table:: Expected incremental gains",
        "   :header: Population,Sources,Mean(pp),Lower(pp),Exact lower",
        "",
    ]
    for name, row in inference["populations"].items():
        lines.append(
            f"   {name},{row['sources']},{float(Fraction(row['mean']) * 100):.12g},{float(Fraction(row['lower']) * 100):.12g},{row['lower']}"
        )
    lines += [
        "",
        ".. csv-table:: Fixed equal-opponent mixtures",
        "   :header: Parent,Mean(pp),Lower(pp),Positive lower bound",
        "",
    ]
    for name, row in inference["parent_mixtures"].items():
        lines.append(
            f"   {name},{float(Fraction(row['mean']) * 100):.12g},{float(Fraction(row['lower']) * 100):.12g},{row['criterion_met']}"
        )
    lines += [
        "",
        f"Primary criterion (both mixture lower bounds positive): {'PASSED' if inference['criterion_met'] else 'FAILED'}.",
        f"All four individual population lower bounds positive: {inference['all_population_lowers_positive']}.",
        "",
        "The previous controller-versus-parent 0.1pp criterion remains a separate",
        "failed test. A positive incremental result does not establish that practical",
        "threshold or absolute superiority over arbitrary opponents.",
        "",
        ".. csv-table:: Realized score differences (descriptive)",
        "   :header: Population,Hybrid minus old controller(pp)",
        "",
    ]
    for name, value in summary["actual_mean_difference_descriptive"].items():
        lines.append(f"   {name},{float(Fraction(value) * 100):.12g}")
    lines += [
        "",
        "Source outcomes and separate work ledgers",
        "----------------------------------------",
        "",
        "Planning physical transitions and represented belief mass are separate.",
        "Source matches are not augmented by oracle or replay completions.",
        "Learner executor calls do not purport to count underlying neural calls.",
        "",
        ".. code-block:: json",
        "",
    ]
    lines += [
        "   " + line
        for line in json.dumps(
            {"status_counts": summary["status_counts"], "work": summary["work"]},
            indent=2,
            sort_keys=True,
        ).splitlines()
    ]
    return "\n".join(lines) + "\n"


def export(base):
    if (base / "export.json").exists() or (base / "results.rst").exists():
        raise FileExistsError("Incremental export already exists")
    summary = read(base / "summary.json")
    if reduce_study(base) != summary:
        raise ValueError("Complete-source reduction differs from saved summary")
    (base / "results.rst").write_bytes(report_text(summary).encode("utf-8"))
    result = {
        "status": "passed",
        "script_sha256": sha(__file__),
        "protocol_sha256": sha(base / "protocol.json"),
        "summary_sha256": sha(base / "summary.json"),
        "report_sha256": sha(base / "results.rst"),
        "sources": summary["sources"],
        "criterion_met": summary["inference"]["criterion_met"],
    }
    write(base / "export.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    export(args.base)
