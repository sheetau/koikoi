"""Export only the complete, independently audited incremental study to TeX."""

import argparse
from fractions import Fraction
from pathlib import Path

from scripts import export_frontier_scope as study_export
from scripts.frontier_scope_study import reduce_study
from scripts.run_arbitration_study import read, sha, write
from scripts.uniform_controller_design import POPULATIONS

PROTOCOL_SHA = "793b6aa2fdae6f76397a8f17b1fb5bee0197ad0b311829a02728b7e43a872a1f"
SOURCE_MATCHES = 4096
RETAINED_SOURCES = 15


def checked_result(base):
    if read(base / "queue.json")["status"] != "complete":
        raise ValueError("Complete queue required; no interim paper export")
    if sha(base / "protocol.json") != PROTOCOL_SHA:
        raise ValueError("Unexpected frozen protocol")
    summary = read(base / "summary.json")
    if reduce_study(base) != summary:
        raise ValueError("Complete-source reduction differs from saved summary")
    receipt = read(base / "export.json")
    expected = {
        "status": "passed",
        "script_sha256": sha(study_export.__file__),
        "protocol_sha256": PROTOCOL_SHA,
        "summary_sha256": sha(base / "summary.json"),
        "report_sha256": sha(base / "results.rst"),
        "sources": summary["sources"],
        "criterion_met": summary["inference"]["criterion_met"],
    }
    if receipt != expected or (base / "results.rst").read_text(
        encoding="utf-8"
    ) != study_export.report_text(summary):
        raise ValueError("Study export binding or report differs")
    if (
        summary["sources"] != SOURCE_MATCHES
        or summary["status_counts"].get("retained_v1", 0) != RETAINED_SOURCES
        or summary["status_counts"].get("new_v2", 0)
        != SOURCE_MATCHES - RETAINED_SOURCES
    ):
        raise ValueError("Complete cohort acquisition counts differ")
    return summary


def pp(value, *, lower=False):
    """Six significant digits in pp; floor lower bounds using exact arithmetic.

    Scientific notation keeps arbitrarily small positive bounds visible. The
    exact fractions, never these display strings, decide the study criterion.
    """
    value = Fraction(value) * 100
    if not value:
        return "0"
    magnitude, exponent = abs(value), 0
    while magnitude >= 10:
        magnitude /= 10
        exponent += 1
    while magnitude < 1:
        magnitude *= 10
        exponent -= 1
    scaled = magnitude * 100_000
    if lower:
        signed = scaled if value > 0 else -scaled
        units = signed.numerator // signed.denominator
    else:
        units = (2 * scaled.numerator + scaled.denominator) // (2 * scaled.denominator)
        if value < 0:
            units = -units
    if abs(units) >= 1_000_000:
        units //= 10
        exponent += 1
    sign, digits = ("-" if units < 0 else ""), f"{abs(units):06d}"
    if exponent < -3 or exponent > 5:
        return rf"${sign}{digits[0]}.{digits[1:]}\times 10^{{{exponent}}}$"
    point = exponent + 1
    if point <= 0:
        return sign + "0." + "0" * (-point) + digits
    if point == 6:
        return sign + digits
    return sign + digits[:point] + "." + digits[point:]


def tables(summary):
    inference = summary["inference"]
    primary = [
        "% Complete incremental study: hybrid minus existing controller.",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Parent & Opponent & Sources & Mean (pp) & Lower bound (pp) \\",
        r"\midrule",
    ]
    actual = [
        "% Realized outcomes are descriptive; not the primary inference.",
        r"\begin{tabular}{llr}",
        r"\toprule",
        r"Parent & Opponent & Realized difference (pp) \\",
        r"\midrule",
    ]
    for name in POPULATIONS:
        parent, opponent = name.split(":")
        label = "WP" if opponent == "wp" else "Point"
        row = inference["populations"][name]
        primary.append(
            f"{int(parent) + 1} & {label} & {row['sources']:,} & "
            f"{pp(row['mean'])} & {pp(row['lower'], lower=True)}" + r" \\"
        )
        actual.append(
            f"{int(parent) + 1} & {label} & "
            f"{pp(summary['actual_mean_difference_descriptive'][name])}" + r" \\"
        )
    primary.append(r"\midrule")
    for parent in ("0", "1"):
        row = inference["parent_mixtures"][parent]
        count = sum(
            inference["populations"][f"{parent}:{opponent}"]["sources"]
            for opponent in ("wp", "point")
        )
        primary.append(
            f"{int(parent) + 1} & Equal mixture & {count:,} & "
            f"{pp(row['mean'])} & {pp(row['lower'], lower=True)}" + r" \\"
        )
    for lines in (primary, actual):
        lines.extend([r"\bottomrule", r"\end{tabular}"])
    result = {
        "frontier_increment_primary_table.tex": "\n".join(primary) + "\n",
        "frontier_increment_actual_table.tex": "\n".join(actual) + "\n",
    }
    outcome = "met" if inference["criterion_met"] else "not met"
    constants = [
        "% Complete audited study only. No population claim from development cases.",
        rf"\newcommand{{\FrontierIncrementSources}}{{{summary['sources']:,}}}",
        rf"\newcommand{{\FrontierIncrementCriterion}}{{{outcome}}}",
    ]
    for parent, label in (("0", "One"), ("1", "Two")):
        row = inference["parent_mixtures"][parent]
        constants.extend(
            [
                rf"\newcommand{{\FrontierIncrementMean{label}}}{{{pp(row['mean'])}}}",
                rf"\newcommand{{\FrontierIncrementLower{label}}}{{{pp(row['lower'], lower=True)}}}",
            ]
        )
    result["frontier_increment_constants.tex"] = "\n".join(constants) + "\n"
    return result


def export(base, destination):
    summary = checked_result(base)
    outputs = tables(summary)
    ledger = destination / "frontier_increment_results_evidence.json"
    if ledger.exists() or any((destination / name).exists() for name in outputs):
        raise FileExistsError("Paper result artifacts already exist")
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (destination / name).write_bytes(content.encode("utf-8"))
    inputs = {
        str(base / name): sha(base / name)
        for name in (
            "protocol.json",
            "queue.json",
            "summary.json",
            "export.json",
            "results.rst",
        )
    }
    inputs[str(Path(__file__).resolve())] = sha(__file__)
    write(
        ledger,
        {
            "status": "passed",
            "inputs": inputs,
            "outputs": {
                str(destination / name): sha(destination / name) for name in outputs
            },
            "summary": summary,
            "scope": "Complete hybrid-minus-old-controller conditional-expectation study; realized outcomes descriptive.",
            "acquisition": {
                "retained_before_correction": RETAINED_SOURCES,
                "new_after_correction": SOURCE_MATCHES - RETAINED_SOURCES,
            },
            "display": "Six significant digits in pp; lower bounds rounded downward; exact fractions determine the criterion.",
            "separate_closed_test": "The base-controller 0.1pp joint criterion failed and is not retested here.",
        },
    )
    return ledger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("runs/frontier_incremental_scope_v2")
    )
    parser.add_argument(
        "--destination", type=Path, default=Path("docs/paper/generated")
    )
    args = parser.parse_args()
    print(export(args.base, args.destination))
