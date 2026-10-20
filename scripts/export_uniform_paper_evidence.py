"""Transcribe a complete uniform-deal study into checked manuscript tables."""

import argparse
from collections import Counter
from fractions import Fraction
from pathlib import Path

from scripts.export_belief_tail import score
from scripts.run_arbitration_study import read, sha, write
from scripts.uniform_controller_design import POPULATIONS, summarize
from scripts.uniform_controller_study import check, chunk_records, report_text

PROTOCOL_SHA = "bc5fce90a1fefb146f766e2be7177c67992292307cf896413b525ba1183dbfc9"


def checked_result(base):
    """Recheck the full denominator and immutable receipts before transcription."""
    if read(base / "queue.json")["status"] != "complete":
        raise ValueError("Complete queue required; no interim paper export")
    if sha(base / "protocol.json") != PROTOCOL_SHA:
        raise ValueError("Unexpected frozen protocol")
    protocol = check(base)
    config = protocol["design"]
    summary, receipt = read(base / "summary.json"), read(base / "export.json")
    if (
        summary["audit_status"] != "passed"
        or receipt["status"] != "passed"
        or summary["protocol_sha256"] != PROTOCOL_SHA
        or receipt["protocol_sha256"] != PROTOCOL_SHA
        or receipt["summary_sha256"] != sha(base / "summary.json")
        or receipt["report_sha256"] != sha(base / "results.rst")
    ):
        raise ValueError("Study export binding differs")
    bindings, seen, statuses = {}, [], Counter()
    gains = {name: [] for name in POPULATIONS}
    actual = {name: [] for name in POPULATIONS}
    for chunk in range(config["source_matches"] // config["chunk_size"]):
        folder = base / f"chunk_{chunk:04}"
        rows = chunk_records(base, chunk, protocol)
        audit = read(folder / "audit.json")
        if (
            audit["status"] != "passed"
            or audit["protocol_sha256"] != PROTOCOL_SHA
            or audit["complete_sha256"] != sha(folder / "complete.json")
            or audit["fresh_external_chance_draws"] != 0
            or len(audit["gains"]) != len(rows)
            or len(audit["actual_differences"]) != len(rows)
        ):
            raise ValueError("Incomplete or mismatched audit")
        for name in ("audit.json", "complete.json"):
            bindings[str(folder / name)] = sha(folder / name)
        for row, gain, difference in zip(
            rows, audit["gains"], audit["actual_differences"], strict=True
        ):
            case = row["case"]
            expected = row["selection"]["gain"] if row["status"] == "evaluated" else "0"
            if gain != {"case": case, "gain": expected}:
                raise ValueError("Audited gain differs from source selection")
            realized = Fraction(0)
            if row["status"] == "evaluated":
                outcomes = row["assessment"]["outcomes"]

                realized = score(outcomes["controller"]["margin"]) - score(
                    outcomes["parent"]["margin"]
                )
            if Fraction(difference) != realized:
                raise ValueError("Audited realized difference differs")
            seen.append(case)
            statuses[row["status"]] += 1
            gains[case["population"]].append(Fraction(expected))
            actual[case["population"]].append(realized)
    if (
        seen != protocol["cases"]
        or len(seen) != config["source_matches"]
        or summary["sources"] != len(seen)
        or receipt["source_matches"] != len(seen)
        or summary["bindings"] != bindings
        or summary["status_counts"] != dict(statuses)
    ):
        raise ValueError("Full cohort or audit binding set differs")
    inference = summarize(gains, expected_n=config["sources_per_population"])
    descriptive = {
        name: str(sum(values, Fraction(0)) / len(values))
        for name, values in actual.items()
    }
    if (
        inference != summary["inference"]
        or descriptive != summary["actual_mean_difference_descriptive"]
        or receipt["criterion_met"] != inference["criterion_met"]
        or (base / "results.rst").read_text(encoding="utf-8") != report_text(summary)
    ):
        raise ValueError("Recomputed result differs from study export")
    return summary


def pp(value, *, lower=False):
    """Four decimal places in percentage points, without binary float rounding."""
    scaled = Fraction(value) * 1_000_000
    if lower:
        units = scaled.numerator // scaled.denominator
    else:
        magnitude = abs(scaled)
        units = (2 * magnitude.numerator + magnitude.denominator) // (
            2 * magnitude.denominator
        )
        units *= -1 if scaled < 0 else 1
    sign = "-" if units < 0 else ""
    return f"{sign}{abs(units) // 10000}.{abs(units) % 10000:04d}"


def tables(summary):
    inference = summary["inference"]
    primary = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Parent & Opponent & Sources & Mean (pp) & Lower bound (pp) \\",
        r"\midrule",
    ]
    for name in POPULATIONS:
        parent, opponent = name.split(":")
        row = inference["populations"][name]
        label = "WP" if opponent == "wp" else "Point"
        primary.append(
            f"{int(parent) + 1} & {label} & {row['sources']:,} & "
            f"{pp(row['mean'])} & {pp(row['lower'], lower=True)}" + r" \\"
        )
    primary.append(r"\midrule")
    for parent in ("0", "1"):
        row = inference["parent_mixtures"][parent]
        count = sum(
            inference["populations"][f"{parent}:{o}"]["sources"]
            for o in ("wp", "point")
        )
        primary.append(
            f"{int(parent) + 1} & Equal mixture & {count:,} & "
            f"{pp(row['mean'])} & {pp(row['lower'], lower=True)}" + r" \\"
        )
    primary += [r"\bottomrule", r"\end{tabular}"]
    actual = [
        r"\begin{tabular}{llr}",
        r"\toprule",
        r"Parent & Opponent & Realized difference (pp) \\",
        r"\midrule",
    ]
    for name in POPULATIONS:
        parent, opponent = name.split(":")
        label = "WP" if opponent == "wp" else "Point"
        actual.append(
            f"{int(parent) + 1} & {label} & "
            f"{pp(summary['actual_mean_difference_descriptive'][name])}" + r" \\"
        )
    actual += [r"\bottomrule", r"\end{tabular}"]
    return {
        "uniform_primary_table.tex": "\n".join(primary) + "\n",
        "uniform_actual_table.tex": "\n".join(actual) + "\n",
    }


def export(base, destination):
    summary = checked_result(base)
    outputs = tables(summary)
    ledger = destination / "uniform_results_evidence.json"
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
            "scope": "Complete audited conditional-expectation study; realized differences are descriptive.",
            "display": "Lower bounds rounded down to four decimals in pp; exact fractions decide the criterion.",
        },
    )
    return ledger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("runs/uniform_controller_evaluation")
    )
    parser.add_argument(
        "--destination", type=Path, default=Path("docs/paper/generated")
    )
    args = parser.parse_args()
    print(export(args.base, args.destination))
