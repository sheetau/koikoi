"""Re-reduce complete extension cohorts without changing frozen evidence."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from koikoi.layout import POINTS


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound_summary(base, filename):
    summary = read(base / filename)
    protocol = read(base / "protocol.json")
    require(
        summary["protocol_sha256"] == sha(base / "protocol.json"), "Protocol changed"
    )
    for path, digest in protocol["inputs"].items():
        require(sha(path) == digest, f"Frozen input changed: {path}")
    for path, digest in summary["result_bindings"].items():
        require(sha(path) == digest, f"Bound result changed: {path}")
    require(
        summary["criterion_met"]
        == all(v["lower"] > 0 for v in summary["comparisons"].values()),
        "Criterion and recorded bounds disagree",
    )
    return summary, protocol


def score(final, seat):
    margin = final[POINTS + seat] - final[POINTS + 1 - seat]
    return Fraction(1 + (margin > 0) - (margin < 0), 2)


def bootstrap_lower(strata, rng):
    """Independent strata; each vector's entries are intact sampling units."""
    samples = np.zeros(20000)
    for values in strata:
        values = np.asarray(values, dtype=float)
        samples += np.concatenate(
            [
                values[rng.integers(len(values), size=(1000, len(values)))].mean(axis=1)
                for _ in range(20)
            ]
        ) / len(strata)
    return float(np.quantile(samples, 0.025))


def frontier(base):
    summary, protocol = bound_summary(base, "summary.json")
    arms = protocol["arms"]
    totals = defaultdict(Counter)
    gains = {a: defaultdict(list) for a in arms}
    actual = {a: [] for a in arms}
    statuses = Counter()
    positives = Counter()
    for case in protocol["cases"]:
        i, pop = case["index"], case["population"]
        path = base / f"chunk_{i // 8:03}" / f"result_{i:04}.json"
        row = read(path)
        require(row["case"] == case, "Case changed")
        require(sha(row["source"]) == row["source_sha256"], "Source changed")
        source = read(row["source"])
        statuses[row["status"]] += 1
        totals["source_acquisition"].update(row["source_work"])
        totals["source_replay"].update(row["source_audit"]["replay_work"])
        if "support" in row:
            totals["support_reconstruction_once"].update(row["support"]["counts"])
            if row["status"] == "evaluated":
                totals["support_second_check"].update(row["support"]["counts"])
        for arm in arms:
            result = row["arms"].get(arm)
            gain, actual_delta = Fraction(0), Fraction(0)
            if result is not None:
                require(
                    sha(result["program"]) == result["program_sha256"],
                    "Program changed",
                )
                oracle = result["oracle"]
                gain = Fraction(oracle["exact_increment"])
                require(
                    gain >= Fraction(result["gain_lower"]) >= 0, "Invalid gain bound"
                )
                recomputed = Fraction(0)
                totals[arm + ":prefix_oracle"].update(oracle.get("prefix_physical", {}))
                for local in oracle.get("local_checks", []):
                    pair = local["oracle"]
                    delta = Fraction(pair["candidate"]["value"]) - Fraction(
                        pair["parent"]["value"]
                    )
                    require(
                        delta == Fraction(local["gain"]) and delta > 0,
                        "Invalid local gain",
                    )
                    recomputed += Fraction(local["reach"]) * delta
                    for value in pair.values():
                        totals[arm + ":local_oracle"].update(value["physical_work"])
                require(gain == recomputed, "Weighted local gains disagree")
                require(
                    result["decisions"] <= protocol["max_decisions"],
                    "Planning cap exceeded",
                )
                require(
                    result["adopted_nodes"] == len(oracle.get("local_checks", [])),
                    "Missing local audit",
                )
                totals[arm + ":planning"].update(
                    {k: result[k] for k in ("decisions", "seconds", "adopted_nodes")}
                )
                execution = result["execution"]
                require(
                    execution["actual_work"] == execution["replay_work"],
                    "Tail replay work differs",
                )
                totals[arm + ":actual_tails"].update(execution["actual_work"])
                totals[arm + ":tail_replay"].update(execution["replay_work"])
                outcomes = execution["outcomes"]
                actual_delta = score(
                    outcomes["controller"]["final"], source["seat"]
                ) - score(outcomes["parent"]["final"], source["seat"])
            gains[arm][pop].append(gain)
            actual[arm].append(actual_delta)
            positives[arm] += gain > 0
    for arm in arms:
        for pop, values in gains[arm].items():
            require(
                len(values) == protocol["design"]["sources_per_population"],
                "Incomplete population",
            )
            mean = sum(values, Fraction(0)) / len(values)
            require(
                abs(float(mean) - summary["means"][arm][pop]) < 1e-14, "Mean differs"
            )
        require(
            positives[arm] == summary["positive_sources"][arm], "Positive count differs"
        )
    require(dict(statuses) == summary["counts"], "Status count differs")
    mixtures = {
        a: {
            str(parent): sum(
                (
                    sum(gains[a][f"{parent}:{op}"], Fraction(0))
                    / len(gains[a][f"{parent}:{op}"])
                    for op in ("wp", "point")
                ),
                Fraction(0),
            )
            / 2
            for parent in (0, 1)
        }
        for a in arms
    }
    rng = np.random.default_rng(470310)
    for parent, contrast in summary["comparisons"].items():
        require(
            abs(
                float(mixtures[arms[1]][parent] - mixtures[arms[0]][parent])
                - contrast["mean_gain"]
            )
            < 1e-14,
            "Paired mean differs",
        )
        strata = [
            [
                float(n - o)
                for n, o in zip(
                    gains[arms[1]][f"{parent}:{op}"],
                    gains[arms[0]][f"{parent}:{op}"],
                    strict=True,
                )
            ]
            for op in ("wp", "point")
        ]
        require(
            abs(bootstrap_lower(strata, rng) - contrast["lower"]) < 1e-14,
            "Fresh-source bootstrap bound differs",
        )
    return {
        "sources": len(protocol["cases"]),
        "statuses": statuses,
        "positive_sources": positives,
        "work": totals,
        "exact_parent_mixture_gains": {
            a: {p: str(v) for p, v in g.items()} for a, g in mixtures.items()
        },
        "realized": {
            a: {
                "positive": sum(v > 0 for v in vs),
                "negative": sum(v < 0 for v in vs),
                "mean": str(sum(vs, Fraction(0)) / len(vs)),
            }
            for a, vs in actual.items()
        },
        "comparisons": summary["comparisons"],
        "criterion_met": summary["criterion_met"],
    }


def rules(base):
    summary, protocol = bound_summary(base, "confirmation_summary.json")
    totals, replay = Counter(), Counter()
    paired = defaultdict(dict)
    for path in summary["result_bindings"]:
        row = read(path)
        trace = Path(path).parent / "trace.npz"
        require(sha(trace) == row["trace_sha256"], "Rule trace changed")
        require(row["status"] == "passed", "Rule audit failed")
        with np.load(trace) as arrays:
            final = arrays["final"]
        size = row["pairs"]
        scores = [score(s.tolist(), int(i >= size)) for i, s in enumerate(final)]
        values = [(scores[i] + scores[i + size]) / 2 for i in range(size)]
        require(
            [float(v) for v in values] == row["paired_scores"],
            "Rule score transcription differs",
        )
        group = paired[row["variant"], str(row["parent"]), row["opponent"]]
        for index, value in enumerate(values):
            seed = row["seed"] + index
            require(seed not in group, "Duplicate rule stream")
            group[seed] = value
        totals.update(row["work"])
        replay.update(row["python_replay_work"])
    require(dict(totals) == summary["work"], "Rule work differs")
    exact = {}
    seeds = range(
        protocol["confirmation"]["seed"],
        protocol["confirmation"]["seed"] + protocol["confirmation"]["pairs"],
    )
    for key, group in paired.items():
        require(set(group) == set(seeds), "Rule stream coverage differs")
        paired[key] = [group[seed] for seed in seeds]
    rng = np.random.default_rng(40821)
    for parent in ("0", "1"):
        difference = Fraction(0)
        for opponent in protocol["opponents"]:
            for name in ("paper_control", summary["selected"]):
                vs = paired[name, parent, opponent]
                require(
                    len(vs) == protocol["confirmation"]["pairs"],
                    "Incomplete rule cohort",
                )
                mean = sum(vs, Fraction(0)) / len(vs)
                require(
                    float(mean) == summary["means"][name][parent][opponent],
                    "Rule mean differs",
                )
                difference += (
                    mean
                    * (-1 if name == "paper_control" else 1)
                    / len(protocol["opponents"])
                )
        require(
            float(difference) == summary["comparisons"][parent]["mean_gain"],
            "Rule contrast differs",
        )
        deltas = np.mean(
            [
                [
                    float(n - o)
                    for n, o in zip(
                        paired[summary["selected"], parent, opponent],
                        paired["paper_control", parent, opponent],
                        strict=True,
                    )
                ]
                for opponent in protocol["opponents"]
            ],
            axis=0,
        )
        require(
            abs(
                bootstrap_lower([deltas], rng) - summary["comparisons"][parent]["lower"]
            )
            < 1e-14,
            "Rule cluster-bootstrap bound differs",
        )
        exact[parent] = str(difference)
    return {
        "selected": summary["selected"],
        "unique_streams": summary["unique_deal_streams"],
        "means": summary["means"],
        "comparisons": summary["comparisons"],
        "criterion_met": summary["criterion_met"],
        "exact_mean_contrasts": exact,
        "work": totals,
        "python_replay_work": replay,
    }


def frontier_table(f):
    rows = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Parent & Optimized (pp) & Parent-default (pp) & Difference (pp) & Lower (pp) \\",
        r"\midrule",
    ]
    for parent, c in f["comparisons"].items():
        means = [
            100 * float(Fraction(f["exact_parent_mixture_gains"][a][parent]))
            for a in ("optimized_proposal", "parent_default_proposal")
        ]
        rows.append(
            f"{parent} & {means[0]:.6f} & {means[1]:.6f} & {100 * c['mean_gain']:.6f} & {100 * c['lower']:.6f}"
            + r" \\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(rows) + "\n"


def rule_table(report):
    rows = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Cohort & Parent & Mean gain (pp) & Lower bound (pp) \\",
        r"\midrule",
    ]
    for label, name in (
        ("First confirmation", "rules_first"),
        ("Independent replication", "rules_replication"),
    ):
        for parent, c in report[name]["comparisons"].items():
            rows.append(
                f"{label} & {parent} & {100 * c['mean_gain']:.6f} & {100 * c['lower']:.6f}"
                + r" \\"
            )
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(rows) + "\n"


def paper_tables(report):
    """Render measured values directly; exact values remain in the JSON ledger."""
    tables = {}
    if "frontier" in report:
        tables["research_frontier_comparison.tex"] = frontier_table(report["frontier"])
    if "rules_first" in report:
        tables["research_rule_comparison.tex"] = rule_table(report)
    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--tex-directory", type=Path)
    parser.add_argument("--study", choices=("all", "frontier", "rules"), default="all")
    args = parser.parse_args()
    require(not args.destination.exists(), "Refusing to overwrite prior evidence")
    bases = [
        Path("runs") / p
        for p in (
            "prospective_frontier_proposals_v1",
            "rule_variants_v1",
            "rule_variants_replication_v1",
        )
    ]
    report = {
        "scope": "Independent result reduction and binding checks, not new matches or policy replay",
    }
    paths = [Path(__file__)]
    if args.study in ("all", "frontier"):
        report["frontier"] = frontier(bases[0])
        paths.append(bases[0] / "summary.json")
    if args.study in ("all", "rules"):
        report["rules_first"] = rules(bases[1])
        report["rules_replication"] = rules(bases[2])
        paths.extend(b / "confirmation_summary.json" for b in bases[1:])
    tables = paper_tables(report) if args.tex_directory else {}
    for name in tables:
        require(
            not (args.tex_directory / name).exists(),
            "Refusing to overwrite prior table",
        )
    report["bindings"] = {str(p): sha(p) for p in paths}
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    if tables:
        args.tex_directory.mkdir(parents=True, exist_ok=True)
        for name, content in tables.items():
            with (args.tex_directory / name).open(
                "x", encoding="utf-8", newline="\n"
            ) as file:
                file.write(content)
    print(json.dumps({"status": "passed", "output": str(args.destination)}))
