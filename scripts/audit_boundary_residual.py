"""Independent scalar/table reconstruction for the boundary-residual diagnostic."""

import argparse
import json
import math
import statistics as st
from pathlib import Path

import numpy as np

from koikoi.layout import TERMINAL
from koikoi.reference import ReferenceEnv
from scripts.run_arbitration_study import read, sha, write

KINDS = ("uniform", "identified_history")


def scalar_statistics(high, cheap, reference):
    x, y, c = high["high"], high["proxy"], cheap["proxy"]
    assert (len(x), len(y), len(c), len(reference)) == (512, 512, 2048, 2048)
    residual = [a - b for a, b in zip(x, y)]
    mx, my, mc, mr = map(st.mean, (x, y, c, residual))
    vx, vc, vr = map(st.variance, (x, c, residual))
    variance = vc / 2048 + vr / 512
    ordinary = [st.mean(x[i : i + 32]) for i in range(0, 512, 32)]
    corrected = [
        st.mean(c[4 * i : 4 * i + 128]) + st.mean(residual[i : i + 32])
        for i in range(0, 512, 32)
    ]
    ref = st.mean(reference)
    return {
        "mean": mc + mr,
        "standard_error": math.sqrt(variance),
        "cheap_mean": mc,
        "residual_mean": mr,
        "high_mean": mx,
        "cheap_variance": vc,
        "residual_variance": vr,
        "high_variance": vx,
        "high_proxy_covariance": sum((a - mx) * (b - my) for a, b in zip(x, y)) / 511,
        "alpha": 1.0,
        "reference_mean": ref,
        "reference_standard_error": st.stdev(reference) / math.sqrt(2048),
        "corrected_block_means": corrected,
        "ordinary_block_means": ordinary,
        "corrected_block_mse": st.mean((v - ref) ** 2 for v in corrected),
        "ordinary_block_mse": st.mean((v - ref) ** 2 for v in ordinary),
        "corrected_block_variance": st.variance(corrected),
        "ordinary_block_variance": st.variance(ordinary),
    }


def verify_part(part, labels, seat, table, terminal):
    fields = part["boundary_fields"]
    assert len(fields) == len(labels)
    assert part["field_order"] == ["round", "points0", "dealer", "phase"]
    values, ended = [], 0
    for pair, label in zip(fields, labels):
        assert len(pair) == 2 and label in (0, 1)
        estimate = []
        for r, p, d, phase in pair:
            assert 0 <= r < 8 and d in (0, 1) and 0 <= phase <= TERMINAL
            if phase == TERMINAL:
                value = (int(p > 30) - int(p < 30)) * (1 - 2 * seat)
                ended += 1
            else:
                assert 1 <= p <= 59 and r >= 1
                value = (
                    float(table[0, r, seat, label, p - 1, d])
                    + float(table[1, r, seat, label, p - 1, d])
                ) / 2
            estimate.append(value)
        values.append(estimate)
    np.testing.assert_allclose(
        part["paired_boundary_values"], values, rtol=0, atol=1e-12
    )
    proxy = [(b - a) / 2 for a, b in values]
    np.testing.assert_allclose(part["proxy"], proxy, rtol=0, atol=1e-12)
    first, work = part["first_round_work"], part["work"]
    assert (
        first["round_endings"] == len(labels) * 2
        and first["terminal_match_endings"] == ended
    )
    assert first["decisions"] >= len(labels) * 2
    if terminal:
        assert work["terminal_endings"] == len(labels) * 2
        assert (
            work["round_endings"] >= first["round_endings"]
            and work["decisions"] >= first["decisions"]
        )
        points = part["paired_points"]
        assert len(points) == len(labels)
        x = [
            ((int(b > 0) - int(b < 0)) - (int(a > 0) - int(a < 0))) / 2
            for a, b in points
        ]
        assert part["high"] == x
    else:
        assert work == {
            "decisions": first["decisions"],
            "round_endings": first["round_endings"],
            "terminal_endings": ended,
        }
        assert part["high"] is None and part["paired_points"] is None
    return proxy


def audit(output):
    receipt = output / "audit.json"
    if receipt.exists():
        raise ValueError("Preserve completed audits")
    report, protocol = read(output / "report.json"), read(output / "protocol.json")
    assert report["status"] == "complete" and report["inputs"] == protocol["inputs"]
    assert report["protocol_sha256"] == sha(output / "protocol.json")
    inputs = dict(report["inputs"])
    for path in (
        Path(__file__),
        Path("tests/test_audit_boundary_residual.py"),
        Path("runs/audit_boundary_residual_tests.xml"),
        output / "report.json",
        output / "protocol.json",
    ):
        inputs[str(path)] = sha(path)
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    tables = {}
    for p in (1, 2):
        data = []
        for name in ("A", "B"):
            with np.load(
                f"runs/round_value_accuracy/parent{p}_{name}/kernels.npz",
                allow_pickle=False,
            ) as f:
                data.append(f["values"].copy())
        tables[p] = np.stack(data)
    originals = [
        json.loads(line)
        for line in Path("runs/sampled_history_diagnostic/records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    originals = [r for r in originals if r["status"] == "eligible"]
    assert len(originals) == 25 and len(report["rows"]) == 50
    totals = {
        category: dict.fromkeys(("decisions", "round_endings", "terminal_endings"), 0)
        for category in ("high", "cheap", "reference")
    }
    results, prefix = [], 0
    for index, original in enumerate(originals):
        p, case = original["parent"], original["case_index"]
        source = json.loads(
            Path(f"runs/final_round_training_study/parent{p}/source.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[case]
        )
        env = ReferenceEnv(source["seed"], "paper8")
        for act in source["actions"][: original["decision"]]:
            assert env.legal()[act]
            env.step(act)
            prefix += 1
        base = source["actions"][original["decision"]]
        alternative = next(int(a) for a in np.flatnonzero(env.legal()) if a != base)
        for offset, kind in enumerate(KINDS):
            entry = report["rows"][index * 2 + offset]
            path = output / entry["file"]
            assert sha(path) == entry["sha256"]
            inputs[str(path)] = entry["sha256"]
            row = read(path)
            for key in (
                "parent",
                "case_index",
                "opponent",
                "seed",
                "seat",
                "decision",
                "public_identification",
            ):
                assert row[key] == original[key]
            assert (row["base"], row["alternative"], row["kind"]) == (
                base,
                alternative,
                kind,
            )
            assert row["public_root_key"] == original["proposals"]["root_key"]
            assert len({s["worlds_sha256"] for s in row["samples"].values()}) == 3
            for category, n in (("high", 512), ("cheap", 2048), ("reference", 2048)):
                sample = row["samples"][category]
                assert len(sample["labels"]) == n and set(sample["labels"]) <= {0, 1}
                if kind == "identified_history":
                    assert set(sample["labels"]) == {
                        ("point", "wp").index(
                            original["public_identification"]["model"]
                        )
                    }
                chunk = 2048 if category == "cheap" else 512
                assert len(sample["parts"]) == n // chunk and sample["seconds"] > 0
                for j, part in enumerate(sample["parts"]):
                    verify_part(
                        part,
                        sample["labels"][j * chunk : (j + 1) * chunk],
                        row["seat"],
                        tables[p],
                        category != "cheap",
                    )
                    for key, value in part["work"].items():
                        totals[category][key] += value
            high = row["samples"]["high"]["parts"][0]
            cheap = row["samples"]["cheap"]["parts"][0]
            ref = [
                x for part in row["samples"]["reference"]["parts"] for x in part["high"]
            ]
            stats = scalar_statistics(high, cheap, ref)
            assert set(stats) == set(row["statistics"])
            for key, value in stats.items():
                np.testing.assert_allclose(
                    row["statistics"][key], value, rtol=0, atol=1e-12, err_msg=key
                )
            cost_decisions = (
                high["work"]["decisions"] + cheap["work"]["decisions"]
            ) / high["work"]["decisions"]
            cost_time = (
                row["samples"]["high"]["seconds"] + row["samples"]["cheap"]["seconds"]
            ) / row["samples"]["high"]["seconds"]
            variance_ratio = (
                stats["standard_error"] ** 2 / (stats["high_variance"] / 512)
                if stats["high_variance"] > 0
                else None
            )
            results.append(
                {
                    "parent": p,
                    "case_index": case,
                    "opponent": original["opponent"],
                    "kind": kind,
                    "variance_ratio": variance_ratio,
                    "cost_ratio_decisions": cost_decisions,
                    "cost_ratio_time": cost_time,
                    "efficiency_ratio_decisions": variance_ratio * cost_decisions
                    if variance_ratio is not None
                    else None,
                    "efficiency_ratio_time": variance_ratio * cost_time
                    if variance_ratio is not None
                    else None,
                    "corrected_block_mse": stats["corrected_block_mse"],
                    "ordinary_block_mse": stats["ordinary_block_mse"],
                }
            )
    assert (
        report["work"] == totals and report["source_prefix_replay_decisions"] == prefix
    )
    assert all(
        report[k] == 0
        for k in (
            "new_historical_neural_calls",
            "actual_completions",
            "new_evaluation_deals",
            "training_updates",
        )
    )
    aggregate = {}
    for scope in ("all", "parent1", "parent2", "wp", "point"):
        aggregate[scope] = {}
        for kind in KINDS:
            rows = [
                r
                for r in results
                if r["kind"] == kind
                and (
                    scope == "all"
                    or scope == f"parent{r['parent']}"
                    or scope == r["opponent"]
                )
            ]
            group = {"cells": len(rows)}
            for key in (
                "variance_ratio",
                "efficiency_ratio_decisions",
                "efficiency_ratio_time",
                "corrected_block_mse",
                "ordinary_block_mse",
            ):
                values = [r[key] for r in rows if r[key] is not None]
                group[key] = {
                    "count": len(values),
                    "mean": st.mean(values) if values else None,
                    "median": st.median(values) if values else None,
                }
            group["lower_variance_cost_cells"] = sum(
                r["efficiency_ratio_time"] is not None
                and r["efficiency_ratio_time"] < 1
                for r in rows
            )
            aggregate[scope][kind] = group
    write(
        receipt,
        {
            "status": "passed",
            "inputs": inputs,
            "results": results,
            "aggregate": aggregate,
            "additional_source_replay_decisions": prefix,
            "additional_neural_calls": 0,
            "scope": "Independent public-table and scalar reconstruction; work totals checked, no neural trajectory rerun.",
        },
    )
    print(json.dumps({"status": "passed", "aggregate": aggregate["all"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
