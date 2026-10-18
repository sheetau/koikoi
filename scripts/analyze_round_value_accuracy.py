"""Fixed shared-deal uncertainty and complete terminal-reference comparison."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.round_value_uncertainty import bootstrap_values, deal_weights
from scripts.run_arbitration_study import sha, write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_bound(report):
    assert all(sha(path) == digest for path, digest in report["inputs"].items())


def load_table(folder, parent, table, inputs):
    path = folder / f"parent{parent}_{table}"
    report, independent, protocol, process = (
        read(path / name)
        for name in (
            "audit.json",
            "independent_audit.json",
            "protocol.json",
            "process.json",
        )
    )
    assert (
        report["status"] == independent["status"] == "passed"
        and process["status"] == "complete"
    )
    assert process["audit_sha256"] == sha(path / "audit.json")
    assert report["protocol_sha256"] == sha(path / "protocol.json")
    assert report["kernels_sha256"] == sha(path / "kernels.npz")
    assert protocol["parent"] == parent and protocol["samples_per_context"] == 128
    assert protocol["seed"] == {"A": 42110001, "B": 52110001}[table]
    verify_bound(report)
    verify_bound(independent)
    for name in (
        "audit.json",
        "independent_audit.json",
        "protocol.json",
        "kernels.npz",
    ):
        inputs[str(path / name)] = sha(path / name)
    with np.load(path / "kernels.npz", allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def load_reference(folder, parent, inputs):
    """Independently check endpoints, split files, utilities and exact round counts."""
    path = folder / f"reference_parent{parent}"
    protocol, report, process = (
        read(path / name) for name in ("protocol.json", "audit.json", "process.json")
    )
    assert report["status"] == "passed" and process["status"] == "complete"
    assert process["audit_sha256"] == sha(path / "audit.json")
    assert report["protocol_sha256"] == sha(path / "protocol.json")
    assert report["outcomes_sha256"] == sha(path / "outcomes.npz")
    verify_bound(report)
    assert report["inputs"] == protocol["inputs"]
    contexts = [[r, p, d] for r in (0, 3, 7) for p in (24, 30, 36) for d in (0, 1)]
    assert protocol["contexts"] == contexts and protocol["parent"] == parent
    assert (
        protocol["samples"] == 512
        and protocol["batch"] == 256
        and protocol["seed"] == 62110001
    )
    assert protocol["modes"] == ["native", "independent"] and protocol["models"] == [
        "point",
        "wp",
    ]
    for name in ("audit.json", "protocol.json", "outcomes.npz"):
        inputs[str(path / name)] = sha(path / name)
    with np.load(path / "outcomes.npz", allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    assert set(arrays) == {"points0", "terminal_round", "dealer", "signed_utility"}
    assert all(a.shape == (2, 2, 2, 18, 512) for a in arrays.values())
    q, r, d, utility = (
        arrays[k] for k in ("points0", "terminal_round", "dealer", "signed_utility")
    )
    assert all(a.dtype == np.int32 for a in (q, r, d)) and utility.dtype == np.int64
    starts = np.array([c[0] for c in contexts])[None, None, None, :, None]
    assert ((r >= starts) & (r <= 7)).all()
    assert ((r == 7) | (q <= 0) | (q >= 60)).all() and np.isin(d, (0, 1)).all()
    np.testing.assert_array_equal(
        utility,
        ((q > 30).astype(int) - (q < 30).astype(int))
        * np.array([1, -1])[None, :, None, None, None],
    )
    for a in arrays.values():
        np.testing.assert_array_equal(a[0, :, :, 12:], a[1, :, :, 12:])
    for mode_index, mode in enumerate(protocol["modes"]):
        for seat in (0, 1):
            for model in (0, 1):
                group = path / f"group_{mode}_{seat}_{model}.npz"
                inputs[str(group)] = sha(group)
                with np.load(group, allow_pickle=False) as data:
                    assert set(data.files) == set(arrays) - {"signed_utility"}
                    for name in data.files:
                        np.testing.assert_array_equal(
                            data[name], arrays[name][mode_index, seat, model]
                        )
    totals = dict.fromkeys(
        ("decisions", "round_endings", "terminal_paths", "reseeded_rounds"), 0
    )
    seen = set()
    for row in report["rows"]:
        mode, seat, model, offset = (
            row[k] for k in ("mode", "parent_seat", "model", "offset")
        )
        key = (mode, seat, model, offset)
        assert (
            key not in seen
            and mode in protocol["modes"]
            and seat in (0, 1)
            and model in (0, 1)
        )
        assert offset in range(0, 18 * 512, 256)
        seen.add(key)
        terminal_rounds = r[protocol["modes"].index(mode), seat, model].reshape(-1)[
            offset : offset + 256
        ]
        round_count = int((terminal_rounds - contexts[offset // 512][0] + 1).sum())
        work = row["work"]
        assert set(work) == set(totals) and all(
            type(v) is int and v >= 0 for v in work.values()
        )
        assert work["terminal_paths"] == 256 and work["round_endings"] == round_count
        assert work["reseeded_rounds"] == (
            round_count - 256 if mode == "independent" else 0
        )
        assert round_count <= work["decisions"] <= 64 * round_count
        for name in totals:
            totals[name] += work[name]
    assert (
        len(seen) == 288
        and totals == report["work"]
        and totals["terminal_paths"] == 73728
    )
    assert report["independent_complete_matches"] == report["policy_updates"] == 0
    return arrays, contexts, totals


def summary(values):
    values = np.asarray(values)
    return {
        "mean": float(values.mean()),
        "mae": float(np.abs(values).mean()),
        "rmse": float(np.sqrt(np.square(values).mean())),
        "max_abs": float(np.abs(values).max()),
    }


def interval(values, axis=0):
    return np.quantile(values, [0.025, 0.975], axis=axis)


def analyze(folder):
    if (folder / "analysis.json").exists() or (
        folder / "bootstrap_values.npz"
    ).exists():
        raise ValueError("Preserve previous analysis artifacts")
    process, tables_report = (
        read(folder / "tables_process.json"),
        read(folder / "tables_audit.json"),
    )
    assert process["status"] == "complete" and process["audit_sha256"] == sha(
        folder / "tables_audit.json"
    )
    verify_bound(tables_report)
    inputs = {
        str(p): sha(p)
        for p in (
            Path(__file__),
            Path("scripts/round_value_uncertainty.py"),
            Path("tests/test_round_value_uncertainty.py"),
            Path("docs/ROUND_VALUE_ACCURACY.rst"),
            folder / "tables_audit.json",
        )
    }
    for job in tables_report["completed"]:
        assert sha(folder / job["name"] / "audit.json") == job["audit_sha256"]
        assert (
            sha(folder / job["name"] / "independent_audit.json")
            == job["independent_audit_sha256"]
        )
    rng = np.random.default_rng(62000001)
    weights = np.stack([deal_weights(128, 256, rng) for _ in range(2)])
    ref_weights = np.random.default_rng(62000002).multinomial(
        512, np.full(512, 1 / 512), size=256
    )
    np.savez_compressed(
        folder / "bootstrap_weights.npz",
        table_weights=weights,
        reference_weights=ref_weights,
    )
    fitted, boot, references, contexts, reference_work = [], [], [], None, []
    for parent in (1, 2):
        parent_tables, parent_boot = [], []
        for table_index, table in enumerate(("A", "B")):
            data = load_table(folder, parent, table, inputs)
            parent_tables.append(data["values"])
            parent_boot.append(
                bootstrap_values(
                    data["points0"],
                    data["next_dealer"],
                    data["terminal"],
                    weights[table_index],
                )
            )
        fitted.append(parent_tables)
        boot.append(parent_boot)
        reference, current_contexts, work = load_reference(folder, parent, inputs)
        if contexts is not None:
            assert current_contexts == contexts
        contexts = current_contexts
        references.append(reference["signed_utility"])
        reference_work.append(work)
        print(
            json.dumps(
                {"parent": parent, "inputs_verified": True, "bootstrap_complete": True}
            ),
            flush=True,
        )
    fitted, boot, references = map(np.asarray, (fitted, boot, references))
    np.savez_compressed(folder / "bootstrap_values.npz", values=boot)
    table_difference = fitted[:, 0] - fitted[:, 1]
    table_difference_boot = boot[:, 0] - boot[:, 1]
    adjacent_difference = np.diff(table_difference, axis=4)
    adjacent_difference_boot = np.diff(table_difference_boot, axis=5)
    np.savez_compressed(
        folder / "table_differences.npz",
        A_minus_B=table_difference,
        A_minus_B_interval=interval(table_difference_boot, axis=1),
        adjacent_point_A_minus_B=adjacent_difference,
        adjacent_point_A_minus_B_interval=interval(adjacent_difference_boot, axis=1),
    )
    table_rows, comparisons, aggregates, chance_rows = [], [], [], []
    ref_boot = np.einsum("...n,bn->b...", references, ref_weights, optimize=True) / 512
    for p in range(2):
        for r in range(8):
            for seat in (0, 1):
                for model in (0, 1):
                    key = (p, r, seat, model)
                    table_rows.append(
                        {
                            "parent": p + 1,
                            "round": r,
                            "seat": seat,
                            "model": ("point", "wp")[model],
                            "A_minus_B": summary(table_difference[key]),
                            "adjacent_point_A_minus_B": summary(
                                adjacent_difference[key]
                            ),
                        }
                    )
        # All 72 contexts per parent/mode/table, in seat/model/context order.
        select = [
            (seat, model, c, r, q, d)
            for seat in (0, 1)
            for model in (0, 1)
            for c, (r, q, d) in enumerate(contexts)
        ]
        for t, table in enumerate(("A", "B")):
            for m, mode in enumerate(("native", "independent")):
                errors, error_boot = [], []
                for seat, model, c, r, q, d in select:
                    value = fitted[p, t, r, seat, model, q - 1, d]
                    empirical = references[p, m, seat, model, c].mean()
                    sampled_error = (
                        boot[p, t, :, r, seat, model, q - 1, d]
                        - ref_boot[:, p, m, seat, model, c]
                    )
                    differences = float(value - empirical)
                    comparisons.append(
                        {
                            "parent": p + 1,
                            "table": table,
                            "mode": mode,
                            "seat": seat,
                            "opponent": ("point", "wp")[model],
                            "round": r,
                            "points0": q,
                            "dealer": d,
                            "table_value": float(value),
                            "terminal_mean": float(empirical),
                            "signed_error": differences,
                            "error_interval": interval(sampled_error).tolist(),
                            "table_bootstrap_se": float(
                                boot[p, t, :, r, seat, model, q - 1, d].std(ddof=1)
                            ),
                            "reference_se": float(
                                references[p, m, seat, model, c].std(ddof=1)
                                / np.sqrt(512)
                            ),
                        }
                    )
                    errors.append(differences)
                    error_boot.append(sampled_error)
                matrix = np.array(error_boot).T
                aggregates.append(
                    {
                        "parent": p + 1,
                        "table": table,
                        "mode": mode,
                        "observed_error": summary(errors),
                        "mean_error_interval": interval(matrix.mean(1)).tolist(),
                        "observed_rmse_interval": interval(
                            np.sqrt(np.square(matrix).mean(1))
                        ).tolist(),
                    }
                )
        for seat, model, c, r, q, d in select:
            diff = references[p, 0, seat, model, c] - references[p, 1, seat, model, c]
            sampled_diff = (
                ref_boot[:, p, 0, seat, model, c] - ref_boot[:, p, 1, seat, model, c]
            )
            chance_rows.append(
                {
                    "parent": p + 1,
                    "seat": seat,
                    "opponent": ("point", "wp")[model],
                    "round": r,
                    "points0": q,
                    "dealer": d,
                    "native_minus_independent": float(diff.mean()),
                    "interval": interval(sampled_diff).tolist(),
                }
            )
    assert len(comparisons) == 576 and len(table_rows) == 64 and len(chance_rows) == 144
    assert all(sha(path) == digest for path, digest in inputs.items())
    result = {
        "status": "passed",
        "inputs": inputs,
        "units": "signed match utility; divide changes by two for draw-adjusted win score",
        "bootstrap": {
            "replicates": 256,
            "table_seed": 62000001,
            "reference_seed": 62000002,
            "coupling": "shared deal columns across all contexts and parents; independent table/round weights; reference paths resampled as common sample-index blocks",
        },
        "table_difference_summary": [summary(a) for a in table_difference],
        "adjacent_point_difference_summary": [summary(a) for a in adjacent_difference],
        "table_strata": table_rows,
        "reference_aggregates": aggregates,
        "reference_comparisons": comparisons,
        "chance_comparisons": chance_rows,
        "table_work": tables_report["work"],
        "reference_work": reference_work,
        "independent_complete_matches": 0,
        "policy_updates": 0,
        "artifacts": {
            name: sha(folder / name)
            for name in (
                "bootstrap_weights.npz",
                "bootstrap_values.npz",
                "table_differences.npz",
            )
        },
        "scope": "All estimator comparisons retained; raw RMSE includes terminal-reference sampling error. Bootstrap intervals are descriptive, not simultaneous guarantees. Saved reference endpoints/accounting verified without rerunning trajectories or remeasuring decisions. No policy choice, strength inference or promotion.",
    }
    write(folder / "analysis.json", result)
    print(
        json.dumps(
            {
                "status": "passed",
                "analysis_sha256": sha(folder / "analysis.json"),
                "reference_aggregates": aggregates,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    analyze(parser.parse_args().folder)
