"""Independently reconstruct saved round-table arithmetic and accounting.

Does not import the producer's backup implementation or rerun neural trajectories.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit(folder):
    folder = Path(folder)
    output = folder / "independent_audit.json"
    if output.exists():
        raise ValueError("Preserve the previous independent audit")
    protocol, report, process = (
        read(folder / name) for name in ("protocol.json", "audit.json", "process.json")
    )
    assert process["status"] == "complete" and report["status"] == "passed"
    assert process["audit_sha256"] == sha(folder / "audit.json")
    assert report["protocol_sha256"] == sha(folder / "protocol.json")
    assert report["kernels_sha256"] == sha(folder / "kernels.npz")
    assert report["inputs"] == protocol["inputs"]
    inputs = protocol["inputs"].copy()
    inputs.update(
        {
            str(folder / name): sha(folder / name)
            for name in ("protocol.json", "audit.json", "kernels.npz", "process.json")
        }
    )
    inputs[str(Path(__file__))] = sha(__file__)
    assert all(sha(p) == h for p, h in inputs.items())
    source_digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "koikoi").rglob("*")):
        if path.suffix in (".py", ".h", ".cpp", ".cu"):
            source_digest.update(path.relative_to(root).as_posix().encode())
            source_digest.update(path.read_bytes())
    assert source_digest.hexdigest() == protocol["provenance"]["source_sha256"]
    n = protocol["samples_per_context"]
    assert type(n) is int and n >= 1
    assert protocol["opponent_models"] == ["point", "wp"]
    assert protocol["axes"] == [
        "round",
        "parent_seat",
        "opponent_model",
        "points0_minus1",
        "dealer",
        "sample",
    ]
    with np.load(folder / "kernels.npz", allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    assert set(arrays) == {"points0", "next_dealer", "terminal", "next_round", "values"}
    points, dealers, done, rounds = (
        arrays[k] for k in ("points0", "next_dealer", "terminal", "next_round")
    )
    shape = (8, 2, 2, 59, 2, n)
    assert all(a.shape == shape for a in (points, dealers, done, rounds))
    assert all(a.dtype == np.int32 for a in (points, dealers, rounds))
    assert done.dtype == bool and arrays["values"].dtype == np.float64
    assert arrays["values"].shape == shape[:-1]
    assert np.isin(dealers, (0, 1)).all()
    assert ((points[~done] >= 1) & (points[~done] <= 59)).all()
    for r in range(8):
        assert (rounds[r][done[r]] == r).all()
        assert (rounds[r][~done[r]] == r + 1).all()
        expected_terminal = (points[r] <= 0) | (points[r] >= 60) | (r == 7)
        np.testing.assert_array_equal(done[r], expected_terminal)
        path = folder / f"round_{r}.npz"
        inputs[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as part:
            assert set(part.files) == set(arrays) - {"values"}
            for name in part.files:
                np.testing.assert_array_equal(part[name], arrays[name][r])
    # Scalar context recursion deliberately differs from the producer's vector backup.
    expected = np.empty(shape[:-1], dtype=np.float64)
    for r in reversed(range(8)):
        for seat in (0, 1):
            for model in (0, 1):
                for p in range(59):
                    for dealer in (0, 1):
                        key = (r, seat, model, p, dealer)
                        total = 0.0
                        for sample in range(n):
                            k = (*key, sample)
                            q = int(points[k])
                            if done[k]:
                                total += ((q > 30) - (q < 30)) * (1 - 2 * seat)
                            else:
                                total += expected[r + 1, seat, model, q - 1, dealers[k]]
                        expected[key] = total / n
    assert np.isfinite(arrays["values"]).all()
    assert (np.abs(arrays["values"]) <= 1).all()
    np.testing.assert_allclose(arrays["values"], expected, rtol=0, atol=2e-14)
    groups = {}
    totals = dict.fromkeys(("decisions", "round_endings", "terminal_match_endings"), 0)
    for row in report["rows"]:
        key = tuple(row[k] for k in ("round", "parent_seat", "model"))
        assert key not in groups
        groups[key] = row["work"]
        work = row["work"]
        assert set(work) == set(totals)
        assert all(type(v) is int and v >= 0 for v in work.values())
        assert work["round_endings"] == 118 * n
        assert work["terminal_match_endings"] == int(done[key].sum())
        assert 118 * n <= work["decisions"] <= 64 * 118 * n
        for name in totals:
            totals[name] += work[name]
    assert set(groups) == {(r, s, m) for r in range(8) for s in (0, 1) for m in (0, 1)}
    assert totals == report["work"]
    assert totals["round_endings"] == 3776 * n
    assert report["contexts"] == 3776 and report["distinct_round_deal_inputs"] == 8 * n
    assert report["independent_complete_matches"] == report["policy_updates"] == 0
    assert report["calibrated"] is False
    assert all(sha(p) == h for p, h in inputs.items())
    result = {
        "status": "passed",
        "inputs": inputs,
        "parent": protocol["parent"],
        "samples_per_context": n,
        "contexts": 3776,
        "work": totals,
        "maximum_backup_difference": float(np.abs(expected - arrays["values"]).max()),
        "new_simulation_decisions": 0,
        "independent_complete_matches": 0,
        "scope": "Public endpoints, saved splits, scalar backups, source identities and reported accounting verified; neural trajectories and timing not rerun; decision counts checked for consistency, not independently remeasured; no calibration or strength claim",
    }
    output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"status": "passed", "sha256": sha(output), "work": totals}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    audit(parser.parse_args().folder)
