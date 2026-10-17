"""Recompute stored gradient arithmetic without producer/statistics imports."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

BASE = Path("runs/objective_gradient_diagnostic")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(left, right):
    assert math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-14), (left, right)


def main():
    report = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    protocol = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    assert report["protocol"] == protocol["protocol"]
    assert report["status"] == "complete"
    assert report["script_sha256"] == sha("scripts/diagnose_objective_gradients.py")
    assert report["raw_sha256"] == sha(BASE / "gradients.npz")
    source = hashlib.sha256()
    for path in sorted(Path("koikoi").rglob("*")):
        if path.suffix in (".py", ".h", ".cpp", ".cu"):
            source.update(path.as_posix().encode())
            source.update(path.read_bytes())
    assert source.hexdigest() == report["source_sha256"]
    raw = np.load(BASE / "gradients.npz", allow_pickle=False)
    totals = {"decisions": 0, "rounds": 0, "matches": 0, "updates": 0}
    results = {}
    for label, path, digest, seed in protocol["protocol"]["parents"]:
        assert sha(path) == digest
        case = report["cases"][label]
        assert case["parent_sha256"] == digest and case["config"]["seed"] == seed
        assert case["weights_unchanged"] and case["optimizer_empty"]
        positions = []
        offset = 0
        for name, shape in case["parameter_layout"]:
            count = math.prod(shape)
            if name.startswith("body."):
                positions.extend(range(offset, offset + count))
            offset += count
        assert len(case["rollouts"]) == 4
        for index, row in enumerate(case["rollouts"]):
            assert row["index"] == index and row["rows"] == 131072
            assert row["counters"]["decisions"] == 131072 * (index + 1)
            assert row["counters"]["updates"] == 0
            assert row["production_gradient_max_error"] < 2e-7
            assert 0 <= row["forced_rows"] <= row["rows"]
            assert 0 <= row["koi_rows"] <= row["rows"]
            for term in protocol["protocol"]["gradient_terms"]:
                full = raw[f"{label}_{index}_all_{term}"]
                body = raw[f"{label}_{index}_body_{term}"]
                assert full.shape == (offset,) and np.isfinite(full).all()
                assert np.array_equal(full[positions], body)
            for scope in ("all", "body"):
                vectors = {
                    key: raw[f"{label}_{index}_{scope}_{key}"].astype(np.float64)
                    for key in protocol["protocol"]["gradient_terms"]
                }
                np.testing.assert_allclose(
                    vectors["policy"],
                    vectors["card_policy"] + vectors["koi_policy"],
                    rtol=2e-4,
                    atol=2e-7,
                )
                norms = {
                    key: math.sqrt(math.fsum(float(x) ** 2 for x in vector))
                    for key, vector in vectors.items()
                }
                for key, norm in norms.items():
                    close(norm, row[scope]["norms"][key])
                for key, comparison in row[scope]["comparisons"].items():
                    left, right = key.split("_vs_")
                    dot = math.fsum(
                        float(x) * float(y)
                        for x, y in zip(vectors[left], vectors[right])
                    )
                    close(dot, comparison["dot"])
                    product = norms[left] * norms[right]
                    if product:
                        close(dot / product, comparison["cosine"])
                    else:
                        assert comparison["cosine"] is None
        assert case["counters"] == case["rollouts"][-1]["counters"]
        for key in totals:
            totals[key] += case["counters"][key]
        rows = case["rollouts"]
        results[label] = {
            "policy_value_body_cosines": [
                r["body"]["comparisons"]["policy_vs_value"]["cosine"] for r in rows
            ],
            "forced_fraction": sum(r["forced_rows"] for r in rows)
            / sum(r["rows"] for r in rows),
            "max_production_gradient_error": max(
                r["production_gradient_max_error"] for r in rows
            ),
        }
    assert totals["decisions"] == 1048576 and totals["updates"] == 0
    audit = {
        "status": "passed",
        "report_sha256": sha(BASE / "report.json"),
        "auditor_sha256": sha(__file__),
        "totals": totals,
        "results": results,
        "scope": "Independent stored-vector arithmetic, body slicing, phase additivity, hashes and counter identities. "
        "Producer directly checked unchanged weights and production backward. This audit does not replay collection, gradients or training trajectories.",
    }
    (BASE / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
