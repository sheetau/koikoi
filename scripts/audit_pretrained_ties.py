"""Post-hoc exact-score-tie supplement using frozen diagnostic scores only.

Checks vector probabilities against a separate scalar partition calculation.
Exact score ties do not certify strategically equivalent game actions.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

BASE = Path("runs/pretrained_policy_diagnostic")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector_probabilities(scores, mask, temperature):
    values = np.where(mask, scores.astype(float), -np.inf)
    best = values.max(1, keepdims=True)
    tied = mask & (values == best)
    weights = np.exp((values - best) / temperature)
    probabilities = weights / weights.sum(1, keepdims=True)
    return probabilities.max(1), (probabilities * tied).sum(1), tied.sum(1)


def scalar_probabilities(scores, mask, temperature):
    output = []
    for row, legal in zip(scores, mask):
        values = [float(value) for value, keep in zip(row, legal) if keep]
        best = max(values)
        denominator = math.fsum(
            math.exp((value - best) / temperature) for value in values
        )
        count = values.count(best)
        output.append((1 / denominator, count / denominator, count))
    return tuple(np.array(output).T)


def main():
    assert (
        sha(BASE / "scores.npz")
        == "b34b56d132bb691eb1cc1dfe85d6f88e90b941548e4694a778ed0724ee2859c8"
    )
    assert (
        sha(BASE / "report.json")
        == "b10b32a59fbe9c8c0085ea428a7855a5e5a71354ffbe09f75512aec390930142"
    )
    assert (
        sha(BASE / "audit.json")
        == "21aac38fb9d5c45715fbcccf8e15eafbafa78a15e9d8c4583c9cca153721fdc5"
    )
    original = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    with np.load(BASE / "scores.npz", allow_pickle=False) as archive:
        raw = {k: archive[k] for k in archive.files}
    mask = raw["mask"]
    selected = (raw["sources"] >= 0) & (mask.sum(1) > 1)
    result = {}
    for device in ("cpu", "cuda"):
        initial = np.where(mask, raw[device + "_initial"], -np.inf)
        initial_best = mask & (initial == initial.max(1, keepdims=True))
        for name in ("initial", "seed6681001", "seed6691001"):
            key = device + "_" + name
            scores = raw[key]
            values = np.where(mask, scores, -np.inf)
            chosen = values.argmax(1)
            best = mask & (values == values.max(1, keepdims=True))
            result[key] = {}
            for phase in range(4):
                rows = selected & (raw["phases"] == phase)
                subset, legal = scores[rows], mask[rows]
                counts = best[rows].sum(1)
                phase_result = {
                    "rows": int(rows.sum()),
                    "rows_with_multiple_maxima": int((counts > 1).sum()),
                    "maximum_count_histogram": {
                        str(n): int((counts == n).sum()) for n in np.unique(counts)
                    },
                    "greedy_action_outside_initial_maximum_set": int(
                        (~initial_best[np.arange(len(scores)), chosen] & rows).sum()
                    ),
                    "disjoint_initial_current_maximum_sets": int(
                        (~(best & initial_best).any(1) & rows).sum()
                    ),
                    "temperatures": {},
                }
                for temperature in (0.05, 0.025, 0.01):
                    vector = vector_probabilities(subset, legal, temperature)
                    scalar = scalar_probabilities(subset, legal, temperature)
                    for first, second in zip(vector, scalar):
                        np.testing.assert_allclose(first, second, rtol=0, atol=1e-12)
                    previous = original["cases"][key]["temperatures"][str(temperature)][
                        str(phase)
                    ]
                    np.testing.assert_allclose(
                        vector[0].mean(),
                        previous["mean_greedy_probability"],
                        rtol=0,
                        atol=1e-12,
                    )
                    phase_result["temperatures"][str(temperature)] = {
                        "mean_single_argmax_probability": float(vector[0].mean()),
                        "mean_exact_maximum_set_probability": float(vector[1].mean()),
                        "mean_outside_exact_maximum_set_probability": float(
                            (1 - vector[1]).mean()
                        ),
                    }
                result[key][str(phase)] = phase_result
    output = {
        "status": "passed",
        "scope": "post-hoc scalar/vector tie-mass arithmetic on saved scores; no inference rerun or new games; exact ties need not be strategically equivalent",
        "script_sha256": sha(__file__),
        "parent_report_sha256": sha(BASE / "report.json"),
        "parent_audit_sha256": sha(BASE / "audit.json"),
        "cases": result,
    }
    path = BASE / "tie_audit.json"
    assert not path.exists()
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "passed", "output": str(path), "sha256": sha(path)}))


if __name__ == "__main__":
    main()
