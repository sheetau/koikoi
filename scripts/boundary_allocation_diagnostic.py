"""Post-hoc, plug-in allocation analysis; never a fitted policy or observed gain."""

import argparse
import math
from pathlib import Path

import numpy as np

from scripts.run_arbitration_study import read, sha, write


def optimal_independent_control(vx, vy, covariance, cheap_cost, high_cost):
    """Asymptotic variance*cost optimum for an independent two-level estimator.

    The moments supplied by the diagnostic are sample estimates. Their plug-in
    optimum is optimistic and requires an independent study before use.
    """
    values = (vx, vy, covariance, cheap_cost, high_cost)
    if (
        not all(math.isfinite(v) for v in values)
        or min(vx, vy) < 0
        or min(cheap_cost, high_cost) <= 0
    ):
        raise ValueError("Finite nonnegative variances and positive costs required")
    if covariance**2 > vx * vy + 1e-12:
        raise ValueError("Invalid covariance matrix")
    result = {"alpha": 0.0, "variance_cost_ratio": 1.0, "cheap_per_high": 0.0}
    if vx == 0 or vy == 0:
        return result
    rho = max(-1.0, min(1.0, covariance / math.sqrt(vx * vy)))
    r = cheap_cost / high_cost
    if r >= 1 or rho * rho <= r:
        return result
    u = abs(rho) - math.sqrt(r * (1 - rho * rho) / (1 - r))
    alpha = math.copysign(u * math.sqrt(vx / vy), rho)
    residual = max(0.0, vx - 2 * alpha * covariance + alpha * alpha * vy)
    # None denotes an infinite asymptotic allocation, not permission to omit
    # the mandatory high-fidelity samples in a finite experiment.
    allocation = (
        math.sqrt(alpha * alpha * vy * high_cost / (residual * cheap_cost))
        if residual > 0
        else None
    )
    return {
        "alpha": alpha,
        "variance_cost_ratio": (
            math.sqrt(r) * abs(rho) + math.sqrt((1 - r) * (1 - rho * rho))
        )
        ** 2,
        "cheap_per_high": allocation,
    }


def run(folder):
    output = folder / "allocation_diagnostic.json"
    if output.exists():
        raise ValueError("Preserve prior diagnostics")
    audit = read(folder / "audit.json")
    assert audit["status"] == "passed"
    inputs = dict(audit["inputs"])
    for path in (
        folder / "audit.json",
        Path(__file__),
        Path("tests/test_boundary_allocation_diagnostic.py"),
        Path("runs/boundary_allocation_diagnostic_tests.xml"),
    ):
        inputs[str(path)] = sha(path)
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    rows = []
    for entry in read(folder / "report.json")["rows"]:
        row = read(folder / entry["file"])
        high = row["samples"]["high"]
        cheap = row["samples"]["cheap"]
        x, y = (
            np.asarray(high["parts"][0]["high"]),
            np.asarray(high["parts"][0]["proxy"]),
        )
        vx, vy, cov = (
            float(x.var(ddof=1)),
            float(y.var(ddof=1)),
            float(np.cov(x, y, ddof=1)[0, 1]),
        )
        result = {k: row[k] for k in ("parent", "case_index", "kind", "opponent")}
        for cost in ("seconds", "decisions"):
            c = (
                cheap["seconds"]
                if cost == "seconds"
                else cheap["parts"][0]["work"]["decisions"]
            )
            h = (
                high["seconds"]
                if cost == "seconds"
                else high["parts"][0]["work"]["decisions"]
            )
            result[cost] = optimal_independent_control(vx, vy, cov, c / 2048, h / 512)
        rows.append(result)
    write(
        output,
        {
            "status": "complete",
            "inputs": inputs,
            "results": rows,
            "scope": "Post-hoc fitted moment/cost diagnostic only; no new trajectories, parameter adoption, measured accuracy or strength claim.",
            "new_neural_calls": 0,
            "new_simulation_decisions": 0,
        },
    )
    print(
        {
            cost: {
                "nonzero_coefficients": sum(r[cost]["alpha"] != 0 for r in rows),
                "mean_predicted_ratio": float(
                    np.mean([r[cost]["variance_cost_ratio"] for r in rows])
                ),
            }
            for cost in ("seconds", "decisions")
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
