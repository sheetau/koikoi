"""Visualize the audited finite-game expectation using exact enumeration."""

import json
import sys
from fractions import Fraction as F
from pathlib import Path

from scripts.train_branch_policy_gradient import sha, write


def plot():
    report_path = Path("runs/shared_observer_bootstrap_cpu.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    inputs = report["inputs"] | {
        str(report_path): sha(report_path),
        str(Path(__file__)): sha(__file__),
    }
    assert all(sha(path) == digest for path, digest in inputs.items())
    means = ((F(1, 5), F(21, 100)), (F(1), F(-1)))
    coefficients = (F(1), F(-1, 100))
    root_values = [sum(row) / 2 for row in means]
    opponent_values = [sum(row[action] for row in means) / 2 for action in (0, 1)]

    def expected(lam):
        return sum(
            coefficients[x]
            * (action - F(1, 2))
            * (
                lam * means[x][action]
                + (1 - lam) * opponent_values[action]
                - root_values[x]
            )
            / 4
            for x in (0, 1)
            for action in (0, 1)
        )

    truth = F(report["true_gradient"])
    threshold = -expected(F(0)) / (expected(F(1)) - expected(F(0)))
    assert expected(F(1)) == truth
    assert expected(F(19, 20)) == F(
        report["cases"]["opponent_observation_lambda095"]["gradient"]
    )
    assert threshold == F(report["lambda_positive_gradient_threshold"])
    assert expected(threshold) == 0
    grid = [F(9, 10) + F(index, 1000) for index in range(101)]
    values = [expected(lam) for lam in grid]
    sys.path.insert(0, str(Path("external/plotting").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, ax = plt.subplots(figsize=(7.0, 3.6))
    x, y = [float(lam) for lam in grid], [float(value) for value in values]
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.axhline(
        float(truth),
        color="#00796B",
        linestyle="--",
        linewidth=1.8,
        label="True gradient / learner-history return",
    )
    ax.plot(x, y, color="#285F9B", linewidth=2, label="Opponent-observation GAE")
    ax.fill_between(
        x, y, 0, where=[value < 0 for value in y], color="#B3453F", alpha=0.10
    )
    point = (0.95, float(expected(F(19, 20))))
    ax.scatter(*point, color="#B3453F", s=32, zorder=4)
    ax.annotate(
        r"$\lambda=0.95$: update $=-0.002594$",
        xy=point,
        xytext=(0.904, -0.0025),
        arrowprops={"arrowstyle": "-", "color": "#B3453F"},
        fontsize=9,
        color="#87342F",
    )
    ax.vlines(float(threshold), -0.0099, 0, color="#777777", linestyle=":", linewidth=1)
    ax.text(
        float(threshold) + 0.0015,
        -0.0094,
        r"Sign changes at" + "\n" + r"$\lambda=6567/6767$",
        fontsize=9,
    )
    ax.set(
        xlim=(0.9, 1),
        ylim=(-0.0102, 0.0053),
        xlabel=r"GAE trace parameter $\lambda$",
        ylabel=r"Expected update at $\theta=0$",
    )
    ax.set_xticks([0.9, 0.925, 0.95, 0.975, 1])
    ax.set_yticks([-0.01, -0.005, 0, 0.005])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        frameon=False,
        ncol=1,
        fontsize=9,
    )
    figure.tight_layout()
    output = Path("docs/paper/generated/observer_reversal.pdf")
    preview = Path("runs/observer_reversal.png")
    figure.savefig(
        output,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    figure.savefig(preview, dpi=160, bbox_inches="tight")
    plt.close(figure)
    write(
        Path("docs/paper/generated/observer_reversal_evidence.json"),
        {
            "status": "passed",
            "inputs": inputs,
            "outputs": {str(output): sha(output), str(preview): sha(preview)},
            "true_gradient": str(truth),
            "zero_crossing": str(threshold),
            "coordinates": [
                {"lambda": str(lam), "expected_gradient": str(value)}
                for lam, value in zip(grid, values, strict=True)
            ],
            "scope": "Exact four-event expectation of the existing finite example; no simulations, fitting or new strength measurements.",
        },
    )


if __name__ == "__main__":
    plot()
