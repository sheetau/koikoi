"""Plot observed choices and controller-only context sensitivity separately."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def plot(report_path, case, output):
    # Isolated plotting dependencies; never modify the live training environment.
    sys.path.insert(0, str(Path("external/plotting").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    report = json.loads(report_path.read_text(encoding="utf-8"))
    row = report["models"][case]
    plt.rcParams.update(
        {
            "font.size": 10,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [1, 1.5]}
    )
    groups = [row["groups"][name] for name in ("trailing", "tied", "leading")]
    values = [100 * group["point_fraction_at_disagreement"] for group in groups]
    axes[0].bar(range(3), values, width=0.6, color="#31779b")
    axes[0].set_xticks(
        range(3),
        [
            f"{name}\n(n={group['disagreements']})"
            for name, group in zip(("Trailing", "Tied", "Leading"), groups, strict=True)
        ],
    )
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Point expert selected (%)")
    axes[0].set_title("Observed cached states", loc="left", fontweight="bold")
    for index, value in enumerate(values):
        axes[0].text(index, value + 3, f"{value:.1f}", ha="center")
    grid = np.array(row["context_point_fraction"]) * 100
    im = axes[1].imshow(grid, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    axes[1].set_xticks(range(len(report["own_points"])), report["own_points"])
    axes[1].set_yticks(
        range(len(report["round_indices"])),
        [value + 1 for value in report["round_indices"]],
    )
    axes[1].set_xlabel("Controller's own-point input (total points = 60)")
    axes[1].set_ylabel("Round input")
    axes[1].set_title("Controller input intervention", loc="left", fontweight="bold")
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            axes[1].text(
                c,
                r,
                f"{grid[r, c]:.1f}",
                ha="center",
                va="center",
                color="white" if grid[r, c] > 60 else "black",
            )
    figure.colorbar(im, ax=axes[1], shrink=0.85, label="Point selected (%)")
    figure.suptitle(
        case.replace("_", " ") + ": choices at expert disagreements", fontsize=12
    )
    figure.text(
        0.5,
        0.025,
        "Right: expert proposals and all other inputs held fixed. Neither panel measures playing strength.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.report, args.case, args.output)


if __name__ == "__main__":
    main()
