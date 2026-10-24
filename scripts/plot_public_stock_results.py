"""Publication figure and compact table from independently reduced evidence."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES = {
    "preview0_choice0_stakes0": "Control",
    "preview0_choice0_stakes1": "Stakes",
    "preview0_choice1_stakes0": "Choice",
    "preview0_choice1_stakes1": "Choice + stakes",
    "preview1_choice0_stakes0": "Preview",
    "preview1_choice0_stakes1": "Preview + stakes",
    "preview1_choice1_stakes0": "Preview + choice",
    "preview1_choice1_stakes1": "All three",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evidence", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    data = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        2, 4, figsize=(12, 6), sharex=True, sharey=True, layout="constrained"
    )
    for ax, (profile, title) in zip(axes.flat, NAMES.items(), strict=True):
        curves = data["development_curve_gain_intervals_exploratory"][profile]
        for parent, color in enumerate(("#1764a0", "#ba5610")):
            x = np.arange(4) + (parent - 0.5) * 0.06
            mean = np.array([r["mean"] for r in curves[parent]]) * 100
            low = np.array([r["lower"] for r in curves[parent]]) * 100
            high = np.array([r["upper"] for r in curves[parent]]) * 100
            ax.errorbar(
                x,
                mean,
                yerr=[mean - low, high - mean],
                color=color,
                marker="o",
                markersize=3,
                linewidth=1.3,
                capsize=2,
                label=f"Parent {parent}",
            )
        ax.axhline(0, color=".55", linewidth=0.65)
        ax.set_title(title)
        ax.set_xticks(range(4), ["0", "1x", "4x", "16x"])
        ax.grid(axis="y", alpha=0.18)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.supylabel("Win-score gain over no search (percentage points)")
    fig.supxlabel(
        "Particle budget per legal action (0, 2, 8, 32); horizon 32 decisions"
    )
    fig.suptitle(
        "Public-stock rule development: 64 duplicate streams, two adapted parents\nPaired 95% exploratory intervals; no multiplicity adjustment",
        fontsize=11,
    )
    fig.savefig(
        out / "public_stock_scaling.pdf",
        metadata={"Title": "Public-stock rule development"},
    )
    fig.savefig(out / "public_stock_scaling.png", dpi=180)
    plt.close(fig)
    rows = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Rule & $1\times$ & $4\times$ & $16\times$ & $16\times$ work \\",
        r"\midrule",
    ]
    for profile, title in NAMES.items():
        curves = np.asarray(data["curves"][profile])
        gains = (curves[:, 1:] - curves[:, :1]) * 100
        values = [f"{gains[0, j]:+.2f}/{gains[1, j]:+.2f}" for j in range(3)]
        work = [
            data["screen_endpoints"][f"{profile}_parent{parent}_p32"]
            for parent in range(2)
        ]
        cost = "/".join(f"{r['transitions'] / r['matches'] / 1000:.1f}" for r in work)
        rows.append(title + " & " + " & ".join(values + [cost]) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (out / "public_stock_curves.tex").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
