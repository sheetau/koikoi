"""V4 figures and complete tables from the frozen independent reduction."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.export_public_stock_results import read, sha, write
from scripts.public_stock_robustness import ALL_THREE, CONTROL, PREVIEW_CHOICE, PROFILES

NAMES = {CONTROL: "Control", PREVIEW_CHOICE: "Preview + choice", ALL_THREE: "All three"}
COLORS = ("#707070", "#1764a0", "#ba5610")


def table(path, header, rows, columns):
    lines = [
        rf"\begin{{tabular}}{{{columns}}}", r"\toprule",
        " & ".join(header) + r" \\", r"\midrule",
        *(" & ".join(row) + r" \\" for row in rows),
        r"\bottomrule", r"\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def render(data, out):
    plt.rcParams.update({
        "pdf.fonttype": 42, "font.size": 12,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True, layout="constrained")
    for parent, ax in enumerate(axes):
        for profile, color in zip(PROFILES, COLORS, strict=True):
            work = [
                data["evaluation_endpoints"][f"evaluation_{profile}_parent{parent}_b{b}"]
                for b in range(3)
            ]
            curve = np.asarray(data["curves"][profile][parent])
            ax.plot(
                [r["transitions"] / r["matches"] / 1000 for r in work],
                (curve - curve[0]) * 100, color=color, marker="o",
                linewidth=1.5, label=NAMES[profile],
            )
        ax.axhline(0, color=".6", linewidth=0.7)
        ax.set_title(f"Parent {parent}")
        ax.set_xlabel("Observed transitions / match (thousands)")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Gain over no search (percentage points)")
    axes[0].legend(frameon=False, fontsize=10)
    fig.suptitle("Changed-opponent evaluation: observed mean work and score gains")
    fig.savefig(out / "public_stock_robustness_work.pdf")
    fig.savefig(out / "public_stock_robustness_work.png", dpi=180)
    plt.close(fig)

    labels, low, high = [], [], []
    rows = []
    for profile in PROFILES:
        for parent in range(2):
            label = f"{profile}_parent{parent}"
            labels.append(f"{NAMES[profile]}\nParent {parent}")
            for budget, deviations in ((1, low), (2, high)):
                check = data["work_admissibility"][f"{label}_b{budget}"]
                deviations.append(check["relative_deviation"] * 100)
                rows.append([
                    NAMES[profile], str(parent), str(check["target"] // 1000),
                    str(data["calibration"]["particles"][label][budget]),
                    f"{check['actual_per_match'] / 1000:.2f}",
                    f"{check['relative_deviation'] * 100:+.2f}",
                    "Yes" if check["passed"] else "No",
                ])
    table(
        out / "public_stock_robustness_costs.tex",
        ["Rule", "Parent", "Target (k)", "Worlds", "Actual (k)", r"Error (\%)", "Pass"],
        rows, "llrrrrl",
    )
    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    positions = np.arange(len(labels))
    ax.axhspan(-10, 10, color="#dce9df", alpha=0.6, label="Allowed interval")
    ax.axhline(-10, color="#477551", linestyle="--", linewidth=0.9)
    ax.axhline(10, color="#477551", linestyle="--", linewidth=0.9)
    ax.bar(positions - 0.18, low, width=0.34, color=COLORS[1], label="60k target")
    ax.bar(positions + 0.18, high, width=0.34, color=COLORS[2], label="240k target")
    ax.axhline(0, color=".4", linewidth=0.7)
    ax.set_xticks(positions, labels, fontsize=10)
    ax.set_ylabel("Observed work deviation from target (%)")
    ax.set_title("Held-out cost admissibility: every endpoint must be within 10%")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncols=3, fontsize=10,
               loc="outside lower center")
    fig.savefig(out / "public_stock_robustness_costs.pdf")
    fig.savefig(out / "public_stock_robustness_costs.png", dpi=180)
    plt.close(fig)

    rows = []
    for profile in (PREVIEW_CHOICE, ALL_THREE):
        for parent in range(2):
            for key, name in (("gain", "High--zero"), ("excess", "Excess over control"), ("high_minus_low", "High--low")):
                row = data["comparisons"][f"{profile}_parent{parent}_{key}"]
                rows.append([
                    NAMES[profile], str(parent), name,
                    *(f"{row[k] * 100:+.2f}" for k in ("mean", "lower", "upper")),
                ])
    for parent in range(2):
        row = data["comparisons"][f"preview_minus_all_parent{parent}"]
        rows.append([
            "Preview + choice minus all three", str(parent), "Gain difference",
            *(f"{row[k] * 100:+.2f}" for k in ("mean", "lower", "upper")),
        ])
    table(
        out / "public_stock_robustness_comparisons.tex",
        ["Rule", "Parent", "Comparison", "Mean", "Lower", "Upper"], rows, "lllrrr",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = read(args.evidence)
    if data["method"] != "public_stock_robustness_v4" or len(data["comparisons"]) != 14:
        raise ValueError("Expected the complete independently reduced V4 study")
    for path, digest in data["bindings"].items():
        if sha(path) != digest:
            raise ValueError(f"Evidence binding changed: {path}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    render(data, out)
    outputs = {p.as_posix(): sha(p) for p in sorted(out.iterdir())}
    write(out / "figure_evidence.json", {
        "inputs": {args.evidence: sha(args.evidence), str(Path(__file__)): sha(__file__)},
        "outputs": outputs,
        "scope": "Means at observed work; complete fourteen simultaneous paired intervals and twelve work-admissibility endpoints. No new strength evaluation or statistical selection.",
    })


if __name__ == "__main__":
    main()
