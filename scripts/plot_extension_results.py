"""Plot paired changes from unchanged parents using audited study outcomes."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scripts.analyze_arbitration_study import interval, outcome, read, sha


def load_changes(base, study, parents):
    protocol = read(base / "protocol.json")
    if study == "arbitration":
        arms = {
            "full": "Learned selection",
            "phase": "Phase-only selection",
            "ppo": "Continued PPO",
        }
    else:
        arms = {
            "full": "Paired supervision",
            "no_match": "Without match inputs",
            "ppo_gate": "PPO selection",
            "ppo_actor": "Continued PPO",
        }
    # Permit a clearly marked first-parent development figure while replication runs.
    audit_path = base / "analysis.json"
    audit = None
    if parents == [1, 2]:
        audit = read(audit_path)
        if audit["status"] != "passed" or audit["protocol_sha256"] != sha(
            base / "protocol.json"
        ):
            raise ValueError("Need a passing audit of this completed study")
    elif parents != [1]:
        raise ValueError(
            "Use both parents, or the explicitly partial first-parent figure"
        )
    changes, inputs = {}, {str(base / "protocol.json"): sha(base / "protocol.json")}
    for index in parents:
        parent = f"parent{index}"
        for opponent in ("rl-wp-assisted", "rl-point-assisted"):
            report, base_wins, base_points = outcome(base, parent, opponent, protocol)
            if (
                audit is not None
                and report != audit["outcomes"][parent + "/" + opponent]
            ):
                raise ValueError("Parent outcome differs from completed audit")
            for suffix in (".json", ".jsonl"):
                path = base / parent / (opponent + suffix)
                inputs[str(path)] = sha(path)
            for arm in arms:
                case = parent + "_" + arm
                report, wins, points = outcome(base, case, opponent, protocol)
                if (
                    audit is not None
                    and report != audit["outcomes"][case + "/" + opponent]
                ):
                    raise ValueError("Candidate outcome differs from completed audit")
                for suffix in (".json", ".jsonl"):
                    path = base / case / (opponent + suffix)
                    inputs[str(path)] = sha(path)
                gain, point_gain = wins - base_wins, points - base_points
                changes[(index, arm, opponent)] = {
                    "wins": [
                        float(gain.mean()) * 100,
                        *np.multiply(interval(gain), 100).tolist(),
                    ],
                    "points": [float(point_gain.mean()), *interval(point_gain)],
                }
    if audit_path.exists():
        inputs[str(audit_path)] = sha(audit_path)
    return protocol, arms, changes, inputs


def plot(base, study, parents, output):
    protocol, arms, changes, inputs = load_changes(base, study, parents)
    sys.path.insert(0, str(Path("external/plotting").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.size": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.2), sharey=True)
    colors, markers = {1: "#0072B2", 2: "#D55E00"}, {1: "o", 2: "s"}
    for row, metric in enumerate(("wins", "points")):
        extent = max(
            abs(value) for result in changes.values() for value in result[metric]
        )
        extent = max(extent * 1.15, 0.5)
        for col, opponent in enumerate(("rl-wp-assisted", "rl-point-assisted")):
            ax = axes[row, col]
            ax.axvline(0, color="#666666", lw=1, ls="--")
            for y, arm in enumerate(arms):
                for index in parents:
                    offset = (-0.12 if index == 1 else 0.12) if len(parents) == 2 else 0
                    mean, low, high = changes[(index, arm, opponent)][metric]
                    ax.hlines(y + offset, low, high, color=colors[index], lw=1.6)
                    ax.vlines(
                        [low, high],
                        y + offset - 0.05,
                        y + offset + 0.05,
                        color=colors[index],
                        lw=1,
                    )
                    ax.plot(
                        mean,
                        y + offset,
                        marker=markers[index],
                        color=colors[index],
                        ms=6,
                    )
            ax.set_yticks(range(len(arms)), arms.values())
            ax.set_ylim(len(arms) - 0.55, -0.55)
            ax.set_xlim(-extent, extent)
            ax.grid(axis="x", alpha=0.18)
            ax.set_axisbelow(True)
            ax.set_xlabel(
                "Win-score change (percentage points)"
                if metric == "wins"
                else "Mean point-difference change"
            )
            if row == 0:
                ax.set_title(
                    "Against " + ("RL-WP" if col == 0 else "RL-Point"),
                    fontweight="bold",
                )
    title = "Paired changes from each unchanged parent"
    if parents == [1]:
        title += "\nFirst parent only; replication pending"
    fig.suptitle(title, fontsize=13)
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color=colors[i],
                marker=markers[i],
                ls="none",
                label=f"Parent {i}",
            )
            for i in parents
        ],
        loc="lower center",
        bbox_to_anchor=(0.57, 0.055),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.5,
        0.02,
        f"Development deals: {protocol['evaluation']['pairs']:,} duplicate pairs. Bars: 95% pair-bootstrap intervals. Positive changes favor the new model.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.095, 1, 0.99))
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=180)
    plt.close(fig)
    receipt = {
        "study": study,
        "parents": parents,
        "complete_replication": parents == [1, 2],
        "script_sha256": sha(__file__),
        "inputs": inputs,
        "outputs": {
            str(output.with_suffix(s)): sha(output.with_suffix(s))
            for s in (".png", ".svg", ".pdf")
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--study", choices=("arbitration", "paired"), required=True)
    parser.add_argument("--parents", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.base, args.study, args.parents, args.output)


if __name__ == "__main__":
    main()
