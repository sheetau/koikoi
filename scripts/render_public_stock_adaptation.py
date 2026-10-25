"""Render the complete V5A qualification tables without selecting outcomes."""

import argparse
from pathlib import Path

from scripts.export_public_stock_results import read, sha, write
from scripts.public_stock_robustness import ALL_THREE, CONTROL, PREVIEW_CHOICE, PROFILES

NAMES = {CONTROL: "Control", PREVIEW_CHOICE: "Preview + choice", ALL_THREE: "All three"}


def table(path, columns, rows, alignment):
    lines = [
        r"\begin{tabular}{" + alignment + "}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = read(args.evidence)
    if (
        data["method"] != "public_stock_adaptation_v5a"
        or len(data["comparisons"]) != 12
        or data["qualification_totals"]["matches"] != 36864
    ):
        raise ValueError("Complete V5A evidence is required")
    for path, digest in data["bindings"].items():
        if sha(path) != digest:
            raise ValueError(f"Evidence binding changed: {path}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    scores, contrasts = [], []
    for profile in PROFILES:
        for parent in range(2):
            label = f"{profile}_parent{parent}"
            row = data["scores"][label]
            scores.append(
                [
                    NAMES[profile],
                    str(parent),
                    *(
                        f"{row[k] * 100:.2f}"
                        for k in ("new_own", "new_other", "old_other")
                    ),
                ]
            )
            intervals = []
            for key in ("own", "other"):
                interval = data["comparisons"][f"{label}_{key}"]
                intervals.append(
                    f"{interval['mean'] * 100:+.2f} [{interval['lower'] * 100:+.2f}]"
                )
            contrasts.append([NAMES[profile], str(parent), *intervals])
    table(
        out / "adaptation_scores.tex",
        ["Rule", "Parent", r"New/own (\%)", r"New/other (\%)", r"Old/other (\%)"],
        scores,
        "llrrr",
    )
    table(
        out / "adaptation_contrasts.tex",
        ["Rule", "Parent", "Own change [LB]", "Other change [LB]"],
        contrasts,
        "llrr",
    )
    write(
        out / "table_evidence.json",
        {
            "inputs": {
                args.evidence: sha(args.evidence),
                str(Path(__file__)): sha(__file__),
            },
            "outputs": {p.as_posix(): sha(p) for p in sorted(out.iterdir())},
            "scope": "All six zero-search score triplets and all twelve one-sided paired qualification contrasts. No new strength evaluation or selection.",
        },
    )


if __name__ == "__main__":
    main()
