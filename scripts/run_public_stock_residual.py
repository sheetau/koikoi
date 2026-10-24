"""Frozen-parent correction matrix, quality gates, and a fixed-budget rule screen."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from koikoi.public_stock import PROFILES
from scripts.evaluate_public_stock_residual import sha, write
from scripts.run_arbitration_study import PARENTS

CONTROL = "preview0_choice0_stakes0"


def run(cmd, log):
    print(json.dumps({"command": cmd}), flush=True)
    with Path(log).open("w", encoding="utf-8", newline="\n") as stream:
        subprocess.run(
            [sys.executable, "-X", "utf8", "-m", *cmd],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        "profiles": list(PROFILES),
        "parents": PARENTS,
        "training_seeds": [931, 932],
        "training_deal_seeds": [823000000, 824000000],
        "updates": 128,
        "decisions_per_model": 4194304,
        "qualification_seed": 825000000,
        "qualification_pairs": 256,
        "development_seed": 826000000,
        "development_pairs": 64,
        "particles": [0, 2, 8, 32],
        "horizon": 32,
        "confidence": 1.0,
        "search_seed": 941,
        "selection": "maximum worse-parent high-budget gain; lexicographic ties",
        "confirmation_reserved_seed": 827000000,
        "confirmation_pairs": 512,
        "sources": {
            p: sha(p)
            for p in (
                "scripts/run_public_stock_residual.py",
                "scripts/train_public_stock_residual.py",
                "scripts/evaluate_public_stock_residual.py",
                "koikoi/public_stock_residual.py",
                "koikoi/public_stock_residual_search.py",
                "koikoi/public_stock_search.py",
                "docs/PUBLIC_STOCK_STUDY_V2.rst",
            )
        },
    }
    write(out / "protocol.json", protocol)

    def checkpoint(profile, parent):
        return out / f"{profile}_parent{parent}" / "policy_00128.pt"

    for profile in PROFILES:
        for parent in range(2):
            run(
                [
                    "scripts.train_public_stock_residual",
                    "--parent",
                    str(parent),
                    "--profile",
                    profile,
                    "--out",
                    str(checkpoint(profile, parent).parent),
                    "--updates",
                    "128",
                    "--save-every",
                    "128",
                    "--seed",
                    str(931 + parent),
                    "--deal-seed",
                    str(protocol["training_deal_seeds"][parent]),
                ],
                out / f"training_{profile}_parent{parent}.log",
            )
            print(json.dumps({"trained": profile, "parent": parent}), flush=True)
        if profile == CONTROL:
            quality = []
            for parent in range(2):
                for opponent in ("greedy", "published_wp"):
                    dest = out / f"qualification_parent{parent}_{opponent}"
                    run(
                        [
                            "scripts.evaluate_public_stock_residual",
                            "evaluate",
                            "--checkpoint",
                            str(checkpoint(profile, parent)),
                            "--opponent",
                            opponent,
                            "--out",
                            str(dest),
                            "--deal-seed",
                            "825000000",
                            "--pairs",
                            "256",
                            "--batch",
                            "64",
                        ],
                        out / f"{dest.name}.log",
                    )
                    result = json.loads(
                        (dest / "result.json").read_text(encoding="utf-8")
                    )
                    quality.append(
                        {
                            "parent": parent,
                            "opponent": opponent,
                            "win_rate": result["win_rate"],
                            "passed": result["win_rate"]
                            >= (0.65 if opponent == "greedy" else 0.45),
                        }
                    )
            write(out / "qualification.json", quality)
            print(json.dumps({"qualification": quality}), flush=True)
            if not all(x["passed"] for x in quality):
                write(
                    out / "closed.json",
                    {"reason": "qualification failed; no rule screen"},
                )
                return
    curves = {}
    for profile in PROFILES:
        curves[profile] = []
        for parent in range(2):
            values = []
            for particles in protocol["particles"]:
                dest = out / f"screen_{profile}_parent{parent}_p{particles}"
                run(
                    [
                        "scripts.evaluate_public_stock_residual",
                        "evaluate",
                        "--checkpoint",
                        str(checkpoint(profile, parent)),
                        "--out",
                        str(dest),
                        "--deal-seed",
                        "826000000",
                        "--pairs",
                        "64",
                        "--batch",
                        "64",
                        "--particles",
                        str(particles),
                        "--horizon",
                        "32",
                        "--search-seed",
                        "941",
                    ],
                    out / f"{dest.name}.log",
                )
                result = json.loads((dest / "result.json").read_text(encoding="utf-8"))
                values.append(result["win_rate"])
                print(
                    json.dumps(
                        {
                            "profile": profile,
                            "parent": parent,
                            "particles": particles,
                            "win_rate": result["win_rate"],
                        }
                    ),
                    flush=True,
                )
            curves[profile].append(values)
        write(out / "curves_partial.json", curves)
    gains = {
        profile: (np.asarray(value)[:, -1] - np.asarray(value)[:, 0]).tolist()
        for profile, value in curves.items()
    }
    selected = min(
        (p for p in PROFILES if p != CONTROL), key=lambda p: (-min(gains[p]), p)
    )
    passes = min(gains[selected]) > max(0.0, min(gains[CONTROL]))
    write(
        out / "screen_result.json",
        {
            "curves": curves,
            "gains": gains,
            "selected": selected,
            "confirmation_gate_passed": passes,
        },
    )
    print(
        json.dumps({"selected": selected, "gains": gains[selected], "gate": passes}),
        flush=True,
    )


if __name__ == "__main__":
    main()
