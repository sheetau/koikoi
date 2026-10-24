"""Fixed completed-round rule screen with audited reusable continuation tables."""

import argparse
import json
from pathlib import Path

import numpy as np

from koikoi.public_stock import PROFILES
from scripts.evaluate_public_stock_residual import sha, write
from scripts.run_public_stock_residual import CONTROL, run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trained", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    trained, out = Path(args.trained), Path(args.out)
    quality = json.loads((trained / "qualification.json").read_text(encoding="utf-8"))
    if not all(row["passed"] for row in quality):
        raise ValueError("Adapted control policies did not qualify")
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        "trained": str(trained),
        "training_protocol_sha256": sha(trained / "protocol.json"),
        "profiles": list(PROFILES),
        "table_samples": 512,
        "table_seed": 900000000,
        "development_seed": 930000000,
        "development_pairs": 32,
        "particles": [0, 8, 32, 128],
        "horizon": 80,
        "confidence": 2.5,
        "search_seed": 971,
        "confirmation_seed": 932000000,
        "confirmation_pairs": 128,
        "confirmation_particles": [0, 8, 128],
        "bootstrap_seed": 981,
        "checkpoints": {
            f"{profile}_parent{parent}": sha(
                trained / f"{profile}_parent{parent}" / "policy_00128.pt"
            )
            for profile in PROFILES
            for parent in range(2)
        },
        "sources": {
            path: sha(path)
            for path in (
                "scripts/run_public_stock_rounds.py",
                "scripts/build_auditable_round_values.py",
                "scripts/audit_public_round_values.py",
                "scripts/evaluate_public_stock_rounds.py",
                "koikoi/public_stock_round_value.py",
                "koikoi/public_stock_residual.py",
                "docs/PUBLIC_STOCK_STUDY_V3.rst",
            )
        },
    }
    write(out / "protocol.json", protocol)
    curves = {}
    for index, profile in enumerate(PROFILES):
        curves[profile] = []
        for parent in range(2):
            label = f"{profile}_parent{parent}"
            checkpoint = trained / label / "policy_00128.pt"
            table = out / f"table_{label}"
            run(
                [
                    "scripts.build_auditable_round_values",
                    "--checkpoint",
                    str(checkpoint),
                    "--out",
                    str(table),
                    "--samples",
                    "512",
                    "--seed",
                    str(900000000 + 2000000 * index + 1000000 * parent),
                ],
                out / f"build_{label}.log",
            )
            run(
                ["scripts.audit_public_round_values", "--table-dir", str(table)],
                out / f"audit_{label}.log",
            )
            print(json.dumps({"audited_table": label}), flush=True)
            values = []
            for particles in protocol["particles"]:
                dest = out / f"screen_{label}_p{particles}"
                run(
                    [
                        "scripts.evaluate_public_stock_rounds",
                        "evaluate",
                        "--checkpoint",
                        str(checkpoint),
                        "--table",
                        str(table / "table.npz"),
                        "--out",
                        str(dest),
                        "--deal-seed",
                        "930000000",
                        "--pairs",
                        "32",
                        "--batch",
                        "32",
                        "--particles",
                        str(particles),
                        "--horizon",
                        "80",
                        "--confidence",
                        "2.5",
                        "--search-seed",
                        "971",
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
        (profile for profile in PROFILES if profile != CONTROL),
        key=lambda profile: (-min(gains[profile]), profile),
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
        json.dumps({"selected": selected, "gate": passes, "gains": gains[selected]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
