"""Recover V5A's pre-evaluation CLI failure without changing its frozen study."""

import argparse
import json
from pathlib import Path

from scripts.export_public_stock_adaptation import (
    audit_training,
    checked_endpoint,
    reduce_study,
)
from scripts.export_public_stock_results import read, sha, write
from scripts.public_stock_adaptation import EVALUATE, verify_protocol
from scripts.run_public_stock_robustness import run


def evaluation_command(options):
    command = ["scripts.evaluate_public_stock_residual", "evaluate"]
    for key, value in options.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base = Path(args.study)
    target = Path(args.out)
    recovery_path = base / "qualification_recovery.json"
    if target.exists() or recovery_path.exists() or (base / "finished.json").exists():
        raise FileExistsError(
            "Recovery or completed evidence exists; inspect before retry"
        )
    if any(p.is_dir() for p in base.glob("qualification_*")):
        raise ValueError(
            "Qualification already consumed streams; this recovery is pre-evaluation only"
        )
    failed = base / "qualification_preview0_choice0_stakes0_parent0_new_own.log"
    failure = failed.read_text(encoding="utf-8")
    if "invalid choice" not in failure or "choose from evaluate" not in failure:
        raise ValueError("Expected preserved argparse failure before evaluation")
    study = read(base / "protocol.json")
    verify_protocol(study)
    audits = {
        label: audit_training(base, study, label, {}) for label in study["systems"]
    }
    if read(base / "training_audit.json") != {
        "protocol_sha256": sha(base / "protocol.json"),
        "systems": audits,
    }:
        raise ValueError("Frozen training audit differs")
    bound = {
        str(path): sha(path)
        for path in (
            Path(__file__),
            Path("scripts/run_public_stock_robustness.py"),
            base / "protocol.json",
            base / "training_audit.json",
            failed,
        )
    }
    recovery = {
        "reason": "Original runner omitted required evaluate subcommand; argparse exited before creating any qualification endpoint.",
        "scope": "All six final models and training audits retained unchanged; zero qualification endpoints existed at recovery start. Original V5A scientific protocol, cohorts, evaluator, comparisons and exporter remain unchanged.",
        "bindings": bound,
        "command_prefix": ["scripts.evaluate_public_stock_residual", "evaluate"],
    }
    write(recovery_path, recovery)
    frozen_recovery = sha(recovery_path)
    for label, row in study["systems"].items():
        other = study["systems"][f"{row['profile']}_parent{1 - row['parent']}"][
            "checkpoint"
        ]
        new = audits[label]["checkpoint"]
        for kind, checkpoint, opponent in (
            ("new_own", new, row["checkpoint"]),
            ("new_other", new, other),
            ("old_other", row["checkpoint"], other),
        ):
            verify_protocol(study)
            if (
                sha(recovery_path) != frozen_recovery
                or any(sha(p) != h for p, h in bound.items())
                or any(
                    sha(a["checkpoint"]) != a["checkpoint_sha256"]
                    for a in audits.values()
                )
            ):
                raise ValueError(
                    "Recovery inputs or qualified model candidates changed"
                )
            name = f"qualification_{label}_{kind}"
            folder = base / name
            run(
                evaluation_command(
                    {
                        **EVALUATE,
                        "checkpoint": checkpoint,
                        "opponent": opponent,
                        "out": str(folder),
                    }
                ),
                base / f"recovered_{name}.log",
            )
            checked_endpoint(folder, study, checkpoint, opponent, row["profile"], {})
            print(json.dumps({"completed": name}), flush=True)
    result = reduce_study(base)
    if sha(recovery_path) != frozen_recovery or any(
        sha(p) != h for p, h in bound.items()
    ):
        raise ValueError("Recovery changed before independent export")
    result["bindings"].update(bound)
    result["bindings"][str(recovery_path)] = frozen_recovery
    result["recovery"] = recovery
    write(target, result)
    write(
        base / "finished.json",
        {"status": "complete", "evidence": str(target), "sha256": sha(target)},
    )
    print(
        json.dumps(
            {k: result[k] for k in ("qualified_rules", "eligible_search_finalists")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
