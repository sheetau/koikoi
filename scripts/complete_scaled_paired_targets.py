"""Retain completed scaled datasets and finish with bounded-resource replay.

The predecessor's in-memory code stays untouched. Wait for its terminal receipt,
then reuse its completed datasets instead of repeating expensive branch work.
"""

import argparse
import contextlib
import json
import shutil
import time
from pathlib import Path

from koikoi.train import provenance
from scripts.analyze_paired_arbitration_study import data_report
from scripts.collect_reusable_paired_targets import collect
from scripts.replay_reusable_paired_features import replay
from scripts.run_arbitration_study import sha, write
from scripts.run_scaled_paired_targets import make_plan


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def recovery_plan(predecessor):
    previous = read(predecessor / "protocol.json")
    plan = make_plan()
    for key in (
        "parents",
        "collection",
        "point_parameters_sha256",
        "point_model_hashes",
    ):
        if previous[key] != plan[key]:
            raise ValueError(f"Predecessor protocol mismatch: {key}")
    if previous["source"]["source_sha256"] != plan["source"]["source_sha256"]:
        raise ValueError("Predecessor production source mismatch")
    for path, digest in previous["inputs"].items():
        if sha(path) != digest:
            raise ValueError(f"Predecessor input changed: {path}")
    for path in (
        Path(__file__),
        Path("scripts/replay_reusable_paired_features.py"),
        predecessor / "protocol.json",
    ):
        plan["inputs"][str(path)] = sha(path)
    plan["predecessor"] = str(predecessor)
    plan["resource_revision"] = (
        "Retain completed predecessor datasets; use one branch workspace and one "
        "feature replay workspace. Reused branch work is not new simulation. "
        "Any predecessor failed replay work is additional unmeasured computation."
    )
    return plan


def run(base, predecessor):
    if base.exists():
        raise ValueError("Use a fresh recovery directory")
    plan = recovery_plan(predecessor)
    base.mkdir(parents=True)
    write(base / "protocol.json", plan)
    reports = {}
    try:
        write(base / "process.json", {"status": "waiting_for_predecessor"})
        while True:
            previous = read(predecessor / "process.json")
            if previous["status"] in ("complete", "failed"):
                break
            time.sleep(5)
        if previous[
            "status"
        ] == "failed" and "filename or extension is too long" not in previous.get(
            "error", ""
        ):
            raise ValueError(f"Unexpected predecessor failure: {previous}")
        write(base / "predecessor_terminal.json", previous)
        for index, parent in enumerate(plan["parents"], 1):
            if provenance()["source_sha256"] != plan["source"]["source_sha256"]:
                raise ValueError("Production changed")
            for path, digest in plan["inputs"].items():
                if sha(path) != digest:
                    raise ValueError(f"Recovery dependency changed: {path}")
            folder = base / "data" / f"parent{index}"
            folder.mkdir(parents=True)
            old_folder = predecessor / "data" / f"parent{index}"
            reused = (old_folder / "dataset_audit.json").exists()
            if reused:
                _, old_report = data_report(predecessor / "data", index, plan)
                if old_report != {
                    k: v
                    for k, v in read(old_folder / "dataset_audit.json").items()
                    if k != "status"
                }:
                    raise ValueError("Predecessor dataset receipt mismatch")
                shutil.copyfile(old_folder / "targets.pt", folder / "targets.pt")
                if sha(folder / "targets.pt") != old_report["dataset_sha256"]:
                    raise ValueError("Dataset copy mismatch")
            else:
                write(base / "process.json", {"status": "collecting", "parent": index})
                with (
                    (folder / "collect.log").open("w", encoding="utf-8") as log,
                    contextlib.redirect_stdout(log),
                ):
                    collect(
                        parent["path"],
                        folder / "targets.pt",
                        seed=plan["collection"]["seeds"][index - 1],
                        roots=plan["collection"]["roots_per_parent"],
                        batch=plan["collection"]["batch"],
                        device="cuda",
                    )
            data, report = data_report(base / "data", index, plan)
            point = data["experts"][1]["current"]
            if (
                point["parameters_sha256"] != plan["point_parameters_sha256"]
                or point["model_hashes"] != plan["point_model_hashes"]
            ):
                raise ValueError("Point expert changed")
            write(folder / "dataset_audit.json", {"status": "passed", **report})
            del data, point
            write(
                base / "process.json", {"status": "replaying_features", "parent": index}
            )
            cache_path = base / "features" / f"parent{index}.pt"
            cache = replay(folder / "targets.pt", cache_path)
            reports[str(index)] = {
                "dataset": report,
                "reused_predecessor_dataset": reused,
                "new_collection_source_work": None if reused else report["source_work"],
                "new_collection_branch_work": None if reused else report["branch_work"],
                "cache_sha256": sha(cache_path),
                "verified_replay_roots": cache["verified_roots"],
                "feature_replay_work": cache["new_source_replay_work"],
                "feature_replay_seconds": cache["elapsed_seconds"],
                "new_feature_branch_decisions": cache["new_branch_decisions"],
            }
            del cache
            write(base / "summary.json", reports)
            print(
                {"completed_parent": index, "report": reports[str(index)]}, flush=True
            )
        write(base / "process.json", {"status": "complete", "parents": 2})
    except Exception as error:
        write(base / "process.json", {"status": "failed", "error": repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("runs/scaled_paired_targets_recovered")
    )
    parser.add_argument(
        "--predecessor", type=Path, default=Path("runs/scaled_paired_targets_reusable")
    )
    args = parser.parse_args()
    run(args.base, args.predecessor)


if __name__ == "__main__":
    main()
