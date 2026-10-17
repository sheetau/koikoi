"""Fourfold paired-data collection with fixed parents and fresh source streams."""

import argparse
import contextlib
from pathlib import Path

from koikoi.published_actor import PublishedActor
from koikoi.train import provenance
from scripts.analyze_paired_arbitration_study import data_report
from scripts.cache_paired_expert_features import replay
from scripts.run_arbitration_study import PARENTS, sha, write
from scripts.train_paired_arbitration import collect


def make_plan():
    paths = [
        Path(__file__),
        Path("scripts/train_paired_arbitration.py"),
        Path("scripts/paired_rollout_targets.py"),
        Path("scripts/cache_paired_expert_features.py"),
        Path("scripts/analyze_paired_arbitration_study.py"),
        Path("scripts/analyze_arbitration_study.py"),
        Path("scripts/run_arbitration_study.py"),
        Path("runs/paired_arbitration_study/protocol.json"),
    ]
    parents = []
    for path, digest in PARENTS:
        if sha(path) != digest:
            raise ValueError("Parent changed")
        paths.append(Path(path))
        parents.append({"path": path, "sha256": digest})
    point = PublishedActor.from_published("point", "cpu").checkpoint()
    return {
        "source": provenance(),
        "parents": parents,
        "inputs": {str(path): sha(path) for path in paths},
        "point_parameters_sha256": point["current"]["parameters_sha256"],
        "point_model_hashes": point["current"]["model_hashes"],
        "collection": {
            "roots_per_parent": 262144,
            "seeds": [19610001, 19620001],
            "batch": 256,
            "envs": 128,
            "device": "cuda",
            "behavior_parent_probability": 0.9,
            "continuation": "greedy parent both seats",
        },
        "scope": "Fixed fourfold target-data scale experiment. Fresh source streams; same parents/behavior/coupled frozen continuation as the smaller study. Count collection, branch and feature-replay work separately. No fitting, match evaluation or automatic model selection.",
        "rationale": "Small-data paired and marginal targets have not established replicated one-action gains. Only about9.6% of paired roots change win outcome. Test whether larger source coverage provides a more learnable signal; improvement is not assumed.",
    }


def run(base):
    if base.exists():
        raise ValueError("Use a fresh scaled-data directory")
    plan = make_plan()
    base.mkdir(parents=True)
    write(base / "protocol.json", plan)
    reports = {}
    try:
        for index, parent in enumerate(plan["parents"], 1):
            if provenance()["source_sha256"] != plan["source"]["source_sha256"]:
                raise ValueError("Production changed")
            for path, digest in plan["inputs"].items():
                if sha(path) != digest:
                    raise ValueError(f"Collection dependency changed: {path}")
            folder = base / "data" / f"parent{index}"
            folder.mkdir(parents=True)
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
                raise ValueError("Point expert changed during collection")
            write(folder / "dataset_audit.json", {"status": "passed", **report})
            del data, point
            write(
                base / "process.json", {"status": "replaying_features", "parent": index}
            )
            cache_path = base / "features" / f"parent{index}.pt"
            cache = replay(folder / "targets.pt", cache_path)
            reports[str(index)] = {
                "dataset": report,
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
    parser.add_argument("--base", type=Path, default=Path("runs/scaled_paired_targets"))
    args = parser.parse_args()
    run(args.base)


if __name__ == "__main__":
    main()
