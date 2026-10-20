"""Complete-cohort comparison of explicit and certified parent-default programs."""

import argparse
import gzip
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.layout import POINTS
from koikoi.upstream import PublishedAgent
from scripts.belief_tail import KoiTailDynamics
from scripts.certified_tail_replay import check as check_source
from scripts.compare_symbolic_programs import compare_common
from scripts.diagnose_history_support import reference
from scripts.export_belief_tail import replay_actions
from scripts.export_prospective_incumbent import audit_arm
from scripts.prospective_incumbent import (
    compressed_save,
    contrasts,
    describe_selection,
    evaluate_arm,
)
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.symbolic_final_turn_tail import (
    RULE,
    SymbolicFinalTurnPlanner,
    SymbolicPlan,
)
from scripts.symbolic_tail_assessment import assess_three, verify_adoption


def freeze(base, source, receipt):
    if base.exists():
        raise ValueError("Use a fresh output directory")
    previous, _ = check_source(source)
    summary = read(source / "summary.json")
    checked = read(receipt)
    assert read(source / "queue.json")["status"] == "complete"
    assert summary["status"] == "passed"
    assert checked["status"] == "passed"
    assert summary["reused_sources"] == 512 and summary["new_source_matches"] == 0
    assert summary["protocol_sha256"] == sha(source / "protocol.json")
    assert checked["summary_sha256"] == sha(source / "summary.json")
    assert checked["chunk_bindings"] == summary["chunk_bindings"]
    bindings = {
        str(source / name): sha(source / name)
        for name in ("protocol.json", "summary.json", "queue.json")
    }
    bindings.update(summary["chunk_bindings"])
    bindings[str(receipt)] = sha(receipt)
    inputs = dict(previous["inputs"])
    inputs.update(
        {
            p: sha(p)
            for p in (
                "scripts/symbolic_tail_replay.py",
                "scripts/symbolic_final_turn_tail.py",
                "scripts/verify_symbolic_tail.py",
                "scripts/symbolic_tail_assessment.py",
                "scripts/compare_symbolic_programs.py",
                "scripts/export_certified_tail_replay.py",
                "scripts/export_symbolic_tail_replay.py",
                "tests/test_symbolic_tail_replay.py",
                "tests/test_export_symbolic_tail_replay.py",
                "tests/test_symbolic_final_turn_tail.py",
                "tests/test_symbolic_tail_assessment.py",
                "tests/test_compare_symbolic_programs.py",
                "docs/SYMBOLIC_TAIL_REPLAY.rst",
            )
        }
    )
    base.mkdir(parents=True)
    protocol = {
        "source": str(source),
        "bindings": bindings,
        "inputs": inputs,
        "chunks": list(range(32)),
        "workers": 4,
        "scope": "All512 saved sources; explicit opponent-only versus symbolic parent-default programs; no fresh strength test.",
    }
    write(base / "protocol.json", protocol)
    return protocol


def check(base):
    protocol = read(base / "protocol.json")
    for mapping in ("inputs", "bindings"):
        assert all(sha(p) == h for p, h in protocol[mapping].items())
    _, original = check_source(Path(protocol["source"]))
    return protocol, original


def load(path):
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def source_records(folder):
    complete = read(folder / "complete.json")
    assert complete["records_sha256"] == sha(folder / "records.json")
    return read(folder / "records.json")


def check_programs(phases):
    """Check rule identity even for roots absent from the previous version."""
    for phase in phases.values():
        plan = phase.result.plan
        if plan is not None and (
            not isinstance(plan, SymbolicPlan) or plan.leaf_rule != RULE
        ):
            raise ValueError("Expected the recognized symbolic program")


def worker(base, chunk):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    protocol, original = check(base)
    if chunk not in protocol["chunks"]:
        raise ValueError("Unknown chunk")
    source = Path(protocol["source"]) / f"chunk_{chunk:03}"
    old_complete = read(source / "complete.json")
    assert old_complete["records_sha256"] == sha(source / "records.json")
    folder = base / f"chunk_{chunk:03}"
    folder.mkdir()

    def stage(name, index=None):
        write(
            folder / "process.json",
            {"status": "running", "pid": os.getpid(), "stage": name, "source": index},
        )

    try:
        stage("setup")
        parents = [NeuralAgent(p, "cpu", deterministic=True) for p, _ in PARENTS]
        opponents = {
            name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")
        }
        records, artifacts = [], {}
        for old in source_records(source):
            index, src = old["index"], old["source"]
            row = {"index": index, "source": src, "status": old["status"]}
            if old["status"] != "evaluated":
                records.append(row)
                write(folder / "progress.json", records)
                continue
            row.update(arms={}, order=old["order"])
            selections = {}
            identity = src["selected_root"]["public_identification"]["model"]
            for arm in row["order"]:
                stage(arm, index)
                prior = old["arms"][arm]
                assert (
                    sha(source / prior["artifact"])
                    == old_complete["artifacts"][prior["artifact"]]
                )
                evidence = load(source / prior["artifact"])
                previous = audit_arm(evidence, prior, src, arm, original)
                worlds = evidence["worlds"]
                dynamics = KoiTailDynamics(
                    original["capacity"],
                    worlds.model_names,
                    {identity: opponents[identity]},
                    parents[src["parent"]],
                )
                phases, selection, timing = evaluate_arm(
                    worlds,
                    dynamics,
                    PARENTS[src["parent"]][1],
                    {identity: original["opponent_identities"][identity]},
                    arm,
                    original,
                    planner_type=SymbolicFinalTurnPlanner,
                )
                assert selection.binding == previous.binding
                if previous.parent_q is not None and selection.parent_q is not None:
                    assert selection.parent_q == previous.parent_q
                check_programs(phases)
                common = compare_common(evidence["phases"], phases)
                stage(f"{arm}_oracle", index)
                started = time.perf_counter()
                oracle = verify_adoption(
                    worlds,
                    phases,
                    selection,
                    parents[src["parent"]],
                    {identity: opponents[identity]},
                    dynamics.policy_cache,
                )
                record = {
                    "selection": describe_selection(selection),
                    "seconds": timing,
                    "common_program_structure": common,
                    "phases": {
                        name: {"work": p.result.work, "refusal": p.result.refusal}
                        for name, p in phases.items()
                    },
                    "parent_adapter_calls": dynamics.baseline_calls,
                    "opponent_adapter_calls": dynamics.policy_calls,
                    "oracle": oracle,
                    "oracle_seconds": time.perf_counter() - started,
                }
                path = folder / f"root_{index:04}_{arm}.pkl.gz"
                compressed_save(
                    path,
                    {
                        "worlds": worlds,
                        "phases": phases,
                        "selection": selection,
                        "opponent_cache": dynamics.policy_cache,
                    },
                )
                artifacts[path.name] = sha(path)
                record["artifact"] = path.name
                row["arms"][arm] = record
                selections[arm] = selection
                del evidence, dynamics, worlds, phases
            path = folder / f"selection_{index:04}.json"
            write(path, row["arms"])
            artifacts[path.name] = sha(path)
            stage("actual_tails", index)
            row["assessment"] = assess_three(
                src,
                selections,
                parents[src["parent"]],
                opponents[identity],
                PARENTS[src["parent"]][1],
            )
            records.append(row)
            write(folder / "progress.json", records)
        check(base)
        write(folder / "records.json", records)
        write(
            folder / "complete.json",
            {
                "status": "complete",
                "protocol_sha256": sha(base / "protocol.json"),
                "records_sha256": sha(folder / "records.json"),
                "artifacts": artifacts,
            },
        )
        write(folder / "process.json", {"status": "complete", "pid": os.getpid()})
    except BaseException as error:
        write(folder / "process.json", {"status": "failed", "error": repr(error)})
        raise


def launch(base):
    protocol, _ = check(base)
    if (base / "queue.json").exists():
        raise ValueError("Queue already exists; inspect the existing process")
    started = time.time()
    write(
        base / "queue.json",
        {"status": "running", "pid": os.getpid(), "started_unix": started},
    )

    def one(chunk):
        if shutil.disk_usage(base).free < 20 * 1024**3:
            raise RuntimeError("Less than20GiB free")
        with (base / f"worker_{chunk:03}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.symbolic_tail_replay",
                    "--base",
                    str(base),
                    "--worker",
                    str(chunk),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return chunk

    done = []
    try:
        with ThreadPoolExecutor(max_workers=protocol["workers"]) as pool:
            futures = [pool.submit(one, chunk) for chunk in protocol["chunks"]]
            try:
                for future in as_completed(futures):
                    done.append(future.result())
                    write(base / "progress.json", {"completed_chunks": sorted(done)})
                    print(json.dumps({"completed_chunks": len(done)}), flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        write(
            base / "queue.json",
            {"status": "complete", "seconds": time.time() - started},
        )
    except BaseException as error:
        write(
            base / "queue.json",
            {
                "status": "failed",
                "error": repr(error),
                "completed_chunks": sorted(done),
            },
        )
        raise


def audit_and_summarize(base):
    protocol, original = check(base)
    assert read(base / "queue.json")["status"] == "complete"
    rows, old_rows, bindings = [], [], {}
    costs = {
        version: {arm: Counter() for arm in ("controller", "root_only")}
        for version in ("opponent_only", "symbolic")
    }
    work = {
        name: Counter()
        for name in (
            "oracle_physical",
            "oracle_represented",
            "oracle_inference",
            "actual_inference",
            "common_program_structure",
            "actual_tails",
            "tail_replay",
            "export_tail_replay",
        )
    }
    for chunk in protocol["chunks"]:
        folder = base / f"chunk_{chunk:03}"
        complete = read(folder / "complete.json")
        assert (
            complete["status"] == read(folder / "process.json")["status"] == "complete"
        )
        assert complete["protocol_sha256"] == sha(base / "protocol.json")
        assert complete["records_sha256"] == sha(folder / "records.json")
        for name, digest in complete["artifacts"].items():
            assert sha(folder / name) == digest
        bindings[str(folder / "complete.json")] = sha(folder / "complete.json")
        old_folder = Path(protocol["source"]) / f"chunk_{chunk:03}"
        previous = source_records(old_folder)
        current = read(folder / "records.json")
        assert len(previous) == len(current) == original["chunk_size"]
        for old, row in zip(previous, current, strict=True):
            assert (row["index"], row["source"], row["status"]) == (
                old["index"],
                old["source"],
                old["status"],
            )
            if row["status"] == "evaluated":
                assert set(row["arms"]) == {"controller", "root_only"}
                assert read(folder / f"selection_{row['index']:04}.json") == row["arms"]
                selections = {}
                for arm, record in row["arms"].items():
                    evidence = load(folder / record["artifact"])
                    check_programs(evidence["phases"])
                    selections[arm] = audit_arm(
                        evidence, record, row["source"], arm, original
                    )
                    prior = old["arms"][arm]
                    old_complete = read(old_folder / "complete.json")
                    assert (
                        sha(old_folder / prior["artifact"])
                        == old_complete["artifacts"][prior["artifact"]]
                    )
                    old_evidence = load(old_folder / prior["artifact"])
                    assert record["common_program_structure"] == compare_common(
                        old_evidence["phases"], evidence["phases"]
                    )
                    work["common_program_structure"].update(
                        record["common_program_structure"]
                    )
                    del old_evidence, evidence
                    for version, entry in (("opponent_only", old), ("symbolic", row)):
                        data, total = entry["arms"][arm], costs[version][arm]
                        total["parent_valued"] += (
                            data["selection"]["parent_q"] is not None
                        )
                        total["adoptions"] += data["selection"]["action"] is not None
                        total["planning_seconds"] += sum(data["seconds"].values())
                        total["parent_adapter_calls"] += data["parent_adapter_calls"]
                        total["opponent_adapter_calls"] += data[
                            "opponent_adapter_calls"
                        ]
                        for phase in data["phases"].values():
                            total["capped_phases"] += phase["refusal"] is not None
                            for key, value in phase["work"].items():
                                if key != "maximum_frontier_rows":
                                    total[key] += value
                    for oracle in record["oracle"].values():
                        work["oracle_physical"].update(oracle["physical_work"])
                        work["oracle_represented"].update(oracle["represented_work"])
                        work["oracle_inference"]["new_parent_adapter_calls"] += oracle[
                            "new_parent_adapter_calls"
                        ]
                        work["oracle_inference"]["new_opponent_adapter_calls"] += (
                            oracle["new_opponent_adapter_calls"]
                        )
                assert (
                    selections["controller"].binding == selections["root_only"].binding
                )
                outcomes = row["assessment"]["outcomes"]
                assert set(outcomes) == {"parent", "controller", "root_only"}
                assert outcomes["parent"] == old["assessment"]["outcomes"]["parent"]
                for arm, selection in selections.items():
                    if selection.plan is None:
                        assert outcomes[arm] == outcomes["parent"]
                row_replay = Counter()
                for outcome in outcomes.values():
                    seat = row["source"]["seat"]
                    assert (
                        outcome["margin"]
                        == outcome["final"][POINTS + seat]
                        - outcome["final"][POINTS + 1 - seat]
                    )
                    replay = replay_actions(
                        reference(row["source"]["selected_root"]["state"]),
                        outcome["actions"],
                        outcome["final"],
                    )
                    assert replay == outcome["work"]
                    work["export_tail_replay"].update(replay)
                    row_replay.update(replay)
                assert dict(row_replay) == row["assessment"]["actual_work"]
                assert dict(row_replay) == row["assessment"]["replay_work"]
                work["actual_inference"].update(row["assessment"]["calls"])
                for name, key in (
                    ("actual_tails", "actual_work"),
                    ("tail_replay", "replay_work"),
                ):
                    work[name].update(row["assessment"][key])
            rows.append(row)
            old_rows.append(old)
    assert [r["index"] for r in rows] == list(range(512))
    for row, case in zip(rows, original["cases"], strict=True):
        assert {key: row["source"][key] for key in case} == case
    old_keys, old_actual, old_conditional = contrasts(old_rows)
    keys, actual, conditional = contrasts(rows)
    assert keys == old_keys and len(keys) == 128
    result = {
        "status": "passed",
        "protocol_sha256": sha(base / "protocol.json"),
        "chunk_bindings": bindings,
        "reused_sources": 512,
        "new_source_matches": 0,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "costs": {
            version: {arm: dict(c) for arm, c in arms.items()}
            for version, arms in costs.items()
        },
        "work": {name: dict(counts) for name, counts in work.items()},
        "contrast_order": read(Path(protocol["source"]) / "summary.json")[
            "contrast_order"
        ],
        "opponent_only_actual_mean": old_actual.mean(0).tolist(),
        "symbolic_actual_mean": actual.mean(0).tolist(),
        "opponent_only_conditional_mean": old_conditional.mean(0).tolist(),
        "symbolic_conditional_mean": conditional.mean(0).tolist(),
        "scope": protocol["scope"],
    }
    # Reused-cohort estimates are descriptive: no fresh strength gate or CI.
    assert np.isfinite(actual).all() and np.isfinite(conditional).all()
    write(base / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("runs/symbolic_tail_replay"))
    parser.add_argument(
        "--source", type=Path, default=Path("runs/certified_tail_replay")
    )
    parser.add_argument(
        "--receipt", type=Path, default=Path("runs/certified_tail_replay_export.json")
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--launch", action="store_true")
    group.add_argument("--worker", type=int)
    group.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze(args.base, args.source, args.receipt)
    elif args.launch:
        launch(args.base)
    elif args.worker is not None:
        worker(args.base, args.worker)
    else:
        audit_and_summarize(args.base)
