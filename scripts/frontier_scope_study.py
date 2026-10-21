"""Frozen fresh-source comparison of the selected hybrid against the old controller."""

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

from koikoi.env import BatchEnv
from koikoi.train import provenance
from scripts.check_selected_frontier_integration import checked as check_integration
from scripts.export_belief_tail import score
from scripts.frontier_retained_sources import (
    acquisition,
    audit_acquisition,
    collect_case,
)
from scripts.frontier_retained_sources import bindings as retained_bindings
from scripts.frontier_scope_assessment import assess_pair, audit_plan, plan_source
from scripts.frontier_scope_design import (
    cases,
    specification,
    summarize,
)
from scripts.logged_public_source import audit as audit_source
from scripts.prospective_incumbent import compressed_save
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.uniform_controller_design import POPULATIONS
from scripts.uniform_controller_study import (
    actors,
    chunk_records,
    load,
    opponent_identities,
    source_files,
)
from scripts.uniform_controller_study import dependencies as old_dependencies

INTEGRATION = Path("runs/selected_frontier_integration")
REVIEW = Path("runs/selected_frontier_integration_review.json")
PIPELINE = Path("runs/frontier_incremental_pipeline_tests.xml")
LEDGER = Path("runs/frontier_scope_tests.xml")


def prerequisites():
    import xml.etree.ElementTree as ET

    check_integration(INTEGRATION)
    review = read(REVIEW)
    if review["status"] != "passed" or any(
        sha(path) != digest for path, digest in review["bindings"].items()
    ):
        raise ValueError("Selected integration must be complete and verified")
    for report, minimum in ((PIPELINE, 12), (LEDGER, 12)):
        root = ET.parse(report).getroot()
        suites = list(root.iter("testsuite"))
        if (
            not suites
            or sum(int(s.get("tests", "0")) for s in suites) < minimum
            or any(
                int(s.get(k, "0"))
                for s in suites
                for k in ("failures", "errors", "skipped")
            )
        ):
            raise ValueError("Complete incremental pipeline and ledger tests required")
    return {
        **{str(p): sha(p) for p in (REVIEW, PIPELINE, LEDGER)},
        **retained_bindings(),
    }


def dependencies():
    inputs = old_dependencies()
    pending = [
        Path(__file__).relative_to(Path.cwd()),
        Path("tests/test_frontier_incremental_pipeline.py"),
        Path("tests/test_frontier_incremental_design.py"),
        Path("tests/test_frontier_incremental_ledger.py"),
        Path("scripts/export_frontier_scope.py"),
        Path("tests/test_frontier_scope.py"),
        Path("tests/test_frontier_scope_ledger.py"),
        Path("docs/FRONTIER_INCREMENTAL_SCOPE_AMENDMENT.rst"),
        Path("docs/FRONTIER_INCREMENTAL_EVALUATION.rst"),
    ]
    found = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        if path.suffix != ".py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            names = (
                [node.module]
                if isinstance(node, ast.ImportFrom) and not node.level
                else [n.name for n in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            for name in names:
                if name and name.split(".")[0] in ("koikoi", "scripts", "tests"):
                    target = Path(*name.split(".")).with_suffix(".py")
                    if target.exists():
                        pending.append(target)
    inputs.update({p.as_posix(): sha(p) for p in sorted(found)})
    return inputs


def freeze(base):
    if base.exists():
        raise FileExistsError(base)
    original = read("runs/prospective_incumbent/protocol.json")
    if (
        sha("runs/prospective_incumbent/protocol.json")
        != "700e11608e3ced60919ca7ef31da460785ff9391459ac83c7e8a16f4a1a0c170"
        or opponent_identities() != original["opponent_identities"]
    ):
        raise ValueError("Pinned opponent identity changed")
    protocol = {
        "design": specification(),
        "cases": cases(),
        "inputs": dependencies(),
        "runtime": provenance(),
        "opponent_identities": opponent_identities(),
        "prerequisites": prerequisites(),
        "created_unix": time.time(),
    }
    base.mkdir(parents=True)
    write(base / "protocol.json", protocol)
    return protocol


def check(base):
    p = read(base / "protocol.json")
    if (
        p["design"] != specification()
        or p["cases"] != cases()
        or p["inputs"] != dependencies()
        or p["prerequisites"] != prerequisites()
    ):
        raise ValueError("Frozen incremental study changed")
    if p["opponent_identities"] != opponent_identities() or sys.flags.optimize:
        raise ValueError("Pinned opponents or assertions changed")
    current = provenance()
    if any(current[k] != p["runtime"][k] for k in ("source_sha256", "torch", "numpy")):
        raise ValueError("Frozen runtime changed")
    return p


def oracle_work(report):
    result = {
        "prefix_physical": {},
        "prefix_represented": {},
        "local_physical": Counter(),
        "local_represented": Counter(),
    }
    if report is not None:
        for name in ("physical", "represented"):
            result["prefix_" + name] = report["prefix_" + name]
        for local in report["local_checks"]:
            for pair in local["oracle"].values():
                for name in ("physical", "represented"):
                    result["local_" + name].update(pair[name + "_work"])
    return {k: dict(v) for k, v in result.items()}


def audit_chunk(base, chunk):
    protocol = check(base)
    folder = base / f"chunk_{chunk:04}"
    if (folder / "audit.json").exists():
        raise FileExistsError(folder / "audit.json")
    rows = chunk_records(base, chunk, protocol)
    parents, opponents, predictors = actors()
    work = {}

    def add(name, counts):
        work.setdefault(name, Counter()).update(counts)

    gains, actual = [], []
    for row in rows:
        case, source = row["case"], row["source"]
        index, parent, opponent = (
            case["index"],
            parents[source["parent"]],
            opponents[source["opponent"]],
        )
        if row["acquisition"] != acquisition(case):
            raise ValueError("Source acquisition origin differs")
        source_files(folder / f"source_{index:05}", case, source, protocol["design"])
        audit_acquisition(folder / f"source_{index:05}", case)
        verified = audit_source(source, predictors, parent, opponent)
        for name, key in (
            ("source_replay", "replay_work"),
            ("actor_verification", "policy_verification_calls"),
            ("identification", "identification_calls"),
        ):
            add(name, verified[key])
        if verified["fresh_external_chance_draws"] != 0:
            raise ValueError("Audit used new source randomness")
        saved = {
            k: v for k, v in row.items() if k not in ("selection_sha256", "assessment")
        }
        path = folder / f"selection_{index:05}.json"
        if sha(path) != row["selection_sha256"] or read(path) != saved:
            raise ValueError("Selection-before-execution record differs")
        verification = {
            "exact_increment": "0",
            "reason": "missing_root",
            "oracle": None,
        }
        difference = Fraction(0)
        if row["status"] == "missing_root":
            if (
                source["selected_root"] is not None
                or "artifact" in row
                or "assessment" in row
                or row["increment_lower"] != "0"
            ):
                raise ValueError("Unjustified missing-root fallback")
        else:
            if source["selected_root"] is None:
                raise ValueError("Unexpected rootless evaluation")
            evidence = load(folder / row["artifact"])
            verification, counts = audit_plan(
                row, evidence, source, parent, opponents, protocol
            )
            add("history_reconstruction", counts)
            for name, counts in oracle_work(verification["oracle"]).items():
                add(name, counts)
            if row["status"] == "evaluated":
                result = assess_pair(
                    source,
                    evidence["hybrid"],
                    parent,
                    opponent,
                    PARENTS[source["parent"]][1],
                )
                if result != row["assessment"]:
                    raise ValueError("Saved old/hybrid actual execution differs")
                for name in ("actual_work", "replay_work", "calls"):
                    add("tails_" + name, result[name])
                difference = score(result["outcomes"]["hybrid"]["margin"]) - score(
                    result["outcomes"]["old"]["margin"]
                )
            elif "assessment" in row:
                raise ValueError("Unexpected refused-support execution")
        gains.append(
            {
                "case": case,
                "gain": verification["exact_increment"],
                "verification": verification,
            }
        )
        actual.append(str(difference))
    check(base)
    write(
        folder / "audit.json",
        {
            "status": "passed",
            "protocol_sha256": sha(base / "protocol.json"),
            "complete_sha256": sha(folder / "complete.json"),
            "gains": gains,
            "actual_differences": actual,
            "work": {k: dict(v) for k, v in work.items()},
            "fresh_external_chance_draws": 0,
        },
    )


def reduce_study(base):
    protocol = check(base)
    if read(base / "queue.json")["status"] != "complete":
        raise ValueError("Complete queue required")
    config = protocol["design"]
    values, actual = ({p: [] for p in POPULATIONS} for _ in range(2))
    bindings, seen, costs, statuses = {}, [], {}, Counter()

    def add(name, counts):
        costs.setdefault(name, Counter()).update(
            {k: v for k, v in counts.items() if "maximum" not in k and "peak" not in k}
        )

    for chunk in range(config["source_matches"] // config["chunk_size"]):
        folder = base / f"chunk_{chunk:04}"
        rows = chunk_records(base, chunk, protocol)
        audit = read(folder / "audit.json")
        if (
            audit["status"] != "passed"
            or audit["protocol_sha256"] != sha(base / "protocol.json")
            or audit["complete_sha256"] != sha(folder / "complete.json")
            or audit["fresh_external_chance_draws"] != 0
            or len(audit["gains"]) != len(rows)
            or len(audit["actual_differences"]) != len(rows)
        ):
            raise ValueError("Incomplete or mismatched chunk audit")
        for name in ("audit.json", "complete.json"):
            bindings[str(folder / name)] = sha(folder / name)
        for row, gain, difference in zip(
            rows, audit["gains"], audit["actual_differences"], strict=True
        ):
            case, source = row["case"], row["source"]
            exact = Fraction(gain["gain"])
            if (
                gain["case"] != case
                or gain["gain"] != gain["verification"]["exact_increment"]
                or not 0 <= Fraction(row["increment_lower"]) <= exact <= 1
            ):
                raise ValueError("Audited incremental value differs")
            oracle = gain["verification"]["oracle"]
            if (
                oracle is None
                and exact != 0
                or oracle is not None
                and oracle["exact_increment"] != str(exact)
            ):
                raise ValueError("Missing incremental decomposition")
            expected_actual = Fraction(0)
            if row["status"] == "evaluated":
                outcomes = row["assessment"]["outcomes"]
                expected_actual = score(outcomes["hybrid"]["margin"]) - score(
                    outcomes["old"]["margin"]
                )
                for name, counts in row["work"].items():
                    add("planning_" + name, counts)
                add(
                    "planning_inference",
                    {
                        "parent": row["parent_adapter_calls"],
                        "opponent": row["opponent_adapter_calls"],
                    },
                )
                for name in ("actual_work", "replay_work", "calls"):
                    add("tails_" + name, row["assessment"][name])
                statuses["old_" + row["old_selection"]["reason"]] += 1
                statuses["positive_increment"] += int(exact > 0)
            if Fraction(difference) != expected_actual:
                raise ValueError("Actual difference differs")
            seen.append(case)
            values[case["population"]].append(exact)
            actual[case["population"]].append(Fraction(difference))
            statuses[row["status"]] += 1
            add("sources", source["work"])
            origin = acquisition(case)
            if row["acquisition"] != origin:
                raise ValueError("Source acquisition accounting differs")
            statuses[origin] += 1
            add(origin + "_source_execution", source["work"])
            add(origin + "_source_native", source["native_work"])
            add(
                origin + "_sampling",
                {**source["sampling_work"], "seat_bounded_draws": 1},
            )
            add(origin + "_source_identification", source["identification_calls"])
            add("history_support", row.get("history_counts", {}))
        for name, counts in audit["work"].items():
            add("audit_" + name, counts)
    if (
        seen != protocol["cases"]
        or costs["sources"]["completions"] != config["source_matches"]
    ):
        raise ValueError("Complete independent source denominator differs")
    result = {
        "audit_status": "passed",
        "protocol_sha256": sha(base / "protocol.json"),
        "bindings": bindings,
        "sources": len(seen),
        "status_counts": dict(statuses),
        "inference": summarize(values, expected_n=config["sources_per_population"]),
        "actual_mean_difference_descriptive": {
            k: str(sum(v, Fraction(0)) / len(v)) for k, v in actual.items()
        },
        "work": {k: dict(v) for k, v in costs.items()},
    }
    return result


def summarize_study(base):
    if (base / "summary.json").exists():
        raise ValueError("Summary already exists")
    result = reduce_study(base)
    write(base / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("freeze", "run", "summarize"):
        mode.add_argument("--" + name, action="store_true")
    mode.add_argument("--worker", type=int)
    mode.add_argument("--audit-chunk", type=int)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.base)
    elif args.run:
        run_queue(args.base)
        summarize_study(args.base)
    elif args.summarize:
        summarize_study(args.base)
    elif args.worker is not None:
        worker(args.base, args.worker)
    else:
        audit_chunk(args.base, args.audit_chunk)


def worker(base, chunk):
    protocol = check(base)
    config = protocol["design"]
    selected = protocol["cases"][
        chunk * config["chunk_size"] : (chunk + 1) * config["chunk_size"]
    ]
    if len(selected) != config["chunk_size"] or chunk < 0:
        raise ValueError("Invalid complete chunk")
    folder = base / f"chunk_{chunk:04}"
    folder.mkdir()
    parents, opponents, predictors = actors()
    native, records = BatchEnv(1, debug=True), []
    try:
        for case in selected:
            index = case["index"]
            write(
                folder / "process.json",
                {
                    "status": "running",
                    "pid": os.getpid(),
                    "source": index,
                    "stage": "source",
                },
            )
            parent_index, opponent = case["population"].split(":")
            source = collect_case(
                folder / f"source_{index:05}",
                case,
                config,
                parents[int(parent_index)],
                opponents[opponent],
                predictors,
                native,
            )
            write(
                folder / "process.json",
                {
                    "status": "running",
                    "pid": os.getpid(),
                    "source": index,
                    "stage": "planning",
                },
            )
            row, evidence = plan_source(source, parents, opponents, protocol)
            row.update(case=case, source=source, acquisition=acquisition(case))
            if evidence is not None:
                row["artifact"] = f"program_{index:05}.pkl.gz"
                compressed_save(folder / row["artifact"], evidence)
            # Freeze selection and source before any actual conditional execution.
            selection_path = folder / f"selection_{index:05}.json"
            write(selection_path, row)
            row["selection_sha256"] = sha(selection_path)
            if row["status"] == "evaluated":
                write(
                    folder / "process.json",
                    {
                        "status": "running",
                        "pid": os.getpid(),
                        "source": index,
                        "stage": "actual_tails",
                    },
                )
                row["assessment"] = assess_pair(
                    source,
                    evidence["hybrid"],
                    parents[int(parent_index)],
                    opponents[opponent],
                    PARENTS[int(parent_index)][1],
                )
            records.append(row)
            write(folder / "progress.json", {"completed_sources": len(records)})
        check(base)
        write(folder / "records.json", records)
        artifacts = {
            p.relative_to(folder).as_posix(): sha(p)
            for p in folder.rglob("*")
            if p.is_file()
            and p.name not in ("process.json", "progress.json", "records.json")
        }
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
        write(
            folder / "process.json",
            {"status": "failed", "pid": os.getpid(), "error": repr(error)},
        )
        raise


def run_queue(base):
    protocol = check(base)
    if (base / "queue.json").exists():
        raise ValueError("Queue already started; no automatic restart")
    config = protocol["design"]
    write(
        base / "queue.json",
        {"status": "running", "pid": os.getpid(), "started_unix": time.time()},
    )

    def one(chunk):
        if shutil.disk_usage(base).free < 20 * 1024**3:
            raise RuntimeError("Less than20GiB free; study stopped")
        for mode in ("worker", "audit-chunk"):
            with (base / f"{mode}_{chunk:04}.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.frontier_scope_study",
                        "--base",
                        str(base),
                        f"--{mode}",
                        str(chunk),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        return chunk

    completed, started = [], time.time()
    try:
        with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
            futures = [
                pool.submit(one, c)
                for c in range(config["source_matches"] // config["chunk_size"])
            ]
            try:
                for future in as_completed(futures):
                    completed.append(future.result())
                    write(base / "progress.json", {"audited_chunks": sorted(completed)})
                    print(json.dumps({"audited_chunks": len(completed)}), flush=True)
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
                "audited_chunks": sorted(completed),
            },
        )
        raise


if __name__ == "__main__":
    main()
