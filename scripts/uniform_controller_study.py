"""Fixed, independently sampled match-prefix evaluation of the symbolic controller."""

import argparse
import ast
import gzip
import hashlib
import json
import os
import pickle
import secrets
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.env import BatchEnv
from koikoi.train import provenance
from koikoi.upstream import UPSTREAM_PATH, PublishedAgent
from scripts.belief_tail import KoiTailDynamics, visible_keys
from scripts.deferred_belief_tail import deferred_worlds
from scripts.diagnose_history_support import reference
from scripts.export_belief_tail import score
from scripts.history_conditioned_worlds import enumerate_support
from scripts.incumbent_tail_controller import bind_worlds, select_improvement
from scripts.logged_public_source import audit as audit_source
from scripts.logged_public_source import collect
from scripts.prospective_incumbent import (
    compressed_save,
    dependency_files,
    describe_selection,
    evaluate_arm,
)
from scripts.revealed_opponent_models import StrictPublishedPrediction
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.symbolic_final_turn_tail import (
    RULE,
    SymbolicFinalTurnPlanner,
    SymbolicPlan,
)
from scripts.uniform_controller_assessment import assess_pair, verify_pair
from scripts.uniform_controller_design import (
    POPULATIONS,
    cases,
    specification,
    summarize,
)
from scripts.uniform_deal_source import LoggedDealSource, sample_valid_deck


def dependencies():
    inputs = dependency_files()
    pending = [Path(__file__).relative_to(Path.cwd())] + [
        Path(p)
        for p in (
            "docs/UNIFORM_CONTROLLER_EVALUATION.rst",
            "docs/LOGGED_POSTERIOR_COMPOSITION.rst",
            "tests/test_uniform_controller_study.py",
            "tests/test_uniform_controller_neural_pipeline.py",
            "tests/test_uniform_controller_assessment.py",
            "tests/test_uniform_controller_design.py",
            "tests/test_logged_source_posterior.py",
            "tests/test_logged_public_source.py",
            "tests/test_uniform_deal_source.py",
            "tests/test_certified_gain_bounds.py",
        )
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
                else (
                    [n.name for n in node.names] if isinstance(node, ast.Import) else []
                )
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
        raise ValueError("Use a fresh study directory")
    # Reuse only the pinned opponent identities, never its source records/law.
    original = read("runs/prospective_incumbent/protocol.json")
    if (
        sha("runs/prospective_incumbent/protocol.json")
        != "700e11608e3ced60919ca7ef31da460785ff9391459ac83c7e8a16f4a1a0c170"
    ):
        raise ValueError("Unexpected opponent identity provenance")
    if opponent_identities() != original["opponent_identities"]:
        raise ValueError("Pinned opponent implementation or checkpoints changed")
    protocol = {
        "design": specification(),
        "cases": cases(),
        "inputs": dependencies(),
        "runtime": provenance(),
        "opponent_identities": original["opponent_identities"],
        "created_unix": time.time(),
    }
    base.mkdir(parents=True)
    write(base / "protocol.json", protocol)
    return protocol


def check(base):
    p = read(base / "protocol.json")
    if p["design"] != specification() or p["cases"] != cases():
        raise ValueError("Protocol differs from the fixed design")
    if p["opponent_identities"] != opponent_identities() or sys.flags.optimize:
        raise ValueError("Opponent identity mismatch or disabled runtime assertions")
    if p["inputs"] != dependencies():
        raise ValueError("Frozen dependency changed")
    current = provenance()
    for key in ("source_sha256", "torch", "numpy"):
        if current[key] != p["runtime"][key]:
            raise ValueError("Source/runtime differs from the frozen protocol")
    return p


def opponent_identities():
    return {
        name: hashlib.sha256(
            json.dumps(
                {
                    **{
                        phase: sha(UPSTREAM_PATH / f"model_agent/{phase}_rl_{name}.pt")
                        for phase in ("discard", "pick", "koikoi")
                    },
                    "adapter": sha("koikoi/upstream.py"),
                    "guard": sha("koikoi/agents.py"),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        for name in ("wp", "point")
    }


def actors():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    parents = [NeuralAgent(path, "cpu", deterministic=True) for path, _ in PARENTS]
    if not all(p.requires_endgame for p in parents):
        raise ValueError("Expected the pinned assisted parents")
    published = {name: PublishedAgent(name) for name in ("wp", "point")}
    return (
        parents,
        {k: EndgameAgent(v) for k, v in published.items()},
        {k: StrictPublishedPrediction(v) for k, v in published.items()},
    )


def collect_case(
    folder,
    case,
    config,
    parent,
    opponent,
    predictors,
    native,
    *,
    seat_draw=None,
    deck_draw=None,
):
    """Persist the seat and every accepted deck BEFORE using it in the match."""
    folder.mkdir()  # Refuse resampling/replacing an existing source, even partial.
    seat = (secrets.randbelow if seat_draw is None else seat_draw)(2)
    if type(seat) is not int or seat not in (0, 1):
        raise ValueError("Invalid sampled seat")
    write(folder / "seat.json", {"case": case, "seat": seat, "bounded_draws": 1})
    position = 0

    def draw():
        nonlocal position
        sample = (sample_valid_deck if deck_draw is None else deck_draw)()
        write(folder / f"deal_{position:02}.json", asdict(sample))
        position += 1
        return sample

    source = LoggedDealSource(config["environment_seed"], config["profile"], draw)
    record = collect(
        source,
        f"source-{case['index']:08}",
        seat,
        parent,
        opponent,
        predictors,
        native,
        max_draws=config["max_draws"],
    )
    record.update(
        parent=int(case["population"].split(":")[0]),
        opponent=case["population"].split(":")[1],
    )
    write(folder / "record.json", record)
    return record


def source_files(folder, case, source, config):
    seat = read(folder / "seat.json")
    if seat != {"case": case, "seat": source["seat"], "bounded_draws": 1}:
        raise ValueError("Seat ledger mismatch")
    if source != read(folder / "record.json"):
        raise ValueError("Saved source mismatch")
    if (
        source["source_key"] != f"source-{case['index']:08}"
        or source["seed"] != config["environment_seed"]
        or source["profile"] != config["profile"]
        or source["max_draws"] != config["max_draws"]
        or f"{source['parent']}:{source['opponent']}" != case["population"]
    ):
        raise ValueError("Source differs from the declared case/law")
    expected = {"seat.json", "record.json"}
    for i, deal in enumerate(source["deals"]):
        name = f"deal_{i:02}.json"
        expected.add(name)
        if read(folder / name) != {
            k: deal[k] for k in ("deck", "attempts", "bounded_draws")
        }:
            raise ValueError("Accepted-deck ledger mismatch")
    if {p.name for p in folder.iterdir()} != expected:
        raise ValueError("Extra or missing source ledger files")


def build_support(source, opponents, config):
    root = reference(source["selected_root"]["state"])
    identity = source["selected_root"]["public_identification"]["model"]
    if identity != source["opponent"]:
        raise ValueError("Public inference disagrees with the declared actor")
    support = enumerate_support(
        root, {identity: opponents[identity]}, max_hands=config["max_hands"]
    )
    worlds = (
        None
        if support.status == "too_many_hands"
        else deferred_worlds(
            root, support, max_draws=config["max_draws"], max_hands=config["max_hands"]
        )
    )
    return support, worlds, identity


def plan_source(source, parents, opponents, protocol):
    config = protocol["design"]
    row = {"status": "missing_root"}
    if source["selected_root"] is None:
        return row, None
    started = time.perf_counter()
    support, worlds, identity = build_support(source, opponents, config)
    row.update(
        history_counts=support.counts, history_seconds=time.perf_counter() - started
    )
    if worlds is None:
        row["status"] = "support_refused"
        return row, {"support": asdict(support)}
    parent = parents[source["parent"]]
    dyn = KoiTailDynamics(
        config["capacity"], worlds.model_names, {identity: opponents[identity]}, parent
    )
    phases, selection, timing = evaluate_arm(
        worlds,
        dyn,
        PARENTS[source["parent"]][1],
        {identity: protocol["opponent_identities"][identity]},
        "controller",
        config,
        planner_type=SymbolicFinalTurnPlanner,
    )
    started = time.perf_counter()
    oracle = verify_pair(worlds, selection, parent, {identity: opponents[identity]})
    row.update(
        status="evaluated",
        selection=describe_selection(selection),
        seconds=timing,
        phases={
            k: {"work": p.result.work, "refusal": p.result.refusal}
            for k, p in phases.items()
        },
        parent_adapter_calls=dyn.baseline_calls,
        opponent_adapter_calls=dyn.policy_calls,
        oracle=oracle,
        oracle_seconds=time.perf_counter() - started,
    )
    return row, {
        "support": asdict(support),
        "worlds": worlds,
        "phases": phases,
        "selection": selection,
    }


def load(path):
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def audit_plan(row, evidence, source, opponents, protocol):
    """Rebuild the posterior independently and validate saved selection/certificates."""
    config = protocol["design"]
    support, worlds, identity = build_support(source, opponents, config)
    if (
        asdict(support) != evidence["support"]
        or support.counts != row["history_counts"]
    ):
        raise ValueError("Reconstructed hand support differs")
    if row["status"] == "support_refused":
        if worlds is not None or set(evidence) != {"support"}:
            raise ValueError("Unjustified support fallback")
        return None, support.counts
    if worlds is None or row["status"] != "evaluated":
        raise ValueError("Unexpected planning status")
    stored = evidence["worlds"]
    for name in ("states", "models", "weights"):
        np.testing.assert_array_equal(getattr(worlds, name), getattr(stored, name))
    if (worlds.model_names, worlds.viewer, worlds.draws) != (
        stored.model_names,
        stored.viewer,
        stored.draws,
    ):
        raise ValueError("Belief metadata differs")
    binding = bind_worlds(
        worlds,
        SimpleNamespace(keys=visible_keys, legal=KoiTailDynamics.legal),
        PARENTS[source["parent"]][1],
        {identity: protocol["opponent_identities"][identity]},
    )
    phases = evidence["phases"]
    if set(phases) not in ({"fixed"}, {"fixed", "optimized"}):
        raise ValueError("Unexpected controller phases")
    fixed_plan = phases["fixed"].result.plan
    optimize = fixed_plan is not None and fixed_plan.values[fixed_plan.action] < 1
    if ("optimized" in phases) != optimize:
        raise ValueError("Controller phase schedule differs")
    for name, phase in phases.items():
        if phase.binding != binding or phase.mode != name:
            raise ValueError("Saved phase binding differs")
        if row["phases"][name] != {
            "work": phase.result.work,
            "refusal": phase.result.refusal,
        }:
            raise ValueError("Saved phase work differs")
        if phase.result.work["decisions"] > config["phase_decisions"]:
            raise ValueError("Phase exceeded transition cap")
        if (
            phase.result.work.get("live_observation_nodes", 0)
            > config["phase_nodes"] + 1
        ):
            raise ValueError("Phase exceeded node cap including refusal probe")
        plan = phase.result.plan
        if plan is not None and (
            not isinstance(plan, SymbolicPlan) or plan.leaf_rule != RULE
        ):
            raise ValueError("Unknown executable leaf rule")
    selection = select_improvement(phases["fixed"], phases.get("optimized"))
    if (
        selection != evidence["selection"]
        or describe_selection(selection) != row["selection"]
    ):
        raise ValueError("Saved selection differs")
    if selection.plan is None:
        if row["oracle"]:
            raise ValueError("Unexpected fallback oracle")
    else:
        if set(row["oracle"]) != {"parent", "selected"}:
            raise ValueError("Missing independent value verification")
        for name, q in (
            ("parent", selection.parent_q),
            ("selected", selection.selected_q),
        ):
            checked = row["oracle"][name]
            if checked["value"] != str(q) or checked["represented_work"][
                "completions"
            ] != sum(map(int, worlds.weights)):
                raise ValueError("Independent value receipt differs")
    return selection, support.counts


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
                    "stage": "planning_and_reference_values",
                },
            )
            row, evidence = plan_source(source, parents, opponents, protocol)
            row.update(case=case, source=source)
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
                    evidence["selection"],
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


def chunk_records(base, chunk, protocol):
    folder = base / f"chunk_{chunk:04}"
    complete = read(folder / "complete.json")
    if read(folder / "process.json")["status"] != "complete":
        raise ValueError("Chunk process did not complete")
    present = {
        p.relative_to(folder).as_posix()
        for p in folder.rglob("*")
        if p.is_file()
        and p.name
        not in (
            "process.json",
            "progress.json",
            "records.json",
            "complete.json",
            "audit.json",
        )
    }
    if present != set(complete["artifacts"]):
        raise ValueError("Incomplete artifact binding set")
    if (
        complete["status"] != "complete"
        or complete["protocol_sha256"] != sha(base / "protocol.json")
        or complete["records_sha256"] != sha(folder / "records.json")
    ):
        raise ValueError("Incomplete or changed chunk")
    for path, digest in complete["artifacts"].items():
        if sha(folder / path) != digest:
            raise ValueError("Changed chunk artifact")
    rows = read(folder / "records.json")
    size = protocol["design"]["chunk_size"]
    if [r["case"] for r in rows] != protocol["cases"][
        chunk * size : (chunk + 1) * size
    ]:
        raise ValueError("Missing, reordered or duplicated source cases")
    return rows


def audit_chunk(base, chunk):
    protocol = check(base)
    folder = base / f"chunk_{chunk:04}"
    if (folder / "audit.json").exists():
        raise ValueError("Audit already exists")
    rows = chunk_records(base, chunk, protocol)
    parents, opponents, predictors = actors()
    work = {
        name: Counter()
        for name in (
            "source_replay",
            "actor_verification",
            "identification",
            "history_reconstruction",
            "tail_execution",
            "tail_replay",
            "tail_inference",
        )
    }
    gains, actual = [], []
    for row in rows:
        case, source = row["case"], row["source"]
        index, parent, opponent = (
            case["index"],
            parents[source["parent"]],
            opponents[source["opponent"]],
        )
        source_files(folder / f"source_{index:05}", case, source, protocol["design"])
        verified = audit_source(source, predictors, parent, opponent)
        work["source_replay"].update(verified["replay_work"])
        work["actor_verification"].update(verified["policy_verification_calls"])
        work["identification"].update(verified["identification_calls"])
        assert verified["fresh_external_chance_draws"] == 0
        saved = dict(row)
        del saved["selection_sha256"]
        saved.pop("assessment", None)
        path = folder / f"selection_{index:05}.json"
        if sha(path) != row["selection_sha256"] or read(path) != saved:
            raise ValueError("Selection-before-execution record differs")
        gain, difference = Fraction(0), Fraction(0)
        if row["status"] == "missing_root":
            if (
                source["selected_root"] is not None
                or "artifact" in row
                or "assessment" in row
            ):
                raise ValueError("Unjustified missing-root fallback")
        else:
            if source["selected_root"] is None:
                raise ValueError("Unexpected rootless planning record")
            selection, counts = audit_plan(
                row, load(folder / row["artifact"]), source, opponents, protocol
            )
            work["history_reconstruction"].update(counts)
            if selection is not None:
                result = assess_pair(
                    source, selection, parent, opponent, PARENTS[source["parent"]][1]
                )
                if result != row["assessment"]:
                    raise ValueError("Saved executable tail assessment differs")
                work["tail_execution"].update(result["actual_work"])
                work["tail_replay"].update(result["replay_work"])
                work["tail_inference"].update(result["calls"])
                gain = selection.gain or Fraction(0)
                difference = score(result["outcomes"]["controller"]["margin"]) - score(
                    result["outcomes"]["parent"]["margin"]
                )
            elif "assessment" in row:
                raise ValueError("Unexpected execution after support refusal")
        gains.append({"case": case, "gain": str(gain)})
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
                        "scripts.uniform_controller_study",
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


def summarize_study(base):
    protocol = check(base)
    if read(base / "queue.json")["status"] != "complete":
        raise ValueError("Complete queue required before inference")
    if (base / "summary.json").exists():
        raise ValueError("Summary already exists")
    config = protocol["design"]
    values = {name: [] for name in POPULATIONS}
    actual = {name: [] for name in POPULATIONS}
    bindings, seen, costs, statuses = {}, [], {}, Counter()

    def add(name, counters):
        costs.setdefault(name, Counter()).update(counters)

    for chunk in range(config["source_matches"] // config["chunk_size"]):
        folder = base / f"chunk_{chunk:04}"
        rows = chunk_records(base, chunk, protocol)
        checked = read(folder / "audit.json")
        if (
            checked["status"] != "passed"
            or checked["protocol_sha256"] != sha(base / "protocol.json")
            or checked["complete_sha256"] != sha(folder / "complete.json")
            or checked["fresh_external_chance_draws"] != 0
        ):
            raise ValueError("Missing or mismatched chunk audit")
        bindings[str(folder / "audit.json")] = sha(folder / "audit.json")
        bindings[str(folder / "complete.json")] = sha(folder / "complete.json")
        if len(checked["gains"]) != len(rows) or len(
            checked["actual_differences"]
        ) != len(rows):
            raise ValueError("Incomplete audited denominator")
        for row, gain, difference in zip(
            rows, checked["gains"], checked["actual_differences"], strict=True
        ):
            case, source = row["case"], row["source"]
            if gain["case"] != case:
                raise ValueError("Gain belongs to another source")
            expected = row["selection"]["gain"] if row["status"] == "evaluated" else "0"
            if gain["gain"] != expected:
                raise ValueError("Audited gain differs from saved selection")
            expected_actual = Fraction(0)
            if row["status"] == "evaluated":
                outcomes = row["assessment"]["outcomes"]
                expected_actual = score(outcomes["controller"]["margin"]) - score(
                    outcomes["parent"]["margin"]
                )
            if Fraction(difference) != expected_actual:
                raise ValueError(
                    "Audited actual difference differs from saved outcomes"
                )
            seen.append(case)
            values[case["population"]].append(Fraction(gain["gain"]))
            actual[case["population"]].append(Fraction(difference))
            statuses[row["status"]] += 1
            add("sources", source["work"])
            add("source_native", source["native_work"])
            add("sampling", {**source["sampling_work"], "seat_bounded_draws": 1})
            add("source_identification", source["identification_calls"])
            add("history_support", row.get("history_counts", {}))
            if row["status"] == "evaluated":
                add(
                    "planning_inference",
                    {
                        "parent": row["parent_adapter_calls"],
                        "opponent": row["opponent_adapter_calls"],
                    },
                )
                for phase in row["phases"].values():
                    add("planning", phase["work"])
                for oracle in row["oracle"].values():
                    add("oracle_physical", oracle["physical_work"])
                    add("oracle_represented", oracle["represented_work"])
                    add("oracle_inference", oracle["new_actor_calls"])
                add("actual_tails", row["assessment"]["actual_work"])
                add("actual_tail_replay", row["assessment"]["replay_work"])
                add("actual_inference", row["assessment"]["calls"])
        for name, counters in checked["work"].items():
            add(f"audit_{name}", counters)
    if (
        seen != protocol["cases"]
        or costs["sources"]["completions"] != config["source_matches"]
    ):
        raise ValueError("Complete source denominator differs")
    result = summarize(values, expected_n=config["sources_per_population"])
    write(
        base / "summary.json",
        {
            "audit_status": "passed",
            "protocol_sha256": sha(base / "protocol.json"),
            "bindings": bindings,
            "sources": len(seen),
            "status_counts": dict(statuses),
            "inference": result,
            "actual_mean_difference_descriptive": {
                k: str(sum(v, Fraction(0)) / len(v)) for k, v in actual.items()
            },
            "work": {k: dict(v) for k, v in costs.items()},
        },
    )
    return result


def report_text(summary):
    result = summary["inference"]
    lines = [
        "Uniform-deal controller evaluation",
        "==================================",
        "",
        f"Complete audited source matches: {summary['sources']}.",
        f"Protocol SHA256: {summary['protocol_sha256']}.",
        f"Joint prespecified criterion: {'PASSED' if result['criterion_met'] else 'FAILED'}.",
        "",
        "Primary values are conditional-expectation improvements over each unchanged",
        "parent under independent uniform valid round deals and a randomized learner",
        "seat. All missing-root and fallback sources remain in the denominator.",
        "The four lower bounds are simultaneous at least95% under the declared law.",
        "No observed-win, absolute-superiority or arbitrary-opponent claim follows.",
        "",
        ".. csv-table:: Expected improvement by parent and opponent",
        "   :header: Population,Sources,Mean(pp),Lower(pp),Exact lower",
        "",
    ]
    for key, row in result["populations"].items():
        lines.append(
            f"   {key},{row['sources']},{float(Fraction(row['mean']) * 100):.6f},{float(Fraction(row['lower']) * 100):.6f},{row['lower']}"
        )
    lines += [
        "",
        ".. csv-table:: Equal WP/Point mixtures",
        "   :header: Parent,Mean(pp),Lower(pp),Lower exceeds0.1pp",
        "",
    ]
    for key, row in result["parent_mixtures"].items():
        lines.append(
            f"   {key},{float(Fraction(row['mean']) * 100):.6f},{float(Fraction(row['lower']) * 100):.6f},{row['practical_criterion_met']}"
        )
    lines += [
        "",
        "Success requires positive lower bounds in all four populations AND a mixture",
        "lower bound above0.1percentage point for each parent. Exact fractions decide",
        "the criterion. Display rounding has no role in the decision.",
        "",
        ".. csv-table:: Realized conditional-tail differences (descriptive only)",
        "   :header: Population,Controller minus parent(pp)",
        "",
    ]
    for key, value in summary["actual_mean_difference_descriptive"].items():
        lines.append(f"   {key},{float(Fraction(value) * 100):.6f}")
    lines += [
        "",
        "Work accounting",
        "---------------",
        "",
        "Source matches, native verification, history reconstruction, planning,",
        "full Python value verification, actual tails and their audit/replays",
        "are separate. Simulated completions and represented mass are not new",
        "source observations. Bounded entropy draws exclude discarded internal",
        "seeded auto-deals. Adapter calls include public endgame wrappers.",
        "",
        ".. code-block:: json",
        "",
    ]
    lines.extend(
        "   " + line
        for line in json.dumps(summary["work"], indent=2, sort_keys=True).splitlines()
    )
    return "\n".join(lines) + "\n"


def export_report(base):
    check(base)
    summary = read(base / "summary.json")
    if summary["audit_status"] != "passed" or summary["protocol_sha256"] != sha(
        base / "protocol.json"
    ):
        raise ValueError("Summary is not bound to the audited protocol")
    if any(sha(p) != digest for p, digest in summary["bindings"].items()):
        raise ValueError("Summary input binding changed")
    if (base / "results.rst").exists() or (base / "export.json").exists():
        raise ValueError("Export already exists")
    (base / "results.rst").write_bytes(report_text(summary).encode("utf-8"))
    write(
        base / "export.json",
        {
            "status": "passed",
            "protocol_sha256": sha(base / "protocol.json"),
            "summary_sha256": sha(base / "summary.json"),
            "report_sha256": sha(base / "results.rst"),
            "source_matches": summary["sources"],
            "criterion_met": summary["inference"]["criterion_met"],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("freeze", "run", "summarize", "export"):
        mode.add_argument(f"--{name}", action="store_true")
    mode.add_argument("--worker", type=int)
    mode.add_argument("--audit-chunk", type=int)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.base)
    elif args.run:
        run_queue(args.base)
        summarize_study(args.base)
        export_report(args.base)
    elif args.summarize:
        summarize_study(args.base)
    elif args.export:
        export_report(args.base)
    elif args.worker is not None:
        worker(args.base, args.worker)
    else:
        audit_chunk(args.base, args.audit_chunk)


if __name__ == "__main__":
    main()
