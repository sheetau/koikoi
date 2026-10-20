"""Fresh, fixed-cohort comparison of contingent and root-only tail search."""

import argparse
import ast
import gzip
import hashlib
import json
import os
import pickle
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
from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.train import provenance
from koikoi.upstream import UPSTREAM_PATH, PublishedAgent
from scripts.belief_tail import KoiTailDynamics, TailBudgetExceeded
from scripts.check_belief_tail import add_step, empty_work
from scripts.check_earlier_belief_tail import collect
from scripts.deferred_belief_tail import deferred_worlds
from scripts.diagnose_history_support import reference
from scripts.export_belief_tail import read, replay_actions, score
from scripts.history_conditioned_worlds import enumerate_support
from scripts.incumbent_belief_tail import IncumbentBeliefPlanner
from scripts.incumbent_tail_controller import (
    ContingentExecutor,
    EvaluatedPhase,
    bind_worlds,
    evaluate_controller,
    select_improvement,
)
from scripts.reference_tail_policy import verify_deferred_policy_values
from scripts.revealed_opponent_models import StrictPublishedPrediction
from scripts.run_arbitration_study import PARENTS, sha, write


def cases():
    # One replicate contains both parents/opponents/seats, before the next replicate.
    return [
        {
            "parent": parent,
            "opponent": opponent,
            "seat": seat,
            "replicate": replicate,
            "seed": 123110001 + 1000 * opponent_index + replicate,
        }
        for replicate in range(64)
        for opponent_index, opponent in enumerate(("wp", "point"))
        for parent in (0, 1)
        for seat in (0, 1)
    ]


def dependency_files():
    """Bind local absolute imports transitively and all production source files."""
    pending = [Path(__file__).relative_to(Path.cwd())]
    found = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            names = (
                [node.module]
                if isinstance(node, ast.ImportFrom) and not node.level
                else [n.name for n in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            for name in names:
                if name and name.split(".")[0] in ("scripts", "koikoi"):
                    target = Path(*name.split(".")).with_suffix(".py")
                    if target.exists():
                        pending.append(target)
    found.update(
        p for p in Path("koikoi").rglob("*") if p.suffix in (".py", ".h", ".cpp", ".cu")
    )
    found.update(
        Path(p)
        for p in (
            "docs/PROSPECTIVE_INCUMBENT.rst",
            "tests/test_prospective_incumbent.py",
        )
    )
    found.update(UPSTREAM_PATH.glob("*.py"))
    for path, expected in PARENTS:
        assert sha(path) == expected
        found.add(Path(path))
    found.update(
        UPSTREAM_PATH / f"model_agent/{phase}_rl_{name}.pt"
        for phase in ("discard", "pick", "koikoi")
        for name in ("wp", "point")
    )
    return {p.as_posix(): sha(p) for p in sorted(found)}


def freeze(base):
    if base.exists():
        raise ValueError("Use a fresh study directory")
    inputs = dependency_files()
    identities = {
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
    base.mkdir(parents=True)
    write(
        base / "protocol.json",
        {
            "inputs": inputs,
            "source": provenance(),
            "cases": cases(),
            "opponent_identities": identities,
            "source_matches": 512,
            "distinct_streams": 128,
            "chunk_size": 16,
            "workers": 4,
            "max_draws": 5,
            "max_hands": 1000,
            "capacity": 4096,
            "phase_nodes": 500000,
            "phase_decisions": 2000000,
            "depth": 32,
            "bootstrap_seed": 123990001,
            "bootstrap_samples": 20000,
            "scope": "Fixed development comparison; no automatic sample extension or promotion.",
        },
    )
    return read(base / "protocol.json")


def check(base):
    protocol = read(base / "protocol.json")
    assert protocol["cases"] == cases()
    assert all(sha(path) == digest for path, digest in protocol["inputs"].items())
    assert provenance()["source_sha256"] == protocol["source"]["source_sha256"]
    return protocol


def evaluate_arm(
    worlds,
    dynamics,
    parent_id,
    identities,
    arm,
    protocol,
    planner_type=IncumbentBeliefPlanner,
):
    """Independent cold caches per arm; equal summed node/transition allowances."""
    phases, timing = {}, {}

    class Tracked(planner_type):
        def solve(self, worlds, optimize_continuation=True):
            name = "optimized" if optimize_continuation else "fixed"
            start = time.perf_counter()
            result = super().solve(worlds, optimize_continuation)
            timing[name] = time.perf_counter() - start
            return result

    binding = bind_worlds(worlds, dynamics, parent_id, identities)
    if arm == "controller":
        fixed, optimized, selection = evaluate_controller(
            worlds,
            dynamics,
            parent_id,
            identities,
            max_nodes=protocol["phase_nodes"],
            max_depth=protocol["depth"],
            max_decisions_per_phase=protocol["phase_decisions"],
            planner_type=Tracked,
        )
        phases["fixed"] = fixed
        if optimized is not None:
            phases["optimized"] = optimized
    elif arm == "root_only":
        planner = Tracked(
            dynamics,
            2 * protocol["phase_nodes"],
            protocol["depth"],
            2 * protocol["phase_decisions"],
        )
        fixed = EvaluatedPhase(binding, "fixed", planner.solve(worlds, False))
        assert bind_worlds(worlds, dynamics, parent_id, identities) == binding
        phases["fixed"] = fixed
        selection = select_improvement(fixed)
    else:
        raise ValueError("Unknown study arm")
    return phases, selection, timing


def describe_selection(selection):
    return {
        "mode": selection.mode,
        "reason": selection.reason,
        "parent_q": None if selection.parent_q is None else str(selection.parent_q),
        "selected_q": None
        if selection.selected_q is None
        else str(selection.selected_q),
        "gain": str(selection.gain or 0),
        "action": None if selection.plan is None else selection.plan.action,
        "policy_nodes": 0 if selection.plan is None else len(selection.plan.policy),
    }


def verify_adoption(worlds, phases, selection, opponent_cache):
    """Check the two policy values that justify each actual adoption."""
    if selection.plan is None:
        return {}
    parent = phases["fixed"].result.plan
    selected = selection.plan
    checks = {}
    for label, plan, action in (
        ("parent", parent, parent.baseline_action),
        ("selected", selected, selected.action),
    ):
        view = SimpleNamespace(
            root_policies={action: plan.root_policies[action]},
            values={action: plan.values[action]},
        )
        checks[label] = verify_deferred_policy_values(worlds, view, opponent_cache)
    return checks


def assess_three(source, selections, parent, opponent, parent_id):
    outcomes, actual, replay, calls = {}, Counter(), Counter(), Counter()
    for arm in ("parent", "root_only", "controller"):
        env = reference(source["selected_root"]["state"])
        executor = (
            None
            if arm == "parent"
            else ContingentExecutor(selections[arm], parent, parent_id)
        )
        actions, work = [], empty_work()
        for _ in range(32):
            if env.state[PHASE] == TERMINAL:
                break
            if env.state[PLAYER] == source["seat"]:
                action = int(parent.act(env)) if executor is None else executor.act(env)
                calls["learner_decisions"] += 1
                if executor is None or executor.policy is None:
                    calls["parent_adapter_calls"] += 1
            else:
                action = int(opponent.act(env))
                calls["opponent_adapter_calls"] += 1
            actions.append(action)
            add_step(work, env.step(action))
        assert env.state[PHASE] == TERMINAL
        outcome = {
            "actions": actions,
            "work": work,
            "final": env.state.tolist(),
            "margin": int(
                env.state[POINTS + source["seat"]]
                - env.state[POINTS + 1 - source["seat"]]
            ),
        }
        outcomes[arm] = outcome
        actual.update(work)
        audited = replay_actions(
            reference(source["selected_root"]["state"]), actions, outcome["final"]
        )
        assert audited == work
        replay.update(audited)
    assert outcomes["parent"]["final"] == source["final"]
    assert (
        outcomes["parent"]["actions"]
        == source["actions"][source["selected_root"]["decision"] :]
    )
    for arm, selection in selections.items():
        if selection.plan is None:
            assert outcomes[arm] == outcomes["parent"]
    return {
        "outcomes": outcomes,
        "actual_work": dict(actual),
        "replay_work": dict(replay),
        "calls": dict(calls),
    }


def compressed_save(path, value):
    # Plain pickle with compression avoids repeated 640-byte key storage on disk.
    with gzip.open(path, "wb", compresslevel=1) as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def worker(base, chunk):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    protocol = check(base)
    start = chunk * protocol["chunk_size"]
    selected_cases = protocol["cases"][start : start + protocol["chunk_size"]]
    if not selected_cases:
        raise ValueError("Invalid chunk")
    output = base / f"chunk_{chunk:03}"
    if output.exists():
        raise ValueError("Chunk output already exists")
    output.mkdir()

    def stage(name, index=None):
        write(
            output / "process.json",
            {
                "status": "running",
                "pid": os.getpid(),
                "stage": name,
                "source": index,
            },
        )

    try:
        stage("setup")
        parents = [NeuralAgent(path, "cpu", deterministic=True) for path, _ in PARENTS]
        assert all(p.requires_endgame for p in parents)
        published = {name: PublishedAgent(name) for name in ("wp", "point")}
        opponents = {name: EndgameAgent(agent) for name, agent in published.items()}
        predictors = {
            name: StrictPublishedPrediction(a) for name, a in published.items()
        }
        native = BatchEnv(1, "cpu", debug=True)
        records, artifacts = [], {}
        for index, case in enumerate(selected_cases, start):
            stage("source", index)
            source = collect(
                case,
                parents[case["parent"]],
                opponents[case["opponent"]],
                predictors,
                native,
                max_draws=protocol["max_draws"],
            )
            row = {"index": index, "source": source, "status": "missing_root"}
            if source["selected_root"] is None:
                records.append(row)
                write(output / "progress.json", records)
                continue
            stage("history", index)
            root = reference(source["selected_root"]["state"])
            identity = source["selected_root"]["public_identification"]["model"]
            # Actor identity is inferred from past public play, not the case label.
            assert identity == case["opponent"]  # Evaluator-only consistency check.
            started = time.perf_counter()
            support = enumerate_support(
                root, {identity: opponents[identity]}, max_hands=protocol["max_hands"]
            )
            row.update(
                history_counts=support.counts,
                history_seconds=time.perf_counter() - started,
            )
            try:
                if support.status == "too_many_hands":
                    raise TailBudgetExceeded("Public hand cap exceeded")
                worlds = deferred_worlds(
                    root,
                    support,
                    max_draws=protocol["max_draws"],
                    max_hands=protocol["max_hands"],
                )
            except TailBudgetExceeded as error:
                row.update(status="support_refused", reason=str(error))
                records.append(row)
                write(output / "progress.json", records)
                continue
            row.update(
                status="evaluated",
                arms={},
                order=(
                    ["controller", "root_only"]
                    if index % 2 == 0
                    else ["root_only", "controller"]
                ),
            )
            selections = {}
            for arm in row["order"]:
                stage(arm, index)
                dynamics = KoiTailDynamics(
                    protocol["capacity"],
                    worlds.model_names,
                    {identity: opponents[identity]},
                    parents[case["parent"]],
                )
                phases, selection, timing = evaluate_arm(
                    worlds,
                    dynamics,
                    PARENTS[case["parent"]][1],
                    {identity: protocol["opponent_identities"][identity]},
                    arm,
                    protocol,
                )
                selections[arm] = selection
                stage(f"{arm}_oracle", index)
                begun = time.perf_counter()
                oracle = verify_adoption(
                    worlds, phases, selection, dynamics.policy_cache
                )
                row["arms"][arm] = {
                    "selection": describe_selection(selection),
                    "seconds": timing,
                    "phases": {
                        mode: {
                            "work": phase.result.work,
                            "refusal": phase.result.refusal,
                        }
                        for mode, phase in phases.items()
                    },
                    "parent_adapter_calls": dynamics.baseline_calls,
                    "opponent_adapter_calls": dynamics.policy_calls,
                    "oracle": oracle,
                    "oracle_seconds": time.perf_counter() - begun,
                }
                path = output / f"root_{index:04}_{arm}.pkl.gz"
                compressed_save(
                    path,
                    {
                        "worlds": worlds,
                        "support": asdict(support),
                        "phases": phases,
                        "selection": selection,
                        "opponent_cache": dynamics.policy_cache,
                    },
                )
                artifacts[path.name] = sha(path)
                row["arms"][arm]["artifact"] = path.name
                del dynamics, phases
            left, right = selections["controller"], selections["root_only"]
            assert left.binding == right.binding
            if left.parent_q is not None and right.parent_q is not None:
                assert left.parent_q == right.parent_q
            # Freeze both choices before executing either actual conditional tail.
            write(output / f"selection_{index:04}.json", row["arms"])
            artifacts[f"selection_{index:04}.json"] = sha(
                output / f"selection_{index:04}.json"
            )
            stage("actual_tails", index)
            row["assessment"] = assess_three(
                source,
                selections,
                parents[case["parent"]],
                opponents[case["opponent"]],
                PARENTS[case["parent"]][1],
            )
            records.append(row)
            write(output / "progress.json", records)
            print(json.dumps({"source": index, "status": row["status"]}), flush=True)
        check(base)
        write(output / "records.json", records)
        write(
            output / "complete.json",
            {
                "status": "complete",
                "protocol_sha256": sha(base / "protocol.json"),
                "records_sha256": sha(output / "records.json"),
                "artifacts": artifacts,
            },
        )
        write(output / "process.json", {"status": "complete", "pid": os.getpid()})
    except BaseException as error:
        write(
            output / "process.json",
            {
                "status": "failed",
                "pid": os.getpid(),
                "error": repr(error),
            },
        )
        raise


def launch(base):
    protocol = check(base)
    chunks = list(range(len(protocol["cases"]) // protocol["chunk_size"]))
    if (base / "queue.json").exists():
        raise ValueError("Queue already launched; inspect existing jobs")
    started = time.time()
    write(
        base / "queue.json",
        {
            "status": "running",
            "pid": os.getpid(),
            "chunks": chunks,
            "started_unix": started,
        },
    )

    def one(chunk):
        if shutil.disk_usage(base).free < 20 * 1024**3:
            raise RuntimeError("Insufficient disk space for next chunk")
        with (base / f"worker_{chunk:03}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.prospective_incumbent",
                    "--base",
                    str(base),
                    "--worker",
                    str(chunk),
                ],
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"Worker {chunk} failed with {result.returncode}")
        return chunk

    done = []
    try:
        with ThreadPoolExecutor(max_workers=protocol["workers"]) as pool:
            futures = [pool.submit(one, chunk) for chunk in chunks]
            try:
                for future in as_completed(futures):
                    done.append(future.result())
                    write(base / "progress.json", {"completed_chunks": sorted(done)})
                    print(json.dumps({"completed_chunks": len(done)}), flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        summarize(base)
        write(
            base / "queue.json",
            {
                "status": "complete",
                "pid": os.getpid(),
                "chunks": chunks,
                "seconds": time.time() - started,
            },
        )
    except BaseException as error:
        write(
            base / "queue.json",
            {
                "status": "failed",
                "pid": os.getpid(),
                "error": repr(error),
                "completed_chunks": sorted(done),
                "seconds": time.time() - started,
            },
        )
        raise


def contrasts(rows):
    """Average paired seats within each deal; keep both parents on the same clusters."""
    blocks = {}
    conditional = {}
    for row in rows:
        source = row["source"]
        parent, opponent, replicate = (
            source["parent"],
            source["opponent"],
            source["replicate"],
        )
        margin = int(
            source["final"][POINTS + source["seat"]]
            - source["final"][POINTS + 1 - source["seat"]]
        )
        scores = dict.fromkeys(("parent", "root_only", "controller"), score(margin))
        gains = dict.fromkeys(("root_only", "controller"), Fraction(0))
        if row["status"] == "evaluated":
            scores.update(
                {
                    arm: score(outcome["margin"])
                    for arm, outcome in row["assessment"]["outcomes"].items()
                }
            )
            gains.update(
                {
                    arm: Fraction(data["selection"]["gain"])
                    for arm, data in row["arms"].items()
                }
            )
        key = (opponent, replicate)
        blocks.setdefault(key, np.zeros(4))
        conditional.setdefault(key, np.zeros(4))
        for offset, control in enumerate(("parent", "root_only")):
            column = 2 * parent + offset
            blocks[key][column] += float(scores["controller"] - scores[control]) / 2
            conditional[key][column] += (
                float(gains["controller"] - gains.get(control, Fraction(0))) / 2
            )
    keys = sorted(blocks)
    return (
        keys,
        np.stack([blocks[k] for k in keys]),
        np.stack([conditional[k] for k in keys]),
    )


def summarize(base):
    protocol = check(base)
    rows, bindings = [], {}
    for chunk in range(len(protocol["cases"]) // protocol["chunk_size"]):
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
        rows.extend(read(folder / "records.json"))
    assert [r["index"] for r in rows] == list(range(len(protocol["cases"])))
    assert [
        {k: r["source"][k] for k in c}
        for r, c in zip(rows, protocol["cases"], strict=True)
    ] == protocol["cases"]
    keys, actual, conditional = contrasts(rows)
    assert len(keys) == 128
    # Fixed stratified cluster bootstrap; same indices for all four contrasts.
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    bootstrap, secondary = [], []
    for _ in range(protocol["bootstrap_samples"] // 1000):
        indices = np.concatenate(
            [
                rng.integers(0, 64, (1000, 64)),
                rng.integers(64, 128, (1000, 64)),
            ],
            axis=1,
        )
        bootstrap.append(actual[indices].mean(1))
        secondary.append(conditional[indices].mean(1))
    lower, upper = np.quantile(np.concatenate(bootstrap), [0.00625, 0.99375], axis=0)
    clower, cupper = np.quantile(np.concatenate(secondary), [0.00625, 0.99375], axis=0)
    work = {
        name: Counter()
        for name in (
            "source",
            "native_source_verification",
            "history",
            "planning",
            "actual_tails",
            "tail_replay",
            "oracle_physical",
            "oracle_represented",
        )
    }
    arm_costs = {arm: Counter() for arm in ("controller", "root_only")}
    identification_calls = Counter()
    adoptions = Counter()
    changed_wins = Counter()
    for row in rows:
        work["source"].update(row["source"]["work"])
        work["native_source_verification"].update(row["source"]["work"])
        work["history"].update(row.get("history_counts", {}))
        identification_calls.update(row["source"]["identification_calls"])
        if row["status"] == "evaluated":
            for name, arm in row["arms"].items():
                parent = row["source"]["parent"] + 1
                group = f"parent{parent}:{name}"
                adoptions[group] += int(arm["selection"]["action"] is not None)
                outcomes = row["assessment"]["outcomes"]
                changed_wins[group] += int(
                    score(outcomes[name]["margin"])
                    != score(outcomes["parent"]["margin"])
                )
                arm_costs[name].update(
                    {
                        "planning_seconds": sum(arm["seconds"].values()),
                        "parent_adapter_calls": arm["parent_adapter_calls"],
                        "opponent_adapter_calls": arm["opponent_adapter_calls"],
                    }
                )
                for phase in arm["phases"].values():
                    work["planning"].update(
                        {
                            k: v
                            for k, v in phase["work"].items()
                            if k != "maximum_frontier_rows"
                        }
                    )
                    arm_costs[name]["physical_decisions"] += phase["work"]["decisions"]
                    arm_costs[name]["capped_phases"] += int(
                        phase["refusal"] is not None
                    )
                for oracle in arm["oracle"].values():
                    work["oracle_physical"].update(oracle["physical_work"])
                    work["oracle_represented"].update(oracle["represented_work"])
            work["actual_tails"].update(row["assessment"]["actual_work"])
            work["tail_replay"].update(row["assessment"]["replay_work"])
    result = {
        "status": "complete",
        "protocol_sha256": sha(base / "protocol.json"),
        "chunk_bindings": bindings,
        "sources": len(rows),
        "streams": len(keys),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "contrast_order": [
            "parent1:controller-parent",
            "parent1:controller-root_only",
            "parent2:controller-parent",
            "parent2:controller-root_only",
        ],
        "actual_mean": actual.mean(0).tolist(),
        "actual_simultaneous95_lower": lower.tolist(),
        "actual_simultaneous95_upper": upper.tolist(),
        "conditional_mean": conditional.mean(0).tolist(),
        "conditional_secondary_lower": clower.tolist(),
        "conditional_secondary_upper": cupper.tolist(),
        "development_strength_gate": bool(np.all(lower > 0)),
        "adoptions": dict(adoptions),
        "changed_actual_win_scores": dict(changed_wins),
        "arm_costs": {arm: dict(costs) for arm, costs in arm_costs.items()},
        "public_identification_calls": dict(identification_calls),
        "work": {name: dict(counts) for name, counts in work.items()},
        "scope": protocol["scope"],
    }
    write(base / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("runs/prospective_incumbent"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--launch", action="store_true")
    group.add_argument("--worker", type=int)
    group.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze(args.base)
    elif args.launch:
        launch(args.base)
    elif args.worker is not None:
        worker(args.base, args.worker)
    else:
        summarize(args.base)
