"""Frozen development comparison of complete learned continuation programs."""

import argparse
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

from koikoi.layout import PHASE, PLAYER, TERMINAL, TURN
from scripts.belief_tail import KoiTailDynamics
from scripts.check_belief_tail import add_step, empty_work
from scripts.continuation_frontier import (
    ContinuationActors,
    ContinuationFrontierExecutor,
    solve_continuation_frontier,
    verify_continuation_patch,
)
from scripts.diagnose_history_support import reference
from scripts.export_belief_tail import replay_actions
from scripts.prospective_incumbent import compressed_save
from scripts.run_arbitration_study import PARENTS, read, sha, write
from scripts.uniform_controller_study import actors, load

FAMILIES = ("other_parent", "policy_pool")


def freeze(base, smoke=False):
    if base.exists():
        raise FileExistsError(base)
    previous = read("runs/resumable_frontier_dev_v1/protocol.json")
    fixtures = [dict(f) for f in previous["fixtures"]]
    baseline = read("runs/resumable_frontier_dev_v1/summary.json")["rows"]
    for i, f in enumerate(fixtures):
        f["baseline"] = f"runs/resumable_frontier_dev_v1/fixture_{i:02}/audit.json"
    for i in (133, 216, 241):
        path = f"runs/prospective_frontier_proposals_v1/chunk_{i // 8:03}/result_{i:04}.json"
        old = read(path)
        arm = old["arms"]["parent_default_proposal"]
        fixtures.append(
            dict(draws=5, source=old["source"], program=arm["program"], baseline=path)
        )
        baseline.append(
            dict(
                fixture=len(baseline),
                exact_increment=arm["oracle"]["exact_increment"],
                decisions=arm["decisions"],
                seconds=arm["seconds"],
            )
        )
    if smoke:
        fixtures, baseline = fixtures[:1], baseline[:1]
    paths = set(previous["inputs"])
    paths.update(
        (
            "runs/resumable_frontier_dev_v1/protocol.json",
            "runs/resumable_frontier_dev_v1/summary.json",
            "runs/prospective_frontier_proposals_v1/protocol.json",
            "scripts/study_continuation_frontier.py",
            "scripts/continuation_frontier.py",
            "tests/test_continuation_frontier.py",
            "docs/CONTINUATION_PORTFOLIO.rst",
        )
    )
    paths.update(f[k] for f in fixtures for k in ("source", "program", "baseline"))
    protocol = dict(
        scope=(
            "Pipeline smoke only; no strength inference"
            if smoke
            else "19 reused development sources: 16 earlier fixtures plus closed-comparison misses 133/216/241; no fresh strength inference"
        ),
        smoke=smoke,
        fixtures=fixtures,
        baseline=baseline,
        families=list(FAMILIES),
        inputs={p: sha(p) for p in sorted(paths)},
        shared_transitions=20000 if smoke else 2000000,
        prefix_transitions=250000,
        local_transitions=2000 if smoke else 100000,
        capacity=4096,
        workers=2,
        continuation_identities=read(
            "runs/uniform_controller_evaluation/protocol.json"
        )["opponent_identities"],
        candidate_order="ordinary root alternatives, other parent, WP then Point; first strict certificate",
        criterion="At least one new positive source, no loss of baseline positive sources, total exact gain at least baseline, and gain per physical transition at least baseline. Select eligible family with greatest total exact gain; ties prefer other_parent. Smoke cannot qualify.",
        timing="Sequential family order within source, two concurrent workers; descriptive elapsed time, not isolated latency",
    )
    base.mkdir(parents=True)
    write(base / "protocol.json", protocol)


def checked(base):
    p = read(base / "protocol.json")
    for path, digest in p["inputs"].items():
        if sha(path) != digest:
            raise ValueError(f"Frozen dependency changed: {path}")
    return p


def assess(source, patch, registry, opponent):
    outcomes, actual, replay = {}, Counter(), Counter()
    for label in ("parent", "controller"):
        env = reference(source["selected_root"]["state"])
        executor = ContinuationFrontierExecutor(patch, registry)
        actions, work = [], empty_work()
        for _ in range(32):
            if env.state[PHASE] == TERMINAL:
                break
            if env.state[PLAYER] != source["seat"]:
                action = int(opponent.act(env))
            else:
                action = (
                    int(registry.parent.act(env))
                    if label == "parent"
                    else executor.act(env)
                )
            actions.append(action)
            add_step(work, env.step(action))
        if env.state[PHASE] != TERMINAL:
            raise ValueError("Actual tail did not terminate")
        final = env.state.tolist()
        verified = replay_actions(
            reference(source["selected_root"]["state"]), actions, final
        )
        if verified != work:
            raise ValueError("Actual replay differs")
        actual.update(work)
        replay.update(verified)
        outcomes[label] = dict(actions=actions, final=final, work=work)
    if (
        outcomes["parent"]["final"] != source["final"]
        or outcomes["parent"]["actions"]
        != source["actions"][source["selected_root"]["decision"] :]
    ):
        raise ValueError("Parent no longer reproduces the source suffix")
    if patch.gain_lower == 0 and outcomes["parent"] != outcomes["controller"]:
        raise ValueError("Fallback differs from parent")
    return dict(outcomes=outcomes, actual_work=dict(actual), replay_work=dict(replay))


def worker(base, index):
    p = checked(base)
    f = p["fixtures"][index]
    folder = base / f"fixture_{index:02}"
    folder.mkdir()
    source, evidence = read(f["source"]), load(f["program"])
    worlds = evidence["worlds"]
    parents, opponents, _ = actors()
    parent_index, opponent = source["parent"], source["opponent"]
    parent = parents[parent_index]
    identities = {opponent: p["continuation_identities"][opponent]}
    for family in p["families"]:
        destination = folder / family
        destination.mkdir()
        alternatives = {PARENTS[1 - parent_index][1]: parents[1 - parent_index]}
        if family == "policy_pool":
            alternatives.update(
                (p["continuation_identities"][name], opponents[name])
                for name in ("wp", "point")
            )
        registry = ContinuationActors(PARENTS[parent_index][1], parent, alternatives)
        dyn = KoiTailDynamics(
            p["capacity"], worlds.model_names, {opponent: opponents[opponent]}, parent
        )
        started = time.perf_counter()
        patch = solve_continuation_frontier(
            worlds,
            dyn,
            registry,
            identities,
            draws=14 - int(worlds.states[0, TURN]),
            max_decisions=p["shared_transitions"],
            prefix_decisions=p["prefix_transitions"],
            local_decisions=p["local_transitions"],
        )
        seconds = time.perf_counter() - started
        compressed_save(destination / "program.pkl.gz", patch)
        defaults = Counter(
            getattr(
                r["selection"].candidate.program,
                "continuation_identity",
                registry.parent_identity,
            )
            for _, r in patch.comparisons
            if r["selection"].candidate is not None
        )
        row = dict(
            fixture=index,
            family=family,
            draws=f["draws"],
            decisions=patch.decisions,
            seconds=seconds,
            gain_lower=str(patch.gain_lower),
            frontier_complete=patch.frontier.complete,
            refusal=patch.frontier.refusal,
            frontier_nodes=len(patch.frontier.nodes),
            attempted_nodes=len(patch.comparisons),
            adopted_nodes=sum(defaults.values()),
            adopted_continuations=dict(defaults),
            continuation_actor_calls=dict(registry.calls),
            program_sha256=sha(destination / "program.pkl.gz"),
        )
        write(destination / "selection.json", row)
        print({"planned": row}, flush=True)
        if patch.frontier.complete:
            oracle = verify_continuation_patch(
                worlds, patch, registry, {opponent: opponents[opponent]}
            )
        else:
            if patch.comparisons or patch.gain_lower:
                raise ValueError("Refused prefix cannot adopt a policy")
            oracle = dict(exact_increment="0", refusal=patch.frontier.refusal)
        execution = assess(source, patch, registry, opponents[opponent])
        write(
            destination / "audit.json",
            dict(
                status="passed",
                oracle=oracle,
                execution=execution,
                selection_sha256=sha(destination / "selection.json"),
            ),
        )
        print(
            {"verified": index, "family": family, "gain": oracle["exact_increment"]},
            flush=True,
        )


def summarize(base):
    p = checked(base)
    rows = []
    for i in range(len(p["fixtures"])):
        for family in p["families"]:
            folder = base / f"fixture_{i:02}" / family
            row, audit = read(folder / "selection.json"), read(folder / "audit.json")
            if (
                audit["status"] != "passed"
                or audit["selection_sha256"] != sha(folder / "selection.json")
                or row["program_sha256"] != sha(folder / "program.pkl.gz")
            ):
                raise ValueError("Result identity mismatch")
            rows.append(
                {
                    **row,
                    "exact_increment": audit["oracle"]["exact_increment"],
                    "audit_sha256": sha(folder / "audit.json"),
                }
            )
    baseline_gain = sum(Fraction(r["exact_increment"]) for r in p["baseline"])
    baseline_work = sum(r["decisions"] for r in p["baseline"])
    old_positive = {
        r["fixture"] for r in p["baseline"] if Fraction(r["exact_increment"]) > 0
    }
    comparison = {}
    for family in p["families"]:
        group = [r for r in rows if r["family"] == family]
        gain = sum(Fraction(r["exact_increment"]) for r in group)
        work = sum(r["decisions"] for r in group)
        positive = {r["fixture"] for r in group if Fraction(r["exact_increment"]) > 0}
        eligible = bool(
            not p["smoke"]
            and positive - old_positive
            and old_positive <= positive
            and gain >= baseline_gain
            and gain * baseline_work >= baseline_gain * work
        )
        comparison[family] = dict(
            exact_gain_sum=str(gain),
            transitions=work,
            seconds=sum(r["seconds"] for r in group),
            positives=sorted(positive),
            new_positives=sorted(positive - old_positive),
            lost_positives=sorted(old_positive - positive),
            eligible=eligible,
        )
    eligible = [f for f in p["families"] if comparison[f]["eligible"]]
    selected = (
        max(eligible, key=lambda f: Fraction(comparison[f]["exact_gain_sum"]))
        if eligible
        else None
    )
    write(
        base / "summary.json",
        dict(
            scope=p["scope"],
            rows=rows,
            comparison=comparison,
            selected_for_fresh_confirmation=selected,
            protocol_sha256=sha(base / "protocol.json"),
        ),
    )


def run(base):
    p = checked(base)

    def launch(index):
        with (base / f"worker_{index:02}.log").open(
            "x", encoding="utf-8", newline="\n"
        ) as log:
            subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-m",
                    "scripts.study_continuation_frontier",
                    "--base",
                    str(base),
                    "--worker",
                    str(index),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    with ThreadPoolExecutor(max_workers=p["workers"]) as pool:
        list(pool.map(launch, range(len(p["fixtures"]))))
    summarize(base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze(args.base, args.smoke)
    elif args.run:
        run(args.base)
    else:
        worker(args.base, args.worker)
