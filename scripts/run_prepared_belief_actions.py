"""Four-condition action diagnostic on all audited exact development roots."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent
from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.upstream import PublishedAgent
from scripts.complete_sampled_history_exact import batch_from_record, select_records
from scripts.diagnose_history_support import reference
from scripts.prepared_belief_search import KINDS, PreparedBeliefSearch
from scripts.run_arbitration_study import PARENTS, read, sha, write

EXACT = Path("runs/sampled_history_complete_exact")
ORIGINAL = Path("runs/sampled_history_diagnostic/records.jsonl")


def load_roots():
    originals = select_records(
        [json.loads(line) for line in ORIGINAL.read_text(encoding="utf-8").splitlines()]
    )
    exact = read(EXACT / "report.json")["results"]
    assert len(exact) == len(originals) == 25
    for original, row in zip(originals, exact):
        for key in ("parent", "opponent", "case_index", "seed", "decision"):
            assert original[key] == row[key]
        path = Path(
            f"runs/final_round_training_study/parent{row['parent']}/source.jsonl"
        )
        source = json.loads(
            path.read_text(encoding="utf-8").splitlines()[row["case_index"]]
        )
        env = ReferenceEnv(source["seed"], "paper8")
        for action in source["actions"][: row["decision"]]:
            assert env.legal()[action]
            env.step(action)
        assert int(env.state[PLAYER]) == original["seat"]
        support_path = EXACT / row["support_file"]
        assert sha(support_path) == row["support_sha256"]
        yield original, env, batch_from_record(read(support_path))


def complete(root, action, parent, opponent, seat):
    env = reference(root.state)
    actions, rounds = [], 0
    while env.state[PHASE] != TERMINAL:
        if actions:
            actor = parent if env.state[PLAYER] == seat else opponent
            action = int(actor.act(env))
        assert env.legal()[action]
        result = env.step(action)
        assert int(result[0]) != 99
        rounds += int(result[1])
        actions.append(action)
    points = int(env.state[POINTS] - env.state[POINTS + 1]) * (1 - 2 * seat)
    return {
        "actions": actions,
        "endpoint": env.state.tolist(),
        "points": points,
        "win": (int(np.sign(points)) + 1) / 2,
        "work": {
            "decisions": len(actions),
            "round_endings": rounds,
            "conditional_completions": 1,
        },
    }


def run(output):
    if output.exists():
        raise ValueError("Use a fresh output directory")
    if not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA execution required")
    audit_path = EXACT / "arithmetic_verification.json"
    audit = read(audit_path)
    assert audit["status"] == "passed" and audit["eligible_roots"] == 25
    inputs = dict(audit["inputs"])
    paths = [
        audit_path,
        ORIGINAL,
        Path(__file__),
        Path("scripts/prepared_belief_search.py"),
        Path("docs/PREPARED_BELIEF_ACTION_CHECK.rst"),
        Path("tests/test_prepared_belief_search.py"),
        Path("tests/test_prepared_belief_search_cuda.py"),
        Path("runs/prepared_belief_search_cpu_tests.xml"),
        Path("runs/prepared_belief_search_cuda_tests.xml"),
    ]
    import xml.etree.ElementTree as ET

    suite = ET.parse(paths[-1]).getroot().find("testsuite")
    assert int(suite.attrib["tests"]) > 0
    assert all(int(suite.attrib[key]) == 0 for key in ("failures", "errors", "skipped"))
    paths += (
        list(Path("koikoi").glob("*.py"))
        + list(Path("native").rglob("*.h"))
        + list(Path("native").rglob("*.cu"))
    )
    paths += [
        Path("scripts") / name
        for name in (
            "history_conditioned_search.py",
            "published_action_search.py",
            "published_koi_search.py",
            "fixed_microbatch_continuation.py",
            "collect_reusable_paired_targets.py",
            "paired_rollout_targets.py",
        )
    ]
    for path, expected in PARENTS:
        assert sha(path) == expected
        paths.append(Path(path))
    for path in paths:
        digest = sha(path)
        assert str(path) not in inputs or inputs[str(path)] == digest
        inputs[str(path)] = digest
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "protocol.json",
        {
            "inputs": inputs,
            "kinds": KINDS,
            "particles": 128,
            "confirmation_particles": 512,
            "confidence": 2.5,
            "root_seed": "102100001+10000*parent+case_index",
            "attempts": 32,
            "eligible_roots": 25,
            "ineligible_attempts": 7,
            "scope": "Old development roots; conditional completions, no strength decision.",
            "rollout_execution": "fixed64 CUDA, TF32 off; strict CPU fallback and actual completion",
        },
    )
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    agents = {
        p: PreparedBeliefSearch(
            path, particles=128, confirmation_particles=512, device="cuda"
        )
        for p, (path, _) in enumerate(PARENTS, 1)
    }
    choices = []
    try:
        for row, env, batch in load_roots():
            agent = agents[row["parent"]]
            conditions = {}
            base = int(agent.parent.act(env))
            for kind in KINDS:
                agent.reset_seed(102100001 + 10000 * row["parent"] + row["case_index"])
                agent.prepare(
                    env,
                    batch,
                    kind,
                    row["public_identification"],
                    row["decision"],
                    row["seat"],
                )
                start = time.perf_counter()
                action = int(agent.act(env))
                conditions[kind] = {
                    "action": action,
                    "decision": agent.last_decision,
                    "sampling": agent.last_sampling,
                    "context": agent.last_context,
                    "seconds": time.perf_counter() - start,
                }
            choice = {
                key: row[key]
                for key in (
                    "parent",
                    "case_index",
                    "opponent",
                    "seed",
                    "decision",
                    "seat",
                )
            }
            choice.update(base=base, conditions=conditions)
            choices.append(choice)
            write(output / f"choice_{row['parent']}_{row['case_index']}.json", choice)
            write(
                output / "process.json",
                {"status": "searching", "completed_roots": len(choices)},
            )
            print(json.dumps({"completed_roots": len(choices)}), flush=True)
        write(output / "choices.json", choices)
        frozen_choices = sha(output / "choices.json")
        # Only after all choices are frozen, assess against actual hidden worlds.
        opponents = {
            name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")
        }
        outcomes = []
        for (row, env, _), choice in zip(load_roots(), choices):
            actions = sorted(
                {choice["base"]} | {v["action"] for v in choice["conditions"].values()}
            )
            completions = {
                str(action): complete(
                    env,
                    action,
                    agents[row["parent"]].parent,
                    opponents[row["opponent"]],
                    row["seat"],
                )
                for action in actions
            }
            outcomes.append(
                {
                    "parent": row["parent"],
                    "case_index": row["case_index"],
                    "completions": completions,
                }
            )
        assert sha(output / "choices.json") == frozen_choices
        for path, digest in inputs.items():
            assert sha(path) == digest, path
        write(output / "outcomes.json", outcomes)
        write(
            output / "report.json",
            {
                "status": "complete",
                "inputs": inputs,
                "protocol_sha256": sha(output / "protocol.json"),
                "choices_sha256": frozen_choices,
                "outcomes_sha256": sha(output / "outcomes.json"),
                "source_prefix_replay_decisions": 2
                * sum(c["decision"] for c in choices),
                "new_historical_neural_calls": 0,
                "search_work": {str(p): asdict(a.work) for p, a in agents.items()},
                "confirmation_work": {
                    str(p): asdict(a.confirmation_work) for p, a in agents.items()
                },
                "actual_completion_work": {
                    key: sum(
                        c["work"][key]
                        for r in outcomes
                        for c in r["completions"].values()
                    )
                    for key in ("decisions", "round_endings", "conditional_completions")
                },
                "full_evaluation_matches": 0,
                "new_evaluation_deals": 0,
                "reused_completion_lookups": 125
                - sum(len(r["completions"]) for r in outcomes),
            },
        )
        write(output / "process.json", {"status": "complete"})
    except BaseException as error:
        write(output / "process.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
