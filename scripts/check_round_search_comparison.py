"""Fixed koi-root comparison of round-value and complete-terminal estimators."""

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent, NeuralAgent, solved_koi_action
from koikoi.env import BatchEnv
from koikoi.layout import KOI, PHASE, PLAYER, POINTS, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.train import provenance
from koikoi.upstream import UPSTREAM_PATH, PublishedAgent
from scripts.diagnose_history_support import reference
from scripts.enumerate_history_support import context
from scripts.history_conditioned_search import HistoryConditionedSearch
from scripts.round_value_search import BoundaryTables, RoundValueSearch
from scripts.run_arbitration_study import PARENTS, sha, write


def cases():
    return [
        {
            "parent": p,
            "opponent": name,
            "seat": seat,
            "replicate": replicate,
            "seed": 72110001 + 100 * opponent + replicate,
        }
        for p in (1, 2)
        for opponent, name in enumerate(("wp", "point"))
        for seat in (0, 1)
        for replicate in (0, 1)
    ]


def collect(case, parent, opponent, native):
    env = ReferenceEnv(case["seed"], "paper8")
    native.state.copy_(torch.from_numpy(env.state[None]))
    root, actions = None, []
    work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    while env.state[PHASE] != TERMINAL:
        if len(actions) >= 1024:
            raise RuntimeError("Source match exceeded its fixed bound")
        ours = int(env.state[PLAYER]) == case["seat"]
        if (
            root is None
            and ours
            and env.state[PHASE] == KOI
            and env.legal().sum() == 2
            and solved_koi_action(env) is None
        ):
            unknown, count, _ = context(env.state)
            if math.comb(len(unknown), count) <= 100:
                root = env.state.copy()
        action = int((parent if ours else opponent).act(env))
        result = env.step(action)
        actual = native.step(torch.tensor([action]))
        np.testing.assert_array_equal(actual[0].numpy(), result)
        np.testing.assert_array_equal(native.state[0].numpy(), env.state)
        actions.append(action)
        work["decisions"] += 1
        work["round_endings"] += int(result[1])
        work["completed_matches"] += int(result[2])
    return root, {
        **case,
        "actions": actions,
        "work": work,
        "terminal_points": env.state[POINTS : POINTS + 2].tolist(),
    }


def run(folder, study):
    if folder.exists() or not torch.cuda.is_available():
        raise ValueError("Use a fresh output directory and CUDA")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # Validate both complete study artifacts before selecting any source roots.
    tables = [BoundaryTables.from_accuracy_study(study, p) for p in (1, 2)]
    files = [
        Path(__file__),
        study / "analysis.json",
        study / "completion_audit.json",
    ] + [
        Path(p)
        for p in (
            "scripts/round_value_search.py",
            "scripts/history_conditioned_search.py",
            "scripts/history_conditioned_worlds.py",
            "scripts/round_transition_values.py",
            "scripts/published_action_search.py",
            "scripts/published_koi_search.py",
            "scripts/collect_reusable_paired_targets.py",
            "scripts/paired_rollout_targets.py",
            "scripts/enumerate_history_support.py",
            "scripts/diagnose_history_support.py",
            "scripts/run_arbitration_study.py",
            "tests/test_round_value_search.py",
            "docs/ROUND_SEARCH_COMPARISON.rst",
            "runs/round_value_search_tests.xml",
        )
    ]
    files += [Path(p) for p, expected in PARENTS if sha(p) == expected]
    assert len(files) == 19
    files += [
        UPSTREAM_PATH / f"model_agent/{phase}_rl_{name}.pt"
        for phase in ("discard", "pick", "koikoi")
        for name in ("point", "wp")
    ]
    inputs = {str(p): sha(p) for p in files}
    source = provenance()
    folder.mkdir(parents=True)
    write(
        folder / "protocol.json",
        {
            "inputs": inputs,
            "provenance": source,
            "cases": cases(),
            "source_matches": 16,
            "source_unique_deals": 4,
            "proposal_particles": 512,
            "confirmation_particles": 1024,
            "reference_particles": 4096,
            "reference_chunks": 8,
            "confidence": 2.5,
            "tables": str(study),
            "scope": "Action-value diagnostic, no strength inference or promotion",
        },
    )
    write(folder / "process.json", {"status": "running", "phase": "source"})
    parents = [NeuralAgent(p, deterministic=True) for p, _ in PARENTS]
    opponents = {name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")}
    native = BatchEnv(1, "cpu", "paper8", seed=0)
    roots, source_rows = [], []
    for case in cases():
        root, row = collect(
            case, parents[case["parent"] - 1], opponents[case["opponent"]], native
        )
        row["root_index"] = len(roots) if root is not None else None
        if root is not None:
            roots.append(root)
        source_rows.append(row)
    np.save(folder / "roots.npy", np.array(roots, dtype=np.int32).reshape(-1, 208))
    write(folder / "source.json", source_rows)
    (folder / "rows.jsonl").touch(exist_ok=False)
    write(
        folder / "process.json",
        {"status": "running", "phase": "search", "roots": len(roots)},
    )
    totals, rows = {}, []
    for parent in (1, 2):
        common = {
            "mode": "history",
            "max_hands": 100,
            "particles": 512,
            "confirmation_particles": 1024,
            "confidence": 2.5,
            "device": "cuda",
        }
        terminal = HistoryConditionedSearch(PARENTS[parent - 1][0], **common)
        round_search = RoundValueSearch(
            PARENTS[parent - 1][0], tables[parent - 1], **common
        )
        ref_agent = HistoryConditionedSearch(
            PARENTS[parent - 1][0],
            mode="history",
            particles=512,
            confirmation_particles=0,
            confidence=2.5,
            device="cuda",
        )
        for case_index, source_row in enumerate(source_rows):
            if source_row["parent"] != parent or source_row["root_index"] is None:
                continue
            root = reference(roots[source_row["root_index"]])
            before = root.state.copy()
            comparisons = {}
            for name, agent in (("terminal", terminal), ("round", round_search)):
                agent.reset_seed(72210001 + case_index)
                torch.cuda.synchronize()
                start = time.perf_counter()
                action = agent.act(root)
                torch.cuda.synchronize()
                comparisons[name] = {
                    "action": int(action),
                    "seconds": time.perf_counter() - start,
                    "context": agent.last_context,
                    "sampling": agent.last_sampling,
                    "decision": agent.last_decision,
                }
                np.testing.assert_array_equal(root.state, before)
                assert root.legal()[action]
            assert terminal.support == round_search.support
            reference_record = None
            if terminal.last_decision is not None:
                assert round_search.last_decision is not None
                for stage in ("proposal", "confirmation"):
                    if (
                        stage in terminal.last_sampling
                        and stage in round_search.last_sampling
                    ):
                        for key in ("worlds_sha256", "model_indices"):
                            assert (
                                terminal.last_sampling[stage][key]
                                == round_search.last_sampling[stage][key]
                            )
                base = terminal.last_decision["base"]
                assert base == round_search.last_decision["base"] and base in (48, 49)
                actual_hand = tuple(
                    np.flatnonzero(root.state[:48] == 1 - root.state[PLAYER])
                )
                # Validation AFTER action selection; truth is never a search input.
                assert terminal.support.compatible[
                    terminal.support.hands.index(actual_hand)
                ][terminal.support.model_names.index(source_row["opponent"])]
                ref_agent.support = terminal.support
                rng = np.random.default_rng(72310001 + case_index)
                chunks = []
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(8):
                    result = ref_agent._compare_actions(
                        root, base, [97 - base], 512, rng
                    )[0]
                    chunks.append(
                        {
                            "sampling": ref_agent.last_sampling["proposal"],
                            "comparison": result,
                        }
                    )
                torch.cuda.synchronize()
                points = np.concatenate(
                    [np.array(c["comparison"]["paired_points"]) for c in chunks]
                )
                assert points.shape == (4096, 2)
                diff = (np.sign(points[:, 1]) - np.sign(points[:, 0])) / 2
                mean, se = float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(4096))
                reference_record = {
                    "chunks": chunks,
                    "win_change": mean,
                    "se": se,
                    "lower_margin": mean - 2.5 * se,
                    "seconds": time.perf_counter() - start,
                }
            row = {
                "case_index": case_index,
                "parent": parent,
                "comparisons": comparisons,
                "reference": reference_record,
            }
            rows.append(row)
            with (folder / "rows.jsonl").open(
                "a", encoding="utf-8", newline="\n"
            ) as out:
                out.write(json.dumps(row) + "\n")
            print(
                json.dumps({"case": case_index, "completed_roots": len(rows)}),
                flush=True,
            )
        for name, agent in (
            ("terminal", terminal),
            ("round", round_search),
            ("reference", ref_agent),
        ):
            totals[f"parent{parent}_{name}"] = {
                "work": asdict(agent.work),
                "confirmation_work": asdict(agent.confirmation_work),
                "history_work": agent.history_work,
                "proposed": agent.proposed_decisions,
                "changed": agent.changed_decisions,
                "refusals": agent.refusals,
            }
        del terminal, round_search, ref_agent
    assert all(sha(p) == digest for p, digest in inputs.items())
    assert provenance()["source_sha256"] == source["source_sha256"]
    result = {
        "status": "passed",
        "inputs": inputs,
        "protocol_sha256": sha(folder / "protocol.json"),
        "source_sha256": sha(folder / "source.json"),
        "roots_sha256": sha(folder / "roots.npy"),
        "rows_sha256": sha(folder / "rows.jsonl"),
        "source_work": {
            k: sum(row["work"][k] for row in source_rows)
            for k in source_rows[0]["work"]
        },
        "root_count": len(roots),
        "totals": totals,
        "scope": "All fixed root estimates retained; no match-strength result or promotion",
    }
    write(folder / "audit.json", result)
    write(
        folder / "process.json",
        {"status": "complete", "audit_sha256": sha(folder / "audit.json")},
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "audit_sha256": sha(folder / "audit.json"),
                "roots": len(roots),
                "totals": totals,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.study)
