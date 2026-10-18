"""Fixed uniform/conditional midround card diagnostic with terminal references."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.train import provenance
from koikoi.upstream import UPSTREAM_PATH, PublishedAgent
from scripts.diagnose_history_support import reference
from scripts.history_conditioned_search import HistoryConditionedSearch
from scripts.midround_search import MidroundRoundSearch, is_midround
from scripts.round_value_search import BoundaryTables
from scripts.run_arbitration_study import PARENTS, sha, write


def cases():
    return [
        {
            "parent": p,
            "opponent": name,
            "seat": seat,
            "replicate": replicate,
            "seed": 83110001 + 100 * opponent + replicate,
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
        assert len(actions) < 1024
        ours = int(env.state[PLAYER]) == case["seat"]
        if ours and is_midround(env):
            assert root is None
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
    tables = [BoundaryTables.from_accuracy_study(study, p) for p in (1, 2)]
    files = [
        Path(__file__),
        study / "analysis.json",
        study / "independent_accuracy_audit.json",
    ]
    files += [
        Path(p)
        for p in (
            "scripts/pruned_history_support.py",
            "scripts/midround_search.py",
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
            "docs/MIDROUND_SEARCH.rst",
            "tests/test_pruned_history_support.py",
            "tests/test_midround_search.py",
            "runs/pruned_history_tests.xml",
            "runs/midround_tests.xml",
        )
    ]
    for path, expected in PARENTS:
        assert sha(path) == expected
        files.append(Path(path))
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
            "proposal_particles": 1024,
            "confirmation_particles": 2048,
            "reference_particles": 4096,
            "reference_chunk_particles": 512,
            "confidence": 2.5,
            "tables": str(study),
            "scope": "Conditional card-value diagnostic, no strength promotion",
        },
    )
    write(folder / "process.json", {"status": "running", "phase": "source"})
    parents = [NeuralAgent(p, deterministic=True) for p, _ in PARENTS]
    opponents = {name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")}
    native = BatchEnv(1, "cpu", "paper8", seed=0)
    roots, sources = [], []
    for case in cases():
        root, row = collect(
            case, parents[case["parent"] - 1], opponents[case["opponent"]], native
        )
        row["root_index"] = len(roots) if root is not None else None
        if root is not None:
            roots.append(root)
        sources.append(row)
    np.save(folder / "roots.npy", np.array(roots, dtype=np.int32).reshape(-1, 208))
    write(folder / "source.json", sources)
    (folder / "rows.jsonl").touch(exist_ok=False)
    write(
        folder / "process.json",
        {"status": "running", "phase": "search", "roots": len(roots)},
    )
    totals, rows = {}, []
    for parent in (1, 2):
        common = {
            "particles": 1024,
            "confirmation_particles": 2048,
            "confidence": 2.5,
            "device": "cuda",
        }
        agents = {
            mode: MidroundRoundSearch(
                PARENTS[parent - 1][0], tables[parent - 1], mode=mode, **common
            )
            for mode in ("uniform", "history")
        }
        ref_agent = HistoryConditionedSearch(
            PARENTS[parent - 1][0],
            mode="history",
            particles=512,
            confirmation_particles=0,
            confidence=2.5,
            device="cuda",
        )
        for case_index, source_row in enumerate(sources):
            if source_row["parent"] != parent or source_row["root_index"] is None:
                continue
            root = reference(roots[source_row["root_index"]])
            before, comparisons = root.state.copy(), {}
            for name, agent in agents.items():
                agent.reset_seed(83210001 + case_index)
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
                assert root.legal()[action] and agent.last_decision is not None
                assert agent.last_context["status"] == "searched"
                assert len(agent.last_decision["alternatives"]) == 3
            support = agents["history"].support
            write(folder / f"support_{case_index}.json", asdict(support))
            ref_agent.support = support
            base = agents["history"].last_decision["base"]
            assert base == agents["uniform"].last_decision["base"]
            alternatives = [int(a) for a in np.flatnonzero(root.legal()) if a != base]
            rng, chunks = np.random.default_rng(83310001 + case_index), []
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(8):
                records = ref_agent._compare_actions(root, base, alternatives, 512, rng)
                chunks.append(
                    {
                        "sampling": ref_agent.last_sampling["proposal"],
                        "comparisons": records,
                    }
                )
            torch.cuda.synchronize()
            summaries = []
            for alternative in alternatives:
                points = np.concatenate(
                    [
                        np.array(
                            next(
                                r
                                for r in c["comparisons"]
                                if r["alternative"] == alternative
                            )["paired_points"]
                        )
                        for c in chunks
                    ]
                )
                assert points.shape == (4096, 2)
                diff = (np.sign(points[:, 1]) - np.sign(points[:, 0])) / 2
                mean, se = float(diff.mean()), float(diff.std(ddof=1) / 64)
                summaries.append(
                    {
                        "alternative": alternative,
                        "win_change": mean,
                        "se": se,
                        "lower_margin": mean - 2.5 * se,
                    }
                )
            row = {
                "case_index": case_index,
                "parent": parent,
                "comparisons": comparisons,
                "support_sha256": sha(folder / f"support_{case_index}.json"),
                "reference": {
                    "alternatives": alternatives,
                    "chunks": chunks,
                    "summary": summaries,
                    "seconds": time.perf_counter() - start,
                },
            }
            rows.append(row)
            with (folder / "rows.jsonl").open(
                "a", encoding="utf-8", newline="\n"
            ) as out:
                out.write(json.dumps(row) + "\n")
            print(
                json.dumps(
                    {
                        "case": case_index,
                        "completed_roots": len(rows),
                        "changed": {
                            m: c["decision"]["changed"] for m, c in comparisons.items()
                        },
                    }
                ),
                flush=True,
            )
        for name, agent in (*agents.items(), ("reference", ref_agent)):
            totals[f"parent{parent}_{name}"] = {
                "work": asdict(agent.work),
                "confirmation_work": asdict(agent.confirmation_work),
                "proposed": agent.proposed_decisions,
                "changed": agent.changed_decisions,
                "history_work": agent.history_work,
            }
        del agents, ref_agent
    assert len(rows) == len(roots) and all(sha(p) == h for p, h in inputs.items())
    assert provenance()["source_sha256"] == source["source_sha256"]
    result = {
        "status": "passed",
        "inputs": inputs,
        "protocol_sha256": sha(folder / "protocol.json"),
        "source_sha256": sha(folder / "source.json"),
        "roots_sha256": sha(folder / "roots.npy"),
        "rows_sha256": sha(folder / "rows.jsonl"),
        "root_count": len(roots),
        "totals": totals,
        "source_work": {
            k: sum(r["work"][k] for r in sources) for k in sources[0]["work"]
        },
        "scope": "All alternative conditional references retained; no actual search match evaluation",
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
