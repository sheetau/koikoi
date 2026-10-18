"""Prospective natural final-round search and one-intervention actual pilot."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.env import BatchEnv
from koikoi.layout import PHASE
from koikoi.train import provenance
from koikoi.upstream import UPSTREAM_PATH, PublishedAgent
from scripts.diagnose_history_support import reference
from scripts.final_round_actual_match import play_final_round_match
from scripts.final_round_search import FinalRoundSearch
from scripts.run_arbitration_study import PARENTS, sha, write


def cases():
    return [
        {
            "parent": parent,
            "opponent": name,
            "seat": seat,
            "replicate": replicate,
            "seed": 87110001 + 100 * opponent + replicate,
        }
        for parent in (1, 2)
        for opponent, name in enumerate(("wp", "point"))
        for seat in (0, 1)
        for replicate in range(16)
    ]


def first_per_phase(roots):
    """Input contains eligible (decision index, state) pairs in play order."""
    result = {}
    previous = -1
    for index, state in roots:
        assert index > previous
        previous = index
        phase = int(state[PHASE])
        assert phase in (0, 1, 2)
        if phase not in result:
            result[phase] = (index, state)
    return result


def bindings():
    # Include established capability receipts and the closed pilot that motivates
    # this protocol; every direct local search dependency is pinned explicitly.
    files = [Path(__file__)] + [
        Path(p)
        for p in (
            "scripts/final_round_search.py",
            "scripts/final_round_actual_match.py",
            "scripts/audit_final_round_search.py",
            "scripts/audit_round_search_comparison.py",
            "scripts/fixed_microbatch_continuation.py",
            "scripts/history_conditioned_search.py",
            "scripts/history_conditioned_worlds.py",
            "scripts/published_action_search.py",
            "scripts/published_koi_search.py",
            "scripts/collect_reusable_paired_targets.py",
            "scripts/paired_rollout_targets.py",
            "scripts/enumerate_history_support.py",
            "scripts/diagnose_history_support.py",
            "scripts/run_arbitration_study.py",
            "docs/FINAL_ROUND_STUDY.rst",
            "tests/test_final_round_search.py",
            "tests/test_final_round_search_cuda.py",
            "tests/test_final_round_actual_match.py",
            "tests/test_final_round_protocol.py",
            "runs/final_round_cpu_tests_v2.xml",
            "runs/final_round_cuda_tests.xml",
            "runs/final_round_actual_tests_v2.xml",
            "runs/final_round_protocol_tests.xml",
            "runs/fixed_midround_terminal_completion.json",
            "runs/fixed_midround_terminal_study/summary.json",
            "runs/fixed_midround_terminal_study/independent_audit.json",
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
    return {str(p): sha(p) for p in files}


def run(folder):
    if folder.exists() or not torch.cuda.is_available():
        raise ValueError("Use a fresh output directory and CUDA")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    inputs, runtime = bindings(), provenance()
    folder.mkdir(parents=True)
    write(
        folder / "protocol.json",
        {
            "inputs": inputs,
            "provenance": runtime,
            "cases": cases(),
            "source_matches": 128,
            "source_unique_deals": 32,
            "candidate_matches": 384,
            "phases": [0, 1, 2],
            "root_selection": "first unforced learner card per phase in round8",
            "proposal_particles": 1024,
            "confirmation_particles": 2048,
            "reference_particles": 4096,
            "reference_chunk_particles": 512,
            "confidence": 2.5,
            "microbatch": 64,
            "bootstrap_samples": 10000,
            "bootstrap_seed": 87410001,
            "scope": "Uniform approximate final-round pilot; no model promotion",
        },
    )
    write(folder / "process.json", {"status": "running", "phase": "source"})
    parents = [NeuralAgent(p, deterministic=True) for p, _ in PARENTS]
    opponents = {name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")}
    native = BatchEnv(1, "cpu", "paper8", seed=0)
    roots, sources, metadata = [], [], []
    for ci, case in enumerate(cases()):
        found, row = play_final_round_match(
            case, parents[case["parent"] - 1], opponents[case["opponent"]], native
        )
        first = first_per_phase(found)
        row["root_indices"] = {}
        for phase in range(3):
            ri = len(roots) if phase in first else None
            row["root_indices"][str(phase)] = ri
            if ri is not None:
                index, state = first[phase]
                roots.append(state)
                metadata.append(
                    {
                        "root_index": ri,
                        "case_index": ci,
                        "phase": phase,
                        "decision_index": index,
                    }
                )
        sources.append(row)
    np.save(folder / "roots.npy", np.array(roots, dtype=np.int32).reshape(-1, 208))
    write(folder / "root_metadata.json", metadata)
    write(folder / "source.json", sources)
    (folder / "rows.jsonl").touch(exist_ok=False)
    write(
        folder / "process.json",
        {"status": "running", "phase": "search", "roots": len(roots)},
    )
    print(json.dumps({"source_matches": len(sources), "roots": len(roots)}), flush=True)
    totals, rows = {}, []
    for parent in (1, 2):
        agent = FinalRoundSearch(
            PARENTS[parent - 1][0],
            particles=1024,
            confirmation_particles=2048,
            confidence=2.5,
            device="cuda",
        )
        ref_agent = FinalRoundSearch(
            PARENTS[parent - 1][0],
            particles=512,
            confirmation_particles=0,
            confidence=2.5,
            device="cuda",
        )
        for meta in metadata:
            ci, phase, ri = (meta[k] for k in ("case_index", "phase", "root_index"))
            if sources[ci]["parent"] != parent:
                continue
            root = reference(roots[ri])
            before = root.state.copy()
            offset = 3 * ci + phase
            agent.reset_seed(87210001 + offset)
            torch.cuda.synchronize()
            start = time.perf_counter()
            action = agent.act(root)
            torch.cuda.synchronize()
            comparison = {
                "action": int(action),
                "seconds": time.perf_counter() - start,
                "context": agent.last_context,
                "sampling": agent.last_sampling,
                "decision": agent.last_decision,
            }
            np.testing.assert_array_equal(root.state, before)
            assert root.legal()[action] and agent.last_decision is not None
            base = agent.last_decision["base"]
            assert base == sources[ci]["actions"][meta["decision_index"]]
            alternatives = [int(a) for a in np.flatnonzero(root.legal()) if a != base]
            write(folder / f"search_{ri}.json", comparison)
            rng, chunks = np.random.default_rng(87310001 + offset), []
            torch.cuda.synchronize()
            start = time.perf_counter()
            for chunk_index in range(8):
                records = ref_agent._compare_actions(root, base, alternatives, 512, rng)
                assert all(r["work"]["round_endings"] == 1024 for r in records)
                chunks.append(
                    {
                        "sampling": ref_agent.last_sampling["proposal"],
                        "comparisons": records,
                    }
                )
                write(folder / f"reference_{ri}_{chunk_index}.json", chunks[-1])
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
                **meta,
                "parent": parent,
                "comparison": comparison,
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
                        "completed_roots": len(rows),
                        "root_index": ri,
                        "changed": agent.last_decision["changed"],
                    }
                ),
                flush=True,
            )
        for name, obj in (("search", agent), ("reference", ref_agent)):
            totals[f"parent{parent}_{name}"] = {
                "work": asdict(obj.work),
                "confirmation_work": asdict(obj.confirmation_work),
                "proposed": obj.proposed_decisions,
                "changed": obj.changed_decisions,
            }
        del agent, ref_agent
    write(folder / "process.json", {"status": "running", "phase": "actual_pilot"})
    by_root = {r["root_index"]: r for r in rows}
    pilot = []
    for ci, (case, source) in enumerate(zip(cases(), sources)):
        for phase in range(3):
            ri = source["root_indices"][str(phase)]
            intervention = (
                None
                if ri is None
                else (
                    metadata[ri]["decision_index"],
                    roots[ri],
                    by_root[ri]["comparison"]["action"],
                )
            )
            _, result = play_final_round_match(
                case,
                parents[case["parent"] - 1],
                opponents[case["opponent"]],
                native,
                intervention=intervention,
            )
            pilot.append({"case_index": ci, "candidate_phase": phase, **result})
    assert len(pilot) == 384 and len(rows) == len(roots)
    write(folder / "actual_pilot.json", pilot)
    assert all(sha(p) == h for p, h in inputs.items())
    assert provenance()["source_sha256"] == runtime["source_sha256"]
    result = {
        "status": "passed",
        "inputs": inputs,
        "artifacts": {
            name: sha(folder / name)
            for name in (
                "protocol.json",
                "source.json",
                "roots.npy",
                "root_metadata.json",
                "rows.jsonl",
                "actual_pilot.json",
            )
        },
        "root_count": len(roots),
        "totals": totals,
        "source_work": {
            k: sum(r["work"][k] for r in sources) for k in sources[0]["work"]
        },
        "actual_pilot_work": {
            k: sum(r["work"][k] for r in pilot) for k in pilot[0]["work"]
        },
        "scope": "32 distinct deals; all128 sources and384 candidates retained; no strength claim",
    }
    write(folder / "audit.json", result)
    write(
        folder / "process.json",
        {"status": "complete", "audit_sha256": sha(folder / "audit.json")},
    )
    print(
        json.dumps({"status": "passed", "audit_sha256": sha(folder / "audit.json")}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fresh = not args.output.exists()
    try:
        run(args.output)
    except Exception as error:
        if fresh and args.output.exists():
            write(
                args.output / "process.json",
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        raise
