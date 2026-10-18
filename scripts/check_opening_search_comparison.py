"""Fixed all-card opening proposals with independent selected-action references."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.env import BatchEnv
from koikoi.layout import PLAYER
from koikoi.reference import ReferenceEnv
from koikoi.train import provenance
from koikoi.upstream import UPSTREAM_PATH, PublishedAgent
from scripts.check_round_search_comparison import collect
from scripts.diagnose_history_support import reference
from scripts.opening_search import OpeningRoundSearch, OpeningTerminalSearch
from scripts.round_value_search import BoundaryTables
from scripts.run_arbitration_study import PARENTS, sha, write


def cases():
    return [
        {
            "parent": p,
            "opponent": name,
            "seat": seat,
            "replicate": replicate,
            "seed": 81110001 + 100 * opponent + replicate,
        }
        for p in (1, 2)
        for opponent, name in enumerate(("wp", "point"))
        for seat in (0, 1)
        for replicate in (0, 1)
    ]


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
    ] + [
        Path(p)
        for p in (
            "scripts/opening_search.py",
            "scripts/check_round_search_comparison.py",
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
            "tests/test_opening_search.py",
            "docs/OPENING_SEARCH.rst",
            "docs/OPENING_SEARCH_COMPARISON.rst",
            "runs/opening_search_tests.xml",
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
            "proposal_particles": 256,
            "confirmation_particles": 1024,
            "reference_particles": 4096,
            "reference_chunk_particles": 512,
            "confidence": 2.5,
            "tables": str(study),
            "scope": "Selected opening-action diagnostic, no strength inference or promotion",
        },
    )
    write(folder / "process.json", {"status": "running", "phase": "source"})
    parents = [NeuralAgent(p, deterministic=True) for p, _ in PARENTS]
    opponents = {name: EndgameAgent(PublishedAgent(name)) for name in ("wp", "point")}
    native = BatchEnv(1, "cpu", "paper8", seed=0)
    roots, source_rows = [], []
    for case in cases():
        initial = ReferenceEnv(case["seed"], "paper8")
        # Reuse its full strict-policy/native trajectory checker; its incidental
        # late-koi root is ignored and never used for this experiment.
        _, row = collect(
            case, parents[case["parent"] - 1], opponents[case["opponent"]], native
        )
        eligible = int(initial.state[PLAYER]) == case["seat"]
        row["root_index"] = len(roots) if eligible else None
        if eligible:
            roots.append(initial.state.copy())
        source_rows.append(row)
    assert len(roots) == 8
    np.save(folder / "roots.npy", np.array(roots, dtype=np.int32))
    write(folder / "source.json", source_rows)
    (folder / "rows.jsonl").touch(exist_ok=False)
    write(folder / "process.json", {"status": "running", "phase": "search", "roots": 8})
    totals, rows = {}, []
    for parent in (1, 2):
        common = {
            "particles": 256,
            "confirmation_particles": 1024,
            "confidence": 2.5,
            "device": "cuda",
        }
        terminal = OpeningTerminalSearch(PARENTS[parent - 1][0], **common)
        round_search = OpeningRoundSearch(
            PARENTS[parent - 1][0], tables[parent - 1], **common
        )
        ref_agent = OpeningTerminalSearch(
            PARENTS[parent - 1][0],
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
                agent.reset_seed(81210001 + case_index)
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
                assert (
                    agent.last_context["status"] == "opening_prior"
                    and not agent.history_work
                )
                assert len(agent.last_decision["alternatives"]) == 7
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
            assert base == round_search.last_decision["base"]
            alternatives = sorted(
                {
                    terminal.last_decision["alternative"],
                    round_search.last_decision["alternative"],
                }
            )
            rng = np.random.default_rng(81310001 + case_index)
            chunks = []
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
            reference_rows = []
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
                reference_rows.append(
                    {
                        "alternative": alternative,
                        "win_change": mean,
                        "se": se,
                        "lower_margin": mean - 2.5 * se,
                    }
                )
            reference_record = {
                "alternatives": alternatives,
                "chunks": chunks,
                "summary": reference_rows,
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
            assert not agent.history_work
            totals[f"parent{parent}_{name}"] = {
                "work": asdict(agent.work),
                "confirmation_work": asdict(agent.confirmation_work),
                "proposed": agent.proposed_decisions,
                "changed": agent.changed_decisions,
                "history_work": agent.history_work,
            }
        del terminal, round_search, ref_agent
    assert len(rows) == 8 and all(sha(p) == digest for p, digest in inputs.items())
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
        "root_count": 8,
        "totals": totals,
        "scope": "All opening proposals and selected-action references retained; no full-match strength result or promotion",
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
