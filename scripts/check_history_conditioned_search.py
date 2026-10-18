"""Fixed eight-source-match capability/cost diagnostic, not a strength test."""

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
from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.train import provenance
from koikoi.upstream import UPSTREAM_PATH, PublishedAgent
from scripts.check_published_action_search import audit_rows
from scripts.diagnose_history_support import reference
from scripts.enumerate_history_support import context
from scripts.history_conditioned_search import HistoryConditionedSearch
from scripts.run_arbitration_study import PARENTS, sha, write


def cases():
    return [
        {
            "parent": i,
            "opponent": opponent,
            "seat": seat,
            "seed": 31210001 + 100 * i + j,
        }
        for i in (1, 2)
        for j, opponent in enumerate(("wp", "point"))
        for seat in (0, 1)
    ]


def collect(case, parent, opponent, native):
    env = ReferenceEnv(case["seed"], "paper8")
    native.state.copy_(torch.from_numpy(env.state[None]))
    root, actions = None, []
    work = {"decisions": 0, "round_endings": 0, "completed_matches": 0}
    while env.state[PHASE] != TERMINAL:
        if len(actions) >= 1024:
            raise RuntimeError("Source match exceeded fixed bound")
        ours = int(env.state[PLAYER]) == case["seat"]
        if (
            root is None
            and ours
            and env.legal().sum() > 1
            and solved_koi_action(env) is None
        ):
            unknown, count, _ = context(env.state)
            if math.comb(len(unknown), count) <= 100:
                root = env.state.copy()
        action = int((parent if ours else opponent).act(env))
        result = env.step(action)
        actual = native.step(torch.tensor([action], dtype=torch.int64))
        np.testing.assert_array_equal(native.state[0].numpy(), env.state)
        np.testing.assert_array_equal(actual[0].numpy(), np.asarray(result))
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


def run(folder):
    if folder.exists():
        raise ValueError("Use a fresh diagnostic output directory")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    files = [
        Path(__file__),
        *[
            Path(p)
            for p in (
                "scripts/history_conditioned_search.py",
                "scripts/history_conditioned_worlds.py",
                "scripts/published_action_search.py",
                "scripts/published_koi_search.py",
                "scripts/collect_reusable_paired_targets.py",
                "scripts/paired_rollout_targets.py",
                "scripts/enumerate_history_support.py",
                "scripts/diagnose_history_support.py",
                "scripts/check_published_action_search.py",
                "scripts/run_arbitration_study.py",
                "tests/test_history_conditioned_search.py",
                "tests/test_history_conditioned_worlds.py",
                "docs/HISTORY_CONDITIONED_SEARCH_ADAPTER.rst",
                "runs/history_conditioned_search_tests.xml",
            )
        ],
    ]
    for checkpoint, digest in PARENTS:
        assert sha(checkpoint) == digest
        files.append(Path(checkpoint))
    files += [
        UPSTREAM_PATH / f"model_agent/{phase}_rl_{name}.pt"
        for name in ("wp", "point")
        for phase in ("discard", "pick", "koikoi")
    ]
    inputs = {str(p): sha(p) for p in files}
    source = provenance()
    folder.mkdir(parents=True)
    protocol = {
        "inputs": inputs,
        "provenance": source,
        "cases": cases(),
        "source_matches": 8,
        "source_unique_deal_streams": 4,
        "selection": "First unforced eligible parent decision per fixed match; retain missing roots without replacement",
        "modes": ["uniform", "history"],
        "max_hands": 100,
        "particles": 64,
        "confirmation_particles": 256,
        "confidence": 2.5,
        "search_seed": "31220001 + case_index, reset identically for both modes",
        "scope": "Capability and cost only; no training, checkpoint selection, match-strength inference or final-test use",
    }
    write(folder / "protocol.json", protocol)
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
    np.save(folder / "roots.npy", np.asarray(roots, dtype=np.int32).reshape(-1, 208))
    write(folder / "source.json", source_rows)
    write(
        folder / "process.json",
        {"status": "running", "phase": "search", "roots": len(roots)},
    )
    (folder / "rows.jsonl").touch(exist_ok=False)
    rows, totals = [], {}
    for parent_index in (1, 2):
        for mode in ("uniform", "history"):
            agent = HistoryConditionedSearch(
                PARENTS[parent_index - 1][0],
                mode=mode,
                particles=64,
                confirmation_particles=256,
                confidence=2.5,
                device="cuda",
            )
            group = []
            for case_index, row in enumerate(source_rows):
                if row["parent"] != parent_index or row["root_index"] is None:
                    continue
                root = reference(roots[row["root_index"]])
                before = root.state.copy()
                agent.reset_seed(31220001 + case_index)
                torch.cuda.synchronize()
                start = time.perf_counter()
                action = agent.act(root)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                np.testing.assert_array_equal(root.state, before)
                assert root.legal()[action]
                if mode == "history":
                    # Post-action validation only; truth is not an inference input.
                    actual_hand = tuple(
                        np.flatnonzero(root.state[:48] == 1 - root.state[PLAYER])
                    )
                    assert agent.support.compatible[
                        agent.support.hands.index(actual_hand)
                    ][agent.support.model_names.index(row["opponent"])]
                record = {
                    "case_index": case_index,
                    "parent": parent_index,
                    "mode": mode,
                    "action": int(action),
                    "seconds": elapsed,
                    "context": agent.last_context,
                    "sampling": agent.last_sampling,
                    "decision": agent.last_decision,
                }
                rows.append(record)
                group.append(record)
                with (folder / "rows.jsonl").open(
                    "a", encoding="utf-8", newline="\n"
                ) as out:
                    out.write(json.dumps(record) + "\n")
                print(
                    json.dumps(
                        {"case": case_index, "mode": mode, "completed": len(rows)}
                    ),
                    flush=True,
                )
            checked = audit_rows(
                [row for row in group if row["decision"] is not None], agent.confidence
            )
            assert checked["work"] == asdict(agent.work)
            assert checked["confirmation_work"] == asdict(agent.confirmation_work)
            assert checked["proposed_roots"] == agent.proposed_decisions
            assert checked["changed_roots"] == agent.changed_decisions
            totals[f"parent{parent_index}_{mode}"] = {
                **checked,
                "history_work": agent.history_work,
                "refusals": agent.refusals,
                "roots": len(group),
                "seconds": sum(r["seconds"] for r in group),
            }
            del agent
    assert all(sha(p) == h for p, h in inputs.items())
    assert provenance()["source_sha256"] == source["source_sha256"]
    result = {
        "status": "passed",
        "inputs": inputs,
        "protocol_sha256": sha(folder / "protocol.json"),
        "source_sha256": sha(folder / "source.json"),
        "roots_sha256": sha(folder / "roots.npy"),
        "rows_sha256": sha(folder / "rows.jsonl"),
        "cases": totals,
        "source_work": {
            k: sum(r["work"][k] for r in source_rows) for k in source_rows[0]["work"]
        },
        "source_unique_deal_streams": 4,
        "roots": len(roots),
        "scope": protocol["scope"],
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
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
                "cases": totals,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
