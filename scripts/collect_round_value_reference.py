"""Collect the fixed 72-context terminal reference under both chance models."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import NeuralAgent
from koikoi.layout import DEALER, PHASE, POINTS, ROUND, TERMINAL
from koikoi.published_actor import PublishedActor
from koikoi.train import provenance
from koikoi.upstream import UPSTREAM_PATH
from scripts.history_conditioned_search import MixtureContinuation
from scripts.round_value_reference import FreshDealBank, TerminalReference
from scripts.run_arbitration_study import PARENTS, sha, write


def collect(folder, parent_index):
    if folder.exists() or parent_index not in (1, 2):
        raise ValueError("Use a fresh folder and a retained parent index")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for the fixed reference")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    checkpoint, expected = PARENTS[parent_index - 1]
    assert sha(checkpoint) == expected
    paths = [Path(checkpoint), Path(__file__)] + [
        Path(p)
        for p in (
            "scripts/round_value_reference.py",
            "scripts/round_transition_values.py",
            "scripts/history_conditioned_search.py",
            "scripts/run_arbitration_study.py",
            "tests/test_round_value_reference.py",
            "docs/ROUND_VALUE_ACCURACY.rst",
        )
    ]
    paths += [
        UPSTREAM_PATH / f"model_agent/{phase}_rl_{name}.pt"
        for phase in ("discard", "pick", "koikoi")
        for name in ("point", "wp")
    ]
    inputs = {str(p): sha(p) for p in paths}
    source = provenance()
    contexts = [[r, p, d] for r in (0, 3, 7) for p in (24, 30, 36) for d in (0, 1)]
    samples, batch, seed = 512, 256, 62110001
    modes = ["native", "independent"]
    protocol = {
        "inputs": inputs,
        "provenance": source,
        "parent": parent_index,
        "contexts": contexts,
        "samples": samples,
        "batch": batch,
        "seed": seed,
        "modes": modes,
        "models": ["point", "wp"],
        "axes": [
            "chance_mode",
            "parent_seat",
            "opponent_model",
            "public_context",
            "sample",
        ],
        "scope": "Artificial predeal terminal paths for continuation-value validation, not ordinary full-match strength evaluation",
    }
    folder.mkdir(parents=True)
    write(folder / "protocol.json", protocol)
    write(folder / "process.json", {"status": "running", "completed_groups": 0})
    parent = NeuralAgent(checkpoint, deterministic=True)
    continuation = MixtureContinuation(
        PublishedActor(parent.model.actor.checkpoint(), "cuda"), "cuda"
    )
    bank = FreshDealBank(seed, samples, "cuda")
    workspace = TerminalReference(batch, "cuda")
    starts = bank.starts(
        np.repeat(contexts, samples, axis=0), np.tile(np.arange(samples), len(contexts))
    )
    sample_indices = torch.arange(samples, device="cuda").repeat(len(contexts))
    shape = (2, 2, 2, len(contexts), samples)
    points = np.empty(shape, np.int32)
    rounds, dealers = np.empty_like(points), np.empty_like(points)
    rows = []
    start = time.perf_counter()
    for mode_index, mode in enumerate(modes):
        for seat in (0, 1):
            for model in (0, 1):
                continuation.prepare(seat, np.full(batch // 2, model, np.int32))
                key = (mode_index, seat, model)
                for offset in range(0, len(starts), batch):
                    endpoints, used = workspace.run(
                        starts[offset : offset + batch],
                        continuation,
                        sample_indices[offset : offset + batch],
                        bank if mode == "independent" else None,
                    )
                    end = endpoints.cpu().numpy()
                    assert (end[:, PHASE] == TERMINAL).all()
                    assert (end[:, POINTS : POINTS + 2].sum(1) == 60).all()
                    for array, values in (
                        (points, end[:, POINTS]),
                        (rounds, end[:, ROUND]),
                        (dealers, end[:, DEALER]),
                    ):
                        array[key].reshape(-1)[offset : offset + batch] = values
                    rows.append(
                        {
                            "mode": mode,
                            "parent_seat": seat,
                            "model": model,
                            "offset": offset,
                            "work": asdict(used),
                        }
                    )
                np.savez_compressed(
                    folder / f"group_{mode}_{seat}_{model}.npz",
                    points0=points[key],
                    terminal_round=rounds[key],
                    dealer=dealers[key],
                )
                write(
                    folder / "progress.json",
                    {"rows": rows, "seconds": time.perf_counter() - start},
                )
                write(
                    folder / "process.json",
                    {
                        "status": "running",
                        "completed_groups": len(rows) // (len(starts) // batch),
                    },
                )
                print(
                    json.dumps(
                        {
                            "parent": parent_index,
                            "mode": mode,
                            "seat": seat,
                            "model": model,
                            "seconds": time.perf_counter() - start,
                        }
                    ),
                    flush=True,
                )
    np.testing.assert_array_equal(points[0, :, :, 12:], points[1, :, :, 12:])
    np.testing.assert_array_equal(rounds[0, :, :, 12:], rounds[1, :, :, 12:])
    np.testing.assert_array_equal(dealers[0, :, :, 12:], dealers[1, :, :, 12:])
    utility = (
        np.sign(points.astype(np.int64) - 30)
        * np.array([1, -1])[None, :, None, None, None]
    )
    np.savez_compressed(
        folder / "outcomes.npz",
        points0=points,
        terminal_round=rounds,
        dealer=dealers,
        signed_utility=utility,
    )
    totals = {k: sum(row["work"][k] for row in rows) for k in rows[0]["work"]}
    assert totals["terminal_paths"] == 73728
    assert all(sha(p) == h for p, h in inputs.items())
    assert provenance()["source_sha256"] == source["source_sha256"]
    result = {
        "status": "passed",
        "inputs": inputs,
        "protocol_sha256": sha(folder / "protocol.json"),
        "outcomes_sha256": sha(folder / "outcomes.npz"),
        "work": totals,
        "rows": rows,
        "seconds": time.perf_counter() - start,
        "independent_complete_matches": 0,
        "policy_updates": 0,
        "scope": protocol["scope"],
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
                "work": totals,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    collect(args.output, args.parent)
