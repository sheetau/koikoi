"""Build sampled round kernels for one fixed parent/opponent policy family.

The table is a policy-evaluation artifact under independently sampled round
deals, not an actor checkpoint or a calibrated strength result. Preserve every
sampled endpoint so table uncertainty can be evaluated separately.
"""

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
from scripts.round_transition_values import (
    RoundWorkspace,
    backward_values,
    fresh_round_states,
)
from scripts.run_arbitration_study import PARENTS, sha, write


def run(folder, parent_index, samples, batch_size, seed):
    if folder.exists():
        raise ValueError("Use a fresh output directory")
    if (
        parent_index not in (1, 2)
        or not 1 <= samples <= 1000000
        or batch_size < 2
        or batch_size % 2
    ):
        raise ValueError("Use positive samples and a positive even batch size")
    if not 0 <= seed < 2**31 - 7000000 - samples:
        raise ValueError("Seed range exceeds the simulator")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for this builder")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    checkpoint, expected = PARENTS[parent_index - 1]
    assert sha(checkpoint) == expected
    files = [
        Path(__file__),
        Path(checkpoint),
        *[
            Path(p)
            for p in (
                "scripts/round_transition_values.py",
                "scripts/history_conditioned_search.py",
                "scripts/run_arbitration_study.py",
                "tests/test_round_transition_values.py",
                "docs/ROUND_TRANSITION_VALUES.rst",
            )
        ],
    ]
    files += [
        UPSTREAM_PATH / f"model_agent/{phase}_rl_{name}.pt"
        for name in ("wp", "point")
        for phase in ("discard", "pick", "koikoi")
    ]
    inputs = {str(p): sha(p) for p in files}
    source = provenance()
    protocol = {
        "inputs": inputs,
        "provenance": source,
        "parent": parent_index,
        "samples_per_context": samples,
        "batch_size": batch_size,
        "seed": seed,
        "axes": [
            "round",
            "parent_seat",
            "opponent_model",
            "points0_minus1",
            "dealer",
            "sample",
        ],
        "opponent_models": ["point", "wp"],
        "deal_seed": "seed + 1000000 * round_index + sample_index; shared across scores/dealers/roles/models",
        "round_model": "independent fresh deals; native finite-seed cross-round independence is not asserted",
        "scope": "Sampled frozen-policy evaluation kernels; no policy learning, calibration guarantee or match-strength evidence",
    }
    folder.mkdir(parents=True)
    write(folder / "protocol.json", protocol)
    write(folder / "process.json", {"status": "running", "completed_rounds": []})
    parent = NeuralAgent(checkpoint, deterministic=True)
    actor = PublishedActor(parent.model.actor.checkpoint(), "cuda")
    continuation = MixtureContinuation(actor, "cuda")
    shape = (8, 2, 2, 59, 2, samples)
    points = np.empty(shape, dtype=np.int32)
    dealers = np.empty(shape, dtype=np.int32)
    terminal = np.empty(shape, dtype=bool)
    next_round = np.empty(shape, dtype=np.int32)
    workspaces, rows = {}, []
    start = time.perf_counter()
    for r in range(7, -1, -1):
        contexts = np.array(
            [[r, p, d] for p in range(1, 60) for d in (0, 1) for _ in range(samples)],
            dtype=np.int32,
        )
        seeds = np.tile(seed + 1000000 * r + np.arange(samples), 118)
        starts = fresh_round_states(contexts, seeds)
        assert len(starts) % 2 == 0
        for player in (0, 1):
            for model in (0, 1):
                used = {"decisions": 0, "round_endings": 0, "terminal_match_endings": 0}
                for offset in range(0, len(starts), batch_size):
                    chunk = starts[offset : offset + batch_size]
                    count = len(chunk)
                    if count not in workspaces:
                        workspaces[count] = RoundWorkspace(count, "cuda")
                    continuation.prepare(
                        player, np.full(count // 2, model, dtype=np.int32)
                    )
                    endpoints, measured = workspaces[count].run(
                        torch.from_numpy(chunk).to("cuda"), continuation
                    )
                    end = endpoints.cpu().numpy()
                    done = end[:, PHASE] == TERMINAL
                    assert (end[:, POINTS : POINTS + 2].sum(1) == 60).all()
                    assert (end[~done, ROUND] == r + 1).all()
                    assert (end[done, ROUND] == r).all()
                    assert np.isin(end[:, DEALER], [0, 1]).all()
                    if r == 7:
                        assert done.all()
                    key = (r, player, model)
                    for array, values in (
                        (points, end[:, POINTS]),
                        (dealers, end[:, DEALER]),
                        (terminal, done),
                        (next_round, end[:, ROUND]),
                    ):
                        array[key].reshape(-1)[offset : offset + count] = values
                    for name, value in asdict(measured).items():
                        used[name] += value
                assert used["round_endings"] == 118 * samples
                rows.append(
                    {"round": r, "parent_seat": player, "model": model, "work": used}
                )
        np.savez_compressed(
            folder / f"round_{r}.npz",
            points0=points[r],
            next_dealer=dealers[r],
            terminal=terminal[r],
            next_round=next_round[r],
        )
        write(
            folder / "progress.json",
            {"rows": rows, "seconds": time.perf_counter() - start},
        )
        write(
            folder / "process.json",
            {"status": "running", "completed_rounds": list(range(7, r - 1, -1))},
        )
        print(
            json.dumps(
                {
                    "parent": parent_index,
                    "completed_round": r,
                    "seconds": time.perf_counter() - start,
                }
            ),
            flush=True,
        )
    values = backward_values(points, dealers, terminal)
    np.savez_compressed(
        folder / "kernels.npz",
        points0=points,
        next_dealer=dealers,
        terminal=terminal,
        next_round=next_round,
        values=values,
    )
    totals = {k: sum(row["work"][k] for row in rows) for k in rows[0]["work"]}
    assert totals["round_endings"] == 3776 * samples
    assert all(sha(p) == h for p, h in inputs.items())
    assert provenance()["source_sha256"] == source["source_sha256"]
    result = {
        "status": "passed",
        "inputs": inputs,
        "protocol_sha256": sha(folder / "protocol.json"),
        "kernels_sha256": sha(folder / "kernels.npz"),
        "work": totals,
        "rows": rows,
        "contexts": 3776,
        "distinct_round_deal_inputs": 8 * samples,
        "independent_complete_matches": 0,
        "policy_updates": 0,
        "seconds": time.perf_counter() - start,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "calibrated": False,
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
                "seconds": result["seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=int, choices=(1, 2), required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run(args.output, args.parent, args.samples, args.batch_size, args.seed)
