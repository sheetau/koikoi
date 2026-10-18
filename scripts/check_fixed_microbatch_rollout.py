"""Complete the failed case's fixed particles under the identified fixed64 policy."""

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.layout import PLAYER
from scripts.audit_round_search_comparison import terminal_comparison
from scripts.diagnose_history_support import reference
from scripts.fixed_microbatch_continuation import FixedMicrobatchContinuation
from scripts.history_conditioned_search import HistoryConditionedSearch
from scripts.history_conditioned_worlds import HistorySupport
from scripts.run_arbitration_study import PARENTS, sha, write


def run():
    folder = Path("runs/fixed_microbatch_rollout")
    if folder.exists():
        raise ValueError("Preserve previous diagnostic")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    source = Path("runs/midround_search_comparison")
    sources = json.loads((source / "source.json").read_text(encoding="utf-8"))
    root = reference(
        np.load(source / "roots.npy", allow_pickle=False)[sources[9]["root_index"]]
    )
    raw = json.loads((source / "support_9.json").read_text(encoding="utf-8"))
    for key in ("hands", "compatible"):
        raw[key] = tuple(tuple(r) for r in raw[key])
    raw["model_names"] = tuple(raw["model_names"])
    support = HistorySupport(**raw)
    inputs = {
        str(p): sha(p)
        for p in (
            Path(__file__),
            Path("scripts/fixed_microbatch_continuation.py"),
            Path("scripts/history_conditioned_search.py"),
            Path("scripts/history_conditioned_worlds.py"),
            Path("scripts/collect_reusable_paired_targets.py"),
            Path("scripts/audit_round_search_comparison.py"),
            Path("tests/test_fixed_microbatch_continuation.py"),
            Path("runs/fixed_microbatch_tests.xml"),
            Path(PARENTS[1][0]),
            source / "source.json",
            source / "roots.npy",
            source / "support_9.json",
        )
    }
    folder.mkdir()
    write(
        folder / "protocol.json",
        {
            "inputs": inputs,
            "case": 9,
            "chunk_particles": 512,
            "chunks": 8,
            "microbatch": 64,
            "seed": 83310010,
            "scope": "Execution consistency diagnostic on archived worlds; different numerical continuation, no strength inference",
        },
    )
    agent = HistoryConditionedSearch(
        PARENTS[1][0], particles=512, confirmation_particles=0, device="cuda"
    )
    agent.continuation = FixedMicrobatchContinuation(
        agent.continuation.actors[2], "cuda", 64
    )
    agent.support = support
    base = int(agent.parent.act(root))
    alternatives = [int(a) for a in np.flatnonzero(root.legal()) if a != base]
    rng, chunks = np.random.default_rng(83310010), []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(8):
        records = agent._compare_actions(root, base, alternatives, 512, rng)
        for record in records:
            terminal_comparison(record, 512)
        chunks.append(
            {"sampling": agent.last_sampling["proposal"], "comparisons": records}
        )
        write(folder / "chunks.json", chunks)
        print(json.dumps({"completed_chunks": i + 1}), flush=True)
    torch.cuda.synchronize()
    assert all(sha(p) == h for p, h in inputs.items())
    write(
        folder / "audit.json",
        {
            "status": "passed",
            "inputs": inputs,
            "protocol_sha256": sha(folder / "protocol.json"),
            "chunks_sha256": sha(folder / "chunks.json"),
            "root_player": int(root.state[PLAYER]),
            "base": base,
            "alternatives": alternatives,
            "work": asdict(agent.work),
            "seconds": time.perf_counter() - start,
            "common_baseline_checks": 16,
            "scope": "All24 paired comparisons complete; same baseline endpoints across alternatives and raw terminal arithmetic checked. No change to old experiment or policy promotion.",
        },
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "audit_sha256": sha(folder / "audit.json"),
                "work": asdict(agent.work),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
