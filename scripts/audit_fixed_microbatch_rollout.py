"""Replay fixed64 futures and compare every shared baseline state and action."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.layout import PLAYER
from scripts.audit_round_search_comparison import terminal_comparison
from scripts.diagnose_history_support import reference
from scripts.fixed_microbatch_continuation import FixedMicrobatchContinuation
from scripts.history_conditioned_search import HistoryConditionedSearch
from scripts.history_conditioned_worlds import HistorySupport, sample_joint_worlds
from scripts.run_arbitration_study import PARENTS, sha, write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run():
    folder = Path("runs/fixed_microbatch_rollout")
    output = folder / "trace_audit.json"
    if output.exists():
        raise ValueError("Preserve previous trace audit")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    report = read(folder / "audit.json")
    assert report["status"] == "passed"
    assert all(sha(p) == h for p, h in report["inputs"].items())
    assert sha(folder / "chunks.json") == report["chunks_sha256"]
    chunks = read(folder / "chunks.json")
    assert len(chunks) == 8
    source = Path("runs/midround_search_comparison")
    sources = read(source / "source.json")
    root = reference(
        np.load(source / "roots.npy", allow_pickle=False)[sources[9]["root_index"]]
    )
    raw = read(source / "support_9.json")
    for key in ("hands", "compatible"):
        raw[key] = tuple(tuple(r) for r in raw[key])
    raw["model_names"] = tuple(raw["model_names"])
    support = HistorySupport(**raw)
    agent = HistoryConditionedSearch(
        PARENTS[1][0], particles=512, confirmation_particles=0, device="cuda"
    )
    fixed = FixedMicrobatchContinuation(agent.continuation.actors[2], "cuda", 64)
    rng = np.random.default_rng(83310010)
    totals = dict.fromkeys(report["work"], 0)
    checks = 0
    for chunk_index, chunk in enumerate(chunks):
        worlds, labels = sample_joint_worlds(root, support, rng, 512)
        assert (
            hashlib.sha256(worlds.tobytes()).hexdigest()
            == chunk["sampling"]["worlds_sha256"]
        )
        assert labels.tolist() == chunk["sampling"]["model_indices"]
        states = torch.from_numpy(worlds).to("cuda")
        trace, final_baseline = [], [None]
        for j, record in enumerate(chunk["comparisons"]):
            fixed.prepare(int(root.state[PLAYER]), labels)
            cursor, local_checks = [0], [0]

            def checked(
                env,
                j=j,
                cursor=cursor,
                trace=trace,
                final_baseline=final_baseline,
                local_checks=local_checks,
            ):
                baseline = env.state[::2].cpu().numpy().copy()
                actions = fixed(env)
                base_actions = actions[::2].cpu().numpy().copy()
                i = cursor[0]
                cursor[0] += 1
                if j == 0:
                    trace.append((baseline, base_actions))
                else:
                    if i < len(trace):
                        expected_state, expected_actions = trace[i]
                    else:
                        expected_state = final_baseline[0]
                        expected_actions = np.full(512, -1, dtype=np.int64)
                    np.testing.assert_array_equal(baseline, expected_state)
                    np.testing.assert_array_equal(base_actions, expected_actions)
                    local_checks[0] += 512
                return actions

            actions = torch.tensor(
                [report["base"], record["alternative"]], device="cuda"
            ).expand(512, 2)
            _, points, work = agent.workspaces[512](states, actions, checked)
            np.testing.assert_array_equal(points.cpu().numpy(), record["paired_points"])
            assert asdict(work) == record["work"]
            terminal_comparison(record, 512)
            if j == 0:
                final_baseline[0] = (
                    agent.workspaces[512].env.state[::2].cpu().numpy().copy()
                )
            else:
                np.testing.assert_array_equal(
                    agent.workspaces[512].env.state[::2].cpu().numpy(),
                    final_baseline[0],
                )
            for k, v in asdict(work).items():
                totals[k] += v
            checks += local_checks[0]
        print(json.dumps({"verified_chunks": chunk_index + 1}), flush=True)
    assert totals == report["work"]
    inputs = report["inputs"].copy()
    inputs.update(
        {
            str(p): sha(p)
            for p in (Path(__file__), folder / "audit.json", folder / "chunks.json")
        }
    )
    assert all(sha(p) == h for p, h in inputs.items())
    write(
        output,
        {
            "status": "passed",
            "inputs": inputs,
            "replayed_work": totals,
            "baseline_state_and_action_checks": checks,
            "new_source_deals": 0,
            "scope": "All saved fixed64 future trajectories rerun; every repeated baseline state/action and endpoint matches, including tails. Same continuation implementation, independent comparison/arithmetic; no strength inference.",
        },
    )
    print(
        json.dumps(
            {"status": "passed", "sha256": sha(output), "baseline_checks": checks}
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
