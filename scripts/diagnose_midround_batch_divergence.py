"""Reproduce the failed case9 baseline without weakening the original assertion."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koikoi.layout import PHASE, PLAYER
from scripts.diagnose_history_support import reference
from scripts.history_conditioned_search import HistoryConditionedSearch
from scripts.history_conditioned_worlds import HistorySupport, sample_joint_worlds
from scripts.run_arbitration_study import PARENTS, sha, write


class Divergence(Exception):
    pass


def run():
    folder = Path("runs/midround_batch_divergence")
    if folder.exists():
        raise ValueError("Use fresh diagnostic output")
    folder.mkdir()
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
    agent = HistoryConditionedSearch(
        PARENTS[1][0], particles=512, confirmation_particles=0, device="cuda"
    )
    base = int(agent.parent.act(root))
    alternatives = [int(a) for a in np.flatnonzero(root.legal()) if a != base]
    rng = np.random.default_rng(83310010)
    inputs = {
        str(p): sha(p)
        for p in (
            Path(__file__),
            source / "source.json",
            source / "roots.npy",
            source / "support_9.json",
            Path("scripts/history_conditioned_search.py"),
            Path(PARENTS[1][0]),
        )
    }
    write(
        folder / "protocol.json",
        {
            "inputs": inputs,
            "case": 9,
            "chunks": 8,
            "particles": 512,
            "base": base,
            "alternatives": alternatives,
            "scope": "Reproduction of failed historical worlds, no new source deals or strength result",
        },
    )
    totals, counter = [], 0
    for chunk in range(8):
        worlds, labels = sample_joint_worlds(root, support, rng, 512)
        states = torch.from_numpy(worlds).to("cuda")
        trace, first_points = [], None
        for alternative in alternatives:
            agent.continuation.prepare(int(root.state[PLAYER]), labels)
            cursor = [0]

            def checked(
                env, cursor=cursor, trace=trace, alternative=alternative, chunk=chunk
            ):
                state = env.state.cpu().numpy().copy()
                action = agent.continuation(env)
                actions = action.cpu().numpy().copy()
                i = cursor[0]
                cursor[0] += 1
                if alternative == alternatives[0]:
                    trace.append((state, actions))
                elif i < len(trace):
                    original, original_actions = trace[i]
                    # Baseline states must agree up to their first changed action.
                    np.testing.assert_array_equal(state[::2], original[::2])
                    changed = np.flatnonzero(actions[::2] != original_actions[::2])
                    if len(changed):
                        np.savez_compressed(
                            folder / "divergence.npz",
                            first_states=original,
                            other_states=state,
                            first_actions=original_actions,
                            other_actions=actions,
                            labels=labels,
                            worlds=worlds,
                        )
                        details = {
                            "status": "reproduced",
                            "chunk": chunk,
                            "continuation_step": i,
                            "first_alternative": alternatives[0],
                            "other_alternative": alternative,
                            "root_player": int(root.state[PLAYER]),
                            "particles": changed.tolist(),
                            "first_actions": original_actions[2 * changed].tolist(),
                            "other_actions": actions[2 * changed].tolist(),
                            "phases": state[2 * changed, PHASE].tolist(),
                            "players": state[2 * changed, PLAYER].tolist(),
                            "complete_previous_branch_work": totals,
                            "saved_tensor_sha256": sha(folder / "divergence.npz"),
                            "partial_current_continuation_calls": i + 1,
                            "scope": "Identical baseline state differs in next action under another branch's batch; partial work is explicit",
                        }
                        write(folder / "audit.json", details)
                        print(json.dumps(details), flush=True)
                        raise Divergence()
                return action

            proposals = torch.tensor([base, alternative], device="cuda").expand(512, 2)
            try:
                _, points, work = agent.workspaces[512](states, proposals, checked)
            except Divergence:
                return
            totals.append(
                {"chunk": chunk, "alternative": alternative, "work": asdict(work)}
            )
            current = points[:, 0].cpu().numpy()
            if first_points is None:
                first_points = current.copy()
            else:
                np.testing.assert_array_equal(current, first_points)
            counter += 1
            print(
                json.dumps(
                    {
                        "completed_branches": counter,
                        "chunk": chunk,
                        "alternative": alternative,
                    }
                ),
                flush=True,
            )
    write(
        folder / "audit.json",
        {"status": "not_reproduced", "complete_previous_branch_work": totals},
    )


if __name__ == "__main__":
    run()
