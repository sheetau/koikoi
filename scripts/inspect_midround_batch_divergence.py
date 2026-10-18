"""Inspect scores at a saved identical-state, different-batch action divergence."""

import json
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import EndgameAgent
from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER, TERMINAL
from koikoi.upstream import PublishedAgent
from scripts.diagnose_history_support import reference
from scripts.history_conditioned_search import HistoryConditionedSearch
from scripts.run_arbitration_study import PARENTS, sha, write


@torch.no_grad()
def run():
    folder = Path("runs/midround_batch_divergence")
    output = folder / "score_inspection.json"
    if output.exists():
        raise ValueError("Preserve existing inspection")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    audit = json.loads((folder / "audit.json").read_text(encoding="utf-8"))
    with np.load(folder / "divergence.npz", allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    agent = HistoryConditionedSearch(
        PARENTS[1][0], particles=512, confirmation_particles=0, device="cuda"
    )
    agent.continuation.prepare(audit["root_player"], arrays["labels"])
    env = BatchEnv(1024, "cuda", "paper8")
    rows, features_saved = [], []
    lane = 2 * audit["particles"][0]
    for key in ("first_states", "other_states"):
        env.state.copy_(torch.from_numpy(arrays[key]).to("cuda"))
        features, legal = env.published_observe()
        solved = env.solved_actions()
        legal &= (solved[:, None] < 0) | (
            torch.arange(50, device="cuda")[None] == solved[:, None]
        )
        routes = torch.where(
            env.state[:, PLAYER] == audit["root_player"], 2, agent.continuation.labels
        )
        route, phase = int(routes[lane]), int(env.state[lane, PHASE])
        indices = torch.where((routes == route) & (env.state[:, PHASE] != TERMINAL))[0]
        pos = int(torch.where(indices == lane)[0])
        actor = agent.continuation.actors[route]
        scores = actor(features[indices], env.state[indices, PHASE], legal[indices])[
            pos
        ]
        target_features = features[lane].clone()
        features_saved.append(target_features)
        singleton = actor(
            target_features[None],
            env.state[lane : lane + 1, PHASE],
            legal[lane : lane + 1],
        )[0]
        strict = (
            agent.parent
            if route == 2
            else EndgameAgent(PublishedAgent(("point", "wp")[route]))
        )
        strict_action = strict.act(reference(arrays[key][lane]))
        order = scores.argsort(descending=True)[:2]
        rows.append(
            {
                "batch": key,
                "lane": lane,
                "route": route,
                "phase": phase,
                "route_count": len(indices),
                "head_count": int((env.state[indices, PHASE] == phase).sum()),
                "top_actions": order.tolist(),
                "top_scores": scores[order].tolist(),
                "margin": float(scores[order[0]] - scores[order[1]]),
                "cuda_singleton_action": int(singleton.argmax()),
                "strict_cpu_action": int(strict_action),
                "all_legal_scores": {
                    int(a): float(scores[a]) for a in torch.where(legal[lane])[0]
                },
            }
        )
    torch.testing.assert_close(features_saved[0], features_saved[1], rtol=0, atol=0)
    np.testing.assert_array_equal(
        arrays["first_states"][lane], arrays["other_states"][lane]
    )
    write(
        output,
        {
            "status": "passed",
            "inputs": {
                str(p): sha(p)
                for p in (
                    Path(__file__),
                    folder / "audit.json",
                    folder / "divergence.npz",
                )
            },
            "same_state_and_features": True,
            "rows": rows,
            "scope": "One fixed reproduced state; no universal error bound or performance claim",
        },
    )
    print(json.dumps({"rows": rows, "sha256": sha(output)}), flush=True)


if __name__ == "__main__":
    run()
