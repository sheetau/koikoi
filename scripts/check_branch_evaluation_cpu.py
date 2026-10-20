"""Check the GPU evaluator's first baseline trajectories against CPU adapters."""

import json
from pathlib import Path

import numpy as np
import torch

from koikoi.evaluate import make_agent
from koikoi.layout import PHASE, PLAYER, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.train import provenance
from scripts.run_arbitration_study import PARENTS
from scripts.train_branch_policy_gradient import setup, sha, write


def run():
    directory = Path("runs/branch_policy_gradient_training/evaluation")
    output = directory / "baseline_cpu_check.json"
    if output.exists():
        raise ValueError("CPU baseline check already exists")
    setup(110070001)
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "passed"
    records, mismatches, calls = [], [], 0
    for parent, (path, digest) in enumerate(PARENTS, 1):
        assert sha(path) == digest
        candidate = make_agent(path, 0, deterministic=True)
        for kind in ("wp", "point"):
            opponent = make_agent(f"rl-{kind}-assisted", 0, deterministic=True)
            source = directory / f"parent{parent}_{kind}" / "chunk_0000.pt"
            data = torch.load(source, map_location="cpu", weights_only=False)
            count = 0
            for lane in range(8):
                game = ReferenceEnv()
                game.state = data["initial"][lane].numpy().copy()
                agents = (
                    [candidate, opponent] if lane % 2 == 0 else [opponent, candidate]
                )
                for step, actions in enumerate(data["actions"]):
                    if game.state[PHASE] == TERMINAL:
                        break
                    actual = agents[int(game.state[PLAYER])].act(game)
                    expected = int(actions[lane])
                    if actual != expected:
                        mismatches.append(
                            {
                                "parent": parent,
                                "opponent": kind,
                                "lane": lane,
                                "step": step,
                                "phase": int(game.state[PHASE]),
                                "gpu_action": expected,
                                "cpu_action": actual,
                            }
                        )
                    # Follow the saved GPU trajectory even after a disagreement.
                    game.step(expected)
                    count += 1
                assert np.array_equal(game.state, data["final"][lane].numpy())
            calls += count
            records.append(
                {
                    "path": str(source),
                    "sha256": sha(source),
                    "matches": 8,
                    "checked_action_requests": count,
                }
            )
    write(
        output,
        {
            "status": "complete",
            "source": provenance(),
            "script_sha256": sha(__file__),
            "evaluation_audit_sha256": sha(directory / "audit.json"),
            "records": records,
            "checked_action_requests": calls,
            "mismatches": mismatches,
            "scope": "First four pairs of each parent/opponent baseline cell; CPU singleton original adapters on saved GPU trajectories. No new independent matches.",
        },
    )
    print(json.dumps({"checked_action_requests": calls, "mismatches": len(mismatches)}))


if __name__ == "__main__":
    run()
