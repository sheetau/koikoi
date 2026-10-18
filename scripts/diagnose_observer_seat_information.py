"""Measure public seat-label information omitted by the relative value input.

The transformed states are observation probes, not alternate played matches.
No training, CUDA work, held-out evaluation or production changes are performed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import NeuralAgent
from koikoi.env import BatchEnv
from koikoi.layout import DEALER, EPISODE, KOIS, PHASE, PLAYER, POINTS, SEED
from koikoi.published_train import packed_observation
from koikoi.reference import ReferenceEnv
from koikoi.train import provenance
from scripts.observer_selfplay_returns import ObserverFeatures


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relabel(state):
    """Swap both seat names while preserving cards and public action history.

    Swap original hand blocks for a consistent stored permutation, but preserve
    the stock/field blocks. Seed/episode are unchanged because no future deal is
    executed from this probe. It is not asserted to be a new seeded trajectory.
    """
    result = state.copy()
    for before, after in ((0, 1), (1, 0), (3, 4), (4, 3)):
        result[:48][state[:48] == before] = after
    result[48:56], result[56:64] = state[56:64], state[48:56]
    result[PLAYER] = 1 - state[PLAYER]
    result[DEALER] = 1 - state[DEALER]
    for offset in (POINTS, KOIS):
        result[offset : offset + 2] = state[offset : offset + 2][::-1]
    return result


@torch.no_grad()
def run(base):
    if base.exists():
        raise ValueError("Use a fresh diagnostic directory")
    torch.set_num_threads(2)
    parents = [
        Path(f"runs/cold_ppo_screen/seed{seed}_warm/update_00000520.pt")
        for seed in (6710001, 6720001)
    ]
    protocol_path = Path("runs/observer_selfplay_study/protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    inputs = {
        str(path): sha(path)
        for path in [
            Path(__file__),
            protocol_path,
            *parents,
            Path("scripts/observer_selfplay_returns.py"),
        ]
    }
    assert all(sha(path) == expected for path, expected in protocol["inputs"].items())
    source = provenance()["source_sha256"]
    assert source == protocol["source"]["source_sha256"]
    base.mkdir(parents=True)
    plan = {
        "roots_per_parent": 256,
        "source_seeds": [26080001, 26081001],
        "parents": [str(p) for p in parents],
        "inputs": inputs,
        "production_source_sha256": source,
        "inference": "singleton CPU2,greedy with public endgame mask; sampled-policy TV uses parent temperature0.05",
        "scope": "Representation diagnostic,not playing strength; relabeled positions are never stepped",
    }
    (base / "protocol.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    reports = {}
    for index, path in enumerate(parents):
        name = f"parent{index + 1}"
        agent = NeuralAgent(path, deterministic=True)
        ref = ReferenceEnv(plan["source_seeds"][index], "paper8")
        env = BatchEnv(1, "cpu", "paper8", seed=0)
        views = ObserverFeatures(env)
        rows, states = [], []
        work = {"decisions": 0, "rounds": 0, "matches": 0}
        with (base / f"{name}.jsonl").open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            for root in range(plan["roots_per_parent"]):
                original = ref.state.copy()
                alternate = relabel(original)
                assert np.array_equal(relabel(alternate), original)
                states.append(original)
                packed, masks, features, logits = [], [], [], []
                for state in (original, alternate):
                    env.state.copy_(torch.from_numpy(state).unsqueeze(0))
                    before = env.state.clone()
                    obs, mask = packed_observation(env)
                    feature = views(env)
                    solved = env.solved_actions()
                    mask.logical_and_(
                        (solved[:, None] < 0)
                        | (torch.arange(50)[None] == solved[:, None])
                    )
                    prediction = agent.model(obs, mask)[0]
                    assert torch.equal(before, env.state)
                    packed.append(obs)
                    masks.append(mask)
                    features.append(feature)
                    logits.append(prediction)
                assert torch.equal(features[0], features[1].flip(1))
                assert torch.equal(packed[0][:, :561], packed[1][:, :561])
                assert torch.equal(masks[0], masks[1])
                difference = (packed[0][:, 561:] != packed[1][:, 561:]).reshape(300, 50)
                changed_rows = torch.where(difference.any(-1))[0].tolist()
                assert changed_rows == [50, 51]
                # Erasing only these public absolute-dealer rows removes all
                # published-input differences. Relative dealer is already equal.
                probabilities = [value.softmax(-1) for value in logits]
                variation = float((probabilities[0] - probabilities[1]).abs().sum() / 2)
                actions = [int(value.argmax(-1)) for value in logits]
                assert actions[0] == agent.act(ref)
                row = {
                    "root": root,
                    "phase": int(original[PHASE]),
                    "player": int(original[PLAYER]),
                    "dealer": int(original[DEALER]),
                    "original_action": actions[0],
                    "relabeled_action": actions[1],
                    "total_variation": variation,
                    "changed_published_rows": changed_rows,
                    "legal_actions": int(masks[0].sum()),
                }
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                outcome = ref.step(actions[0])
                work["decisions"] += 1
                work["rounds"] += int(outcome[1])
                work["matches"] += int(outcome[2])
                if ref.state[PHASE] == 4:
                    ref = ReferenceEnv(
                        int(ref.state[SEED]), "paper8", int(ref.state[EPISODE]) + 1
                    )
        np.save(base / f"{name}_states.npy", np.stack(states), allow_pickle=False)
        assert {row["phase"] for row in rows} == {0, 1, 2, 3}
        reports[name] = {
            "source_work": work,
            "greedy_changes": sum(
                r["original_action"] != r["relabeled_action"] for r in rows
            ),
            "mean_total_variation": float(
                np.mean([r["total_variation"] for r in rows])
            ),
            "max_total_variation": max(r["total_variation"] for r in rows),
            "nontrivial_distribution_changes": sum(
                r["total_variation"] > 1e-6 for r in rows
            ),
            "by_phase": {
                str(phase): {
                    "roots": sum(r["phase"] == phase for r in rows),
                    "greedy_changes": sum(
                        r["phase"] == phase
                        and r["original_action"] != r["relabeled_action"]
                        for r in rows
                    ),
                }
                for phase in range(4)
            },
            "rows_sha256": sha(base / f"{name}.jsonl"),
            "states_sha256": sha(base / f"{name}_states.npy"),
        }
        print(json.dumps({"case": name, **reports[name]}), flush=True)
    assert all(sha(path) == expected for path, expected in inputs.items())
    assert all(sha(path) == expected for path, expected in protocol["inputs"].items())
    result = {
        "status": "passed",
        "inputs": inputs,
        "production_source_sha256": source,
        "protocol_sha256": sha(base / "protocol.json"),
        "parents": reports,
        "scope": "Relative value inputs collide under seat relabeling while published inputs retain absolute dealer. Any observed policy discrepancy shows omitted policy context,not proof of unequal true values or measured critic/playing-strength error. No alternate-state transitions,search branches,training or CUDA work.",
    }
    (base / "audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"audit_sha256": sha(base / "audit.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
