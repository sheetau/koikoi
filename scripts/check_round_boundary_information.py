"""Bounded witness search for loss of earlier actions at a round reset.

Enumerate every legal initial action at the first retained root of each parent.
The strict public-observation parent completes the round. This is a feature
diagnostic, not a search policy, value-bias measurement or strength experiment.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import NeuralAgent
from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER, ROUND, TERMINAL, TURN
from koikoi.reference import ReferenceEnv
from koikoi.train import provenance
from scripts.check_observer_score_reconstruction import reconstruct_past_input_action
from scripts.run_arbitration_study import PARENTS, sha, write
from scripts.seat_observer_selfplay import SeatObserverFeatures


def run(folder):
    if folder.exists():
        raise ValueError("Use a fresh output directory")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    source = provenance()
    source_audit = Path("runs/observer_seat_information_v2/audit.json")
    assert (
        sha(source_audit)
        == "ac30ff58a6b921716ee2b8f5b59c93de5d7e3a818f5c4626e7fc626191bd1684"
    )
    original = json.loads(source_audit.read_text(encoding="utf-8"))
    inputs = {
        str(p): sha(p)
        for p in (
            Path(__file__),
            source_audit,
            Path("scripts/check_observer_score_reconstruction.py"),
            Path("scripts/check_observer_temporal_reconstruction.py"),
            Path("scripts/check_seat_feature_reconstruction.py"),
            Path("scripts/seat_observer_selfplay.py"),
            Path("scripts/observer_selfplay_returns.py"),
        )
    }
    folder.mkdir(parents=True)
    native = BatchEnv(1, "cpu", "paper8", debug=True)
    observe = SeatObserverFeatures(native, "public")
    cases = {}
    for parent_index, (checkpoint, expected) in enumerate(PARENTS, 1):
        key = f"parent{parent_index}"
        assert sha(checkpoint) == expected
        inputs[checkpoint] = expected
        path = Path(f"runs/observer_seat_information_v2/{key}_states.npy")
        assert sha(path) == original["parents"][key]["states_sha256"]
        inputs[str(path)] = sha(path)
        root = np.load(path, allow_pickle=False)[0].copy()
        assert (root[PHASE], root[TURN], root[ROUND]) == (0, 0, 0)
        actor = NeuralAgent(checkpoint, device="cpu", deterministic=True)
        env = ReferenceEnv(1, "paper8")
        env.state[:] = root
        legal = np.flatnonzero(env.legal()).tolist()
        native.state.copy_(torch.from_numpy(root[None]))
        root_input = native.published_observe()[0][0].numpy().copy()
        seat = int(root[PLAYER])
        rows, endpoints, features = [], [], []
        for first in legal:
            env.state[:] = root
            native.state.copy_(torch.from_numpy(root[None]))
            trace = []
            work = {"decisions": 0, "round_endings": 0, "completed_matches": 0}
            for step in range(64):
                last = env.state.copy()
                action = first if step == 0 else int(actor.act(env))
                result = env.step(action)
                actual = native.step(torch.tensor([action]))[0].numpy()
                np.testing.assert_array_equal(actual, result)
                np.testing.assert_array_equal(native.state[0].numpy(), env.state)
                trace.append(action)
                work["decisions"] += 1
                work["round_endings"] += int(result[1])
                work["completed_matches"] += int(result[2])
                if result[1]:
                    break
            else:
                raise RuntimeError("Round exceeded64 exposed decisions")
            endpoint = env.state.copy()
            endpoints.append(endpoint)
            features.append(observe(native)[0].numpy().copy())
            native.state.copy_(torch.from_numpy(last[None]))
            past_input, past_action = reconstruct_past_input_action(
                observe(native)[0, seat].numpy(), int(root[ROUND]), 0, 0
            )
            np.testing.assert_array_equal(past_input, root_input)
            assert past_action == first
            rows.append({"first_action": first, "trace": trace, "work": work})
        collisions = []
        for i, j in itertools.combinations(range(len(rows)), 2):
            if endpoints[i][PHASE] == TERMINAL or endpoints[j][PHASE] == TERMINAL:
                continue
            if np.array_equal(features[i][seat], features[j][seat]):
                collisions.append(
                    {
                        "branches": [i, j],
                        "first_actions": [
                            rows[i]["first_action"],
                            rows[j]["first_action"],
                        ],
                        "both_observers_equal": bool(
                            np.array_equal(features[i], features[j])
                        ),
                        "full_successor_state_equal": bool(
                            np.array_equal(endpoints[i], endpoints[j])
                        ),
                    }
                )
        states_path = folder / f"{key}.npz"
        np.savez(
            states_path,
            root=root,
            endpoints=np.stack(endpoints),
            features=np.stack(features),
        )
        cases[key] = {
            "branches": rows,
            "collisions": collisions,
            "before_reset_reconstructions": len(rows),
            "states_sha256": sha(states_path),
            "work": {k: sum(row["work"][k] for row in rows) for k in rows[0]["work"]},
        }
    assert all(sha(p) == h for p, h in inputs.items())
    assert provenance()["source_sha256"] == source["source_sha256"]
    report = {
        "status": "passed",
        "inputs": inputs,
        "provenance": source,
        "cases": cases,
        "scope": "Two fixed saved roots, all legal first actions, public deterministic round continuations. Collisions witness loss of earlier actions, not value bias or stronger play. Equal full successor states also have identical deterministic future continuations; loss of history alone does not establish a need for critic memory. Branch round endings are not independent deals.",
    }
    write(folder / "audit.json", report)
    print(
        json.dumps(
            {
                "output": str(folder / "audit.json"),
                "sha256": sha(folder / "audit.json"),
                "cases": {
                    k: {"collisions": len(v["collisions"]), "work": v["work"]}
                    for k, v in cases.items()
                },
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
