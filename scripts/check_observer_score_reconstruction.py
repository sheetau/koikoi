"""Check recovery of past actor inputs AND actions from later observer rows."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.env import BatchEnv
from koikoi.layout import EPISODE, PHASE, PLAYER, ROUND, TURN
from scripts.check_observer_temporal_reconstruction import reconstruct_past_actor, sha
from scripts.seat_observer_selfplay import SeatObserverFeatures


def reconstruct_past_input_action(features, root_round, root_turn, root_phase):
    """Recover a completed earlier decision in the SAME played round.

    Round numbers repeat between matches; the caller must enforce the same
    played round, not merely equal displayed round numbers.
    """
    actor_input = reconstruct_past_actor(features, root_round, root_turn, root_phase)
    future_turn = round(float(features[532]) * 16)
    future_phase = int(np.argmax(features[528:532]))
    if (root_turn, root_phase) >= (future_turn, future_phase):
        raise ValueError("The requested action must already have occurred")
    if root_phase == 3:
        if features[544 + root_turn] != 1:
            raise ValueError("A same-round successor requires recorded continuation")
        action = 49
    else:
        channel = (7, 9, 10)[root_phase]
        cards = np.asarray(features)[:528].reshape(48, 11)
        found = np.flatnonzero(cards[:, channel] == (root_turn + 1) / 16)
        if len(found) != 1:
            raise ValueError("Missing or ambiguous recorded action")
        action = int(found[0])
    return actor_input, action


@torch.no_grad()
def run(base, output):
    if output.exists():
        raise ValueError("Use a fresh output path")
    torch.set_num_threads(2)
    temporal_path = Path("runs/observer_temporal_reconstruction/audit.json")
    assert (
        sha(temporal_path)
        == "534146a5f301b9fdb617a81fee1493ba8848c5c00d740bc91c2873ff00ae649d"
    )
    temporal = json.loads(temporal_path.read_text(encoding="utf-8"))
    inputs = temporal["inputs"].copy()
    inputs[str(temporal_path)] = sha(temporal_path)
    inputs[str(Path(__file__))] = sha(Path(__file__))
    assert all(sha(p) == h for p, h in inputs.items())
    source = json.loads((base / "audit.json").read_text(encoding="utf-8"))
    reports = {}
    rejected_unexecuted = rejected_round = 0
    for parent, record in source["parents"].items():
        path = base / f"{parent}.jsonl"
        assert sha(path) == record["rows_sha256"]
        inputs[str(path)] = sha(path)
        actions = [
            json.loads(line)["original_action"]
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        states = np.load(base / f"{parent}_states.npy", allow_pickle=False)
        assert len(actions) == len(states) == 256
        env = BatchEnv(len(states), "cpu", "paper8", seed=0)
        env.state.copy_(torch.from_numpy(states))
        views = SeatObserverFeatures(env, "public")(env).numpy()
        actor_inputs, _ = env.published_observe()
        actor_inputs = actor_inputs.numpy()
        count = inactive = 0
        phases = {str(i): 0 for i in range(4)}
        for root, state in enumerate(states):
            seat = int(state[PLAYER])
            address = tuple(int(state[k]) for k in (ROUND, TURN, PHASE))
            for invalid in (address, (address[0] + 1, *address[1:])):
                try:
                    reconstruct_past_input_action(views[root, seat], *invalid)
                except ValueError:
                    if invalid == address:
                        rejected_unexecuted += 1
                    else:
                        rejected_round += 1
                else:
                    raise AssertionError("Invalid address was accepted")
            for future in range(root + 1, len(states)):
                later = states[future]
                if (later[EPISODE], later[ROUND]) != (state[EPISODE], state[ROUND]):
                    break
                restored, action = reconstruct_past_input_action(
                    views[future, seat], *address
                )
                np.testing.assert_array_equal(restored, actor_inputs[root])
                assert action == actions[root], (parent, root, future)
                count += 1
                inactive += int(later[PLAYER] != seat)
                phases[str(state[PHASE])] += 1
        np.testing.assert_array_equal(env.state.numpy(), states)
        assert count == temporal["parents"][parent]["earlier_later_pairs"]
        reports[parent] = {
            "input_action_pairs": count,
            "inactive_viewer_future_pairs": inactive,
            "root_phase_counts": phases,
        }
    assert all(sha(p) == h for p, h in inputs.items())
    report = {
        "status": "passed",
        "inputs": inputs,
        "parents": reports,
        "unexecuted_action_rejections": rejected_unexecuted,
        "different_round_rejections": rejected_round,
        "new_decisions": 0,
        "new_rounds": 0,
        "new_matches": 0,
        "scope": "Exact reconstruction of earlier actor-input/action pairs from later same-viewer features at a supplied public address, within the same played round. No actor forward/backward pass, simulator stepping or training. Equality of these arguments makes a fixed differentiable policy score a function of the later information on checked probes; no value-accuracy, across-round, unbiased-PPO or strength claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha(output),
                "parents": reports,
                "unexecuted_action_rejections": rejected_unexecuted,
                "different_round_rejections": rejected_round,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("runs/observer_seat_information_v2")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.output)
