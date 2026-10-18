"""Recover earlier same-viewer actor inputs from later within-round features.

The decoder receives one observer feature row and a public round/turn/phase
address. It never receives a simulator state or a private opponent hand.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.env import BatchEnv
from koikoi.layout import EPISODE, PHASE, PLAYER, ROUND, TURN
from koikoi.reference import score
from scripts.check_seat_feature_reconstruction import reconstruct_actor
from scripts.seat_observer_selfplay import SeatObserverFeatures


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct_past_actor(features, root_round, root_turn, root_phase):
    """Return the actor input at an earlier active-viewer address in this round.

    The public address identifies which earlier decision is requested. This
    does not reconstruct a previous round, or the complete old critic vector.
    """
    x = np.asarray(features)
    if x.shape != (562,) or not np.isfinite(x).all():
        raise ValueError("Expected a finite public-seat observer feature row")
    if x[561] not in (0, 1) or x[560] not in (0, 1):
        raise ValueError("Public seat and actor-role bits are required")
    if root_phase not in range(4) or root_turn not in range(16):
        raise ValueError("Invalid public decision address")
    future_cards = x[:528].reshape(48, 11)
    future_global = x[528:560]
    future_turn = round(float(future_global[4]) * 16)
    future_phase = int(np.argmax(future_global[:4]))
    if root_round != round(float(future_global[6]) * 12):
        raise ValueError("A previous round cannot be recovered by this decoder")
    if (root_turn, root_phase) > (future_turn, future_phase):
        raise ValueError("Requested decision is after the observation")
    seat = int(x[561])
    dealer = seat if future_global[5] == 1 else 1 - seat
    if (dealer + root_turn) % 2 != seat:
        raise ValueError("Requested actor must be this observer")
    if not np.isclose(future_global[7] * 12, 8) or future_global[15] != 1:
        raise ValueError("Only paper8 is covered")

    def event(turn, channel, required=True):
        found = np.flatnonzero(future_cards[:, channel] == (turn + 1) / 16)
        if len(found) > 1 or (required and len(found) != 1):
            raise ValueError("Missing or ambiguous public event")
        return int(found[0]) if len(found) else None

    # Cards are never added to a hand during a round. Its initial contents are
    # the remaining own hand plus the same viewer's timestamped hand plays.
    hand = set(np.flatnonzero(future_cards[:, 0]))
    for t in range(future_turn + 1):
        if (dealer + t) % 2 == seat:
            card = event(t, 7, required=False)
            if card is not None:
                hand.add(card)
    if len(hand) != 8:
        raise ValueError("Could not recover the initial own hand")
    initial_field = set(np.flatnonzero(future_cards[:, 6]))
    board = initial_field.copy()
    captured = [set(), set()]
    koi = [0, 0]
    history = np.zeros((48, 4), dtype=np.float32)
    continuation = np.zeros(16, dtype=np.float32)

    def snapshot(pending=None):
        result = np.zeros(562, dtype=np.float32)
        cards = result[:528].reshape(48, 11)
        unknown = set(range(48)) - hand - board - captured[0] - captured[1]
        if pending is not None:
            unknown.discard(pending)
            cards[pending, 5] = 1
        for index, group in enumerate((hand, board, *captured, unknown)):
            cards[list(group), index] = 1
        cards[list(initial_field), 6] = 1
        cards[:, 7:11] = history
        g = result[528:560]
        g[root_phase] = 1
        g[4] = root_turn / 16
        g[5:10] = future_global[5:10]
        g[10:12] = np.asarray(koi) / 8
        g[15] = 1
        g[16:32] = continuation
        result[560:] = (1, seat)
        return reconstruct_actor(result)

    for t in range(root_turn + 1):
        relative_actor = 0 if (dealer + t) % 2 == seat else 1
        if t == root_turn and root_phase == 0:
            return snapshot()
        before = score(captured[relative_actor], koi[relative_actor])
        for part in (0, 1):
            card = event(t, 7 + part)
            if part == 0 and relative_actor == 0:
                hand.remove(card)
            history[card, part] = (t + 1) / 16
            paired = {c for c in board if c // 4 == card // 4}
            if len(paired) == 2:
                if t == root_turn and root_phase == part + 1:
                    return snapshot(card)
                chosen = event(t, 9 + part)
                if chosen not in paired:
                    raise ValueError("Recorded capture is not a field alternative")
                history[chosen, part + 2] = (t + 1) / 16
                collected = {chosen}
            else:
                if t == root_turn and root_phase == part + 1:
                    raise ValueError("Requested capture decision did not occur")
                collected = paired
            if collected:
                board -= collected
                captured[relative_actor] |= collected | {card}
            else:
                board.add(card)
        if t == root_turn:
            if root_phase != 3 or t >= 14:
                raise ValueError("Requested continuation decision did not occur")
            if score(captured[relative_actor], koi[relative_actor]) <= before:
                raise ValueError("Requested continuation lacks a new score")
            return snapshot()
        decision = future_global[16 + t]
        if decision not in (0, 1):
            raise ValueError("A completed round cannot precede this observation")
        continuation[t] = decision
        if decision == 1:
            koi[relative_actor] += 1
    raise AssertionError("Decision address was not reached")


@torch.no_grad()
def run(base, output):
    if output.exists():
        raise ValueError("Use a fresh output path")
    torch.set_num_threads(2)
    audit_path = base / "audit.json"
    source = json.loads(audit_path.read_text(encoding="utf-8"))
    assert (
        sha(audit_path)
        == "ac30ff58a6b921716ee2b8f5b59c93de5d7e3a818f5c4626e7fc626191bd1684"
    )
    inputs = source["inputs"].copy()
    for path in (
        Path(__file__),
        audit_path,
        Path("scripts/check_seat_feature_reconstruction.py"),
        Path("scripts/seat_observer_selfplay.py"),
        Path("koikoi/native/core.h"),
        Path("koikoi/native/published.h"),
        Path("koikoi/reference.py"),
    ):
        inputs[str(path)] = sha(path)
    reports = {}
    for parent, record in source["parents"].items():
        path = base / f"{parent}_states.npy"
        assert sha(path) == record["states_sha256"]
        inputs[str(path)] = sha(path)
        states = np.load(path, allow_pickle=False)
        env = BatchEnv(len(states), "cpu", "paper8", seed=0)
        env.state.copy_(torch.from_numpy(states))
        features = SeatObserverFeatures(env, "public")(env).numpy()
        expected, _ = env.published_observe()
        expected = expected.numpy()
        pairs = inactive = 0
        phases = {str(i): 0 for i in range(4)}
        max_distance = 0
        for root, state in enumerate(states):
            seat = int(state[PLAYER])
            address = tuple(int(state[k]) for k in (ROUND, TURN, PHASE))
            for future in range(root + 1, len(states)):
                later = states[future]
                if (later[EPISODE], later[ROUND]) != (state[EPISODE], state[ROUND]):
                    break
                actual = reconstruct_past_actor(features[future, seat], *address)
                np.testing.assert_array_equal(
                    actual, expected[root], err_msg=f"{parent}: {root} -> {future}"
                )
                pairs += 1
                phases[str(state[PHASE])] += 1
                inactive += int(later[PLAYER] != seat)
                max_distance = max(max_distance, future - root)
        np.testing.assert_array_equal(env.state.numpy(), states)
        reports[parent] = {
            "source_roots": len(states),
            "earlier_later_pairs": pairs,
            "inactive_viewer_future_pairs": inactive,
            "root_phase_counts": phases,
            "max_decision_distance": max_distance,
        }
    assert all(sha(p) == h for p, h in inputs.items())
    result = {
        "status": "passed",
        "inputs": inputs,
        "parents": reports,
        "exact_past_actor_inputs": True,
        "new_decisions": 0,
        "new_rounds": 0,
        "new_matches": 0,
        "scope": "Observation-only reconstruction at a supplied public earlier decision address, within the same round and for the same viewer. All retained chronological pairs checked against native actor inputs. No actor inference or simulator stepping. This is measured temporal information retention, not a sufficient-history, exact-value, unbiased-gradient or strength guarantee.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({"output": str(output), "sha256": sha(output), "parents": reports})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("runs/observer_seat_information_v2")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.output)
