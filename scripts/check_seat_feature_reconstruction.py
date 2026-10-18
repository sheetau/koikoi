"""Reconstruct active actor inputs using only public-seat critic features.

The decoder never receives a simulator state, hidden cards or an actor output.
The check compares it with the native builder on retained observation probes.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER
from koikoi.reference import BLUE, KASU, RED, RIBBONS, TANE, score
from scripts.diagnose_observer_seat_information import relabel
from scripts.seat_observer_selfplay import SeatObserverFeatures


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct_actor(features):
    """Decode one reachable, active-viewer paper8 row to the padded 300x50 input."""
    x = np.asarray(features)
    if x.shape != (562,) or not np.isfinite(x).all():
        raise ValueError("Expected one finite 562-column observation")
    if x[560] != 1 or x[561] not in (0, 1):
        raise ValueError("An active viewer with an explicit public seat is required")
    if not np.isclose(x[535] * 12, 8) or x[543] != 1:
        raise ValueError("Expected the paper8 protocol")
    cards, g = x[:528].reshape(48, 11), x[528:560]
    phase, turn, seat = int(np.argmax(g[:4])), round(float(g[4]) * 16), int(x[561])
    dealer = seat if g[5] == 1 else 1 - seat
    groups = [set(np.flatnonzero(cards[:, i])) for i in range(5)]
    initial_field = set(np.flatnonzero(cards[:, 6]))
    koi = [round(float(g[10 + i]) * 8) for i in (0, 1)]
    broadcast = np.zeros(137, dtype=np.float32)

    def triple(offset, value):
        magnitude, sign = abs(value), (value > 0) - (value < 0)
        broadcast[offset : offset + 3] = (
            math.sqrt(magnitude) * sign,
            magnitude * sign / 2,
            math.pow(magnitude, 1.5) * sign * 0.1,
        )

    triple(17, (round(float(g[8]) * 64) - round(float(g[9]) * 64)) / 2)
    triple(20, score(groups[2], koi[0]))
    triple(23, score(groups[3], koi[1]))
    broadcast[26 + round(float(g[6]) * 12)] = 1
    broadcast[34 + turn] = 1
    broadcast[50 + dealer] = 1
    broadcast[52:56] = (koi[0], koi[0] ** 2, koi[1], koi[1] ** 2)
    for t, decision in enumerate(g[16:32]):
        if decision == 1:
            broadcast[56 + (0 if (dealer + t) % 2 == seat else 8) + t // 2] = 1
    keys = [{c} for c in (0, 8, 28, 40, 44, 32)] + [
        {20, 24, 36},
        TANE,
        RED,
        BLUE,
        RED | BLUE,
        RIBBONS,
        KASU,
    ]
    for group_index, group in enumerate(groups):
        for k, subset in enumerate(keys):
            broadcast[72 + group_index * 13 + k] = len(group & subset)
    out = np.zeros((300, 50), dtype=np.float32)
    offset = 2 if phase == 3 else 0
    out[:137, offset : offset + 48] = broadcast[:, None]
    if offset:
        out[:137, :2] = broadcast[:, None]
        out[0, 0] = out[1, 1] = 1

    def mark(row, subset):
        if subset:
            out[row, np.array(sorted(subset)) + offset] = 1

    for k, subset in enumerate(keys):
        mark(137 + k, subset)
    for month in range(12):
        mark(150 + month, set(range(4 * month, 4 * month + 4)))
    for row, subset in enumerate(
        (
            groups[0],
            initial_field,
            groups[4],
            groups[0],
            groups[2],
            groups[1],
            groups[2],
            groups[4],
        ),
        start=162,
    ):
        mark(row, subset)
    if phase in (1, 2):
        pending = set(np.flatnonzero(cards[:, 5]))
        if len(pending) != 1:
            raise ValueError("Expected one publicly pending card")
        month = next(iter(pending)) // 4
        mark(170, pending)
        mark(171, {c for c in groups[1] if c // 4 == month})
    board = initial_field.copy()
    for t in range(turn + 1):
        when = (t + 1) / 16
        for part in (0, 1):
            played = set(np.flatnonzero(cards[:, 7 + part] == when))
            if not played:
                break
            if len(played) != 1:
                raise ValueError("Each history phase must identify one played card")
            month = next(iter(played)) // 4
            paired = {c for c in board if c // 4 == month}
            row = 172 + (turn - t) * 8 + 4 * part
            mark(row if paired else row + 1, played)
            chosen = set(np.flatnonzero(cards[:, 9 + part] == when))
            if len(paired) == 2 and not chosen:
                break
            collected = chosen if len(paired) == 2 else paired
            mark(row + 2, collected)
            mark(row + 3, paired - collected)
            if paired:
                board -= collected
            else:
                board |= played
    return out


@torch.no_grad()
def run(base, output):
    if output.exists():
        raise ValueError("Use a fresh output path")
    torch.set_num_threads(2)
    source = json.loads((base / "audit.json").read_text(encoding="utf-8"))
    assert source["status"] == "passed"
    inputs = source["inputs"].copy()
    for path in (
        Path(__file__),
        base / "audit.json",
        Path("koikoi/reference.py"),
        Path("koikoi/native/core.h"),
        Path("koikoi/native/published.h"),
        Path("scripts/seat_observer_selfplay.py"),
        Path("scripts/observer_selfplay_returns.py"),
        Path("scripts/diagnose_observer_seat_information.py"),
    ):
        inputs[str(path)] = sha(path)
    probes = []
    for parent, expected in source["parents"].items():
        path = base / f"{parent}_states.npy"
        assert sha(path) == expected["states_sha256"]
        inputs[str(path)] = sha(path)
        for state in np.load(path, allow_pickle=False):
            probes.extend((state, relabel(state)))
    states = np.stack(probes)
    env = BatchEnv(len(states), "cpu", "paper8", seed=0)
    env.state.copy_(torch.from_numpy(states))
    views = SeatObserverFeatures(env, "public")(env).numpy()
    actual, _ = env.published_observe()
    rows = views[np.arange(len(states)), states[:, PLAYER]]
    decoded = np.stack([reconstruct_actor(row) for row in rows])
    np.testing.assert_array_equal(decoded, actual.numpy())
    np.testing.assert_array_equal(env.state.numpy(), states)
    assert set(states[:, PHASE]) == {0, 1, 2, 3}
    # The same masked row must correspond to different actor inputs on each
    # original/relabelled pair. This independently retains the missing-bit witness.
    np.testing.assert_array_equal(rows[::2, :561], rows[1::2, :561])
    changed_rows = np.any(decoded[::2] != decoded[1::2], axis=(0, 2))
    assert np.flatnonzero(changed_rows).tolist() == [50, 51]
    assert all(sha(path) == digest for path, digest in inputs.items())
    result = {
        "status": "passed",
        "inputs": inputs,
        "source_roots": len(states) // 2,
        "active_viewer_reconstructions": len(states),
        "phase_counts": {str(p): int(sum(states[:, PHASE] == p)) for p in range(4)},
        "exact_actor_input": True,
        "masked_seat_collision_pairs": len(states) // 2,
        "changed_rows_on_relabel": [50, 51],
        "new_decisions": 0,
        "new_rounds": 0,
        "new_matches": 0,
        "scope": "An observation-only decoder reproduces all 300x50 actor features exactly on retained active-viewer probes and their seat relabelings. No policy inference, transitions, training or strength evaluation. This is empirical representation coverage, not a proof of sufficient history, exact conditional values or unbiased policy gradients.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha(output),
                "reconstructions": len(states),
                "phase_counts": result["phase_counts"],
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
