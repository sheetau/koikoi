"""Independent scalar reduction plus fixed-stratum Python trajectory replay."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from koikoi.layout import DEALER, PHASE, PLAYER, POINTS, ROUND, TERMINAL
from koikoi.public_stock import PublicStockEnv

PUBLIC_FIELDS = [PHASE, ROUND, DEALER, POINTS, POINTS + 1, 114]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(base):
    meta = json.loads((base / "result.json").read_text(encoding="utf-8"))
    config = meta["config"]
    if sha(base / "table.npz") != meta["table_sha256"]:
        raise ValueError("Table changed")
    for path, digest in meta["sources"].items():
        if sha(path) != digest:
            raise ValueError(f"Table input changed: {path}")
    data = np.load(base / "table.npz")
    value, targets = data["value"], data["targets"]
    if (
        value.shape != (8, 2, 61)
        or not np.isfinite(value).all()
        or np.max(np.abs(value)) > 1.000001
    ):
        raise ValueError("Invalid cache values")
    samples = config["samples"]
    offsets = sorted({0, samples // 3, 2 * samples // 3, samples - 1})
    decisions, replays, replay_decisions, terminals = 0, 0, 0, 0
    for r in range(8):
        path = base / f"round_{r}.npz"
        if sha(path) != meta["traces"][path.name]:
            raise ValueError("Saved table trajectory changed")
        trace = np.load(path)
        final, actions = trace["final_public"], trace["actions"]
        decisions += int((actions >= 0).sum())
        terminal = final[:, 0] == TERMINAL
        terminals += int(terminal.sum())
        if not np.all(final[:, 5] == 1):
            raise ValueError("Cache rollout did not finish exactly one round")
        expected_round = np.where(terminal, r, r + 1)
        np.testing.assert_array_equal(final[:, 1], expected_round)
        actual = np.sign(final[:, 3] - final[:, 4]).astype(np.float32)
        if (~terminal).any():
            live = final[~terminal]
            actual[~terminal] = value[live[:, 1], live[:, 2], live[:, 3]]
        np.testing.assert_array_equal(actual.reshape(2, 59, samples), targets[r])
        np.testing.assert_allclose(
            value[r, :, 1:60], actual.reshape(2, 59, samples).mean(2), rtol=0, atol=2e-7
        )
        for dealer in range(2):
            for points in range(1, 60):
                for offset in offsets:
                    index = (dealer * 59 + points - 1) * samples + offset
                    seed = config["seed"] + r * 131072 + index
                    ref = PublicStockEnv(seed, meta["profile"])
                    ref.state[ROUND] = r
                    ref.state[PLAYER] = ref.state[DEALER] = dealer
                    ref.state[POINTS : POINTS + 2] = [points, 60 - points]
                    ended = False
                    for action in actions[:, index]:
                        if action >= 0:
                            if ended:
                                raise ValueError(
                                    "Actions continue after round settlement"
                                )
                            result = ref.step(int(action))
                            replay_decisions += 1
                            ended = bool(result[1])
                    if not ended:
                        raise ValueError("Reference round did not end")
                    np.testing.assert_array_equal(
                        ref.state[PUBLIC_FIELDS], final[index]
                    )
                    replays += 1
    if decisions != meta["transitions"] or terminals != meta["terminal_round_rollouts"]:
        raise ValueError("Preparation counters differ")
    if meta["conditional_round_rollouts"] != 8 * 118 * samples:
        raise ValueError("Conditional rollout count differs")
    return {
        "status": "passed",
        "table_sha256": meta["table_sha256"],
        "all_target_values_reduced": int(targets.size),
        "transitions": decisions,
        "python_round_replays": replays,
        "python_replay_decisions": replay_decisions,
        "audit_offsets_per_cell": offsets,
        "all_trajectories_replayed": len(offsets) == samples,
        "scope": "All saved targets and means reduced; trajectory coverage is specified by all_trajectories_replayed and audit_offsets_per_cell",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table-dir", required=True)
    args = p.parse_args()
    base = Path(args.table_dir)
    result = audit(base)
    (base / "audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
