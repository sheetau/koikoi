"""Replay complete diagnostic actions on CPU and recompute returns in float64."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER, ROUND

BASE = Path("runs/match_return_diagnostic")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recurrence(data, lam, horizon):
    """Seat-zero recurrence avoids reproducing production's player-switch code."""
    sign = 1 - 2 * data["player"]
    value = data["value"].astype(np.float64) * sign
    rewards = data["result"][:, :, 0]
    terminal = data["result"][:, :, 2].astype(bool) | ~data["valid"]
    result = np.zeros_like(value)
    for start in range(0, len(value), horizon):
        end = min(len(value), start + horizon)
        bootstrap = value[end].copy() if end < len(value) else np.zeros(value.shape[1])
        carry = np.zeros_like(bootstrap)
        for time in reversed(range(start, end)):
            delta = rewards[time] + np.where(terminal[time], 0, bootstrap) - value[time]
            carry = delta + np.where(terminal[time], 0, lam * carry)
            result[time] = carry + value[time]
            bootstrap = value[time]
    return result * sign


def main():
    torch.set_num_threads(2)
    report = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    protocol = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    assert report["protocol"] == protocol["protocol"] and report["status"] == "complete"
    assert sha("scripts/diagnose_match_returns.py") == report["script_sha256"]
    digest = hashlib.sha256()
    for path in sorted(Path("koikoi").rglob("*")):
        if path.suffix in (".py", ".h", ".cpp", ".cu"):
            digest.update(path.as_posix().encode())
            digest.update(path.read_bytes())
    assert digest.hexdigest() == report["source_sha256"]
    totals = {
        "matches": 0,
        "rounds": 0,
        "active_decisions": 0,
        "executed_rows_including_frozen": 0,
        "updates": 0,
    }
    maximum_return_error = 0.0
    for label, path, parent_sha, first_seed in report["protocol"]["parents"]:
        assert sha(path) == parent_sha
        for index, row in enumerate(report["cases"][label]["batches"]):
            raw_path = BASE / f"{label}_{index}.npz"
            assert sha(raw_path) == row["raw_sha256"]
            data = np.load(raw_path, allow_pickle=False)
            assert row["seed"] == first_seed + 1024 * index
            env = BatchEnv(1024, "cpu", "paper8", row["seed"])
            assert np.array_equal(env.state.numpy(), data["initial"])
            for time, action in enumerate(data["action"]):
                state = env.state.numpy()
                assert (
                    hashlib.sha256(state.tobytes()).hexdigest()
                    == data["state_digests"][time]
                )
                assert np.array_equal(state[:, PLAYER], data["player"][time])
                assert np.array_equal(state[:, PHASE], data["phase"][time])
                assert np.array_equal(state[:, ROUND], data["round"][time])
                assert np.array_equal(state[:, PHASE] != 4, data["valid"][time])
                outcome = env.step(
                    torch.from_numpy(action.copy()), freeze_done=True
                ).numpy()
                assert np.array_equal(outcome, data["result"][time])
            assert np.array_equal(env.state.numpy(), data["final"])
            assert (env.state[:, PHASE] == 4).all()
            valid = data["valid"]
            counts = row["counts"]
            assert counts["matches"] == int(data["result"][:, :, 2].sum()) == 1024
            assert (
                counts["rounds"]
                == int(data["result"][:, :, 1].sum())
                == int((env.state[:, ROUND] + 1).sum())
            )
            assert counts["active_decisions"] == int(valid.sum())
            assert counts["executed_rows_including_frozen"] == valid.size
            assert counts["steps"] == len(valid)
            assert counts["updates"] == 0
            for key in totals:
                totals[key] += counts[key]
            terminal_outcome = np.sign(data["final"][:, 106] - data["final"][:, 107])
            truth = terminal_outcome[None] * (1 - 2 * data["player"])
            assert np.array_equal(data["mc"], truth)
            np.testing.assert_allclose(
                recurrence(data, 1, len(valid))[valid], truth[valid], atol=1e-12, rtol=0
            )
            for key, lam, horizon in (
                ("gae95_full", 0.95, len(valid)),
                ("gae95_h128", 0.95, 128),
                ("gae1_h128", 1, 128),
            ):
                expected = recurrence(data, lam, horizon)
                error = float(np.max(np.abs(expected[valid] - data[key][valid])))
                maximum_return_error = max(maximum_return_error, error)
                assert error < 2e-6
            length = valid.sum(0)
            distance = length[None] - np.arange(len(valid))[:, None] - 1
            masks = {
                "all": valid,
                "rounds_1_2": valid & (data["round"] < 2),
                "rounds_3_6": valid & (data["round"] >= 2) & (data["round"] < 6),
                "rounds_7_8": valid & (data["round"] >= 6),
                "terminal_distance_gt64": valid & (distance > 64),
                "terminal_distance_le16": valid & (distance <= 16),
            }
            for group, mask in masks.items():
                stored = row["statistics"][group]
                assert stored["rows"] == int(mask.sum())
                mc_adv = truth[mask] - data["value"][mask].astype(float)
                np.testing.assert_allclose(
                    stored["mc_advantage_std"], np.std(mc_adv), rtol=1e-12
                )
                np.testing.assert_allclose(
                    stored["mean_terminal_trace_weight"],
                    np.average(np.power(0.95, distance[mask])),
                    rtol=1e-12,
                )
                for key, stats in stored["estimators"].items():
                    estimate = data[key][mask].astype(float)
                    advantage = estimate - data["value"][mask]
                    error = estimate - truth[mask]
                    left, right = advantage - advantage.mean(), mc_adv - mc_adv.mean()
                    correlation = np.dot(left, right) / np.sqrt(
                        np.dot(left, left) * np.dot(right, right)
                    )
                    expected = {
                        "return_rmse_vs_realized_mc": np.sqrt(
                            np.dot(error, error) / len(error)
                        ),
                        "advantage_std": np.std(advantage),
                        "advantage_sign_disagreement": np.count_nonzero(
                            advantage * mc_adv < 0
                        )
                        / len(error),
                        "advantage_correlation": correlation,
                    }
                    for metric, value in expected.items():
                        np.testing.assert_allclose(
                            stats[metric], value, rtol=1e-10, atol=1e-14
                        )
                delta = (data["gae95_h128"][mask] - data["gae95_full"][mask]).astype(
                    float
                )
                np.testing.assert_allclose(
                    stored["h128_vs_full95_rmse"],
                    np.sqrt(np.dot(delta, delta) / len(delta)),
                    rtol=1e-12,
                )
    assert totals["matches"] == 8192
    audit = {
        "status": "passed",
        "report_sha256": sha(BASE / "report.json"),
        "auditor_sha256": sha(__file__),
        "totals": totals,
        "maximum_float64_return_difference": maximum_return_error,
        "scope": "All stored actions replayed through native CPU against per-step CUDA-state hashes and raw results; returns recomputed in seat-zero coordinates, all summaries checked. Native transition core is shared; actor inference and action RNG were not independently replayed.",
    }
    (BASE / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
