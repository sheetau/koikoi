"""CPU action replay and separate per-root scalar return/score calculations."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER, POINTS, ROUND
from koikoi.published_train import packed_observation
from scripts.directional_return_statistics import ESTIMATORS, summarize
from scripts.run_arbitration_study import read, sha, write


def scalar_check(data):
    """Forward TD expansion per starting decision, independent of reverse traces."""
    lengths = data["valid"].sum(0)
    counts = data["valid"].shape[1]
    expected_contributions = np.zeros((counts, 4, 2, 5))
    max_error = 0.0
    for lane, length in enumerate(lengths):
        sign = 1 - 2 * data["player"][:length, lane].astype(int)
        values = data["value"][:length, lane].astype(float)
        seat0 = sign * values
        rewards = data["result"][:length, lane, 0]
        payoff = int(
            np.sign(data["final"][lane, POINTS] - data["final"][lane, POINTS + 1])
        )
        assert rewards[-1] == payoff and (rewards[:-1] == 0).all()
        delta = rewards + np.append(seat0[1:], 0) - seat0
        for t in range(length):
            action, phase = int(data["action"][t, lane]), int(data["phase"][t, lane])
            weights = data["weights"][t, lane].astype(float)
            assert np.isfinite(weights).all() and (weights >= 0).all()
            assert (weights[~data["legal"][t, lane]] == 0).all()
            assert weights[action] > 0
            p = weights / sum(weights)
            score = np.zeros(5)
            if phase == 3:
                score[0] = float(action == 49) - p[49]
            score[phase + 1] = p[action] - sum(float(q) ** 2 for q in p)
            np.testing.assert_allclose(
                score, data["scores"][t, lane], atol=2e-15, rtol=0
            )
            cutoff = min(length, t + 128 - ((t + int(data["offsets"][lane])) % 128))
            mc = payoff * sign[t] - values[t]
            expected = {"mc": mc, "gae1_full": mc}
            for key, decay, end in (
                ("gae95_full", 0.95, length),
                ("gae95_window", 0.95, cutoff),
                ("gae1_window", 1.0, cutoff),
            ):
                expected[key] = (
                    np.dot(decay ** np.arange(end - t), delta[t:end]) * sign[t]
                )
            for key, value in expected.items():
                error = abs(value - data["adv_" + key][t, lane])
                max_error = max(max_error, error)
                assert error < 2e-12, (key, lane, t, error)
            seat = int(data["player"][t, lane])
            for k, name in enumerate(ESTIMATORS):
                expected_contributions[lane, k, seat] += score * expected[name]
    np.testing.assert_allclose(
        expected_contributions, data["contributions"], atol=2e-12, rtol=0
    )
    assert (data["scores"][~data["valid"]] == 0).all()
    for key in (*ESTIMATORS, "gae1_full"):
        assert (data["adv_" + key][~data["valid"]] == 0).all()
    return max_error


def replay(data, seed):
    env = BatchEnv(64, "cpu", "paper8", seed)
    np.testing.assert_array_equal(env.state.numpy(), data["initial"])
    lengths = data["valid"].sum(0)
    np.testing.assert_array_equal(
        data["valid"], np.arange(len(data["valid"]))[:, None] < lengths
    )
    assert (lengths > 0).all()
    for t, actions in enumerate(data["action"]):
        state = env.state.numpy()
        assert hashlib.sha256(state.tobytes()).hexdigest() == data["state_hashes"][t]
        for key, column in (("player", PLAYER), ("phase", PHASE), ("round", ROUND)):
            np.testing.assert_array_equal(state[:, column], data[key][t])
        np.testing.assert_array_equal(state[:, PHASE] != 4, data["valid"][t])
        obs, legal = packed_observation(env)
        assert (
            hashlib.sha256(obs.numpy().tobytes()).hexdigest()
            == data["public_hashes"][t]
        )
        guards = env.solved_actions()
        legal &= (guards[:, None] == -1) | (torch.arange(50)[None] == guards[:, None])
        np.testing.assert_array_equal(legal.numpy(), data["legal"][t])
        result = env.step(torch.from_numpy(actions.copy()), freeze_done=True).numpy()
        np.testing.assert_array_equal(result, data["result"][t])
    np.testing.assert_array_equal(env.state.numpy(), data["final"])
    assert (data["final"][:, PHASE] == 4).all()
    np.testing.assert_array_equal(
        data["result"][..., 2].astype(bool),
        np.arange(len(data["valid"]))[:, None] == lengths - 1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    base = args.input
    if (base / "audit.json").exists() or (base / "summary.json").exists():
        raise ValueError("Preserve completed diagnostic outputs")
    torch.set_num_threads(2)
    protocol, report = read(base / "protocol.json"), read(base / "report.json")
    assert report["status"] == "complete"
    assert report["protocol_sha256"] == sha(base / "protocol.json")
    for path, digest in protocol["inputs"].items():
        assert sha(path) == digest, path
    for parent, model in enumerate(report["models"]):
        assert (
            model["parent"] == parent
            and model["before_sha256"] == model["after_sha256"]
        )
    assert len(report["models"]) == 2
    assert len(report["batches"]) == 2 * protocol["batches"]
    totals = {
        key: 0
        for key in (
            "matches",
            "rounds",
            "active_decisions",
            "padded_rows",
            "model_forward_batches",
            "updates",
        )
    }
    maximum = 0.0
    contributions = [[], []]
    initial_hashes = {}
    for index, row in enumerate(report["batches"]):
        parent, batch = divmod(index, protocol["batches"])
        assert row["parent"] == parent and row["batch"] == batch
        assert row["deal_seed"] == protocol["first_seed"] + 64 * batch
        assert (
            row["sampling_seed"] == protocol["sampling_first"] + 10000 * parent + batch
        )
        assert row["offset_seed"] == protocol["offset_first"] + batch
        path = base / row["path"]
        assert sha(path) == row["sha256"]
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
            offsets = np.random.default_rng(row["offset_seed"]).integers(
                0, 128, size=64
            )
            np.testing.assert_array_equal(offsets, data["offsets"])
            initial = hashlib.sha256(data["initial"].tobytes()).hexdigest()
            if parent == 0:
                initial_hashes[batch] = initial
            else:
                assert initial_hashes[batch] == initial
            replay(data, row["deal_seed"])
            maximum = max(maximum, scalar_check(data))
            counts = {
                "matches": int(data["result"][..., 2].sum()),
                "rounds": int(data["result"][..., 1].sum()),
                "active_decisions": int(data["valid"].sum()),
                "padded_rows": int((~data["valid"]).sum()),
                "model_forward_batches": len(data["valid"]),
                "updates": 0,
            }
            assert counts == row["counts"] and counts["matches"] == 64
            assert counts["rounds"] == int((data["final"][:, ROUND] + 1).sum())
            for key, value in counts.items():
                totals[key] += value
            contributions[parent].append(data["contributions"].copy())
        print(
            f"Verified parent{parent + 1} batch{batch}: {counts['active_decisions']} decisions",
            flush=True,
        )
    audit = {
        "status": "passed",
        "protocol_sha256": sha(base / "protocol.json"),
        "report_sha256": sha(base / "report.json"),
        "auditor_sha256": sha(__file__),
        "totals": totals,
        "maximum_scalar_advantage_error": maximum,
        "scope": "All recorded actions replayed on CPU with shared transition core; all state, public-input and legal-mask hashes checked. Forward per-root TD expansion, scalar scores and both-seat sums verified. No neural inference or action RNG replay; replay creates no new independent matches. No learning updates.",
    }
    if protocol["mode"] == "main":
        x = np.stack([np.concatenate(parts) for parts in contributions])
        summary = summarize(x)
        summary["protocol_sha256"] = sha(base / "protocol.json")
        summary["report_sha256"] = sha(base / "report.json")
        write(base / "summary.json", summary)
        audit["summary_sha256"] = sha(base / "summary.json")
    for path, digest in protocol["inputs"].items():
        assert sha(path) == digest, path
    write(base / "audit.json", audit)
    print(audit, flush=True)


if __name__ == "__main__":
    main()
