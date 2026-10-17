"""Native CPU boundary replay and independent calibration/statistics checks.

No producer imports. The simulator core remains shared with the input corpus;
actor inference and action sampling are not regenerated.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from koikoi.env import BatchEnv
from koikoi.layout import DEALER, PHASE, PLAYER, POINTS, ROUND, TURN

BASE = Path("runs/round_boundary_calibration")
INPUT = Path("runs/match_return_diagnostic")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-13)


def verify_statistics(rows, probability, reported):
    y = rows[:, 4]
    estimates = {
        "public_logistic": probability,
        "network_with_cards": rows[:, 5].clip(0, 1),
        "constant_half": np.repeat(0.5, len(y)),
    }
    losses = {}
    assert reported["rows"] == len(y)
    seeds = np.unique(rows[:, 0])
    assert reported["matches"] == len(seeds) == 2048
    assert reported["network_probability_clipped"] == np.count_nonzero(
        (rows[:, 5] < 0) | (rows[:, 5] > 1)
    )
    for name, prediction in estimates.items():
        loss = np.square(prediction - y)
        losses[name] = loss
        result = reported["estimators"][name]
        equal(result["brier"], np.average(loss))
        p = prediction.clip(1e-12, 1 - 1e-12)
        equal(
            result["cross_entropy"],
            -np.average(y * np.log(p) + (1 - y) * np.log(1 - p)),
        )
        bins = []
        for i in range(10):
            mask = (prediction >= i / 10) & (
                (prediction < (i + 1) / 10) if i < 9 else (prediction <= 1)
            )
            if mask.any():
                bins.append(
                    (i, int(mask.sum()), prediction[mask].mean(), y[mask].mean())
                )
        assert len(result["bins"]) == len(bins)
        for stored, (index, count, predicted, observed) in zip(result["bins"], bins):
            assert stored["bin"] == index and stored["count"] == count
            equal([stored["predicted"], stored["observed"]], [predicted, observed])
    for i, item in enumerate(reported["by_round"]):
        mask = rows[:, 1] == i
        assert item["round"] == i and item["count"] == int(mask.sum())
        equal(item["public_brier"], losses["public_logistic"][mask].mean())
        equal(item["network_brier"], losses["network_with_cards"][mask].mean())
    # Aggregate with explicit seed selections, independently of producer bincount.
    for name, stored in reported["public_minus_other"].items():
        delta = losses["public_logistic"] - losses[name]
        aggregate = np.array(
            [
                (delta[rows[:, 0] == seed].sum(), (rows[:, 0] == seed).sum())
                for seed in seeds
            ]
        )
        rng = np.random.default_rng(99)
        boot = np.empty(5000)
        for i in range(len(boot)):
            sampled = aggregate[rng.integers(len(seeds), size=len(seeds))].sum(axis=0)
            boot[i] = sampled[0] / sampled[1]
        equal(stored["brier_difference"], np.average(delta))
        equal(stored["match_cluster_ci95"], np.percentile(boot, [2.5, 97.5]))


def main():
    torch.set_num_threads(2)
    report = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    protocol = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    upstream = json.loads((INPUT / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == upstream["status"] == "complete"
    assert report["protocol"] == protocol["protocol"]
    assert (
        report["input_report_sha256"]
        == sha(INPUT / "report.json")
        == ("a107da18790ed4821a1e407295f2f2aac35c22618c26fcedcd783b84e93cc3bd")
    )
    assert report["producer_sha256"] == sha("scripts/calibrate_round_boundaries.py")
    assert report["data_sha256"] == sha(BASE / "data.npz")
    source = hashlib.sha256()
    for path in sorted(Path("koikoi").rglob("*")):
        if path.suffix in (".py", ".h", ".cpp", ".cu"):
            source.update(path.as_posix().encode())
            source.update(path.read_bytes())
    assert source.hexdigest() == report["production_source_sha256"]
    for key in ("producer_sha256", "production_source_sha256", "input_report_sha256"):
        assert protocol[key] == report[key]
    assert protocol["protocol"] == {
        "fit_batches": [0, 1],
        "holdout_batches": [2, 3],
        "ridge": 0.0001,
        "features": ["(points0-30)/30", "2*dealer0-1"],
        "intercept": False,
        "separate_round_fits": 8,
        "bootstrap_matches": 5000,
        "bootstrap_seed": 99,
        "fields": [
            "seed",
            "round",
            "points0",
            "dealer0",
            "outcome0",
            "network_probability",
            "batch",
            "step",
            "env",
        ],
        "new_simulations": 0,
        "policy_updates": 0,
    }
    totals = {
        "matches_replayed": 0,
        "boundaries": 0,
        "active_decisions_replayed": 0,
        "executed_replay_rows": 0,
    }
    maxima = []
    with np.load(BASE / "data.npz", allow_pickle=False) as stored:
        for label, checkpoint, parent_sha, first_seed in upstream["protocol"][
            "parents"
        ]:
            result = report["cases"][label]
            assert result["parent_sha256"] == sha(checkpoint) == parent_sha
            records = []
            for batch, metadata in enumerate(upstream["cases"][label]["batches"]):
                path = INPUT / f"{label}_{batch}.npz"
                assert sha(path) == metadata["raw_sha256"]
                assert metadata["seed"] == first_seed + 1024 * batch
                with np.load(path, allow_pickle=False) as archive:
                    data = {
                        key: archive[key]
                        for key in (
                            "initial",
                            "final",
                            "action",
                            "value",
                            "state_digests",
                            "result",
                        )
                    }
                    env = BatchEnv(1024, "cpu", "paper8", metadata["seed"])
                    np.testing.assert_array_equal(env.state.numpy(), data["initial"])
                    last_round = np.full(1024, -1)
                    for time, actions in enumerate(data["action"]):
                        state = env.state.numpy()
                        assert (
                            hashlib.sha256(state.tobytes()).hexdigest()
                            == data["state_digests"][time]
                        )
                        active = state[:, PHASE] != 4
                        fresh = active & (state[:, ROUND] != last_round)
                        assert np.all(state[fresh, PHASE] == 0) and np.all(
                            state[fresh, TURN] == 0
                        )
                        assert np.all(state[fresh, PLAYER] == state[fresh, DEALER])
                        for index in np.flatnonzero(fresh):
                            s = state[index]
                            final = data["final"][index, POINTS : POINTS + 2]
                            outcome = (
                                0.5
                                if final[0] == final[1]
                                else float(final[0] > final[1])
                            )
                            value = float(data["value"][time, index])
                            chance = (
                                (1 + value) / 2 if s[PLAYER] == 0 else (1 - value) / 2
                            )
                            records.append(
                                [
                                    metadata["seed"] + index,
                                    s[ROUND],
                                    s[POINTS],
                                    int(s[DEALER] == 0),
                                    outcome,
                                    chance,
                                    batch,
                                    time,
                                    index,
                                ]
                            )
                        last_round = state[:, ROUND].copy()
                        totals["active_decisions_replayed"] += int(active.sum())
                        totals["executed_replay_rows"] += len(active)
                        actual = env.step(
                            torch.from_numpy(actions.copy()), freeze_done=True
                        ).numpy()
                        np.testing.assert_array_equal(actual, data["result"][time])
                    np.testing.assert_array_equal(env.state.numpy(), data["final"])
                    assert np.all(env.state.numpy()[:, PHASE] == 4)
                    totals["matches_replayed"] += 1024
            rows = np.asarray(records, dtype=np.float64)
            np.testing.assert_array_equal(rows, stored[f"{label}_rows"])
            totals["boundaries"] += len(rows)
            fit = rows[np.isin(rows[:, 6], [0, 1])]
            test = rows[np.isin(rows[:, 6], [2, 3])]
            assert len(np.unique(fit[:, 0])) == len(np.unique(test[:, 0])) == 2048
            assert not set(fit[:, 0]).intersection(test[:, 0])
            assert result["fit_rows"] == len(fit)
            weights = stored[f"{label}_weights"]
            assert weights.shape == (8, 2) and len(result["fits"]) == 8
            for i in range(8):
                subset = fit[fit[:, 1] == i]
                x = np.stack(
                    [subset[:, 2] / 30 - 1, np.where(subset[:, 3] == 1, 1.0, -1.0)],
                    axis=1,
                )
                z = x @ weights[i]
                p = 1 / (1 + np.exp(-z))
                # Strict convexity from positive ridge makes stationarity sufficient.
                gradient = (
                    np.mean(x * (p - subset[:, 4])[:, None], axis=0) + 1e-4 * weights[i]
                )
                maximum = float(np.abs(gradient).max())
                maxima.append(maximum)
                assert maximum < 1e-10
                item = result["fits"][i]
                assert item["round"] == i and item["rows"] == len(subset)
                equal(item["coefficients"], weights[i])
                equal(item["gradient_max"], maximum)
                equal(
                    item["objective"],
                    np.average(np.logaddexp(0, z) - subset[:, 4] * z)
                    + 5e-5 * np.square(weights[i]).sum(),
                )
            assert (
                weights[0, 0] == 0
                and result["round0_score_coefficient_identifiable"] is False
            )
            prediction = np.array(
                [
                    1
                    / (
                        1
                        + np.exp(
                            -np.dot(
                                weights[int(row[1])],
                                [(row[2] - 30) / 30, 1 if row[3] else -1],
                            )
                        )
                    )
                    for row in test
                ]
            )
            equal(prediction, stored[f"{label}_prediction"])
            counts = Counter(tuple(row[1:4]) for row in fit)
            support = np.array([counts[tuple(row[1:4])] for row in test])
            np.testing.assert_array_equal(support, stored[f"{label}_fit_cell_counts"])
            assert result["holdout_unseen_cells"] == int((support == 0).sum())
            assert result["holdout_cells_with_fewer_than20_fit_rows"] == int(
                (support < 20).sum()
            )
            verify_statistics(test, prediction, result["holdout"])
            print(
                f"{label}: replay, fit stationarity and holdout statistics passed",
                flush=True,
            )
    assert totals == {
        "matches_replayed": 8192,
        "boundaries": 63974,
        "active_decisions_replayed": 960241,
        "executed_replay_rows": 1189888,
    }
    audit = {
        "status": "passed",
        "auditor_sha256": sha(__file__),
        "report_sha256": sha(BASE / "report.json"),
        "protocol_sha256": sha(BASE / "protocol.json"),
        "data_sha256": sha(BASE / "data.npz"),
        "totals": totals,
        "maximum_fit_stationarity_error": max(maxima),
        "production_source_sha256": source.hexdigest(),
        "original_corpus_source_sha256": upstream["source_sha256"],
        "scope": "All original trajectories replayed with native CPU; boundary points/dealer/round and labels reconstructed independently. Fit-only stationarity, predictions, support, holdout losses/bins and whole-match bootstrap checked without producer helpers. Native transition core shared; actor inference/RNG not regenerated. Replay is verification, not new independent simulation or strength evidence.",
    }
    (BASE / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
