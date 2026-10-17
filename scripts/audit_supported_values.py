"""Independent world-stream and raw-value audit for supported-world validation."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from koikoi.layout import DEALER, HISTORY, PLAYER, SEED, STOCK
from koikoi.reference import ReferenceEnv
from koikoi.train import provenance


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct_worlds(state, support, entropy, count, conditioned):
    env = ReferenceEnv.__new__(ReferenceEnv)
    env.state = state.copy()
    obs = env.observe()
    unknown = np.flatnonzero(obs[:528].reshape(48, 11)[:, 4])
    size = round(float(obs[541]) * 8)
    player, dealer = int(state[PLAYER]), int(state[DEALER])
    played = [
        int(state[HISTORY + 5 * turn]) - 1
        for turn in range(16)
        if (dealer + turn) % 2 != player and state[HISTORY + 5 * turn] > 0
    ]
    support = sorted(tuple(sorted(h)) for h in support)
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    worlds, indices = [], []
    for _ in range(count):
        if conditioned:
            index = int(rng.integers(len(support)))
            indices.append(index)
            hand = support[index]
        else:
            while True:
                hand = tuple(rng.choice(unknown, size, replace=False).tolist())
                initial = set(played) | set(hand)
                if not any(
                    set(range(4 * month, 4 * month + 4)) <= initial
                    for month in range(12)
                ):
                    break
        initial = sorted(played + list(hand))
        assert len(initial) == len(set(initial)) == 8
        world = state.copy()
        for card in unknown:
            world[card] = 1 - player if card in hand else -1
        start = 48 + 8 * (1 - player)
        world[start : start + 8] = initial
        stock = np.array([c for c in unknown if c not in hand], dtype=np.int32)
        rng.shuffle(stock)
        world[72 : 49 + state[STOCK]] = stock
        world[SEED] = rng.integers(0, 2**31 - 1)
        worlds.append(world)
    return np.stack(worlds), indices if conditioned else None


def main(pilot):
    path = Path(
        "runs/supported_values_pilot.json"
        if pilot
        else "runs/supported_values_diagnostic.json"
    )
    report = read(path)
    assert report["status"] == "complete" and report["pilot"] == pilot
    n = 2 if pilot else 1024
    assert report["particles"] == n
    meta = report["metadata"]
    for key, file in [
        ("script_sha256", "scripts/validate_supported_values.py"),
        ("rollout_sha256", "scripts/rollout_opponent.py"),
        ("sampler_sha256", "scripts/sample_history_support.py"),
        ("enumeration_sha256", "runs/history_support_enumeration.json"),
        ("audit_sha256", "runs/history_support_enumeration_audit.json"),
    ]:
        assert meta[key] == sha(file)
    assert (
        meta["audit_sha256"]
        == "ff79e702f4871f47a9136d17acde346652a830b4275b6ed72ae770d51926d7e4"
    )
    enum_meta = meta["enumeration_metadata"]
    assert enum_meta["script_sha256"] == sha("scripts/enumerate_history_support.py")
    assert enum_meta["world_helper_sha256"] == sha("scripts/audit_history_support.py")
    assert enum_meta["support_sha256"] == sha("runs/history_support_diagnostic.json")
    assert enum_meta["true_history_sha256"] == sha(
        "runs/history_support_preflight.json"
    )
    history_meta = enum_meta["history_metadata"]
    assert history_meta["script_sha256"] == sha("scripts/diagnose_history_support.py")
    assert (
        history_meta["source_sha256"]
        == provenance()["source_sha256"]
        == "b5fb6911d7a0fee266f98eca186408fad9148e7d481e743e5fb71312ce6d307f"
    )
    for p, h in history_meta["input_hashes"].items():
        assert sha(p) == h
    for name, h in history_meta["published_hashes"].items():
        assert sha(f"external/KoiKoi-AI/model_agent/{name}_rl_wp.pt") == h
    for label, directory in [
        ("seed1", "ppo8_v3"),
        ("seed10001", "ppo8_seed10001_captured"),
    ]:
        assert meta["checkpoints"][label] == sha(f"runs/{directory}/update_00005000.pt")
    manifest = read("runs/supported_values_process.json")
    if manifest["pilot"] == pilot:
        assert manifest["status"] == "complete" and manifest["output_sha256"] == sha(
            path
        )
        assert manifest["metadata"] == meta
    if not pilot:
        assert report["pilot_sha256"] == sha("runs/supported_values_pilot.json")
        assert read("runs/supported_values_pilot.json")["metadata"] == meta
    enumeration = read("runs/history_support_enumeration.json")["results"]
    if pilot:
        enumeration = [
            next(e for e in enumeration if e["identity"]["model"] == label)
            for label in ("seed1", "seed10001")
        ]
    roots = read("runs/rollout_opponent_roots.json")["roots"]
    assert len(report["results"]) == 2 * len(enumeration)
    assert len(report["distribution_differences"]) == len(enumeration)
    budgets, archive_budgets = Counter(), Counter()
    summary = {}
    for j, entry in enumerate(enumeration):
        index = entry["root_index"]
        root = roots[index]
        state = np.array(root["state"], dtype=np.int32)
        assert hashlib.sha256(state.tobytes()).hexdigest() == root["state_sha256"]
        support = [
            h["hand"]
            for h in entry["hands"]
            if h["valid"] and h["replay"]["compatible"]
        ]
        assert len(support) == entry["compatible_count"]
        stats = []
        for distribution, name in enumerate(("uniform", "supported")):
            row = report["results"][2 * j + distribution]
            ident = root["identity"]
            assert (
                row["root_index"] == index
                and row["identity"] == ident == entry["identity"]
            )
            assert row["distribution"] == name
            entropy = [
                94003 if pilot else 94009,
                ["seed1", "seed10001"].index(ident["model"]),
                ident["seed"],
                ident["seat"],
                ident["candidate_decision"],
                distribution,
            ]
            assert row["entropy"] == entropy
            worlds, indices = reconstruct_worlds(
                state, support, entropy, n, bool(distribution)
            )
            assert hashlib.sha256(worlds.tobytes()).hexdigest() == row["worlds_sha256"]
            assert row["hand_indices"] == indices
            values = np.array(row["values"], dtype=np.float64)
            assert values.shape == (4, n) and np.isfinite(values).all()
            assert (np.abs(values) <= 1).all()
            deltas = [values[1] - values[0], values[3] - values[2]]
            deltas.append(deltas[1] - deltas[0])
            current = {}
            for key, delta in zip(
                ("self_gain", "published_gain", "gain_shift"), deltas
            ):
                mean, se = float(delta.mean()), float(delta.std(ddof=1) / np.sqrt(n))
                np.testing.assert_allclose(
                    [row[key]["mean"], row[key]["se"]], [mean, se], rtol=0, atol=1e-15
                )
                current[key] = {"mean": mean, "se": se}
            stats.append(current)
            c = row["counts"]
            assert c["selfplay_exact"]
            assert (
                c["logical_transitions"]
                == c["gold_executed_transitions"]
                == 4 * n * 192
            )
            assert c["executed_transitions"] == 4 * n * c["steps"]
            assert 1 <= c["steps"] <= 192
            assert (
                0
                < c["active_transitions"]
                <= c["executed_transitions"]
                <= c["logical_transitions"]
            )
            assert 0 < c["serial_checks"] <= 4 * c["steps"]
            assert 0 <= c["feature_invariance_checks"] <= 4
            assert (
                c["published_calls"]
                == c["compared_opponent_actions"]
                == sum(p["calls"] for p in c["phases"].values())
            )
            assert c["opponent_action_disagreements"] == sum(
                p["disagreements"] for p in c["phases"].values()
            )
            assert (
                c["total_teacher_calls"]
                == c["published_calls"] + c["verification_teacher_calls"]
            )
            assert c["verification_teacher_calls"] >= c["feature_invariance_checks"]
            assert len(c["unfinished_per_arm"]) == 4
            for arm, count in enumerate(c["unfinished_per_arm"]):
                assert 0 <= count <= n
                if count == 0:
                    assert np.isin(values[arm], [-1, 0, 1]).all()
            budgets.update({k: v for k, v in c.items() if type(v) is int})
            if pilot:
                assert row["archived_exact"]
                archived = row["archive_counts"]
                assert {
                    k: v for k, v in c.items() if k != "verification_teacher_calls"
                } == archived
                archive_budgets.update(
                    {k: v for k, v in archived.items() if type(v) is int}
                )
            else:
                assert not row["archived_exact"] and row["archive_counts"] is None
            summary.setdefault(ident["model"], {}).setdefault(name, []).append(
                {
                    "root_index": index,
                    **current,
                    "unfinished": sum(c["unfinished_per_arm"]),
                }
            )
        difference = report["distribution_differences"][j]
        assert difference["root_index"] == index
        for key in ("self_gain", "published_gain", "gain_shift"):
            expected = [
                stats[1][key]["mean"] - stats[0][key]["mean"],
                (stats[0][key]["se"] ** 2 + stats[1][key]["se"] ** 2) ** 0.5,
            ]
            np.testing.assert_allclose(
                [difference[key]["mean"], difference[key]["se"]],
                expected,
                rtol=0,
                atol=1e-15,
            )
    audit = {
        "passed": True,
        "pilot": pilot,
        "report_sha256": sha(path),
        "auditor_sha256": sha(__file__),
        "budgets": dict(budgets),
        "archive_budgets": dict(archive_budgets),
        "summary": summary,
        "note": "Independent reconstruction of every world stream and raw utility statistics/hash/count audit. Not independent replay of every future trajectory. Production self-play and four serial trajectories checked per group by runner. No aggregate independent-root CI or strength claim.",
    }
    output = Path(
        "runs/supported_values_pilot_audit.json"
        if pilot
        else "runs/supported_values_independent_audit.json"
    )
    assert not output.exists()
    output.write_bytes((json.dumps(audit, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["pilot", "run"])
    main(parser.parse_args().mode == "pilot")
