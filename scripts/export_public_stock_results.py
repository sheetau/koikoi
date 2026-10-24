"""Independently reconstruct public-stock scores/counters and paired intervals."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from koikoi.layout import PHASE, POINTS, TERMINAL
from koikoi.public_stock import PROFILES

CONTROL = "preview0_choice0_stakes0"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def reduce_endpoint(folder, bindings):
    result = read(folder / "result.json")
    config = result["config"]
    pairs, batch_size, seed = (config[k] for k in ("pairs", "batch", "deal_seed"))
    bindings[str(folder / "result.json")] = sha(folder / "result.json")
    bindings[str(folder / "protocol.json")] = sha(folder / "protocol.json")
    if sha(config["checkpoint"]) != result["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    bindings[config["checkpoint"]] = result["checkpoint_sha256"]
    for path, digest in result["source_hashes"].items():
        if sha(path) != digest:
            raise ValueError(f"Source changed: {path}")
        bindings[path] = digest
    scores, margins, decisions, rounds = [], [], 0, 0
    for offset in range(0, pairs, batch_size):
        file = folder / f"trajectories_{offset:05d}.npz"
        bindings[str(file)] = sha(file)
        data = np.load(file)
        size = min(batch_size, pairs - offset)
        np.testing.assert_array_equal(
            data["seeds"], list(range(seed + offset, seed + offset + size)) * 2
        )
        final, actions = data["final"], data["actions"]
        if not np.all(final[:, PHASE] == TERMINAL):
            raise ValueError("Incomplete match included in score")
        decisions += int((actions >= 0).sum())
        rounds += int(final[:, 114].sum())
        margin = (final[:, POINTS] - final[:, POINTS + 1]) * np.repeat([1, -1], size)
        margins.extend(margin.reshape(2, size).mean(0))
        scores.extend(((np.sign(margin) + 1) / 2).reshape(2, size).mean(0))
    scores, margins = np.asarray(scores), np.asarray(margins)
    np.testing.assert_array_equal(scores, result["scores"])
    np.testing.assert_array_equal(margins, result["margins"])
    np.testing.assert_array_equal(result["seeds"], np.arange(seed, seed + pairs))
    if (
        result["matches"] != 2 * pairs
        or decisions != result["decisions"]
        or rounds != result["rounds"]
    ):
        raise ValueError("Recorded counters differ from independent reduction")
    if (
        decisions != result["python_replay_decisions"]
        or scores.mean() != result["win_rate"]
    ):
        raise ValueError("Replay or mean score differs")
    return scores, result


def intervals(values, indices, quantile=0.025):
    samples = values[indices].mean(1)
    return {
        "mean": float(values.mean()),
        "lower": float(np.quantile(samples, quantile)),
        "upper": float(np.quantile(samples, 1 - quantile)),
    }


def reduce_study(base, confirmation=None):
    protocol, screen = read(base / "protocol.json"), read(base / "screen_result.json")
    bindings = {
        str(base / name): sha(base / name)
        for name in ("protocol.json", "screen_result.json", "qualification.json")
    }
    for path, digest in protocol["sources"].items():
        if sha(path) != digest:
            raise ValueError(f"Frozen study input changed: {path}")
        bindings[path] = digest
    curves, endpoints, training = {}, {}, {}
    for profile in PROFILES:
        curves[profile] = []
        for parent in range(2):
            train_dir = base / f"{profile}_parent{parent}"
            train_config = read(train_dir / "config.json")
            train_rows = [
                json.loads(line)
                for line in (train_dir / "progress.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            trained = train_rows[-1]
            if (
                trained["decisions"] != protocol["decisions_per_model"]
                or trained["updates"] != protocol["updates"]
            ):
                raise ValueError("Training budget does not match protocol")
            if (
                train_config["config"]["profile"] != profile
                or train_config["config"]["parent"] != parent
            ):
                raise ValueError("Wrong training profile or parent")
            for name in ("config.json", "progress.jsonl"):
                bindings[str(train_dir / name)] = sha(train_dir / name)
            training[f"{profile}_parent{parent}"] = trained
            rows = []
            for particles in protocol["particles"]:
                label = f"{profile}_parent{parent}_p{particles}"
                scores, result = reduce_endpoint(base / f"screen_{label}", bindings)
                if (
                    result["profile"] != profile
                    or result["config"]["particles"] != particles
                ):
                    raise ValueError("Mislabeled screen endpoint")
                rows.append(scores)
                endpoints[label] = {
                    k: result[k]
                    for k in (
                        "matches",
                        "rounds",
                        "decisions",
                        "rollouts",
                        "transitions",
                        "roots",
                        "seconds",
                        "python_replay_decisions",
                    )
                }
            curves[profile].append(rows)
        curves[profile] = np.asarray(curves[profile])  # parent, budget, deal
        np.testing.assert_array_equal(
            curves[profile].mean(2), screen["curves"][profile]
        )
    gains = {profile: data[:, -1] - data[:, 0] for profile, data in curves.items()}
    selected = min(
        (p for p in PROFILES if p != CONTROL),
        key=lambda p: (-gains[p].mean(1).min(), p),
    )
    if selected != screen["selected"]:
        raise ValueError("Selection does not reproduce")
    gate = gains[selected].mean(1).min() > max(0.0, gains[CONTROL].mean(1).min())
    if bool(gate) != screen["confirmation_gate_passed"]:
        raise ValueError("Gate does not reproduce")
    indices = np.random.default_rng(951).integers(
        0, protocol["development_pairs"], (20000, protocol["development_pairs"])
    )
    summary = {
        "selected": selected,
        "confirmation_gate_passed": bool(gate),
        "curves": {p: x.mean(2).tolist() for p, x in curves.items()},
        "development_gain_intervals_exploratory": {
            p: [intervals(row, indices) for row in x] for p, x in gains.items()
        },
        "screen_endpoints": endpoints,
        "training": training,
        "training_totals": {
            k: sum(row[k] for row in training.values())
            for k in ("decisions", "rounds", "matches", "updates")
        },
        "screen_totals": {
            k: sum(r[k] for r in endpoints.values())
            for k in (
                "matches",
                "rounds",
                "decisions",
                "rollouts",
                "transitions",
                "roots",
                "python_replay_decisions",
            )
        },
    }
    summary["development_curve_gain_intervals_exploratory"] = {
        p: [
            [intervals(budget - parent[0], indices) for budget in parent]
            for parent in data
        ]
        for p, data in curves.items()
    }
    effects = {}
    for name, factors in (
        ("preview", [0]),
        ("choice", [1]),
        ("stakes", [2]),
        ("preview_choice", [0, 1]),
        ("preview_stakes", [0, 2]),
        ("choice_stakes", [1, 2]),
        ("three_way", [0, 1, 2]),
    ):
        contrast = np.zeros_like(gains[CONTROL])
        for profile, values in gains.items():
            levels = [int(part[-1]) for part in profile.split("_")]
            sign = np.prod([2 * levels[factor] - 1 for factor in factors])
            contrast += sign * values * 2 ** len(factors) / len(PROFILES)
        effects[name] = [intervals(row, indices) for row in contrast]
    summary["factorial_gain_contrasts_exploratory"] = effects
    if confirmation is not None:
        cp = read(confirmation / "protocol.json")
        if not gate or cp["profiles"] != [CONTROL, selected]:
            raise ValueError("Unqualified confirmation")
        bindings[str(confirmation / "protocol.json")] = sha(
            confirmation / "protocol.json"
        )
        data, work = {}, {}
        for profile in [CONTROL, selected]:
            data[profile] = []
            for parent in range(2):
                rows = []
                for particles in protocol["particles"]:
                    label = f"{profile}_parent{parent}_p{particles}"
                    scores, result = reduce_endpoint(confirmation / label, bindings)
                    rows.append(scores)
                    work[label] = {
                        k: result[k]
                        for k in (
                            "matches",
                            "rounds",
                            "decisions",
                            "rollouts",
                            "transitions",
                            "python_replay_decisions",
                        )
                    }
                data[profile].append(rows)
            data[profile] = np.asarray(data[profile])
        indices = np.random.default_rng(cp["bootstrap_seed"]).integers(
            0, cp["pairs"], (cp["bootstrap_replicates"], cp["pairs"])
        )
        gain = {p: x[:, -1] - x[:, 0] for p, x in data.items()}
        checks = {
            "selected_gain": [
                intervals(row, indices, cp["lower_quantile"]) for row in gain[selected]
            ],
            "gain_minus_control": [
                intervals(row, indices, cp["lower_quantile"])
                for row in gain[selected] - gain[CONTROL]
            ],
        }
        summary["confirmation"] = {
            "checks": checks,
            "passed": all(x["lower"] > 0 for rows in checks.values() for x in rows),
            "curves": {p: x.mean(2).tolist() for p, x in data.items()},
            "endpoints": work,
        }
    summary["bindings"] = bindings
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", required=True)
    p.add_argument("--confirmation")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    result = reduce_study(
        Path(args.study), Path(args.confirmation) if args.confirmation else None
    )
    write(args.out, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("selected", "confirmation_gate_passed", "screen_totals")
            }
        )
    )


if __name__ == "__main__":
    main()
