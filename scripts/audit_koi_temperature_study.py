"""Saved-learning checks, full recorded-action replay and independent statistics."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

from koikoi.layout import PHASE, POINTS, TERMINAL
from koikoi.published_actor import PublishedActor
from koikoi.reference import ReferenceEnv
from koikoi.upstream_batch import parameter_digest
from scripts.audit_koi_temperature_capability import same
from scripts.run_arbitration_study import PARENTS, read, sha, write

POLICIES = ("parent", "fixed", "calibrated")
OPPONENTS = ("wp", "point")
WORK = ("decisions", "round_endings", "full_matches")


def checked(inputs):
    for name, digest in inputs.items():
        assert sha(name) == digest, name


def independent_interval(values, seed, replicates):
    """Use a separate resampling implementation, preserving whole paired blocks."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = []
    for start in range(0, replicates, 37):
        indices = rng.integers(
            0, len(values), (min(37, replicates - start), len(values))
        )
        means.extend(np.sum(values[indices], axis=1) / len(values))
    return np.percentile(means, [2.5, 97.5]).tolist()


def independent_statistics(points, seed, replicates):
    points = np.asarray(points)
    assert points.ndim == 5 and points.shape[:3] == (2, 3, 2)
    assert points.shape[-1] == 2 and points.shape[3] >= 2
    assert points.dtype.kind in "iu" and seed >= 0 and replicates >= 2
    absolute, contrasts = [], []
    for parent in (1, 2):
        for scope, groups in (("mixture", (0, 1)), ("wp", (0,)), ("point", (1,))):
            values = []
            for policy in range(3):
                sample = points[parent - 1, policy, list(groups)]
                scores = np.where(sample > 0, 1.0, np.where(sample < 0, 0.0, 0.5))
                win = scores.sum(axis=(0, 2)) / (2 * len(groups))
                margin = sample.sum(axis=(0, 2)) / (2 * len(groups))
                values.append((win, margin))
                absolute.append(
                    {
                        "parent": parent,
                        "policy": POLICIES[policy],
                        "scope": scope,
                        "win": float(win.mean()),
                        "points": float(margin.mean()),
                        "win_interval": independent_interval(win, seed, replicates),
                        "point_interval": independent_interval(
                            margin, seed, replicates
                        ),
                    }
                )
            for left, right in ((2, 1), (2, 0), (1, 0)):
                win = values[left][0] - values[right][0]
                margin = values[left][1] - values[right][1]
                contrasts.append(
                    {
                        "parent": parent,
                        "left": POLICIES[left],
                        "right": POLICIES[right],
                        "scope": scope,
                        "win_change": float(win.mean()),
                        "point_change": float(margin.mean()),
                        "win_interval": independent_interval(win, seed, replicates),
                        "point_interval": independent_interval(
                            margin, seed, replicates
                        ),
                    }
                )
    primary = [
        r for r in contrasts if r["scope"] == "mixture" and r["left"] == "calibrated"
    ]
    assert len(primary) == 4
    return {
        "absolute": absolute,
        "contrasts": contrasts,
        "replicated_strength_gain": all(r["win_interval"][0] > 0 for r in primary),
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
    }


def replay(row):
    assert row["seat"] in (0, 1)
    env = ReferenceEnv(row["seed"], "paper8")
    rounds = 0
    for action in row["actions"]:
        assert type(action) is int and 0 <= action < 50
        assert env.state[PHASE] != TERMINAL and env.legal()[action]
        result = env.step(action)
        assert int(result[0]) != 99
        rounds += int(result[1])
    assert env.state[PHASE] == TERMINAL
    assert env.state.tolist() == row["endpoint"]
    difference = int(env.state[POINTS] - env.state[POINTS + 1])
    assert type(row["points"]) is int
    assert row["points"] == (difference if row["seat"] == 0 else -difference)
    work = {
        "decisions": len(row["actions"]),
        "round_endings": rounds,
        "full_matches": 1,
    }
    assert row["work"] == work
    return work


def audit_cell(root, parent, policy, opponent, checkpoint, pairs, seed_base):
    stem = root / "evaluation" / f"parent{parent}_{policy}_{opponent}"
    report_path, rows_path = stem.with_suffix(".json"), stem.with_suffix(".jsonl")
    report = read(report_path)
    assert report["status"] == "complete"
    assert (report["parent"], report["policy"], report["opponent"]) == (
        parent,
        policy,
        opponent,
    )
    assert Path(report["checkpoint"]).resolve() == Path(checkpoint).resolve()
    assert report["checkpoint_sha256"] == sha(checkpoint)
    assert report["rows_sha256"] == sha(rows_path)
    values = np.empty((pairs, 2), dtype=np.int64)
    total = dict.fromkeys(WORK, 0)
    with rows_path.open(encoding="utf-8") as stream:
        for rep in range(pairs):
            for seat in (0, 1):
                row = json.loads(next(stream))
                expected = {
                    "parent": parent,
                    "opponent": opponent,
                    "replicate": rep,
                    "seat": seat,
                    "seed": seed_base + (10000 if opponent == "point" else 0) + rep,
                }
                assert all(row[key] == value for key, value in expected.items())
                for key, value in replay(row).items():
                    total[key] += value
                values[rep, seat] = row["points"]
        assert not stream.read().strip(), "Unexpected extra match rows"
    assert total == report["work"] and total["full_matches"] == 2 * pairs
    return (
        values,
        total,
        {str(p): sha(p) for p in (report_path, rows_path, Path(checkpoint))},
    )


def check_optimizer(payload, parameter_names, update):
    cfg, counters, optimizer = (
        payload["config"],
        payload["counters"],
        payload["optimizer"],
    )
    groups = optimizer["param_groups"]
    assert len(groups) == 2 and (groups[0]["lr"], groups[1]["lr"]) == (
        cfg["lr"],
        cfg["value_lr"],
    )
    assert len(parameter_names) == len(groups[0]["params"])
    steps = cfg["epochs"] * (
        (cfg["envs"] * cfg["horizon"] + cfg["minibatch"] - 1) // cfg["minibatch"]
    )
    assert counters["optimizer_steps_value"] == update * steps
    active = max(0, update - cfg["critic_warmup_updates"])
    phase_names = {"0": "discard", "1": "pick", "3": "koi"}
    expected_indices = set(groups[1]["params"]) if update else set()
    for name, index in zip(parameter_names, groups[0]["params"]):
        expected = counters["optimizer_steps_" + phase_names[name.split(".")[1]]]
        assert 0 <= expected <= active * steps
        if expected:
            expected_indices.add(index)
            assert int(optimizer["state"][index]["step"].item()) == expected
        else:
            assert index not in optimizer["state"]
    for index in groups[1]["params"]:
        if update:
            assert int(optimizer["state"][index]["step"].item()) == update * steps
    assert set(optimizer["state"]) == expected_indices
    assert all(
        bool(torch.isfinite(v).all())
        for item in optimizer["state"].values()
        for v in item.values()
    )


def audit_arm(root, key, config, original_actor, parameter_names):
    folder = root / key
    assert read(folder / "config.json") == config
    logs = [
        json.loads(s)
        for s in (folder / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [r["update"] for r in logs] == list(range(1, 521))
    for update, row in enumerate(logs, 1):
        c = row["counters"]
        assert c["updates"] == update and c["decisions"] == update * 16384
        assert c["warmup_updates"] == min(update, 20)
        assert c["policy_updates"] == max(0, update - 20)
        assert c["warmup_decisions"] == min(update, 20) * 16384
        assert row["metrics"]["critic_only"] == (update <= 20)
        assert all(
            np.isfinite(v)
            for v in row["metrics"].values()
            if isinstance(v, (float, int))
        )
        assert np.isfinite(row["seconds"]) and row["seconds"] >= 0
        assert all(type(v) is int and v >= 0 for v in c.values())
        if update > 1:
            assert row["seconds"] >= logs[update - 2]["seconds"]
            assert all(v >= logs[update - 2]["counters"][k] for k, v in c.items())
        if update > 20:
            assert all(
                c[k] == logs[19]["counters"][k] for k in c if k.startswith("warmup_")
            )
    paths = [folder / n for n in ("config.json", "metrics.jsonl", "complete.json")]
    endpoints = (
        [("initial.pt", 0)]
        + [(f"update_{u:08d}.pt", u) for u in (20, 100, 200, 300, 400, 500, 520)]
        + [("final.pt", 520)]
    )
    initial = last = None
    for filename, update in endpoints:
        path = folder / filename
        payload = torch.load(path, map_location="cpu", weights_only=False)
        paths.append(path)
        assert (
            payload["format"] == "koi_temperature_ppo_v1"
            and payload["config"] == config
        )
        assert payload["actor"]["initialization"] == original_actor["initialization"]
        assert (
            payload["provenance"]["published_initialization"]
            == original_actor["initialization"]
        )
        assert payload["provenance"]["koi_temperature_source"] == sha(
            "scripts/koi_temperature_ppo.py"
        )
        assert payload["actor"]["current"]["parameters_sha256"] == parameter_digest(
            payload["actor"]["current"]["weights"]
        )
        assert all(
            bool(torch.isfinite(t).all())
            for weights in payload["actor"]["current"]["weights"].values()
            for t in weights.values()
        )
        assert all(bool(torch.isfinite(t).all()) for t in payload["value"].values())
        check_optimizer(payload, parameter_names, update)
        if update == 0:
            initial = payload
            same(initial["actor"], original_actor)
            assert set(initial["counters"].values()) == {0}
        else:
            assert payload["counters"] == logs[update - 1]["counters"]
            same(payload["provenance"], initial["provenance"])
            same(payload["runtime"], initial["runtime"])
        if update == 20:
            same(initial["actor"], payload["actor"])
        if filename == "final.pt":
            same(payload, last)
        last = payload
    for phase, weights in initial["actor"]["current"]["weights"].items():
        assert any(
            not torch.equal(v, last["actor"]["current"]["weights"][phase][k])
            for k, v in weights.items()
        )
    receipt = read(folder / "complete.json")
    assert receipt == {
        "status": "complete",
        "counters": last["counters"],
        "checkpoint_sha256": sha(folder / "final.pt"),
    }
    return initial, last["counters"], {str(p): sha(p) for p in paths}


def audit(root):
    output = root / "independent_audit.json"
    if output.exists():
        raise ValueError("Preserve completed independent audit")
    plan = read(root / "protocol.json")
    assert (
        sha(root / "protocol.json")
        == "b9075ac261ce6a3e681a64fa7845f031850ef20fba60f618435b3ad108a74d19"
    )
    assert read(root / "process.json")["status"] == "complete_awaiting_audit"
    assert (
        plan["updates"],
        plan["pairs"],
        plan["evaluation_seed"],
        plan["bootstrap_seed"],
        plan["bootstrap_replicates"],
        plan["full_matches"],
        plan["shared_deals"],
    ) == (520, 2048, 97110001, 97310001, 10000, 49152, 4096)
    assert plan["policies"] == list(POLICIES) and plan["opponents"] == list(OPPONENTS)
    inputs = dict(plan["inputs"])
    for p in (
        root / "protocol.json",
        root / "process.json",
        Path(__file__),
        Path("scripts/audit_koi_temperature_capability.py"),
        Path("tests/test_audit_koi_temperature_study.py"),
        Path("runs/koi_temperature_audit_tests.xml"),
    ):
        inputs[str(p)] = sha(p)
    checked(inputs)
    suite = ET.parse("runs/koi_temperature_audit_tests.xml").getroot().find("testsuite")
    assert int(suite.attrib["tests"]) >= 3
    assert all(int(suite.attrib[k]) == 0 for k in ("errors", "failures", "skipped"))
    original = PublishedActor.from_published("wp", "cpu")
    original_actor = original.checkpoint()
    names = [name for name, _ in original.named_parameters()]
    training = {}
    for parent in (1, 2):
        initial = []
        for mode in ("fixed", "calibrated"):
            key = f"seed{parent}_{mode}"
            config = plan["configs"][key]
            assert config["seed"] == (96110001 if parent == 1 else 96210001)
            assert config["koi_temperature"] == (
                0.05 if mode == "fixed" else 0.21989584747027907
            )
            state, counters, bindings = audit_arm(
                root, key, config, original_actor, names
            )
            initial.append(state)
            training[key] = counters
            inputs.update(bindings)
        a, b = initial
        assert {k for k in a["config"] if a["config"][k] != b["config"][k]} == {
            "koi_temperature"
        }
        for key in (
            "actor",
            "value",
            "optimizer",
            "env",
            "counters",
            "torch_rng",
            "cuda_rng",
            "python_rng",
            "numpy_rng",
            "runtime",
        ):
            same(a[key], b[key])
    points = np.empty((2, 3, 2, 2048, 2), dtype=np.int64)
    total = dict.fromkeys(WORK, 0)
    cells = {}
    for parent in (1, 2):
        for pi, policy in enumerate(POLICIES):
            checkpoint = (
                Path(PARENTS[parent - 1][0])
                if policy == "parent"
                else root / f"seed{parent}_{policy}/final.pt"
            )
            if policy == "parent":
                assert sha(checkpoint) == PARENTS[parent - 1][1]
            for oi, opponent in enumerate(OPPONENTS):
                values, work, bindings = audit_cell(
                    root, parent, policy, opponent, checkpoint, 2048, 97110001
                )
                points[parent - 1, pi, oi] = values
                cells[f"parent{parent}_{policy}_{opponent}"] = work
                inputs.update(bindings)
                for k in total:
                    total[k] += work[k]
                print(
                    json.dumps(
                        {
                            "replayed_cells": len(cells),
                            "full_matches": total["full_matches"],
                        }
                    ),
                    flush=True,
                )
    assert total["full_matches"] == 49152
    result = independent_statistics(points, 97310001, 10000)
    from scripts.koi_temperature_analysis import analyse

    produced = analyse(points)
    assert all(result[key] == produced[key] for key in result)
    checked(inputs)
    write(
        output,
        {
            "status": "passed",
            "inputs": inputs,
            "analysis": result,
            "training_counters": training,
            "evaluation_work": total,
            "cells": cells,
            "evaluation_replay_work": total,
            "shared_evaluation_deals": 4096,
            "new_neural_inferences": 0,
            "new_independent_matches": 0,
            "scope": "All saved schedules, initial weights/RNG, Adam steps, match actions/endpoints/work and paired statistics. No replay of training, neural action selection or independent learning run.",
        },
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "replicated_strength_gain": result["replicated_strength_gain"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
