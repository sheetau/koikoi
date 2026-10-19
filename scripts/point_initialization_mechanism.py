"""Fixed-input endpoint diagnostics for the completed Point initialization study."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from koikoi.published_actor import PublishedActor
from koikoi.published_opponent import runtime_identity
from scripts.koi_temperature_mechanism import (
    CORPUS,
    CORPUS_SHA,
    checked,
    public_inputs,
    score_public,
)
from scripts.phase_exploration_calibration import choice_statistics
from scripts.run_arbitration_study import read, sha

STUDY = Path("runs/point_initialization_study")
MAIN = Path("runs/point_initialization_evaluation/evaluation_protocol.json")
MAIN_SHA = "1df245b6fea1799c0ed9f91518642ccb3d1e4bf9c0f7e682a1208819e609b7e7"
ARMS = tuple(
    f"seed{seed}_{mode}"
    for seed in (1, 2)
    for mode in ("wp_fixed", "point_fixed", "point_matched")
)


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validation_indices(sources):
    sources = np.asarray(sources)
    return np.flatnonzero((sources >= 0) & (sources % 4 == 3))


def greedy(scores, mask):
    result = np.full(len(mask), -1, dtype=np.int64)
    eligible = mask.sum(1) > 1
    values = np.where(mask[eligible], scores[eligible], -np.inf).astype(np.float32)
    assert np.isfinite(values[mask[eligible]]).all()
    result[eligible] = (values / np.float32(0.05)).argmax(1)
    return result


def describe(
    scores, mask, phases, actions, wp_actions, point_actions, variant, temperatures
):
    if len(temperatures) != 4:
        raise ValueError("Four declared phase divisors required")
    original = wp_actions if variant == "wp" else point_actions
    if variant not in ("wp", "point"):
        raise ValueError("Unknown initialization")
    result = {}
    for phase in range(4):
        all_rows = phases == phase
        rows = all_rows & (mask.sum(1) > 1)
        record = {"all_rows": int(all_rows.sum()), "unforced_rows": int(rows.sum())}
        if rows.any():
            values, legal = scores[rows], mask[rows]
            ordered = np.sort(np.where(legal, values.astype(float), -np.inf), axis=1)
            current, initial = actions[rows], original[rows]
            assert (current >= 0).all() and legal[np.arange(len(legal)), current].all()
            record.update(
                {
                    "mean_top_two_raw_score_gap": float(
                        (ordered[:, -1] - ordered[:, -2]).mean()
                    ),
                    "common_temperature": choice_statistics(values, legal, 0.05),
                    "declared_temperature": choice_statistics(
                        values, legal, temperatures[phase]
                    ),
                    "changed_vs_wp": int((current != wp_actions[rows]).sum()),
                    "changed_vs_point": int((current != point_actions[rows]).sum()),
                    "changed_vs_initial": int((current != initial).sum()),
                    "stop_to_continue_vs_initial": int(
                        ((initial == 48) & (current == 49)).sum()
                    ),
                    "continue_to_stop_vs_initial": int(
                        ((initial == 49) & (current == 48)).sum()
                    ),
                }
            )
        result[str(phase)] = record
    return result


def disagreements(raw):
    result = {}
    for seed in (1, 2):
        for first, second in (
            ("wp_fixed", "point_fixed"),
            ("wp_fixed", "point_matched"),
            ("point_fixed", "point_matched"),
        ):
            a, b = f"seed{seed}_{first}", f"seed{seed}_{second}"
            result[f"{a}__{b}"] = {
                str(phase): int(
                    np.count_nonzero(
                        (raw["phases"] == phase)
                        & (raw["mask"].sum(1) > 1)
                        & (raw[a + "_actions"] != raw[b + "_actions"])
                    )
                )
                for phase in range(4)
            }
    return result


def prepare(root):
    if root.exists():
        raise ValueError("Use a fresh diagnostic directory")
    assert sha(MAIN) == MAIN_SHA and sha(CORPUS) == CORPUS_SHA
    main, complete = read(MAIN), read(STUDY / "complete.json")
    assert complete["status"] == "complete" and set(complete["arms"]) == set(ARMS)
    inputs = dict(main["inputs"])
    for path in (
        MAIN,
        Path(__file__),
        Path("scripts/audit_point_initialization_mechanism.py"),
        Path("tests/test_point_initialization_mechanism.py"),
        Path("runs/point_initialization_mechanism_tests.xml"),
        Path("docs/POINT_INITIALIZATION_MECHANISM.rst"),
        Path("scripts/audit_koi_temperature_mechanism.py"),
        Path("scripts/koi_temperature_mechanism.py"),
        CORPUS,
        STUDY / "complete.json",
    ):
        inputs[str(path)] = sha(path)
    models = {}
    for arm in ARMS:
        path = STUDY / arm / "update_00000520.pt"
        assert sha(path) == complete["arms"][arm]["checkpoint_sha256"]
        config = read(STUDY / arm / "config.json")
        assert config["seed"] in (102110001, 102210001)
        models[arm] = {
            "path": str(path),
            "variant": config["initial_variant"],
            "temperatures": config["phase_temperatures"],
        }
    temperatures = read("runs/point_initialization_calibration.json")[
        "phase_temperatures"
    ]
    with np.load(CORPUS, allow_pickle=False) as old:
        indices = validation_indices(old["sources"])
        assert len(indices) == 3258 and len(np.unique(old["sources"][indices])) == 25
    checked(inputs)
    root.mkdir(parents=True, exist_ok=False)
    write_new(
        root / "protocol.json",
        {
            "inputs": inputs,
            "models": models,
            "initial_point_temperatures": temperatures,
            "indices": indices.tolist(),
            "scope": "Descriptive reused public inputs only; no outcomes, selection, new trajectories or strength claim.",
        },
    )
    print(
        {
            "prepared": str(root),
            "protocol_sha256": sha(root / "protocol.json"),
            "inputs": len(inputs),
        },
        flush=True,
    )


def run(root):
    if any(
        (root / name).exists()
        for name in ("started.json", "scores.npz", "report.json", "failed.json")
    ):
        raise ValueError("Preserve completed or partial diagnostics")
    plan = read(root / "protocol.json")
    checked(plan["inputs"])
    write_new(
        root / "started.json",
        {"pid": os.getpid(), "protocol_sha256": sha(root / "protocol.json")},
    )
    try:
        with np.load(CORPUS, allow_pickle=False) as old:
            indices = np.array(plan["indices"], dtype=int)
            np.testing.assert_array_equal(indices, validation_indices(old["sources"]))
            features, phases, mask, native_mask, guards = public_inputs(
                old["states"][indices]
            )
            np.testing.assert_array_equal(native_mask.numpy(), old["masks"][indices])
            np.testing.assert_array_equal(guards.numpy(), old["guards"][indices])
            for i, expected in enumerate(old["oracle_features_sha256"][indices]):
                assert (
                    hashlib.sha256(features[i].numpy().tobytes()).digest()
                    == expected.tobytes()
                )
            raw = {
                "indices": indices,
                "phases": phases.numpy(),
                "mask": mask.numpy(),
                "guards": guards.numpy(),
                "sources": old["sources"][indices],
                "wp_scores": old["cpu_wp"][indices],
                "point_scores": old["cpu_point"][indices],
            }
        wp, point = [greedy(raw[v + "_scores"], raw["mask"]) for v in ("wp", "point")]
        result, calls = {}, 0
        for name, variant, temperatures in (
            ("initial_wp", "wp", [0.05] * 4),
            ("initial_point_fixed", "point", [0.05] * 4),
            ("initial_point_matched", "point", plan["initial_point_temperatures"]),
        ):
            actions = wp if variant == "wp" else point
            result[name] = describe(
                raw[variant + "_scores"],
                raw["mask"],
                raw["phases"],
                actions,
                wp,
                point,
                variant,
                temperatures,
            )
        for name, info in plan["models"].items():
            payload = torch.load(info["path"], map_location="cpu", weights_only=False)
            assert payload["counters"]["updates"] == 520
            assert payload["config"]["initial_variant"] == info["variant"]
            assert list(payload["config"]["phase_temperatures"]) == info["temperatures"]
            actor = PublishedActor(payload["actor"], "cpu").eval()
            scores, actions, count = score_public(actor, features, phases, mask)
            np.testing.assert_array_equal(actions, greedy(scores, raw["mask"]))
            raw[name + "_scores"], raw[name + "_actions"] = scores, actions
            result[name] = describe(
                scores,
                raw["mask"],
                raw["phases"],
                actions,
                wp,
                point,
                info["variant"],
                info["temperatures"],
            )
            calls += count
            print({"scored_model": name, "singleton_calls": count}, flush=True)
        assert calls == 18270
        checked(plan["inputs"])
        np.savez_compressed(root / "scores.npz", **raw)
        write_new(
            root / "report.json",
            {
                "status": "complete",
                "results": result,
                "disagreements": disagreements(raw),
                "inputs": plan["inputs"],
                "protocol_sha256": sha(root / "protocol.json"),
                "scores_sha256": sha(root / "scores.npz"),
                "runtime": runtime_identity(torch.device("cpu")),
                "work": {
                    "singleton_neural_calls": calls,
                    "new_matches": 0,
                    "simulator_steps": 0,
                    "training_updates": 0,
                },
                "scope": plan["scope"],
            },
        )
    except BaseException as error:
        write_new(root / "failed.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.command](args.output)
