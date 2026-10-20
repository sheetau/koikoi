"""Predeclared auxiliary-loss curves and fixed-public-input actor diagnostics."""

import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET
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
from scripts.point_initialization_mechanism import greedy, validation_indices, write_new
from scripts.run_arbitration_study import read, sha

STUDY = Path("runs/score_auxiliary_study")
STUDY_SHA = "9ba0bd7b1d187d59269af336537ceab890bf8884a969a9df073dcd626e686f46"
ARMS = tuple(
    f"seed{seed}_{kind}" for seed in (1, 2) for kind in ("none", "win", "score")
)


def describe(scores, mask, phases, actions, original):
    result = {}
    for phase in range(4):
        all_rows = phases == phase
        selected = all_rows & (mask.sum(1) > 1)
        item = {
            "all_rows": int(all_rows.sum()),
            "unforced_rows": int(selected.sum()),
            "forced_rows": int((all_rows & (mask.sum(1) == 1)).sum()),
        }
        assert item["all_rows"] == item["unforced_rows"] + item["forced_rows"]
        if selected.any():
            current, initial = actions[selected], original[selected]
            legal = mask[selected]
            assert (current >= 0).all() and legal[
                np.arange(len(current)), current
            ].all()
            item.update(
                {
                    "changed_vs_point": int((current != initial).sum()),
                    "stop_to_continue_vs_point": int(
                        ((initial == 48) & (current == 49)).sum()
                    ),
                    "continue_to_stop_vs_point": int(
                        ((initial == 49) & (current == 48)).sum()
                    ),
                    "common_temperature": choice_statistics(
                        scores[selected], legal, 0.05
                    ),
                }
            )
        result[str(phase)] = item
    return result


def disagreements(raw):
    result = {}
    for seed in (1, 2):
        for first, second in (("none", "win"), ("none", "score"), ("win", "score")):
            left, right = f"seed{seed}_{first}", f"seed{seed}_{second}"
            result[left + "__" + right] = {
                str(phase): int(
                    np.count_nonzero(
                        (raw["phases"] == phase)
                        & (raw["mask"].sum(1) > 1)
                        & (raw[left + "_actions"] != raw[right + "_actions"])
                    )
                )
                for phase in range(4)
            }
    return result


def learning_diagnostics(rows, expected_cycles=100):
    phases = [
        row["auxiliary_phase"] for row in rows if row["auxiliary_phase"] is not None
    ]
    assert len(phases) == expected_cycles
    assert [row["cycle"] for row in phases] == list(range(1, expected_cycles + 1))
    return phases


def prepare(root):
    if root.exists():
        raise ValueError("Preserve existing diagnostic directory")
    assert sha(STUDY / "protocol.json") == STUDY_SHA and sha(CORPUS) == CORPUS_SHA
    audit = read(STUDY / "training_audit.json")
    assert (
        audit["status"] == "passed"
        and audit["format"] == "score_auxiliary_training_audit_v1"
    )
    assert audit["protocol_sha256"] == STUDY_SHA
    assert audit["complete_sha256"] == sha(STUDY / "complete.json")
    assert audit["auditor_sha256"] == sha("scripts/audit_score_auxiliary_training.py")
    assert set(audit["arms"]) == set(ARMS)
    inputs = dict(audit["inputs"])
    paths = [
        Path(__file__),
        Path("scripts/audit_score_auxiliary_mechanism.py"),
        Path("scripts/koi_temperature_mechanism.py"),
        Path("scripts/phase_exploration_calibration.py"),
        Path("scripts/point_initialization_mechanism.py"),
        Path("scripts/audit_koi_temperature_mechanism.py"),
        Path("tests/test_score_auxiliary_mechanism.py"),
        Path("runs/score_auxiliary_mechanism_tests.xml"),
        Path("docs/SCORE_AUXILIARY_MECHANISM.rst"),
        CORPUS,
        STUDY / "training_audit.json",
        STUDY / "protocol.json",
    ]
    inputs.update({str(path): sha(path) for path in paths})
    suite = (
        ET.parse("runs/score_auxiliary_mechanism_tests.xml").getroot().find("testsuite")
    )
    assert int(suite.attrib["tests"]) == 4
    assert all(int(suite.attrib[k]) == 0 for k in ("errors", "failures", "skipped"))
    models = {}
    for arm in ARMS:
        folder = STUDY / arm / "update_00000520"
        assert (
            sha(folder / "manifest.json")
            == audit["arms"][arm]["endpoint_manifest_sha256"]
        )
        path = folder / "ppo.pt"
        assert inputs[str(path)] == sha(path)
        models[arm] = str(path)
    with np.load(CORPUS, allow_pickle=False) as data:
        indices = validation_indices(data["sources"])
        assert len(indices) == 3258 and len(np.unique(data["sources"][indices])) == 25
    checked(inputs)
    root.mkdir(parents=True, exist_ok=False)
    write_new(
        root / "protocol.json",
        {
            "format": "score_auxiliary_mechanism_v1",
            "inputs": inputs,
            "models": models,
            "indices": indices.tolist(),
            "temperature": 0.05,
            "scope": "All100auxiliary training-phase loss summaries plus fixed old public validation inputs. Descriptive action/entropy changes,not held-out prediction,visitation,causal mediation or playing strength. No selection or new training.",
        },
    )
    print(
        {"prepared": str(root), "protocol_sha256": sha(root / "protocol.json")},
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
    assert set(plan["models"]) == set(ARMS) and plan["temperature"] == 0.05
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
                "point_scores": old["cpu_point"][indices],
            }
        point = greedy(raw["point_scores"], raw["mask"])
        results = {
            "point": describe(
                raw["point_scores"], raw["mask"], raw["phases"], point, point
            )
        }
        calls, learning = 0, {}
        for arm, path in plan["models"].items():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            assert payload["counters"]["updates"] == 520
            assert payload["config"]["initial_variant"] == "point"
            assert payload["config"]["score_temperature"] == 0.05
            actor = PublishedActor(payload["actor"], "cpu").eval()
            scores, actions, count = score_public(actor, features, phases, mask)
            np.testing.assert_array_equal(actions, greedy(scores, raw["mask"]))
            raw[arm + "_scores"], raw[arm + "_actions"] = scores, actions
            results[arm] = describe(scores, raw["mask"], raw["phases"], actions, point)
            rows = [
                json.loads(line)
                for line in (STUDY / arm / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            assert len(rows) == 520
            learning[arm] = learning_diagnostics(rows)
            calls += count
            print({"scored_model": arm, "singleton_calls": count}, flush=True)
        assert calls == 18270
        checked(plan["inputs"])
        np.savez_compressed(root / "scores.npz", **raw)
        write_new(
            root / "report.json",
            {
                "status": "complete",
                "results": results,
                "learning": learning,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    {"prepare": prepare, "run": run}[args.command](args.output)
