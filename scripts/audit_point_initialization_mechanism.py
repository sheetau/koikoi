"""Reconstruct the old population and verify endpoint diagnostics with scalar math."""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from koikoi.env import BatchEnv
from koikoi.layout import PHASE
from scripts.audit_koi_temperature_mechanism import equal_scalars, scalar_statistics
from scripts.run_arbitration_study import read, sha

ARMS = tuple(
    f"seed{s}_{m}" for s in (1, 2) for m in ("wp_fixed", "point_fixed", "point_matched")
)


def scalar_description(scores, mask, phases, actions, wp, point, variant, temperatures):
    reference = wp if variant == "wp" else point
    result = {}
    for phase in range(4):
        all_rows = [i for i, p in enumerate(phases) if p == phase]
        rows = [i for i in all_rows if np.count_nonzero(mask[i]) > 1]
        record = {"all_rows": len(all_rows), "unforced_rows": len(rows)}
        if rows:
            gaps = []
            for i in rows:
                legal = sorted(
                    float(x) for x, yes in zip(scores[i], mask[i], strict=True) if yes
                )
                gaps.append(legal[-1] - legal[-2])
            record.update(
                {
                    "mean_top_two_raw_score_gap": statistics.mean(gaps),
                    "common_temperature": scalar_statistics(
                        scores[rows], mask[rows], 0.05
                    ),
                    "declared_temperature": scalar_statistics(
                        scores[rows], mask[rows], temperatures[phase]
                    ),
                    "changed_vs_wp": sum(int(actions[i] != wp[i]) for i in rows),
                    "changed_vs_point": sum(int(actions[i] != point[i]) for i in rows),
                    "changed_vs_initial": sum(
                        int(actions[i] != reference[i]) for i in rows
                    ),
                    "stop_to_continue_vs_initial": sum(
                        int(reference[i] == 48 and actions[i] == 49) for i in rows
                    ),
                    "continue_to_stop_vs_initial": sum(
                        int(reference[i] == 49 and actions[i] == 48) for i in rows
                    ),
                }
            )
        result[str(phase)] = record
    return result


def audit(root):
    output = root / "independent_audit.json"
    if output.exists():
        raise ValueError("Preserve completed audits")
    assert not (root / "failed.json").exists()
    plan, report = read(root / "protocol.json"), read(root / "report.json")
    assert report["status"] == "complete" and report["inputs"] == plan["inputs"]
    assert report["protocol_sha256"] == sha(root / "protocol.json")
    assert report["scores_sha256"] == sha(root / "scores.npz")
    assert set(plan["models"]) == set(ARMS)
    assert report["scope"] == plan["scope"]
    main_path = Path("runs/point_initialization_evaluation/evaluation_protocol.json")
    assert (
        sha(main_path)
        == "1df245b6fea1799c0ed9f91518642ccb3d1e4bf9c0f7e682a1208819e609b7e7"
    )
    assert all(
        plan["inputs"].get(p) == digest
        for p, digest in read(main_path)["inputs"].items()
    )
    calibration = read("runs/point_initialization_calibration.json")
    assert plan["initial_point_temperatures"] == calibration["phase_temperatures"]
    for arm in ARMS:
        info = plan["models"][arm]
        expected_path = (
            Path("runs/point_initialization_study") / arm / "update_00000520.pt"
        )
        assert Path(info["path"]).resolve() == expected_path.resolve()
        assert info["variant"] == ("wp" if arm.endswith("wp_fixed") else "point")
        assert info["temperatures"] == (
            calibration["phase_temperatures"]
            if arm.endswith("point_matched")
            else [0.05] * 4
        )
    inputs = dict(plan["inputs"])
    for path in (
        Path(__file__),
        root / "protocol.json",
        root / "report.json",
        root / "scores.npz",
    ):
        inputs[str(path)] = sha(path)
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    with np.load("runs/published_batch_validation.npz", allow_pickle=False) as old:
        indices = np.array(
            [
                i
                for i, source in enumerate(old["sources"])
                if source >= 0 and source % 4 == 3
            ]
        )
        np.testing.assert_array_equal(indices, plan["indices"])
        assert len(indices) == 3258 and len(set(old["sources"][indices])) == 25
        states, sources = old["states"][indices], old["sources"][indices]
        cached = {v: old["cpu_" + v][indices] for v in ("wp", "point")}
    env = BatchEnv(len(states), "cpu", "paper8")
    env.state.copy_(torch.as_tensor(states, dtype=torch.int32))
    _, native_mask = env.published_observe()
    guards = env.solved_actions().numpy()
    mask = native_mask.numpy() & (
        (guards[:, None] < 0) | (np.arange(50)[None] == guards[:, None])
    )
    np.testing.assert_array_equal(env.state.numpy(), states)
    with np.load(root / "scores.npz", allow_pickle=False) as raw:
        expected_keys = {
            "indices",
            "sources",
            "phases",
            "mask",
            "guards",
            "wp_scores",
            "point_scores",
        }
        expected_keys |= {
            arm + suffix for arm in ARMS for suffix in ("_scores", "_actions")
        }
        assert set(raw.files) == expected_keys
        for name, expected in (
            ("indices", indices),
            ("sources", sources),
            ("phases", states[:, PHASE]),
            ("mask", mask),
            ("guards", guards),
        ):
            np.testing.assert_array_equal(raw[name], expected)
        for variant in ("wp", "point"):
            np.testing.assert_array_equal(raw[variant + "_scores"], cached[variant])
        actions = {}
        for model in ("wp", "point", *ARMS):
            values = raw[model + "_scores"]
            assert values.shape == mask.shape and values.dtype == np.float32
            result = np.full(len(mask), -1, dtype=int)
            for i, legal in enumerate(mask):
                if np.count_nonzero(legal) > 1:
                    assert np.isfinite(values[i, legal]).all()
                    scaled = np.where(legal, values[i], -np.inf) / np.float32(0.05)
                    result[i] = int(np.argmax(scaled))
            actions[model] = result
            if model in ARMS:
                np.testing.assert_array_equal(result, raw[model + "_actions"])
        specifications = {
            "initial_wp": ("wp", "wp", [0.05] * 4),
            "initial_point_fixed": ("point", "point", [0.05] * 4),
            "initial_point_matched": (
                "point",
                "point",
                plan["initial_point_temperatures"],
            ),
            **{
                arm: (arm, info["variant"], info["temperatures"])
                for arm, info in plan["models"].items()
            },
        }
        assert report["results"].keys() == specifications.keys()
        for name, (model, variant, temperatures) in specifications.items():
            expected = scalar_description(
                raw[model + "_scores"],
                mask,
                states[:, PHASE],
                actions[model],
                actions["wp"],
                actions["point"],
                variant,
                temperatures,
            )
            equal_scalars(expected, report["results"][name])
        for phase in range(4):
            for label, reference in (
                ("initial_wp", "wp_fixed"),
                ("initial_point_fixed", "point_fixed"),
                ("initial_point_matched", "point_matched"),
            ):
                equal_scalars(
                    report["results"][label][str(phase)]["declared_temperature"],
                    calibration["phases"][str(phase)]["validation"][reference],
                )
        expected_pairs = {}
        for seed in (1, 2):
            names = [
                f"seed{seed}_{mode}"
                for mode in ("wp_fixed", "point_fixed", "point_matched")
            ]
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = names[i], names[j]
                    expected_pairs[a + "__" + b] = {
                        str(p): sum(
                            int(
                                states[k, PHASE] == p
                                and np.count_nonzero(mask[k]) > 1
                                and actions[a][k] != actions[b][k]
                            )
                            for k in range(len(states))
                        )
                        for p in range(4)
                    }
        assert expected_pairs == report["disagreements"]
        calls = 6 * sum(np.count_nonzero(row) > 1 for row in mask)
        assert calls == 18270
        assert report["work"] == {
            "singleton_neural_calls": calls,
            "new_matches": 0,
            "simulator_steps": 0,
            "training_updates": 0,
        }
    receipt = {
        "status": "passed",
        "inputs": inputs,
        "models_checked": 9,
        "unforced_rows": 3045,
        "source_groups": 25,
        "phases_per_model": 4,
        "new_work": {
            "neural_calls": 0,
            "simulator_steps": 0,
            "matches": 0,
            "training_updates": 0,
        },
        "scope": "Independent population/mask, stored-score argmax, scalar probabilities, gaps and action-count verification. No independent neural rerun or strength evidence.",
    }
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        {"status": "passed", "audit_sha256": sha(output), "inputs": len(inputs)},
        flush=True,
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    audit(parser.parse_args().input)
