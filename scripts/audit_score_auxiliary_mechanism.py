"""Separate population/scalar audit of fixed auxiliary mechanism diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.env import BatchEnv
from koikoi.layout import PHASE
from scripts.audit_koi_temperature_mechanism import equal_scalars, scalar_statistics
from scripts.run_arbitration_study import read, sha

ARMS = tuple(
    f"seed{seed}_{kind}" for seed in (1, 2) for kind in ("none", "win", "score")
)
STUDY = Path("runs/score_auxiliary_study")


def scalar_description(scores, masks, phases, actions, reference):
    result = {}
    for phase in range(4):
        all_ids = [i for i, p in enumerate(phases) if p == phase]
        ids = [i for i in all_ids if np.count_nonzero(masks[i]) > 1]
        forced = [i for i in all_ids if np.count_nonzero(masks[i]) == 1]
        assert len(ids) + len(forced) == len(all_ids)
        item = {
            "all_rows": len(all_ids),
            "unforced_rows": len(ids),
            "forced_rows": len(forced),
        }
        if ids:
            item.update(
                {
                    "changed_vs_point": sum(
                        int(actions[i] != reference[i]) for i in ids
                    ),
                    "stop_to_continue_vs_point": sum(
                        int(actions[i] == 49 and reference[i] == 48) for i in ids
                    ),
                    "continue_to_stop_vs_point": sum(
                        int(actions[i] == 48 and reference[i] == 49) for i in ids
                    ),
                    "common_temperature": scalar_statistics(
                        scores[ids], masks[ids], 0.05
                    ),
                }
            )
        result[str(phase)] = item
    return result


def audit(root):
    output = root / "independent_audit.json"
    if output.exists():
        raise ValueError("Preserve completed diagnostic audit")
    assert not (root / "failed.json").exists()
    plan, report = read(root / "protocol.json"), read(root / "report.json")
    assert (
        plan["format"] == "score_auxiliary_mechanism_v1" and plan["temperature"] == 0.05
    )
    assert report["status"] == "complete" and report["inputs"] == plan["inputs"]
    assert report["protocol_sha256"] == sha(root / "protocol.json")
    assert report["scores_sha256"] == sha(root / "scores.npz")
    assert set(plan["models"]) == set(ARMS)
    assert report["scope"] == plan["scope"]
    assert (
        sha(STUDY / "protocol.json")
        == "9ba0bd7b1d187d59269af336537ceab890bf8884a969a9df073dcd626e686f46"
    )
    training = read(STUDY / "training_audit.json")
    assert training["status"] == "passed"
    assert all(
        plan["inputs"].get(p) == digest for p, digest in training["inputs"].items()
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
    for arm in ARMS:
        expected = STUDY / arm / "update_00000520/ppo.pt"
        assert Path(plan["models"][arm]).resolve() == expected.resolve()
        assert (
            sha(expected.parent / "manifest.json")
            == training["arms"][arm]["endpoint_manifest_sha256"]
        )
    with np.load("runs/published_batch_validation.npz", allow_pickle=False) as data:
        indices = np.array(
            [
                i
                for i, source in enumerate(data["sources"])
                if source >= 0 and source % 4 == 3
            ]
        )
        np.testing.assert_array_equal(indices, plan["indices"])
        assert len(indices) == 3258 and len(set(data["sources"][indices])) == 25
        states, sources, original = (
            data["states"][indices],
            data["sources"][indices],
            data["cpu_point"][indices],
        )
    env = BatchEnv(len(states), "cpu", "paper8")
    env.state.copy_(torch.as_tensor(states, dtype=torch.int32))
    _, legal = env.published_observe()
    guards = env.solved_actions().numpy()
    mask = legal.numpy() & (
        (guards[:, None] < 0) | (np.arange(50)[None] == guards[:, None])
    )
    np.testing.assert_array_equal(env.state.numpy(), states)
    phases = states[:, PHASE]
    with np.load(root / "scores.npz", allow_pickle=False) as raw:
        assert set(raw.files) == {
            "indices",
            "sources",
            "phases",
            "mask",
            "guards",
            "point_scores",
        } | {arm + suffix for arm in ARMS for suffix in ("_scores", "_actions")}
        for key, expected in (
            ("indices", indices),
            ("sources", sources),
            ("phases", phases),
            ("mask", mask),
            ("guards", guards),
            ("point_scores", original),
        ):
            np.testing.assert_array_equal(raw[key], expected)
        actions = {}
        for model in ("point", *ARMS):
            scores = raw[model + "_scores"]
            assert scores.dtype == np.float32 and scores.shape == mask.shape
            selected = np.full(len(mask), -1, dtype=int)
            for i, allowed in enumerate(mask):
                if np.count_nonzero(allowed) > 1:
                    assert np.isfinite(scores[i, allowed]).all()
                    selected[i] = int(
                        np.argmax(
                            np.where(allowed, scores[i], -np.inf) / np.float32(0.05)
                        )
                    )
            actions[model] = selected
            if model != "point":
                np.testing.assert_array_equal(selected, raw[model + "_actions"])
        assert set(report["results"]) == {"point", *ARMS}
        for model in ("point", *ARMS):
            equal_scalars(
                scalar_description(
                    raw[model + "_scores"],
                    mask,
                    phases,
                    actions[model],
                    actions["point"],
                ),
                report["results"][model],
            )
        pairwise = {}
        for seed in (1, 2):
            models = [f"seed{seed}_{kind}" for kind in ("none", "win", "score")]
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = models[i], models[j]
                    pairwise[a + "__" + b] = {
                        str(phase): sum(
                            int(
                                phases[k] == phase
                                and np.count_nonzero(mask[k]) > 1
                                and actions[a][k] != actions[b][k]
                            )
                            for k in range(len(mask))
                        )
                        for phase in range(4)
                    }
        assert pairwise == report["disagreements"]
    assert set(report["learning"]) == set(ARMS)
    for arm in ARMS:
        logs = [
            json.loads(line)
            for line in (STUDY / arm / "metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(logs) == 520
        phases_logged = []
        for update, row in enumerate(logs, 1):
            if update > 20 and (update - 20) % 5 == 0:
                assert row["auxiliary_phase"]["cycle"] == (update - 20) // 5
                phases_logged.append(row["auxiliary_phase"])
            else:
                assert row["auxiliary_phase"] is None
        assert len(phases_logged) == 100 and phases_logged == report["learning"][arm]
    assert report["work"] == {
        "singleton_neural_calls": 6 * int((mask.sum(1) > 1).sum()),
        "new_matches": 0,
        "simulator_steps": 0,
        "training_updates": 0,
    }
    assert report["work"]["singleton_neural_calls"] == 18270
    assert (
        report["runtime"]["device"] == "cpu" and report["runtime"]["cpu_threads"] == 2
    )
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            {
                "status": "passed",
                "inputs": inputs,
                "report_sha256": sha(root / "report.json"),
                "scope": "Original old public population and cached Point scores,scalar entropy/action summaries and all100training log phases checked. No neural-inference or gradient replay;no strength or causal claim.",
            },
            stream,
            indent=2,
        )
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    audit(args.input)
