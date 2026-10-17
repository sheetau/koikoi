"""Training-root branch advantages by opponent and decision type, without fitting."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.diagnose_paired_targets import cluster_interval
from scripts.paired_rollout_targets import match_split
from scripts.run_arbitration_study import read, sha


def summarize(data):
    keep = ~match_split(data["match_groups"])
    public = data["public_observations"][keep].numpy()
    groups = data["match_groups"][keep].numpy()
    opponents = data["opponent_ids"][keep].numpy()
    outcomes = data["outcomes"][keep].double().numpy()
    points = data["point_differences"][keep].double().numpy()
    assert public.shape == (len(groups), 562)
    assert outcomes.shape == points.shape == (len(groups), 2)
    assert np.isin(opponents, [0, 1]).all()
    assert np.isin(outcomes, [-1, 0, 1]).all()
    assert np.array_equal(outcomes, np.sign(points))
    assert (public[:, 560] != public[:, 561]).all()
    unique, inverse = np.unique(groups, axis=0, return_inverse=True)
    low, high = np.ones(len(unique)), np.zeros(len(unique))
    np.minimum.at(low, inverse, opponents)
    np.maximum.at(high, inverse, opponents)
    assert np.array_equal(low, high)  # Opponent stays fixed for a source match.
    phases = public[:, 528:532].argmax(1)
    masks = {"all": np.ones(len(public), dtype=bool)}
    masks.update(
        {
            name: phases == phase
            for phase, name in enumerate(("discard", "pick_hand", "pick_draw", "koi"))
        }
    )
    masks["koi_point_stop"] = (phases == 3) & (public[:, 561] == 48)
    masks["koi_point_continue"] = (phases == 3) & (public[:, 561] == 49)
    assert np.array_equal(
        masks["koi_point_stop"] | masks["koi_point_continue"], masks["koi"]
    )
    win_delta = (outcomes[:, 1] - outcomes[:, 0]) / 2
    point_delta = points[:, 1] - points[:, 0]
    result = {}
    for name, phase_mask in masks.items():
        rows = {}
        for identity, opponent in enumerate(("wp", "point")):
            mask = phase_mask & (opponents == identity)
            count = int(mask.sum())
            wins, margins = win_delta[mask], point_delta[mask]
            rows[opponent] = {
                "roots": count,
                "source_matches": len(np.unique(groups[mask], axis=0)),
                "point_better_win_labels": int((wins > 0).sum()),
                "parent_better_win_labels": int((wins < 0).sum()),
                "win_change": float(wins.mean()) if count else None,
                "win_source_match_ci95": cluster_interval(wins, groups[mask])
                if count
                else None,
                "point_margin_change": float(margins.mean()) if count else None,
                "point_source_match_ci95": cluster_interval(margins, groups[mask])
                if count
                else None,
            }
        result[name] = rows
    assert sum(result["all"][key]["roots"] for key in ("wp", "point")) == len(public)
    return {
        "training_roots": len(public),
        "training_source_matches": len(unique),
        "groups": result,
    }


def run(output):
    if output.exists():
        raise ValueError("Use a fresh diagnostic output")
    torch.set_num_threads(2)
    base = Path("runs/opponent_target_study")
    completed = read(base / "results.json")
    inputs = {
        str(path): sha(path)
        for path in (
            Path(__file__),
            Path("scripts/diagnose_paired_targets.py"),
            Path("scripts/paired_rollout_targets.py"),
            base / "results.json",
        )
    }
    reports = {}
    for index in (1, 2):
        path = base / "data" / f"parent{index}" / "opponent_preserving/targets.pt"
        digest = sha(path)
        assert (
            digest
            == completed[str(index)]["collection"]["conditions"]["opponent_preserving"][
                "dataset_sha256"
            ]
        )
        inputs[str(path)] = digest
        data = torch.load(path, map_location="cpu", weights_only=False)
        assert data["format"] == "paired_expert_targets_v1"
        reports[str(index)] = summarize(data)
        print(
            json.dumps({"parent": index, "groups": reports[str(index)]["groups"]}),
            flush=True,
        )
    assert all(sha(path) == digest for path, digest in inputs.items())
    report = {
        "inputs": inputs,
        "parents": reports,
        "bootstrap": {
            "unit": "source match",
            "estimand": "root-weighted mean",
            "repetitions": 5000,
            "seed": 19730001,
        },
        "scope": "TRAINING source matches, expert-disagreement roots only. Point proposal minus parent proposal under fixed opponent-preserving continuation. Opponent identities are analysis metadata, not policy inputs. Groups have different source states; this is not a causal effect of changing opponent at the same state or evidence of an implementable opponent classifier. Marginal exploratory intervals, without multiple-comparison adjustment. No model, threshold, held-out evaluation or new simulation.",
    }
    output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"output": str(output), "sha256": sha(output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/opponent_conditioning_diagnostic.json"),
    )
    run(parser.parse_args().output)
