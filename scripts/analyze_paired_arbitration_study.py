"""Inspect paired-target accounting and recompute all comparative match results."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.analyze_arbitration_study import interval, outcome, read, sha, tensor_equal
from scripts.paired_rollout_targets import match_split
from scripts.train_paired_arbitration import PairedRolloutAgent, prediction


def data_report(base, index, p):
    path = base / f"parent{index}" / "targets.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    roots = p["collection"]["roots_per_parent"]
    assert payload["format"] == "paired_expert_targets_v1"
    assert payload["collection"]["roots"] == roots
    assert payload["parent"]["sha256"] == p["parents"][index - 1]["sha256"]
    assert payload["provenance"]["source_sha256"] == p["source"]["source_sha256"]
    public, groups = payload["public_observations"], payload["match_groups"]
    wins, points = payload["outcomes"], payload["point_differences"]
    assert public.shape == (roots, 562) and groups.shape == (roots, 2)
    assert wins.shape == points.shape == (roots, 2)
    assert torch.isfinite(public).all() and torch.equal(wins, points.sign().float())
    proposals = public[:, 560:].long()
    assert torch.equal(proposals.float(), public[:, 560:])
    assert ((proposals >= 0) & (proposals < 50)).all()
    assert (proposals[:, 0] != proposals[:, 1]).all()
    first = p["collection"]["seeds"][index - 1]
    assert ((groups[:, 0] >= first) & (groups[:, 0] < first + 128)).all()
    assert (groups[:, 1] >= 0).all()
    work = payload["branch_work"]
    assert work["completed_branches"] == 2 * roots
    assert 2 * roots <= work["decisions"] <= 512 * 2 * roots
    assert (
        work["completed_branches"]
        <= work["round_endings"]
        <= 8 * work["completed_branches"]
    )
    assert payload["source_work"]["decisions"] % 128 == 0
    validation = match_split(groups)
    train_groups = set(map(tuple, groups[~validation].tolist()))
    validation_groups = set(map(tuple, groups[validation].tolist()))
    assert (
        train_groups
        and validation_groups
        and train_groups.isdisjoint(validation_groups)
    )
    delta = wins[:, 1] - wins[:, 0]
    report = {
        "dataset_sha256": sha(path),
        "roots": roots,
        "source_work": payload["source_work"],
        "branch_work": work,
        "training_source_matches": len(train_groups),
        "validation_source_matches": len(validation_groups),
        "target_counts": {
            str(value): int((delta == value).sum()) for value in (-2, -1, 0, 1, 2)
        },
        "target_mean": float(delta.mean()),
        "target_variance": float(delta.var(unbiased=False)),
        "branch_return_covariance": float(
            np.cov(wins.numpy(), rowvar=False, ddof=0)[0, 1]
        ),
    }
    return payload, report


def fit_report(base, case, data, p):
    path = base / case / "final.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    receipt = read(path.parent / "training_complete.json")
    assert sha(path) == receipt["checkpoint_sha256"]
    assert payload["format"] == "paired_rollout_arbitration_v1"
    assert (
        payload["dataset"]["sha256"]
        == receipt["dataset_sha256"]
        == sha(payload["dataset"]["path"])
    )
    assert payload["training_config"]["epochs"] == p["fit"]["epochs"]
    assert payload["provenance"]["source_sha256"] == p["source"]["source_sha256"]
    assert tensor_equal(payload["experts"], data["experts"])
    assert payload["counters"] == receipt["counters"]
    validation = match_split(data["match_groups"])
    counts = payload["counters"]
    assert counts["training_roots"] == int((~validation).sum())
    assert counts["validation_roots"] == int(validation.sum())
    assert (
        counts["optimizer_steps"]
        == math.ceil(counts["training_roots"] / 1024) * p["fit"]["epochs"]
    )
    assert counts["branch_work"] == data["branch_work"]
    assert counts["source_work"] == data["source_work"]
    assert len(payload["history"]) == p["fit"]["epochs"]
    assert [row["epoch"] for row in payload["history"]] == list(
        range(1, p["fit"]["epochs"] + 1)
    )
    assert all(
        np.isfinite(value) for row in payload["history"] for value in row.values()
    )
    model = PairedRolloutAgent(path).model
    with torch.no_grad():
        predicted = torch.cat(
            [
                prediction(model, rows)
                for rows in data["public_observations"][validation].split(1024)
            ]
        )
        actual = (data["outcomes"][:, 1] - data["outcomes"][:, 0])[validation]
        mse = float((predicted - actual).square().mean())
        np.testing.assert_allclose(
            mse, payload["history"][-1]["validation_mse"], rtol=0, atol=1e-5
        )
    return {
        "checkpoint_sha256": sha(path),
        "counters": counts,
        "validation_mse_recomputed_cpu": mse,
        "validation_zero_mse": float(actual.square().mean()),
        "validation_point_fraction": float((predicted > 0).float().mean()),
    }


def analyze(base):
    p = read(base / "protocol.json")
    assert read(base / "process.json")["status"] == "complete"
    assert all(sha(path) == digest for path, digest in p["scripts"].items())
    identities = read(base / "endpoint_identities.json")
    assert all(
        sha(spec["path"]) == identities[case] for case, spec in p["cases"].items()
    )
    datasets, fits = {}, {}
    for index in (1, 2):
        data, datasets[f"parent{index}"] = data_report(base, index, p)
        for context in p["fit"]["contexts"]:
            case = f"parent{index}_{context}"
            fits[case] = fit_report(base, case, data, p)
    reports, samples = {}, {}
    for case, spec in p["cases"].items():
        opponents = p["evaluation"]["opponents"] + (
            ["parent"] if spec["kind"] == "regression" else []
        )
        for opponent in opponents:
            key = case + "/" + opponent
            reports[key], scores, points = outcome(base, case, opponent, p)
            samples[key] = (scores, points)
    comparisons = {}
    for index in (1, 2):
        parent = f"parent{index}"
        for context in p["fit"]["contexts"]:
            case = parent + "_" + context
            for opponent in p["evaluation"]["opponents"]:
                for control in (
                    parent,
                    parent + "_ppo_gate",
                    parent + "_ppo_actor",
                    parent + "_no_match",
                ):
                    if case == control:
                        continue
                    left, right = (
                        samples[case + "/" + opponent],
                        samples[control + "/" + opponent],
                    )
                    delta, points = left[0] - right[0], left[1] - right[1]
                    comparisons[case + "-" + control + "/" + opponent] = {
                        "win_difference": float(delta.mean()),
                        "win_ci95": interval(delta),
                        "point_difference": float(points.mean()),
                        "points_ci95": interval(points),
                    }
    result = {
        "status": "passed",
        "protocol_sha256": sha(base / "protocol.json"),
        "analysis_script_sha256": sha(__file__),
        "datasets": datasets,
        "fits": fits,
        "outcomes": reports,
        "paired_comparisons": comparisons,
        "scope": "Development comparisons. Each dataset is shared by two controllers; generation work is counted once per parent. Validation holds out source matches. Audits inspect recorded artifacts and statistics, not every generated trajectory.",
    }
    target = base / "analysis.json"
    target.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "datasets": datasets,
                "fits": fits,
                "outcomes": reports,
                "comparisons": comparisons,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("runs/paired_arbitration_study")
    )
    analyze(parser.parse_args().base)


if __name__ == "__main__":
    main()
