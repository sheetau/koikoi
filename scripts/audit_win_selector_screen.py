"""Audit fixed paired-win/preference endpoints and recompute held-out losses."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from koikoi.arbitration import read_checkpoint
from koikoi.train import provenance
from scripts.analyze_arbitration_study import tensor_equal
from scripts.feature_score_selector import FeatureScoreSelector, pack_features
from scripts.paired_rollout_targets import match_split
from scripts.run_arbitration_study import sha
from scripts.train_paired_arbitration import setup
from scripts.train_score_distribution import ScoreSelector


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@torch.no_grad()
def audit(base, output):
    if output.exists():
        raise ValueError("Use a fresh audit output")
    setup(1)
    plan = read(base / "plan.json")
    if read(base / "process.json")["status"] != "complete":
        raise ValueError("Fitting is not complete")
    if provenance()["source_sha256"] != plan["source"]["source_sha256"]:
        raise ValueError("Production changed")
    for path, digest in plan["inputs"].items():
        if sha(path) != digest:
            raise ValueError(f"Bound input changed: {path}")
    reports = {}
    for case in plan["cases"]:
        kind = case["kind"]
        if kind not in ("paired_win", "preference"):
            raise ValueError("This auditor covers paired-win/preference screens")
        folder = base / case["name"]
        data, digest = read_checkpoint(case["dataset"])
        fitted, checkpoint_digest = read_checkpoint(folder / "final.pt")
        receipt = read(folder / "fit_complete.json")
        diagnostic = read(folder / "diagnostic.json")
        assert fitted["network_config"]["kind"] == kind
        assert fitted["network_config"]["seed"] == case["seed"]
        assert fitted["dataset"]["sha256"] == digest == diagnostic["dataset_sha256"]
        assert (
            receipt["checkpoint_sha256"]
            == checkpoint_digest
            == diagnostic["checkpoint_sha256"]
        )
        assert tensor_equal(data["experts"], fitted["experts"])
        assert tensor_equal(data["parent"], fitted["parent"])
        valid = match_split(data["match_groups"])
        counts, epochs = fitted["counters"], plan["epochs"]
        assert fitted["training_config"]["weight_decay"] == plan.get(
            "weight_decay", 1e-4
        )
        assert counts == receipt["counters"]
        assert counts["training_roots"] == int((~valid).sum())
        assert counts["validation_roots"] == int(valid.sum())
        assert counts["reused_source_work"] == data["source_work"]
        assert counts["reused_branch_work"] == data["branch_work"]
        assert counts["new_simulation_decisions"] == 0
        assert (
            counts["optimizer_steps"]
            == math.ceil(counts["training_roots"] / 1024) * epochs
        )
        assert len(fitted["history"]) == epochs == fitted["training_config"]["epochs"]
        assert [row["epoch"] for row in fitted["history"]] == list(range(1, epochs + 1))
        assert all(np.isfinite(v) for row in fitted["history"] for v in row.values())
        reference_base = Path(
            "runs/paired_difference_fit_screen"
            if kind == "paired_win"
            else "runs/preference_fit_screen"
        )
        reference, reference_digest = read_checkpoint(
            reference_base / case["name"] / "final.pt"
        )
        for field in ("initial_shared_body_sha256", "minibatch_order_seed"):
            assert (
                fitted["training_config"][field] == reference["training_config"][field]
            )
        public = data["public_observations"]
        if case["feature_cache"]:
            cache, cache_digest = read_checkpoint(case["feature_cache"])
            assert fitted["feature_cache"]["sha256"] == cache_digest
            assert cache["dataset_sha256"] == digest
            public = pack_features(public, cache["embeddings"], cache["scores"])
            model = FeatureScoreSelector(kind)
        else:
            model = ScoreSelector(kind)
        model.load_state_dict(fitted["model"])
        model.eval()
        scores = torch.cat([model.values(rows) for rows in public[valid].split(1024)])
        z = scores[:, 1] - scores[:, 0]
        outcomes = data["outcomes"][valid]
        delta = outcomes[:, 1] - outcomes[:, 0]
        if kind == "paired_win":
            loss = float(((z - delta) ** 2).mean())
        else:
            zero = torch.zeros_like(z)
            errors = torch.where(
                delta > 0, torch.logaddexp(zero, -z), torch.logaddexp(zero, z)
            )
            loss = float((errors * delta.abs() / 2).mean())
            assert "win_delta_mse" not in diagnostic["sections"]["all"]
        np.testing.assert_allclose(
            loss, fitted["history"][-1]["validation_loss"], atol=1e-5, rtol=0
        )
        assert diagnostic["sections"]["all"]["roots"] == counts["validation_roots"]
        reports[case["name"]] = {
            "checkpoint_sha256": checkpoint_digest,
            "reference_20epoch_checkpoint_sha256": reference_digest,
            "recomputed_validation_loss": loss,
            "initialization_and_order_match_20_epochs": True,
            "optimizer_steps": counts["optimizer_steps"],
            "weight_decay": fitted["training_config"]["weight_decay"],
            "fit_seconds": receipt["elapsed_fit_seconds"],
            "all": diagnostic["sections"]["all"],
        }
    result = {
        "status": "passed",
        "scope": "Stored identities, counters, matched initialization/shuffle and independent CPU final-loss formula. No training replay or full-match strength estimate; timing may include concurrent work.",
        "plan_sha256": sha(base / "plan.json"),
        "auditor_sha256": sha(__file__),
        "reports": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.base, args.output)
    print(
        {
            "status": result["status"],
            "cases": len(result["reports"]),
            "sha256": sha(args.output),
        }
    )


if __name__ == "__main__":
    main()
