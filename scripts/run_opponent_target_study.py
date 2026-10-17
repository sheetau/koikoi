"""Fixed four-endpoint continuation-target control; no automatic deployment."""

import argparse
import contextlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from koikoi.arbitration import read_checkpoint
from koikoi.published_actor import PublishedActor
from koikoi.train import provenance
from scripts.analyze_arbitration_study import tensor_equal
from scripts.audit_paired_feature_cache import inspect
from scripts.collect_opponent_paired_targets import CONDITIONS, collect
from scripts.compare_score_training_budgets import evaluate, paired_change
from scripts.diagnose_score_targets import diagnose
from scripts.feature_score_selector import FeatureScoreSelector, pack_features
from scripts.paired_rollout_targets import match_split
from scripts.run_arbitration_study import PARENTS, sha, write
from scripts.train_score_distribution import fit


def make_plan():
    paths = [Path(__file__)] + [
        Path("scripts") / (name + ".py")
        for name in (
            "collect_opponent_paired_targets",
            "opponent_paired_roots",
            "collect_reusable_paired_targets",
            "paired_rollout_targets",
            "cache_paired_expert_features",
            "audit_paired_feature_cache",
            "train_paired_arbitration",
            "train_score_distribution",
            "feature_score_selector",
            "diagnose_score_targets",
            "diagnose_paired_targets",
            "diagnose_arbitration",
            "compare_score_training_budgets",
            "analyze_arbitration_study",
            "run_arbitration_study",
        )
    ]
    parents = []
    for path, digest in PARENTS:
        if sha(path) != digest:
            raise ValueError("Parent changed")
        paths.append(Path(path))
        parents.append({"path": path, "sha256": digest})
    opponents = {}
    for kind in ("wp", "point"):
        snapshot = PublishedActor.from_published(kind, "cpu").checkpoint()["current"]
        opponents[kind] = {
            key: snapshot[key] for key in ("parameters_sha256", "model_hashes")
        }
    return {
        "source": provenance(),
        "inputs": {str(path): sha(path) for path in paths},
        "parents": parents,
        "opponents": opponents,
        "collection": {
            "roots": 65536,
            "batch": 256,
            "envs": 128,
            "device": "cuda",
            "seeds": [19710001, 19720001],
        },
        "fitting": {
            "epochs": 20,
            "weight_decay": 1e-4,
            "kind": "paired_win",
            "seeds": [19410001, 19420001],
        },
        "conditions": list(CONDITIONS),
        "scope": "Same learner roots/features, two continuation labels, two parents. Four fixed final endpoints; no epoch selection, match evaluation or deployment changes. Hold out whole source matches. Main diagnostic evaluates both models on opponent-preserving labels; retain parent-only-label cross-evaluation too.",
        "accounting": "One shared source and one inline public feature extraction per parent. Four branch completions per root across two conditions. Stored duplicated roots/features are not independent data. No new source replay or simulator work during fitting.",
    }


def verify_inputs(plan):
    if provenance()["source_sha256"] != plan["source"]["source_sha256"]:
        raise ValueError("Production changed")
    for path, digest in plan["inputs"].items():
        if sha(path) != digest:
            raise ValueError(f"Study input changed: {path}")


def audit_data(folder, plan, index):
    reports, previous_data, previous_cache = {}, None, None
    for condition in CONDITIONS:
        root = folder / condition
        dataset, cache_path = root / "targets.pt", root / "features.pt"
        data, digest = read_checkpoint(dataset)
        cache = read_checkpoint(cache_path)[0]
        report = inspect(dataset, cache_path)
        count = plan["collection"]["roots"]
        assert data["parent"]["sha256"] == plan["parents"][index - 1]["sha256"]
        for kind, snapshot in (
            ("point", data["experts"][1]),
            ("wp", data["opponent_wp"]),
        ):
            for key, expected in plan["opponents"][kind].items():
                assert snapshot["current"][key] == expected
        assert data["collection"]["continuation"] == condition
        assert data["collection"]["seed"] == plan["collection"]["seeds"][index - 1]
        assert len(data["public_observations"]) == count
        assert data["public_observations"].shape == (count, 562)
        assert data["outcomes"].shape == data["point_differences"].shape == (count, 2)
        assert torch.equal(data["outcomes"], data["point_differences"].sign().float())
        assert ((data["opponent_ids"] == 0) | (data["opponent_ids"] == 1)).all()
        assert data["branch_work"]["completed_branches"] == 2 * count
        source = data["source_work"]
        assert (
            source["learner_decisions"] + source["opponent_decisions"]
            == source["decisions"]
        )
        assert (
            source["wp_completed_matches"] + source["point_completed_matches"]
            == source["completed_matches"]
        )
        if previous_data is not None:
            for field in (
                "public_observations",
                "match_groups",
                "opponent_ids",
                "source_work",
                "experts",
                "parent",
                "opponent_wp",
            ):
                assert tensor_equal(previous_data[field], data[field]), field
            for field in (
                "embeddings",
                "scores",
                "public_dealer",
                "aligned_source_work",
            ):
                assert tensor_equal(previous_cache[field], cache[field]), field
        valid = match_split(data["match_groups"])
        assert valid.any() and (~valid).any()
        report.update(
            dataset_sha256=digest,
            branch_work=data["branch_work"],
            shared_source_work=source,
            validation_roots=int(valid.sum()),
            validation_source_matches=len(data["match_groups"][valid].unique(dim=0)),
            validation_opponent_counts=torch.bincount(
                data["opponent_ids"][valid], minlength=2
            ).tolist(),
        )
        reports[condition] = report
        previous_data, previous_cache = data, cache
    return reports


@torch.no_grad()
def analyze_parent(folder, fitted_folder, plan, index):
    torch.set_num_threads(2)
    datasets = {
        condition: read_checkpoint(folder / condition / "targets.pt")
        for condition in CONDITIONS
    }
    caches = {
        condition: read_checkpoint(folder / condition / "features.pt")
        for condition in CONDITIONS
    }
    reports, samples, loss_checks = {}, {}, {}
    initialization = None
    for trained in CONDITIONS:
        checkpoint = fitted_folder / trained / "final.pt"
        payload, _ = read_checkpoint(checkpoint)
        data, digest = datasets[trained]
        cache, cache_digest = caches[trained]
        assert payload["dataset"]["sha256"] == digest
        assert payload["feature_cache"]["sha256"] == cache_digest
        config = payload["training_config"]
        assert payload["network_config"]["seed"] == plan["fitting"]["seeds"][index - 1]
        assert payload["network_config"]["kind"] == plan["fitting"]["kind"]
        assert config["epochs"] == plan["fitting"]["epochs"]
        assert config["weight_decay"] == plan["fitting"]["weight_decay"]
        current = config["initial_shared_body_sha256"], config["minibatch_order_seed"]
        assert initialization is None or current == initialization
        initialization = current
        valid = match_split(data["match_groups"])
        assert (
            payload["counters"]["optimizer_steps"]
            == math.ceil(int((~valid).sum()) / 1024) * config["epochs"]
        )
        assert payload["counters"]["new_simulation_decisions"] == 0
        model = FeatureScoreSelector("paired_win")
        model.load_state_dict(payload["model"])
        model.eval()
        public = pack_features(
            data["public_observations"], cache["embeddings"], cache["scores"]
        )
        values = torch.cat([model.values(rows) for rows in public[valid].split(1024)])
        target = data["outcomes"][valid, 1] - data["outcomes"][valid, 0]
        loss = float(((values[:, 1] - values[:, 0] - target) ** 2).mean())
        np.testing.assert_allclose(
            loss, payload["history"][-1]["validation_loss"], atol=1e-5, rtol=0
        )
        loss_checks[trained] = {
            "independent_cpu_validation_mse": loss,
            "optimizer_steps": payload["counters"]["optimizer_steps"],
        }
        for evaluated in CONDITIONS:
            eval_data, eval_digest = datasets[evaluated]
            eval_cache, eval_cache_digest = caches[evaluated]
            report, wins, points = evaluate(
                checkpoint, eval_data, eval_digest, eval_cache, eval_cache_digest
            )
            key = trained + "/evaluated_on_" + evaluated
            reports[key], samples[key] = report, (wins, points)
    contrasts = {}
    for evaluated in CONDITIONS:
        groups = datasets[evaluated][0]["match_groups"][
            match_split(datasets[evaluated][0]["match_groups"])
        ].numpy()
        contrasts[evaluated] = paired_change(
            samples["opponent_preserving/evaluated_on_" + evaluated],
            samples["parent_only/evaluated_on_" + evaluated],
            groups,
        )
    return {
        "endpoints": reports,
        "opponent_target_minus_parent_target": contrasts,
        "fit_loss_checks": loss_checks,
    }


def run(base):
    if base.exists():
        raise ValueError("Use a fresh study directory")
    plan = make_plan()
    base.mkdir(parents=True)
    write(base / "protocol.json", plan)
    results = {}
    try:
        for index, parent in enumerate(plan["parents"], 1):
            verify_inputs(plan)
            folder = base / "data" / f"parent{index}"
            write(base / "process.json", {"status": "collecting", "parent": index})
            with (
                (base / f"parent{index}_collection.log").open(
                    "x", encoding="utf-8"
                ) as log,
                contextlib.redirect_stdout(log),
            ):
                collection = collect(
                    parent["path"],
                    folder,
                    seed=plan["collection"]["seeds"][index - 1],
                    **{
                        key: plan["collection"][key]
                        for key in ("roots", "batch", "envs", "device")
                    },
                )
            verify_inputs(plan)
            data_audit = audit_data(folder, plan, index)
            write(
                folder / "audit.json",
                {
                    "status": "passed",
                    "conditions": data_audit,
                    "collection": collection,
                },
            )
            fitted_folder = base / "fits" / f"parent{index}"
            for condition in CONDITIONS:
                verify_inputs(plan)
                output = fitted_folder / condition
                output.mkdir(parents=True)
                write(
                    base / "process.json",
                    {"status": "fitting", "parent": index, "condition": condition},
                )
                dataset = folder / condition / "targets.pt"
                with (
                    (output / "fit.log").open("x", encoding="utf-8") as log,
                    contextlib.redirect_stdout(log),
                ):
                    fit(
                        dataset,
                        output / "final.pt",
                        kind=plan["fitting"]["kind"],
                        seed=plan["fitting"]["seeds"][index - 1],
                        device=plan["collection"]["device"],
                        epochs=plan["fitting"]["epochs"],
                        weight_decay=plan["fitting"]["weight_decay"],
                        feature_cache=folder / condition / "features.pt",
                    )
                diagnose(dataset, output / "final.pt", output / "diagnostic.json")
            write(base / "process.json", {"status": "analyzing", "parent": index})
            analysis = analyze_parent(folder, fitted_folder, plan, index)
            verify_inputs(plan)
            results[str(index)] = {
                "collection": collection,
                "data_audit": data_audit,
                "analysis": analysis,
            }
            write(base / "results.json", results)
            print(
                json.dumps(
                    {
                        "completed_parent": index,
                        "contrasts": analysis["opponent_target_minus_parent_target"],
                    }
                ),
                flush=True,
            )
        write(base / "process.json", {"status": "complete", "parents": len(results)})
    except BaseException as error:
        write(base / "process.json", {"status": "failed", "error": repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("runs/opponent_target_study"))
    run(parser.parse_args().base)


if __name__ == "__main__":
    main()
