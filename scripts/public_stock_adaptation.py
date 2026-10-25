"""Fixed V5A launch, training lineage and qualification contracts."""

from pathlib import Path

import numpy as np

from scripts.export_public_stock_results import intervals, read, sha
from scripts.public_stock_robustness import ALL_THREE, CONTROL, PREVIEW_CHOICE, PROFILES

V3 = "docs/paper/generated/public_stock_v3_evidence.json"
V4 = "docs/paper/generated/public_stock_v4_evidence.json"
V4_BASE = Path("runs/public_stock_robustness_v4")
TRAIN = {
    "updates": 384, "envs": 512, "horizon": 64, "epochs": 3,
    "minibatch": 4096, "save_every": 128, "lr": 1e-4,
    "kl": 0.02, "lam": 0.99, "entropy": 0.01, "device": "cuda",
}
EVALUATE = {
    "deal_seed": 1240000000, "pairs": 1024, "batch": 64,
    "particles": 0, "horizon": 32, "confidence": 1.0, "search_seed": 1103,
}
FIXED = {
    "method": "public_stock_adaptation_v5a", "profiles": list(PROFILES),
    "training": TRAIN, "training_seeds": [1101, 1102],
    "training_deal_seeds": [1220000000, 1230000000],
    "evaluation": EVALUATE, "bootstrap_seed": 1109,
    "bootstrap_replicates": 20000, "lower_quantile": 0.05 / 12,
}
SOURCES = (
    "docs/PUBLIC_STOCK_ADAPTATION_V5.rst", "scripts/public_stock_adaptation.py",
    "scripts/run_public_stock_adaptation.py", "scripts/export_public_stock_adaptation.py",
    "scripts/continue_public_stock_residual.py", "scripts/train_public_stock_residual.py",
    "scripts/evaluate_public_stock_residual.py", "scripts/export_public_stock_results.py",
    "scripts/run_arbitration_study.py", "scripts/public_stock_robustness.py",
    "koikoi/public_stock_residual.py", "koikoi/public_stock_policy.py",
    "koikoi/public_stock_residual_search.py", "koikoi/public_stock_search.py",
    "koikoi/public_stock.py", "koikoi/public_stock_native.py",
    "koikoi/native/public_stock.h", "koikoi/native/core.h",
    "koikoi/train.py", "koikoi/reference.py", "koikoi/rule_variants.py",
    "koikoi/layout.py", "koikoi/env.py", "koikoi/native.py",
    "koikoi/published_actor.py", "external/KoiKoi-AI/koikoinet2L.py",
)


def launch_gate(evidence):
    if evidence.get("method") != "public_stock_robustness_v4":
        raise ValueError("Expected complete independent V4 evidence")
    if len(evidence["comparisons"]) != 14 or evidence["evaluation_totals"]["matches"] != 4608:
        raise ValueError("V4 is incomplete")
    passed = {
        profile: all(
            evidence["comparisons"][f"{profile}_parent{p}_{key}"]["lower"] > 0
            for p in range(2) for key in ("gain", "excess", "high_minus_low")
        ) for profile in (PREVIEW_CHOICE, ALL_THREE)
    }
    if passed != evidence["score_tests_passed"] or not any(passed.values()):
        raise ValueError("V5A requires a complete V4 finalist score pass")
    return passed


def make_protocol():
    v4 = read(V4)
    passes = launch_gate(v4)
    finished = read(V4_BASE / "finished.json")
    if finished["status"] != "complete" or finished["sha256"] != sha(V4):
        raise ValueError("V4 finished receipt differs")
    for path, digest in v4["bindings"].items():
        if sha(path) != digest:
            raise ValueError(f"V4 evidence input changed: {path}")
    prior = read(V4_BASE / "protocol.json")
    systems = {
        label: {k: row[k] for k in ("profile", "parent", "checkpoint", "checkpoint_sha256")}
        for label, row in prior["systems"].items()
    }
    inputs = {row["checkpoint"]: row["checkpoint_sha256"] for row in systems.values()}
    for path in (V3, V4, (V4_BASE / "protocol.json").as_posix(), (V4_BASE / "finished.json").as_posix()):
        inputs[path] = sha(path)
    return {
        **FIXED, "systems": systems, "inputs": inputs,
        "sources": {path: sha(path) for path in SOURCES}, "v4_score_passes": passes,
        "v4_work_targets_passed": v4["all_work_targets_passed"],
    }


def verify_protocol(study):
    if any(study.get(key) != value for key, value in FIXED.items()):
        raise ValueError("V5A fixed design changed")
    if set(study["systems"]) != {f"{p}_parent{i}" for p in PROFILES for i in range(2)}:
        raise ValueError("Wrong V5A system coverage")
    if not set(SOURCES) <= set(study["sources"]):
        raise ValueError("Incomplete V5A source bindings")
    for path, digest in {**study["sources"], **study["inputs"]}.items():
        if sha(path) != digest:
            raise ValueError(f"Frozen V5A input changed: {path}")
    evidence = read(V4)
    if launch_gate(evidence) != study["v4_score_passes"]:
        raise ValueError("V4 launch decision changed")
    if study["v4_work_targets_passed"] != evidence["all_work_targets_passed"]:
        raise ValueError("V4 work-admissibility result changed")
    if study["inputs"].get(V4) != sha(V4) or study["inputs"].get(V3) != sha(V3):
        raise ValueError("Prior evidence is not bound")
    for name in ("protocol.json", "finished.json"):
        path = (V4_BASE / name).as_posix()
        if study["inputs"].get(path) != sha(path):
            raise ValueError("V4 protocol/finish receipt is not bound")
    finished = read(V4_BASE / "finished.json")
    if finished["status"] != "complete" or finished["sha256"] != sha(V4):
        raise ValueError("V4 finish receipt changed")
    retained = {Path(p).resolve(): h for p, h in read(V3)["bindings"].items()}
    for label, row in study["systems"].items():
        if label != f"{row['profile']}_parent{row['parent']}":
            raise ValueError("Wrong rule/parent identity")
        if retained.get(Path(row["checkpoint"]).resolve()) != row["checkpoint_sha256"]:
            raise ValueError("Starting policy is not retained")
        if study["inputs"].get(row["checkpoint"]) != row["checkpoint_sha256"]:
            raise ValueError("Starting policy is not bound")


def training_config(base, study, label):
    row = study["systems"][label]
    parent = row["parent"]
    return {
        **TRAIN, "checkpoint": row["checkpoint"], "profile": row["profile"],
        "parent": parent, "out": str(base / f"train_{label}"),
        "seed": study["training_seeds"][parent], "deal_seed": study["training_deal_seeds"][parent],
    }


def validate_evaluation(result, study, checkpoint, opponent, profile, folder):
    config = result["config"]
    if any(config.get(k) != v for k, v in EVALUATE.items()):
        raise ValueError("Wrong V5A cohort or search settings")
    for key, expected in (("checkpoint", checkpoint), ("opponent", opponent), ("out", folder)):
        if Path(config[key]).resolve() != Path(expected).resolve():
            raise ValueError(f"Wrong evaluation {key}")
    if result["profile"] != profile:
        raise ValueError("Wrong evaluation rule")
    if result["checkpoint_sha256"] != sha(checkpoint) or result["opponent_sha256"] != sha(opponent):
        raise ValueError("Wrong evaluation policy identity")
    if any(result[k] != 0 for k in ("rollouts", "transitions", "roots")):
        raise ValueError("Qualification must not contain search work")
    expected = {
        "scripts/evaluate_public_stock_residual.py", "koikoi/public_stock_residual.py",
        "koikoi/public_stock_residual_search.py", "koikoi/public_stock_search.py",
        "koikoi/public_stock.py", "koikoi/public_stock_native.py",
        "koikoi/native/public_stock.h", "koikoi/native/core.h",
    }
    if set(result["source_hashes"]) != expected or any(
        study["sources"].get(p) != h for p, h in result["source_hashes"].items()
    ):
        raise ValueError("Wrong evaluation source bindings")
    if result["matches"] != 2 * EVALUATE["pairs"]:
        raise ValueError("Incomplete qualification endpoint")


def compare(data, indices):
    """Data has three scores per system: new/own, new/other, old/other."""
    contrasts, passed = {}, {}
    for profile in PROFILES:
        for parent in range(2):
            label = f"{profile}_parent{parent}"
            scores = np.asarray(data[label], dtype=float)
            if scores.ndim != 2 or scores.shape[0] != 3 or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                raise ValueError("Invalid paired qualification scores")
            for key, values in (("own", scores[0] - 0.5), ("other", scores[1] - scores[2])):
                interval = intervals(values, indices, FIXED["lower_quantile"])
                contrasts[f"{label}_{key}"] = {k: interval[k] for k in ("mean", "lower")}
        passed[profile] = all(
            contrasts[f"{profile}_parent{p}_{key}"]["lower"] > 0
            for p in range(2) for key in ("own", "other")
        )
    eligible = [p for p in (PREVIEW_CHOICE, ALL_THREE) if passed[CONTROL] and passed[p]]
    return contrasts, passed, eligible
