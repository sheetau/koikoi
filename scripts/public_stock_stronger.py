"""Conditional V5B contract: improved policies and two-stage work calibration."""

from pathlib import Path

import numpy as np

from scripts.export_public_stock_results import intervals, read, sha
from scripts.public_stock_adaptation import SOURCES as ADAPTATION_SOURCES
from scripts.public_stock_robustness import (
    ALL_THREE,
    CONTROL,
    PREVIEW_CHOICE,
    PROFILES,
    TARGETS,
    validate_result,
)
from scripts.public_stock_robustness import (
    SOURCES as ROBUSTNESS_SOURCES,
)

V5A = "docs/paper/generated/public_stock_v5a_evidence.json"
V5A_BASE = Path("runs/public_stock_adaptation_v5a")
FIXED = {
    "method": "public_stock_stronger_v5b",
    "table_samples": 512,
    "table_seed": 1250000000,
    "initial_seed": 1260000000,
    "initial_pairs": 64,
    "initial_particles": 32,
    "refinement_seed": 1270000000,
    "refinement_pairs": 64,
    "evaluation_seed": 1280000000,
    "evaluation_pairs": 256,
    "targets": list(TARGETS),
    "work_tolerance": 0.1,
    "batch": 32,
    "horizon": 80,
    "confidence": 2.5,
    "search_seed": 1117,
    "bootstrap_seed": 1123,
    "bootstrap_replicates": 20000,
}
SOURCES = tuple(
    sorted(
        set(
            ADAPTATION_SOURCES
            + ROBUSTNESS_SOURCES
            + (
                "docs/PUBLIC_STOCK_STUDY_V5B.rst",
                "docs/PUBLIC_STOCK_WORK_ANALYSIS.rst",
                "scripts/public_stock_stronger.py",
                "scripts/run_public_stock_stronger.py",
                "scripts/export_public_stock_stronger.py",
                "scripts/build_auditable_round_values.py",
                "scripts/audit_public_round_values.py",
                "scripts/complete_public_stock_adaptation.py",
            )
        )
    )
)


def qualified_profiles(evidence):
    if (
        evidence.get("method") != "public_stock_adaptation_v5a"
        or evidence.get("qualification_totals", {}).get("matches") != 36864
        or len(evidence.get("comparisons", {})) != 12
    ):
        raise ValueError("Expected complete independent V5A qualification")
    passed = {
        profile: all(
            evidence["comparisons"][f"{profile}_parent{p}_{key}"]["lower"] > 0
            for p in range(2)
            for key in ("own", "other")
        )
        for profile in PROFILES
    }
    finalists = [
        p for p in (PREVIEW_CHOICE, ALL_THREE) if passed[CONTROL] and passed[p]
    ]
    if (
        passed != evidence["qualified_rules"]
        or finalists != evidence["eligible_search_finalists"]
    ):
        raise ValueError("V5A qualification decision differs")
    if not finalists:
        raise ValueError("V5B requires control and at least one improved finalist")
    return [CONTROL, *finalists]


def make_protocol(base):
    evidence = read(V5A)
    profiles = qualified_profiles(evidence)
    finished = read(V5A_BASE / "finished.json")
    if finished["status"] != "complete" or finished["sha256"] != sha(V5A):
        raise ValueError("V5A finished receipt differs")
    for path, digest in evidence["bindings"].items():
        if sha(path) != digest:
            raise ValueError(f"V5A evidence changed: {path}")
    systems, inputs = {}, {}
    for profile in profiles:
        for parent in range(2):
            label = f"{profile}_parent{parent}"
            trained = evidence["training"][label]
            other = evidence["training"][f"{profile}_parent{1 - parent}"]
            systems[label] = {
                "profile": profile,
                "parent": parent,
                "checkpoint": trained["checkpoint"],
                "checkpoint_sha256": trained["checkpoint_sha256"],
                "opponent": other["checkpoint"],
                "opponent_sha256": other["checkpoint_sha256"],
                "table": (base / f"table_{label}" / "table.npz").as_posix(),
                "table_seed": FIXED["table_seed"]
                + 2000000 * PROFILES.index(profile)
                + 1000000 * parent,
            }
            inputs[trained["checkpoint"]] = trained["checkpoint_sha256"]
    for path in (
        V5A,
        (V5A_BASE / "finished.json").as_posix(),
        (V5A_BASE / "training_audit.json").as_posix(),
    ):
        inputs[path] = sha(path)
    return {
        **FIXED,
        "profiles": profiles,
        "systems": systems,
        "inputs": inputs,
        "sources": {p: sha(p) for p in SOURCES},
    }


def verify_protocol(study):
    if any(study.get(k) != v for k, v in FIXED.items()):
        raise ValueError("V5B fixed design changed")
    if set(study["sources"]) != set(SOURCES):
        raise ValueError("Incomplete V5B sources")
    for path, digest in {**study["inputs"], **study["sources"]}.items():
        if sha(path) != digest:
            raise ValueError(f"Frozen V5B input changed: {path}")
    evidence = read(V5A)
    if study["profiles"] != qualified_profiles(evidence):
        raise ValueError("Eligible profiles changed")
    if set(study["systems"]) != {
        f"{p}_parent{i}" for p in study["profiles"] for i in range(2)
    }:
        raise ValueError("Wrong improved-system coverage")
    for path in (
        V5A,
        (V5A_BASE / "finished.json").as_posix(),
        (V5A_BASE / "training_audit.json").as_posix(),
    ):
        if study["inputs"].get(path) != sha(path):
            raise ValueError("V5A evidence/finish/audit is not bound")
    finished = read(V5A_BASE / "finished.json")
    if finished["status"] != "complete" or finished["sha256"] != sha(V5A):
        raise ValueError("V5A is not finished")
    for label, row in study["systems"].items():
        trained = evidence["training"][label]
        other = evidence["training"][f"{row['profile']}_parent{1 - row['parent']}"]
        if label != f"{row['profile']}_parent{row['parent']}":
            raise ValueError("Wrong parent/profile")
        for key, record in (("checkpoint", trained), ("opponent", other)):
            if (
                Path(row[key]).resolve() != Path(record["checkpoint"]).resolve()
                or row[key + "_sha256"] != record["checkpoint_sha256"]
            ):
                raise ValueError("System does not use the qualified final policy")
            if study["inputs"].get(row[key]) != row[key + "_sha256"]:
                raise ValueError("Qualified checkpoint is not bound")
        expected_seed = (
            FIXED["table_seed"]
            + 2000000 * PROFILES.index(row["profile"])
            + 1000000 * row["parent"]
        )
        if row["table_seed"] != expected_seed:
            raise ValueError("Wrong table seed")


def corrected_count(transitions, matches, particles, target):
    if (
        type(transitions) is not int
        or transitions <= 0
        or matches != 128
        or type(particles) is not int
        or not 2 <= particles <= 256
        or target not in TARGETS
    ):
        raise ValueError("Invalid work-only calibration inputs")
    return min(
        256,
        max(2, (2 * target * matches * particles + transitions) // (2 * transitions)),
    )


def increasing(counts):
    if len(counts) != 2 or not 2 <= counts[0] < counts[1] <= 256:
        raise ValueError("Calibration must yield increasing low/high counts")
    return [0, *counts]


def table_receipt(study, label, bindings):
    row = study["systems"][label]
    folder = Path(row["table"]).parent
    meta, audit, receipt = (
        read(folder / n) for n in ("result.json", "audit.json", "protocol.json")
    )
    config = meta["config"]
    if any(meta.get(k) != v for k, v in receipt.items()):
        raise ValueError("Table pre-run receipt differs")
    if (
        meta["profile"] != row["profile"]
        or meta["checkpoint_sha256"] != row["checkpoint_sha256"]
        or Path(config["checkpoint"]).resolve() != Path(row["checkpoint"]).resolve()
        or Path(config["out"]).resolve() != folder.resolve()
        or config["samples"] != study["table_samples"]
        or config["seed"] != row["table_seed"]
        or meta["table_sha256"] != sha(row["table"])
    ):
        raise ValueError("Wrong new-policy table identity or cohort")
    if (
        audit["status"] != "passed"
        or audit["table_sha256"] != meta["table_sha256"]
        or meta["conditional_round_rollouts"] != 483328
        or audit["all_target_values_reduced"] != 483328
        or audit["python_round_replays"] != 3776
        or audit["audit_offsets_per_cell"] != [0, 170, 341, 511]
        or audit["all_trajectories_replayed"] is not False
        or audit["transitions"] != meta["transitions"]
    ):
        raise ValueError("Incomplete new-table audit")
    canonical = {Path(p).resolve(): h for p, h in study["sources"].items()}
    expected = {
        Path(p).resolve()
        for p in (
            "scripts/build_auditable_round_values.py",
            "koikoi/public_stock_round_value.py",
            "koikoi/public_stock_residual.py",
        )
    }
    if {Path(p).resolve() for p in meta["sources"]} != expected:
        raise ValueError("Incomplete table source receipt")
    for path, digest in meta["sources"].items():
        if canonical.get(Path(path).resolve()) != digest or sha(path) != digest:
            raise ValueError("Table source differs")
    if set(meta["traces"]) != {f"round_{r}.npz" for r in range(8)}:
        raise ValueError("Incomplete table traces")
    for name, digest in meta["traces"].items():
        path = (folder / name).as_posix()
        if sha(path) != digest:
            raise ValueError("Table trace changed")
        bindings[path] = digest
    for name in ("result.json", "audit.json", "protocol.json", "table.npz"):
        path = (folder / name).as_posix()
        bindings[path] = sha(path)
    system = {
        **row,
        "round_table_sha256": meta["table_sha256"],
        "round_table_metadata_sha256": sha(folder / "result.json"),
    }
    counts = {
        k: meta[k]
        for k in (
            "conditional_round_rollouts",
            "terminal_round_rollouts",
            "transitions",
        )
    }
    counts.update(
        {
            k: audit[k]
            for k in (
                "python_round_replays",
                "python_replay_decisions",
                "all_target_values_reduced",
            )
        }
    )
    return system, counts


def validate_endpoint(result, study, tables, label, stage, particles, folder):
    if stage not in ("initial", "refinement", "evaluation"):
        raise ValueError("Unknown V5B stage")
    # The V4 validator is parameterized by the cohort and system identities.
    # Reuse its strict recorded-evaluator contract, without changing frozen code.
    effective = {
        **study,
        "systems": tables["systems"],
        "calibration_seed": study[f"{stage}_seed"],
        "calibration_pairs": study[f"{stage}_pairs"],
    }
    validate_result(
        result,
        effective,
        label,
        "evaluation" if stage == "evaluation" else "calibration",
        particles,
        folder,
    )


def compare(data, indices):
    profiles = [p for p in PROFILES if p in data]
    if (
        not profiles
        or profiles[0] != CONTROL
        or len(profiles) not in (2, 3)
        or set(profiles) != set(data)
    ):
        raise ValueError("Expected control and qualified finalist(s)")
    arrays = {p: np.asarray(data[p], dtype=float) for p in profiles}
    shape = arrays[CONTROL].shape
    if (
        len(shape) != 3
        or shape[:2] != (2, 3)
        or any(
            a.shape != shape or not np.isfinite(a).all() or np.any((a < 0) | (a > 1))
            for a in arrays.values()
        )
    ):
        raise ValueError("Invalid parent/budget/stream score arrays")
    changes = {p: a[:, 2] - a[:, 0] for p, a in arrays.items()}
    vectors = {}
    for profile in profiles[1:]:
        for parent in range(2):
            label = f"{profile}_parent{parent}"
            vectors[f"{label}_gain"] = changes[profile][parent]
            vectors[f"{label}_excess"] = (
                changes[profile][parent] - changes[CONTROL][parent]
            )
            vectors[f"{label}_high_minus_low"] = (
                arrays[profile][parent, 2] - arrays[profile][parent, 1]
            )
    if len(profiles) == 3:
        for parent in range(2):
            vectors[f"preview_minus_all_parent{parent}"] = (
                changes[PREVIEW_CHOICE][parent] - changes[ALL_THREE][parent]
            )
    quantile = 0.05 / (2 * len(vectors))
    comparisons = {k: intervals(v, indices, quantile) for k, v in vectors.items()}
    passes = {
        profile: all(
            comparisons[f"{profile}_parent{p}_{k}"]["lower"] > 0
            for p in range(2)
            for k in ("gain", "excess", "high_minus_low")
        )
        for profile in profiles[1:]
    }
    ranking = "not_tested" if len(profiles) == 2 else "unresolved"
    if len(profiles) == 3:
        if all(
            comparisons[f"preview_minus_all_parent{p}"]["lower"] > 0 for p in range(2)
        ):
            ranking = PREVIEW_CHOICE
        elif all(
            comparisons[f"preview_minus_all_parent{p}"]["upper"] < 0 for p in range(2)
        ):
            ranking = ALL_THREE
    return comparisons, passes, ranking, quantile
