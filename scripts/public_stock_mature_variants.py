"""V5C keeps old control while testing only independently qualified new variants."""

from pathlib import Path

from scripts.export_public_stock_results import read, sha
from scripts.public_stock_robustness import CONTROL, PROFILES
from scripts.public_stock_stronger import (
    FIXED as V5B_FIXED,
)
from scripts.public_stock_stronger import (
    SOURCES as V5B_SOURCES,
)
from scripts.public_stock_stronger import V5A, V5A_BASE, table_receipt

V3 = "docs/paper/generated/public_stock_v3_evidence.json"
V4_PROTOCOL = "runs/public_stock_robustness_v4/protocol.json"
FIXED = {
    **V5B_FIXED,
    "method": "public_stock_mature_variants_v5c",
    "table_seed": 1290000000,
    "initial_seed": 1300000000,
    "refinement_seed": 1310000000,
    "evaluation_seed": 1320000000,
    "search_seed": 1137,
    "bootstrap_seed": 1151,
    "control_policy": "retained_128_updates",
    "finalist_policy": "qualified_512_updates",
}
SOURCES = (
    *V5B_SOURCES,
    "docs/PUBLIC_STOCK_STUDY_V5C.rst",
    "scripts/public_stock_mature_variants.py",
    "scripts/run_public_stock_mature_variants.py",
    "scripts/export_public_stock_mature_variants.py",
)


def qualified_profiles(evidence):
    if (
        evidence.get("method") != "public_stock_adaptation_v5a"
        or evidence.get("qualification_totals", {}).get("matches") != 36864
        or len(evidence.get("comparisons", {})) != 12
    ):
        raise ValueError("V5C requires complete V5A qualification")
    passed = {
        profile: all(
            evidence["comparisons"][f"{profile}_parent{p}_{k}"]["lower"] > 0
            for p in range(2)
            for k in ("own", "other")
        )
        for profile in PROFILES
    }
    if passed != evidence["qualified_rules"]:
        raise ValueError("Qualification decisions differ")
    if passed[CONTROL]:
        raise ValueError(
            "Improved control qualifies; V5C is not the applicable protocol"
        )
    if evidence["eligible_search_finalists"]:
        raise ValueError("The all-improved V5B launch gate must have failed")
    finalists = [p for p in PROFILES[1:] if passed[p]]
    if not finalists:
        raise ValueError("No qualified finalist")
    return [CONTROL, *finalists]


def expected_systems(base, evidence):
    retained = read(V4_PROTOCOL)["systems"]
    systems = {}
    for profile in qualified_profiles(evidence):
        for parent in range(2):
            label = f"{profile}_parent{parent}"
            if profile == CONTROL:
                row = retained[label]
                systems[label] = {
                    **row,
                    "reuse_table": True,
                    "table_seed": read(Path(row["table"]).parent / "result.json")[
                        "config"
                    ]["seed"],
                }
            else:
                own = evidence["training"][label]
                other = evidence["training"][f"{profile}_parent{1 - parent}"]
                systems[label] = {
                    "profile": profile,
                    "parent": parent,
                    "reuse_table": False,
                    "checkpoint": own["checkpoint"],
                    "checkpoint_sha256": own["checkpoint_sha256"],
                    "opponent": other["checkpoint"],
                    "opponent_sha256": other["checkpoint_sha256"],
                    "table": (base / f"table_{label}" / "table.npz").as_posix(),
                    "table_seed": FIXED["table_seed"]
                    + 2000000 * PROFILES.index(profile)
                    + 1000000 * parent,
                }
    return systems


def make_protocol(base):
    evidence = read(V5A)
    profiles = qualified_profiles(evidence)
    finished = read(V5A_BASE / "finished.json")
    if finished["status"] != "complete" or finished["sha256"] != sha(V5A):
        raise ValueError("V5A finish receipt differs")
    for path, digest in evidence["bindings"].items():
        if sha(path) != digest:
            raise ValueError(f"V5A evidence changed: {path}")
    systems = expected_systems(base, evidence)
    inputs = {row["checkpoint"]: row["checkpoint_sha256"] for row in systems.values()}
    for path in (
        V3,
        V4_PROTOCOL,
        V5A,
        (V5A_BASE / "finished.json").as_posix(),
        (V5A_BASE / "training_audit.json").as_posix(),
    ):
        inputs[path] = sha(path)
    study = {
        **FIXED,
        "out": base.as_posix(),
        "profiles": profiles,
        "systems": systems,
        "inputs": inputs,
        "sources": {p: sha(p) for p in SOURCES},
    }
    prior = {Path(p).resolve(): h for p, h in read(V3)["bindings"].items()}
    for label, row in systems.items():
        if row["reuse_table"]:
            bindings = {}
            table_receipt(study, label, bindings)
            for p, h in {
                **bindings,
                row["checkpoint"]: row["checkpoint_sha256"],
            }.items():
                if prior.get(Path(p).resolve()) != h:
                    raise ValueError("Control input is not the retained V3 artifact")
            inputs.update(bindings)
    return study


def verify_protocol(study):
    if any(study.get(k) != v for k, v in FIXED.items()) or set(study["sources"]) != set(
        SOURCES
    ):
        raise ValueError("V5C fixed design or source coverage differs")
    for p, h in {**study["inputs"], **study["sources"]}.items():
        if sha(p) != h:
            raise ValueError(f"Frozen V5C input changed: {p}")
    evidence = read(V5A)
    if study["profiles"] != qualified_profiles(evidence) or study[
        "systems"
    ] != expected_systems(Path(study["out"]), evidence):
        raise ValueError("Retained control or qualified finalist identity differs")
    for p in (
        V3,
        V4_PROTOCOL,
        V5A,
        (V5A_BASE / "finished.json").as_posix(),
        (V5A_BASE / "training_audit.json").as_posix(),
    ):
        if study["inputs"].get(p) != sha(p):
            raise ValueError(
                "Qualification or retained-control provenance is not bound"
            )
    finished = read(V5A_BASE / "finished.json")
    if finished["status"] != "complete" or finished["sha256"] != sha(V5A):
        raise ValueError("V5A is not complete")
    prior = {Path(p).resolve(): h for p, h in read(V3)["bindings"].items()}
    for label, row in study["systems"].items():
        for k in ("checkpoint", "opponent"):
            if study["inputs"].get(row[k]) != row[k + "_sha256"]:
                raise ValueError("Policy is not bound")
        if row["reuse_table"]:
            bindings = {}
            table_receipt(study, label, bindings)
            for p, h in {
                **bindings,
                row["checkpoint"]: row["checkpoint_sha256"],
            }.items():
                if study["inputs"].get(p) != h or prior.get(Path(p).resolve()) != h:
                    raise ValueError("Retained control artifact identity differs")
