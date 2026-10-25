"""Fixed V4 contracts; no game logic or policy changes."""

from pathlib import Path

from scripts.export_public_stock_results import read, sha

CONTROL = "preview0_choice0_stakes0"
PREVIEW_CHOICE = "preview1_choice1_stakes0"
ALL_THREE = "preview1_choice1_stakes1"
PROFILES = (CONTROL, PREVIEW_CHOICE, ALL_THREE)
TARGETS = (60000, 240000)
FIXED = {
    "method": "public_stock_robustness_v4",
    "profiles": list(PROFILES),
    "calibration_seed": 1200000000,
    "calibration_pairs": 32,
    "calibration_particles": 32,
    "evaluation_seed": 1210000000,
    "evaluation_pairs": 128,
    "targets": list(TARGETS),
    "work_tolerance": 0.1,
    "batch": 32,
    "horizon": 80,
    "confidence": 2.5,
    "search_seed": 1091,
    "bootstrap_seed": 1093,
    "bootstrap_replicates": 20000,
    "interval_quantile": 0.05 / 28,
}
COUNTS = (
    "matches", "rounds", "decisions", "rollouts", "transitions", "roots",
    "python_replay_decisions",
)
SOURCES = (
    "docs/PUBLIC_STOCK_STUDY_V4.rst",
    "scripts/public_stock_robustness.py",
    "scripts/run_public_stock_robustness.py",
    "scripts/export_public_stock_robustness.py",
    "scripts/export_public_stock_results.py",
    "scripts/evaluate_public_stock_rounds.py",
    "koikoi/public_stock_round_value.py",
    "koikoi/public_stock_residual.py",
    "koikoi/public_stock_policy.py",
    "koikoi/public_stock_search.py",
    "koikoi/public_stock.py",
    "koikoi/public_stock_native.py",
    "koikoi/native/public_stock.h",
    "koikoi/native/core.h",
    "koikoi/reference.py",
    "koikoi/rule_variants.py",
    "koikoi/layout.py",
)


def calibrate_particles(transitions, matches):
    """Only physical work enters this rule; scores cannot select budgets."""
    if type(transitions) is not int or transitions <= 0 or matches != 2 * FIXED["calibration_pairs"]:
        raise ValueError("Invalid calibration work or match count")
    # Exact integer half-up rounding avoids a floating-point tie changing a budget.
    counts = [
        min(256, max(2, (2 * t * matches * 32 + transitions) // (2 * transitions)))
        for t in TARGETS
    ]
    if counts[0] >= counts[1]:
        raise ValueError("Calibration did not yield distinct increasing budgets")
    return [0, *counts]


def make_protocol(trained, tables):
    """Bind six existing policies and audited tables without repeating preparation."""
    sources = {p: sha(p) for p in SOURCES}
    inputs = {}
    systems = {}
    previous = read("docs/paper/generated/public_stock_v3_evidence.json")
    if not previous["confirmation"]["passed"]:
        raise ValueError("V3 confirmation is not complete and qualified")
    previous_inputs = {Path(p).resolve(): h for p, h in previous["bindings"].items()}
    for profile in PROFILES:
        for parent in range(2):
            label = f"{profile}_parent{parent}"
            checkpoint = trained / label / "policy_00128.pt"
            opponent = trained / f"{profile}_parent{1 - parent}" / "policy_00128.pt"
            folder = tables / f"table_{label}"
            table = folder / "table.npz"
            meta, audit = read(folder / "result.json"), read(folder / "audit.json")
            digest = sha(checkpoint)
            if meta["profile"] != profile or meta["checkpoint_sha256"] != digest:
                raise ValueError("Wrong retained table policy or rule")
            if (
                meta["table_sha256"] != sha(table)
                or audit["table_sha256"] != meta["table_sha256"]
                or audit["status"] != "passed"
                or meta["config"]["samples"] != 512
                or meta["conditional_round_rollouts"] != 8 * 118 * 512
                or audit["all_target_values_reduced"] != meta["conditional_round_rollouts"]
                or audit["python_round_replays"] != 3776
            ):
                raise ValueError("Retained table audit or coverage differs")
            for p, h in meta["sources"].items():
                if sha(p) != h:
                    raise ValueError(f"Retained table source changed: {p}")
                sources[p] = h
            for name, h in meta["traces"].items():
                p = folder / name
                if sha(p) != h:
                    raise ValueError("Retained conditional-round trace changed")
                inputs[p.as_posix()] = h
            for p in (
                checkpoint, opponent, table, folder / "protocol.json",
                folder / "result.json", folder / "audit.json",
            ):
                if previous_inputs.get(p.resolve()) != sha(p):
                    raise ValueError(f"Input is not the retained V3 artifact: {p}")
                inputs[p.as_posix()] = sha(p)
            systems[label] = {
                "profile": profile,
                "parent": parent,
                "checkpoint": checkpoint.as_posix(),
                "checkpoint_sha256": digest,
                "opponent": opponent.as_posix(),
                "opponent_sha256": sha(opponent),
                "table": table.as_posix(),
                "round_table_sha256": meta["table_sha256"],
                "round_table_metadata_sha256": sha(folder / "result.json"),
            }
    for p in (
        "docs/paper/generated/public_stock_v3_evidence.json",
        "runs/public_stock_model_audit_v2.json",
    ):
        inputs[p] = sha(p)
    return {**FIXED, "systems": systems, "sources": sources, "inputs": inputs}


def verify_protocol(study):
    if any(study.get(k) != v for k, v in FIXED.items()):
        raise ValueError("V4 fixed design changed")
    labels = {f"{p}_parent{i}" for p in PROFILES for i in range(2)}
    if set(study["systems"]) != labels or not set(SOURCES) <= set(study["sources"]):
        raise ValueError("Missing V4 systems or source bindings")
    previous = read("docs/paper/generated/public_stock_v3_evidence.json")
    if not previous["confirmation"]["passed"]:
        raise ValueError("V3 confirmation receipt changed")
    retained = {Path(p).resolve(): h for p, h in previous["bindings"].items()}
    for label, system in study["systems"].items():
        opposite = study["systems"][f"{system['profile']}_parent{1 - system['parent']}"]
        if (
            label != f"{system['profile']}_parent{system['parent']}"
            or system["opponent_sha256"] != opposite["checkpoint_sha256"]
            or Path(system["opponent"]).resolve() != Path(opposite["checkpoint"]).resolve()
        ):
            raise ValueError("Opponent is not the other retained parent")
        for path_key, digest_key in (
            ("checkpoint", "checkpoint_sha256"), ("opponent", "opponent_sha256"),
            ("table", "round_table_sha256"),
        ):
            if study["inputs"].get(system[path_key]) != system[digest_key]:
                raise ValueError("System identity is absent from input bindings")
            if retained.get(Path(system[path_key]).resolve()) != system[digest_key]:
                raise ValueError("System does not use the retained V3 artifact")
        meta_path = (Path(system["table"]).parent / "result.json").as_posix()
        if (
            study["inputs"].get(meta_path) != system["round_table_metadata_sha256"]
            or retained.get(Path(meta_path).resolve()) != system["round_table_metadata_sha256"]
        ):
            raise ValueError("Table metadata is absent from input bindings")
    for p, h in {**study["sources"], **study["inputs"]}.items():
        if sha(p) != h:
            raise ValueError(f"Frozen V4 input changed: {p}")


def validate_result(result, study, label, stage, particles, folder):
    """Enforce opponent, table, RNG, cohort, and physical-work identities."""
    if stage not in ("calibration", "evaluation"):
        raise ValueError("Unknown V4 stage")
    system = study["systems"][label]
    config = result["config"]
    expected = {
        "command": "evaluate", "particles": particles,
        "pairs": study[f"{stage}_pairs"], "deal_seed": study[f"{stage}_seed"],
        **{k: study[k] for k in ("batch", "horizon", "confidence", "search_seed")},
    }
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Wrong V4 endpoint settings or stream")
    for k in ("checkpoint", "opponent", "table"):
        if Path(config[k]).resolve() != Path(system[k]).resolve():
            raise ValueError(f"Wrong endpoint {k}")
    if Path(config["out"]).resolve() != Path(folder).resolve():
        raise ValueError("Wrong output endpoint")
    for k in (
        "profile", "checkpoint_sha256", "opponent_sha256",
        "round_table_sha256", "round_table_metadata_sha256",
    ):
        if result.get(k) != system[k]:
            raise ValueError(f"Wrong endpoint identity: {k}")
    if not result.get("source_hashes") or any(
        study["sources"].get(p) != h for p, h in result["source_hashes"].items()
    ):
        raise ValueError("Endpoint source is not bound to V4")
    expected_endpoint_sources = {
        "scripts/evaluate_public_stock_rounds.py", "koikoi/public_stock_residual.py",
        "koikoi/public_stock_round_value.py", "koikoi/public_stock_search.py",
        "koikoi/public_stock.py", "koikoi/public_stock_native.py",
        "koikoi/native/public_stock.h", "koikoi/native/core.h",
    }
    if set(result["source_hashes"]) != expected_endpoint_sources:
        raise ValueError("Incomplete endpoint source receipt")
    if any(type(result.get(k)) is not int or result[k] < 0 for k in COUNTS):
        raise ValueError("Invalid work counters")
    r, t, roots = (result[k] for k in ("rollouts", "transitions", "roots"))
    if particles == 0:
        if (r, t, roots) != (0, 0, 0):
            raise ValueError("No-search endpoint recorded search work")
    elif r % particles or r < 2 * particles * roots or not r <= t <= 80 * r:
        raise ValueError("Inconsistent recorded completed-round work")
    if roots > result["decisions"] or result["matches"] != 2 * expected["pairs"]:
        raise ValueError("Wrong root or match count")


def work_admissibility(transitions, matches, target):
    if target <= 0 or matches <= 0 or transitions < 0:
        raise ValueError("Invalid work-admissibility inputs")
    actual = transitions / matches
    return {
        "target": target, "actual_per_match": actual,
        "relative_deviation": actual / target - 1,
        "passed": abs(actual - target) <= FIXED["work_tolerance"] * target,
    }
