"""V5B protocol and independent work-reduction checks; no strength measurement."""

from copy import deepcopy

import numpy as np
import pytest

from scripts import export_public_stock_stronger as exporter
from scripts.public_stock_robustness import ALL_THREE, CONTROL, PREVIEW_CHOICE, PROFILES
from scripts.public_stock_stronger import (
    FIXED,
    compare,
    corrected_count,
    increasing,
    qualified_profiles,
    table_receipt,
    validate_endpoint,
)
from tests.test_public_stock_robustness import example_endpoint


def qualification():
    return {
        "method": "public_stock_adaptation_v5a",
        "qualification_totals": {"matches": 36864},
        "comparisons": {
            f"{p}_parent{i}_{k}": {"mean": 0.1, "lower": 0.05}
            for p in PROFILES
            for i in range(2)
            for k in ("own", "other")
        },
        "qualified_rules": dict.fromkeys(PROFILES, True),
        "eligible_search_finalists": [PREVIEW_CHOICE, ALL_THREE],
    }


def test_only_complete_qualified_profiles_can_advance():
    e = qualification()
    assert qualified_profiles(e) == list(PROFILES)
    e["comparisons"][f"{PREVIEW_CHOICE}_parent1_other"]["lower"] = -0.01
    e["qualified_rules"][PREVIEW_CHOICE] = False
    e["eligible_search_finalists"] = [ALL_THREE]
    assert qualified_profiles(e) == [CONTROL, ALL_THREE]
    e["comparisons"][f"{CONTROL}_parent0_own"]["lower"] = 0
    e["qualified_rules"][CONTROL] = False
    e["eligible_search_finalists"] = []
    with pytest.raises(ValueError, match="requires control"):
        qualified_profiles(e)


@pytest.mark.parametrize(
    "field,value",
    [
        ("method", "other"),
        ("qualification_totals", {"matches": 36863}),
        ("comparisons", {}),
    ],
)
def test_incomplete_qualification_cannot_launch(field, value):
    e = qualification()
    e[field] = value
    with pytest.raises(ValueError, match="complete"):
        qualified_profiles(e)


def test_qualification_receipt_must_agree_with_all_contrasts():
    e = qualification()
    e["qualified_rules"][ALL_THREE] = False
    with pytest.raises(ValueError, match="decision differs"):
        qualified_profiles(e)


def test_initial_and_once_corrected_work_counts():
    initial = increasing(
        [corrected_count(4000 * 128 * 32, 128, 32, t) for t in (60000, 240000)]
    )
    assert initial == [0, 15, 60]
    assert corrected_count(5000 * 128 * 15, 128, 15, 60000) == 12
    assert corrected_count(4000 * 128 * 60, 128, 60, 240000) == 60
    assert corrected_count(24000 * 128 * 32, 128, 32, 60000) == 3  # exact 2.5 tie
    with pytest.raises(ValueError, match="increasing"):
        increasing([256, 256])


@pytest.mark.parametrize(
    "t,m,p,target",
    [
        (0, 128, 32, 60000),
        (-1, 128, 32, 60000),
        (1000, 64, 32, 60000),
        (1000, 128, 1, 60000),
        (1000, 128, 257, 60000),
        (1000, 128, 32, 1),
    ],
)
def test_invalid_calibration_inputs(t, m, p, target):
    with pytest.raises(ValueError):
        corrected_count(t, m, p, target)


def test_independent_two_stage_reduction_uses_proposed_counts(tmp_path, monkeypatch):
    for name in ("protocol.json", "tables.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    label = f"{ALL_THREE}_parent0"
    study = {**FIXED, "systems": {label: {}}}
    seen = []

    def reduce(base, study, tables, label, stage, budget, particles, bindings):
        seen.append((stage, budget, particles))
        per_particle = 4000 if stage == "initial" or budget == 2 else 5000
        name = exporter.endpoint_name(label, stage, budget)
        folder = base / name
        folder.mkdir()
        (folder / "result.json").write_text("{}", encoding="utf-8")
        return [], {
            "matches": 128,
            "rounds": 512,
            "decisions": 1024,
            "rollouts": particles * 512,
            "transitions": particles * 128 * per_particle,
            "roots": 256,
            "python_replay_decisions": 1024,
            "seconds": 1.0,
        }

    monkeypatch.setattr(exporter, "reduce_checked", reduce)
    initial, _ = exporter.reduce_calibration(tmp_path, study, {}, "initial", {})
    refined, work = exporter.reduce_calibration(
        tmp_path, study, {}, "refinement", {}, initial
    )
    assert initial["particles"][label] == [0, 15, 60]
    assert refined["particles"][label] == [0, 12, 60]
    assert seen == [("initial", 0, 32), ("refinement", 1, 15), ("refinement", 2, 60)]
    assert len(work) == 2


@pytest.mark.parametrize("stage", ["initial", "refinement", "evaluation"])
def test_endpoint_contract_enforces_new_cohorts_and_policies(tmp_path, stage):
    result, old_study, label, folder = example_endpoint(tmp_path)
    study = {**deepcopy(FIXED), "sources": old_study["sources"]}
    tables = {"systems": old_study["systems"]}
    result["config"].update(
        {k: study[k] for k in ("batch", "horizon", "confidence", "search_seed")}
    )
    result["config"].update(
        pairs=study[stage + "_pairs"], deal_seed=study[stage + "_seed"]
    )
    result["matches"] = 2 * study[stage + "_pairs"]
    validate_endpoint(result, study, tables, label, stage, 32, folder)
    result["config"]["deal_seed"] = 1210000000
    with pytest.raises(ValueError, match="settings or stream"):
        validate_endpoint(result, study, tables, label, stage, 32, folder)
    result["config"]["deal_seed"] = study[stage + "_seed"]
    result["opponent_sha256"] = "old retained actor"
    with pytest.raises(ValueError, match="identity"):
        validate_endpoint(result, study, tables, label, stage, 32, folder)


@pytest.mark.parametrize(
    "finalists", [[ALL_THREE], [PREVIEW_CHOICE], [PREVIEW_CHOICE, ALL_THREE]]
)
def test_conditional_comparison_family_keeps_shared_noise(finalists):
    noise = np.array([0, 0.25, 0.5, 0.75])
    base = np.stack([noise, noise, noise])
    better = np.stack([noise, noise + 0.125, noise + 0.25])
    data = {
        CONTROL: np.stack([base, base]),
        **{p: np.stack([better, better]) for p in finalists},
    }
    indices = np.random.default_rng(1).integers(0, 4, (100, 4))
    comparisons, passes, ranking, q = compare(data, indices)
    m = 14 if len(finalists) == 2 else 6
    assert len(comparisons) == m and q == 0.05 / (2 * m)
    assert all(passes.values())
    assert comparisons[f"{finalists[0]}_parent0_gain"] == {
        "mean": 0.25,
        "lower": 0.25,
        "upper": 0.25,
    }
    assert ranking == ("unresolved" if m == 14 else "not_tested")
    data[finalists[0]][1, 2] = noise
    assert not compare(data, indices)[1][finalists[0]]


def test_old_policy_table_cannot_be_used_for_new_policy(tmp_path, monkeypatch):
    from scripts import public_stock_stronger as common

    checkpoint = tmp_path / "new.pt"
    folder = tmp_path / "table"
    folder.mkdir()
    (folder / "table.npz").write_bytes(b"table")
    row = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": "new-sha",
        "profile": ALL_THREE,
        "table": str(folder / "table.npz"),
        "table_seed": 1254000000,
    }
    meta = {
        "config": {
            "checkpoint": str(checkpoint),
            "out": str(folder),
            "samples": 512,
            "seed": 1254000000,
        },
        "profile": ALL_THREE,
        "checkpoint_sha256": "old-sha",
        "table_sha256": common.sha(folder / "table.npz"),
    }
    monkeypatch.setattr(
        common, "read", lambda p: {} if p.name == "audit.json" else meta
    )
    with pytest.raises(ValueError, match="new-policy table identity"):
        table_receipt({"systems": {"test": row}, "table_samples": 512}, "test", {})


def test_empty_comparison_is_rejected():
    with pytest.raises(ValueError):
        compare({}, np.zeros((1, 1), dtype=int))
