"""V5C gate and retained/new policy identities; synthetic, not strength evidence."""

from pathlib import Path

import pytest

from scripts import export_public_stock_mature_variants as exporter
from scripts import public_stock_mature_variants as common
from scripts.public_stock_robustness import ALL_THREE, CONTROL, PREVIEW_CHOICE, PROFILES
from tests.test_public_stock_stronger import qualification


def evidence():
    result = qualification()
    for p in range(2):
        for key in ("own", "other"):
            result["comparisons"][f"{CONTROL}_parent{p}_{key}"]["lower"] = -0.01
    result["qualified_rules"][CONTROL] = False
    result["eligible_search_finalists"] = []
    result["training"] = {
        f"{profile}_parent{p}": {
            "checkpoint": f"new/{profile}_{p}.pt",
            "checkpoint_sha256": f"new-{profile}-{p}",
        }
        for profile in PROFILES
        for p in range(2)
    }
    return result


def test_gate_does_not_promote_failed_control_or_failed_finalist():
    e = evidence()
    assert common.qualified_profiles(e) == list(PROFILES)
    e["comparisons"][f"{PREVIEW_CHOICE}_parent1_other"]["lower"] = 0
    e["qualified_rules"][PREVIEW_CHOICE] = False
    assert common.qualified_profiles(e) == [CONTROL, ALL_THREE]
    e["comparisons"][f"{ALL_THREE}_parent0_own"]["lower"] = -0.1
    e["qualified_rules"][ALL_THREE] = False
    with pytest.raises(ValueError, match="No qualified finalist"):
        common.qualified_profiles(e)


def test_successful_control_must_use_different_prospective_protocol():
    with pytest.raises(ValueError, match="not the applicable protocol"):
        common.qualified_profiles(qualification())


def test_incomplete_or_inconsistent_qualification_is_rejected():
    e = evidence()
    e["qualification_totals"]["matches"] -= 1
    with pytest.raises(ValueError, match="complete"):
        common.qualified_profiles(e)
    e = evidence()
    e["eligible_search_finalists"] = [ALL_THREE]
    with pytest.raises(ValueError, match="gate must have failed"):
        common.qualified_profiles(e)


def test_actual_opponents_and_tables_follow_old_control_new_finalists(
    tmp_path, monkeypatch
):
    retained = {
        f"{CONTROL}_parent{p}": {
            "profile": CONTROL,
            "parent": p,
            "checkpoint": f"old/control{p}.pt",
            "checkpoint_sha256": f"old-{p}",
            "opponent": f"old/control{1 - p}.pt",
            "opponent_sha256": f"old-{1 - p}",
            "table": f"old/table{p}/table.npz",
            "round_table_sha256": f"table-{p}",
            "round_table_metadata_sha256": f"meta-{p}",
        }
        for p in range(2)
    }

    def read(path):
        if str(path) == common.V4_PROTOCOL:
            return {"systems": retained}
        return {
            "config": {"seed": 900000000 + 1000000 * int(Path(path).parent.name[-1])}
        }

    monkeypatch.setattr(common, "read", read)
    rows = common.expected_systems(tmp_path, evidence())
    for parent in range(2):
        old = rows[f"{CONTROL}_parent{parent}"]
        assert old["reuse_table"] and old["checkpoint_sha256"] == f"old-{parent}"
        assert old["opponent_sha256"] == f"old-{1 - parent}"
        assert old["table_seed"] == 900000000 + 1000000 * parent
        for profile in (PREVIEW_CHOICE, ALL_THREE):
            new = rows[f"{profile}_parent{parent}"]
            assert not new["reuse_table"]
            assert new["checkpoint_sha256"] == f"new-{profile}-{parent}"
            assert new["opponent_sha256"] == f"new-{profile}-{1 - parent}"
            assert Path(new["table"]).is_relative_to(tmp_path)


def test_independent_v5c_refinement_preserves_fixed_counts(tmp_path, monkeypatch):
    for name in ("protocol.json", "tables.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    label = f"{ALL_THREE}_parent0"
    study = {**common.FIXED, "systems": {label: {}}}
    seen = []

    def reduce(base, study, tables, label, stage, budget, particles, bindings):
        seen.append(particles)
        folder = base / exporter.endpoint_name(label, stage, budget)
        folder.mkdir()
        (folder / "result.json").write_text("{}", encoding="utf-8")
        return [], {
            "matches": 128,
            "rounds": 512,
            "decisions": 2048,
            "rollouts": particles * 512,
            "transitions": particles * 128 * 5000,
            "roots": 256,
            "python_replay_decisions": 2048,
            "seconds": 1.0,
        }

    monkeypatch.setattr(exporter, "reduce_checked", reduce)
    initial = {"particles": {label: [0, 15, 60]}}
    record, work = exporter.reduce_calibration(
        tmp_path, study, {}, "refinement", {}, initial
    )
    assert record["particles"][label] == [0, 12, 48]
    assert seen == [15, 60] and len(work) == 2
