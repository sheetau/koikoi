import copy
import itertools
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from koikoi.published_train import PublishedConfig
from koikoi.upstream import UPSTREAM_PATH
from scripts.score_auxiliary_trainer import AuxiliaryConfig
from scripts.train_score_auxiliary_study import (
    design,
    prepare,
    run,
    scan_seed_ranges,
    train_arm,
    validate_auxiliary,
    write_new,
)


def test_frozen_design_budgets_matched_controls_and_disjoint_ranges():
    plan = design()
    assert len(plan["arms"]) == 6
    assert plan["updates"] == 520 and plan["warmup"] == 20
    assert plan["total_ppo_decisions"] == 6 * 128 * 128 * 520
    assert plan["total_auxiliary_matches"] == 6 * 100 * 128
    assert plan["evaluation"]["candidate"] == "seed1_score"
    assert plan["evaluation"]["full_matches"] == 2 * 4 * 2 * 4096 * 2
    for index in (1, 2):
        controls = [
            copy.deepcopy(plan["arms"][f"seed{index}_{kind}"])
            for kind in ("none", "win", "score")
        ]
        for value in controls:
            assert value["ppo"]["initial_variant"] == "point"
            assert value["ppo"]["score_temperature"] == 0.05
            assert value["auxiliary"]["interval"] == 5
            value["auxiliary"].pop("kind")
        assert controls[0] == controls[1] == controls[2]
    intervals = sorted(plan["seed_ranges"].values())
    assert all(left[1] < right[0] for left, right in itertools.pairwise(intervals))


def test_scan_checks_nested_seed_fields_and_reports_unreadable(tmp_path):
    (tmp_path / "valid.json").write_text(
        json.dumps(
            {
                "seed": 101,
                "other": {"seeds": [105, 202]},
                "ordinary_count": 103,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    (tmp_path / "legacy.json").write_text(json.dumps({"seed": 108}), encoding="utf-16")
    scan = scan_seed_ranges(tmp_path, {"range": [100, 110]})
    assert sorted(row["value"] for row in scan["overlaps"]) == [101, 105, 108]
    assert len(scan["unreadable"]) == 1
    assert scan["json_files_scanned"] == 3
    assert len(scan["seed_bearing_files"]) == 2


def test_auxiliary_accounting_detects_work_and_optimizer_corruption():
    c = AuxiliaryConfig(kind="score", matches=2, interval=5, minibatch=8, epochs=6)
    rows = [
        {
            "collection_work": {"decisions": 100, "rounds": 12, "matches": 2},
            "retained_rows": 13,
            "example_presentations": 78,
        }
    ]
    counts = {
        "cycles": 1,
        "decisions": 100,
        "rounds": 12,
        "matches": 2,
        "retained_rows": 13,
        "example_presentations": 78,
        "optimizer_steps": 12,
    }
    trainer = SimpleNamespace(
        config=c, counters=counts, ppo=SimpleNamespace(counters={"policy_updates": 5})
    )
    validate_auxiliary(trainer, rows)
    for key, value in counts.items():
        trainer.counters = counts | {key: value + 1}
        with pytest.raises(AssertionError):
            validate_auxiliary(trainer, rows)


def test_existing_study_and_started_receipt_never_restart(tmp_path):
    with pytest.raises(ValueError, match="Preserve"):
        prepare(tmp_path)
    write_new(tmp_path / "protocol.json", {"inputs": {}})
    write_new(tmp_path / "started.json", {"pid": 1})
    with pytest.raises(FileExistsError):
        run(tmp_path)
    assert not (tmp_path / "failed.json").exists()


@pytest.mark.upstream
@pytest.mark.skipif(not UPSTREAM_PATH.exists(), reason="Upstream not fetched")
def test_actual_runner_saved_stages_first_collection_and_accounting(
    tmp_path, record_property
):
    ppo = PublishedConfig(
        device="cpu",
        seed=105950001,
        initial_variant="point",
        envs=2,
        horizon=4,
        minibatch=8,
        epochs=1,
        critic_width=32,
    )
    aux = AuxiliaryConfig(
        kind="none",
        interval=5,
        matches=2,
        epochs=2,
        minibatch=16,
        deal_seed=105960001,
        action_seed=105970001,
        reservoir_seed=105980001,
        order_seed=105990001,
    )
    protocol = {
        "inputs": {},
        "updates": 25,
        "ppo_decisions_per_arm": 200,
        "auxiliary_matches_per_arm": 2,
    }
    write_new(tmp_path / "protocol.json", protocol)
    results = []
    for kind in ("none", "score"):
        spec = {"ppo": asdict(ppo), "auxiliary": asdict(aux) | {"kind": kind}}
        results.append(train_arm(tmp_path, "seed1_" + kind, spec, protocol))
        folder = tmp_path / ("seed1_" + kind)
        assert (folder / "initial/manifest.json").is_file()
        assert (folder / "update_00000020/manifest.json").is_file()
        assert (folder / "update_00000025/manifest.json").is_file()
        assert (folder / "collection_0001.pt").is_file()
        rows = [
            json.loads(line)
            for line in (folder / "metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(rows) == 25 and rows[-1]["auxiliary_counters"]["cycles"] == 1
        assert sum(row["auxiliary_phase"] is not None for row in rows) == 1
    assert results[0]["ppo_work"] == results[1]["ppo_work"]
    for group in ("ppo_work", "auxiliary_work"):
        record_property(
            group,
            json.dumps(
                {
                    key: sum(result[group][key] for result in results)
                    for key in ("decisions", "rounds", "matches")
                }
            ),
        )
