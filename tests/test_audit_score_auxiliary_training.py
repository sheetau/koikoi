"""Exercise the new auditor on closed artifacts; no training or policy rollout."""

import copy
import json
from pathlib import Path

import pytest
import torch

from koikoi.agents import solved_koi_action
from koikoi.layout import KOI, PHASE, PLAYER
from koikoi.reference import ReferenceEnv
from koikoi.upstream import reconstruct
from scripts.audit_score_auxiliary_collection import auxiliary_steps, check_collection
from scripts.audit_score_auxiliary_training import (
    COUNTERS,
    audit,
    check_payloads,
    checkpoint,
    load,
)

CAPABILITY = Path("runs/score_auxiliary_training_capability")


@pytest.fixture(scope="module")
def closed():
    torch.set_num_threads(2)
    return load(CAPABILITY / "score/cycle2/auxiliary.pt")


def test_collection_against_separate_python_reference(closed, record_property):
    record, config = closed["last_collection"], closed["config"]
    checked = check_collection(record, config, 2, return_features=True)
    assert checked["retained_rows"] == 958
    assert auxiliary_steps(record, config, 2)["optimizer_steps"] == 48
    prefix_decisions = 0
    for phase in range(4):
        row = int(torch.where(record["phases"] == phase)[0][0])
        lane, seat, _ = record["identities"][row].tolist()
        ref = ReferenceEnv(record["deal_seed"] + lane, "paper8")
        for action in record["actions"][: int(record["selected_steps"][row]), lane]:
            assert ref.step(int(action))[0] != 99
            prefix_decisions += 1
        assert int(ref.state[PHASE]) == phase and int(ref.state[PLAYER]) == seat
        original = reconstruct(ref)
        width = 50 if phase == KOI else 48
        assert torch.equal(checked["features"][row, :, :width], original.feature_tensor)
        legal = torch.from_numpy(ref.legal()).bool()
        solved = solved_koi_action(ref)
        if solved is not None:
            legal &= torch.arange(50) == solved
        assert torch.equal(record["legal"][row], legal)
    for key, value in record["work"].items():
        record_property("native_replay_" + key, value)
    record_property("python_reference_prefix_decisions", prefix_decisions)
    record_property("new_policy_matches", 0)


@pytest.mark.parametrize("corruption", ["step", "features", "margin", "work"])
def test_collection_corruption_rejected(closed, corruption):
    record = copy.deepcopy(closed["last_collection"])
    if corruption == "step":
        record["selected_steps"][0] = -1
    elif corruption == "features":
        record["features_sha256"] = "0" * 64
    elif corruption == "margin":
        record["margins"][0] += 2
    else:
        record["work"]["matches"] += 1
    with pytest.raises(AssertionError):
        check_collection(record, closed["config"], 2)


def capability_rows(kind):
    folder = CAPABILITY / kind
    capability = json.loads((folder / "capability.json").read_text(encoding="utf-8"))
    rows = []
    for stage, metrics in zip(
        ("warmup", "cycle1", "cycle2"), capability["metrics"], strict=True
    ):
        rows.append(metrics | load(folder / stage / "ppo.pt")["counters"])
    return rows, {
        "ppo": capability["ppo_config"],
        "auxiliary": capability["auxiliary_config"],
    }


@pytest.mark.parametrize("kind", ["none", "win", "score"])
def test_all_closed_checkpoint_stages(kind):
    rows, specification = capability_rows(kind)
    counts = dict.fromkeys(COUNTERS, 0)
    bindings = {}
    for update, stage in enumerate(("initial", "warmup", "cycle1", "cycle2")):
        if update >= 2:
            aux = load(CAPABILITY / kind / stage / "auxiliary.pt")
            collection = aux["last_collection"]
            counts["cycles"] += 1
            counts["retained_rows"] += len(collection["targets"])
            for key, value in (
                collection["work"]
                | auxiliary_steps(collection, specification["auxiliary"], update - 1)
            ).items():
                counts[key] += value
        _, checked = checkpoint(
            CAPABILITY / kind / stage, specification, rows[:update], counts, bindings
        )
        assert checked["auxiliary_counters"] == counts
    assert len(bindings) == 15


def test_optimizer_and_identity_corruption_rejected(closed):
    rows, specification = capability_rows("score")
    original = load(CAPABILITY / "score/cycle2/ppo.pt")
    for corruption in (
        "ppo_step",
        "auxiliary_step",
        "negative_moment",
        "shape",
        "source",
    ):
        ppo, aux = copy.deepcopy(original), copy.deepcopy(closed)
        if corruption == "ppo_step":
            ppo["optimizer"]["state"][0]["step"] += 1
        elif corruption == "auxiliary_step":
            aux["optimizer"]["state"][0]["step"] += 1
        elif corruption == "negative_moment":
            aux["optimizer"]["state"][0]["exp_avg_sq"].view(-1)[0] = -1
        elif corruption == "shape":
            aux["optimizer"]["state"][0]["exp_avg"] = torch.zeros(1)
        else:
            ppo["provenance"]["source_sha256"] = "0" * 64
        with pytest.raises(AssertionError):
            check_payloads(ppo, aux, specification, rows, closed["counters"])


def test_incomplete_study_cannot_authorize_evaluation(tmp_path):
    protocol = Path("runs/score_auxiliary_study/protocol.json")
    (tmp_path / "protocol.json").write_bytes(protocol.read_bytes())
    with pytest.raises(FileNotFoundError):
        audit(tmp_path)
    assert not (tmp_path / "training_audit.json").exists()
