"""Independent saved-state checks; final result checks require a closed pipeline."""

import copy
from pathlib import Path

import pytest

from scripts import audit_seat_observer_optimizer as optimizer
from scripts import audit_seat_observer_study as study


@pytest.fixture(scope="module")
def pipeline():
    receipt = study.read(Path("runs/seat_observer_pipeline_launch.json"))
    base = Path(receipt["base"])
    assert study.sha(base / "protocol.json") == receipt["protocol_sha256"]
    return base


@pytest.fixture(scope="module")
def receipts(pipeline, tmp_path_factory):
    folder = tmp_path_factory.mktemp("seat_optimizer")
    for arm in ("public", "masked"):
        case = f"parent1_{arm}"
        optimizer.run(pipeline, case, folder / f"{case}.json")
    print("OPTIMIZER_RECEIPTS=" + str(folder), flush=True)
    return folder


def test_actual_optimizer_states(pipeline, receipts):
    for arm in ("public", "masked"):
        report = study.read(receipts / f"parent1_{arm}.json")
        assert report["status"] == "passed"
        assert sum(b["parameter_tensors"] for b in report["blocks"].values()) == 96
        for name, block in report["blocks"].items():
            assert block["expected_and_actual_adam_steps"] == (
                4 if name == "value" else 2
            )
        assert (report["seat_column_l1"] > 0) == (arm == "public")


def test_rejects_changed_actual_adam(pipeline, tmp_path, monkeypatch):
    load = optimizer.torch.load

    def bad_state(path, *args, **kwargs):
        payload = load(path, *args, **kwargs)
        if Path(path) == pipeline / "parent1_public/final.pt":
            payload = copy.deepcopy(payload)
            next(iter(payload["optimizer"]["state"].values()))["step"] += 1
        return payload

    monkeypatch.setattr(optimizer.torch, "load", bad_state)
    output = tmp_path / "bad_optimizer.json"
    with pytest.raises(AssertionError):
        optimizer.run(pipeline, "parent1_public", output)
    assert not output.exists()


def test_refuses_nonterminal_results(tmp_path):
    base = tmp_path / "unfinished"
    base.mkdir()
    (base / "process.json").write_text('{"status":"running"}\n', encoding="utf-8")
    output = tmp_path / "premature.json"
    with pytest.raises(ValueError, match="not terminal"):
        study.audit(base, output, allow_smoke=True)
    assert not output.exists()


def test_closed_pipeline_statistics_and_tampering(
    pipeline, receipts, tmp_path, monkeypatch
):
    # Run only after the actual queued pipeline has completed; never synthesize
    # a successful process state to force this evidence-dependent test to pass.
    output = tmp_path / "audit.json"
    study.audit(pipeline, output, allow_smoke=True, optimizer_dir=receipts)
    report = study.read(output)
    assert report["replicated_seat_gain"] is None
    assert report["training_totals"] == {
        "decisions": 256,
        "rounds": 16,
        "matches": 0,
        "value_rows": 512,
    }
    assert report["evaluation_matches"] == 32
    assert report["shared_unique_deal_streams"] == 2
    assert len(report["recomputed"]) == 20
    original_read = study.read

    def changed(path):
        result = original_read(path)
        if Path(path) == pipeline / "analysis.json":
            result["outcomes"]["parent1/rl-point-assisted"]["win_score"] += 0.1
        if Path(path) == pipeline / "parent1/rl-point-assisted.json":
            result["win_score"] += 0.1
        return result

    monkeypatch.setattr(study, "read", changed)
    with pytest.raises(AssertionError):
        study.audit(
            pipeline,
            tmp_path / "bad_outcome.json",
            allow_smoke=True,
            optimizer_dir=receipts,
        )
    print("CLOSED_PIPELINE_AUDIT=" + str(output), flush=True)
