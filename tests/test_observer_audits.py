"""Audit archived actual pipeline outputs; reject incomplete/misreported data."""

import copy
import zipfile
from pathlib import Path

import pytest

from scripts import audit_observer_optimizer as optimizer
from scripts import audit_observer_selfplay_study as study

EVIDENCE = Path("runs/observer_selfplay_pipeline_evidence")


@pytest.fixture(scope="module")
def archived():
    receipt = study.read(EVIDENCE / "audit.json")
    archive = EVIDENCE / "pipeline.zip"
    assert study.sha(archive) == receipt["archive_sha256"]
    original = Path(receipt["original_path"])
    if not original.exists():
        with zipfile.ZipFile(archive) as bundle:
            for name in bundle.namelist():
                assert (original / name).resolve().is_relative_to(original.resolve())
            bundle.extractall(original)
    assert study.sha(original / "analysis.json") == receipt["analysis_sha256"]
    return original


@pytest.fixture(scope="module")
def receipts(archived, tmp_path_factory):
    folder = tmp_path_factory.mktemp("observer_optimizer")
    for arm in ("observer", "signed"):
        case = f"parent1_{arm}"
        optimizer.run(archived, case, folder / f"{case}.json")
    return folder


def test_actual_saved_states_and_joint_bootstrap(archived, receipts, tmp_path):
    output = tmp_path / "audit.json"
    study.audit(archived, output, allow_smoke=True, optimizer_dir=receipts)
    result = study.read(output)
    assert result["status"] == "passed"
    assert result["replicated_observer_gain"] is None
    assert result["training_totals"] == {
        "decisions": 256,
        "rounds": 12,
        "matches": 0,
        "value_rows": 512,
    }
    assert result["evaluation_matches"] == 32 and result["evaluation_decisions"] == 4269
    assert result["shared_unique_deal_streams"] == 2
    assert len(result["recomputed"]) == 20
    assert result["matched_warmup"]["parent1"]["status"] == "passed"
    for arm in ("observer", "signed"):
        blocks = study.read(receipts / f"parent1_{arm}.json")["blocks"]
        assert blocks["value"]["expected_and_actual_adam_steps"] == 4
        assert blocks["koi"]["expected_and_actual_adam_steps"] == 2
    with pytest.raises(ValueError, match="fresh"):
        study.audit(archived, output, allow_smoke=True, optimizer_dir=receipts)
    with pytest.raises(AssertionError):
        study.audit(archived, tmp_path / "not_full.json", optimizer_dir=receipts)


def test_refuses_live_study_without_writing(archived, tmp_path, monkeypatch):
    original_read = study.read

    def changed(path):
        result = original_read(path)
        if Path(path) == archived / "process.json":
            result["status"] = "running"
        return result

    monkeypatch.setattr(study, "read", changed)
    output = tmp_path / "premature.json"
    with pytest.raises(ValueError, match="not terminal"):
        study.audit(archived, output, allow_smoke=True)
    assert not output.exists()


def test_rejects_wrong_value_count_before_accepting_receipt(
    archived, receipts, tmp_path, monkeypatch
):
    original_read = study.read

    def changed(path):
        result = original_read(path)
        if Path(path) == archived / "analysis.json":
            result["training"]["parent1_observer"]["counters"]["value_rows"] += 1
        if Path(path) == archived / "parent1_observer/training_audit.json":
            result["counters"]["value_rows"] += 1
        return result

    monkeypatch.setattr(study, "read", changed)
    with pytest.raises(AssertionError):
        study.audit(
            archived,
            tmp_path / "bad_count.json",
            allow_smoke=True,
            optimizer_dir=receipts,
        )


def test_rejects_misreported_outcome_and_changed_adam(
    archived, receipts, tmp_path, monkeypatch
):
    original_read = study.read

    def changed(path):
        result = original_read(path)
        if Path(path) == archived / "analysis.json":
            result["outcomes"]["parent1/rl-point-assisted"]["win_score"] += 0.1
        if Path(path) == archived / "parent1/rl-point-assisted.json":
            result["win_score"] += 0.1
        return result

    with monkeypatch.context() as context:
        context.setattr(study, "read", changed)
        with pytest.raises(AssertionError):
            study.audit(
                archived,
                tmp_path / "bad_outcome.json",
                allow_smoke=True,
                optimizer_dir=receipts,
            )
    load = optimizer.torch.load

    def bad_state(path, *args, **kwargs):
        payload = load(path, *args, **kwargs)
        if Path(path) == archived / "parent1_observer/final.pt":
            payload = copy.deepcopy(payload)
            next(iter(payload["optimizer"]["state"].values()))["step"] += 1
        return payload

    monkeypatch.setattr(optimizer.torch, "load", bad_state)
    with pytest.raises(AssertionError):
        optimizer.run(archived, "parent1_observer", tmp_path / "bad_optimizer.json")
