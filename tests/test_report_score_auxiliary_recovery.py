"""Identity gates and preservation of fixed analysis in recovery reports."""

import json

import pytest

from scripts import export_score_auxiliary_results as exporter
from scripts import report_score_auxiliary_recovery as adapter
from scripts.run_arbitration_study import sha


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("bad_prefix", [False, True])
def test_recovery_prefix_identity(tmp_path, monkeypatch, bad_prefix):
    monkeypatch.setattr(adapter, "TRAINING", tmp_path)
    put(tmp_path / "recovery_protocol.json", {"inputs": {}})
    monkeypatch.setattr(
        adapter, "RECOVERY_SHA", sha(tmp_path / "recovery_protocol.json")
    )
    put(
        tmp_path / "repeated_prefix.json",
        {"status": "passed", "updates": 163 if bad_prefix else 164, "collections": 28},
    )
    put(
        tmp_path / "complete.json",
        {"repeated_prefix_sha256": sha(tmp_path / "repeated_prefix.json")},
    )
    if bad_prefix:
        with pytest.raises(ValueError, match="prefix"):
            adapter.recovery_inputs()
    else:
        assert str(tmp_path / "repeated_prefix.json") in adapter.recovery_inputs()
        put(tmp_path / "complete.json", {"repeated_prefix_sha256": "wrong"})
        with pytest.raises(ValueError, match="differs"):
            adapter.recovery_inputs()


def test_export_preserves_all_analysis_and_records_extra_work(tmp_path, monkeypatch):
    root, output = tmp_path / "study", tmp_path / "export"
    monkeypatch.setattr(adapter, "TRAINING", root)
    monkeypatch.setattr(adapter, "RESULTS", output)
    monkeypatch.setattr(adapter, "recovery_inputs", dict)
    put(
        root / "recovery_protocol.json",
        {"additional_work": {"ppo_decisions_before_failure": 2703360}},
    )
    put(root / "repeated_prefix.json", {"status": "passed"})
    summary = {
        "qualified": False,
        "primary": list(range(6)),
        "secondary": list(range(120)),
    }
    observed = []

    def load(evaluation, queue, audit):
        observed.append((exporter.TRAINING, exporter.QUEUE_SHA, audit))
        return summary, {"full_matches": 131072}, {"fixed": True}, {}

    monkeypatch.setattr(exporter, "load_completed", load)
    monkeypatch.setattr(exporter, "render", lambda *args: "Fixed original results\n")
    monkeypatch.setattr(exporter, "TRAINING", exporter.TRAINING)
    monkeypatch.setattr(exporter, "QUEUE_SHA", exporter.QUEUE_SHA)
    adapter.export("pinned-audit")
    result = json.loads((output / "evidence.json").read_text(encoding="utf-8"))
    assert result["summary"] == summary
    assert result["recovery_overhead"]["ppo_decisions_before_failure"] == 2703360
    assert result["new_matches"] == result["new_neural_calls"] == 0
    assert result["outputs"]["results.rst"] == sha(output / "results.rst")
    assert observed == [(root, adapter.QUEUE_SHA, "pinned-audit")]
    assert "additional to the retained" in (output / "results.rst").read_text(
        encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Preserve"):
        adapter.export("pinned-audit")
