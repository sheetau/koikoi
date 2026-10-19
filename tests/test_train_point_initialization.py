import json

import pytest

from scripts import train_point_initialization as study


def protocol(root, inputs=None):
    root.mkdir()
    study.write_new(
        root / "protocol.json",
        {
            "inputs": inputs or {},
            "configs": {"first": {}, "second": {}, "third": {}},
            "total_decisions": 3,
        },
    )


def test_failure_preserves_completed_arm_and_stops_queue(tmp_path, monkeypatch):
    root = tmp_path / "study"
    protocol(root)
    called = []

    def fake_arm(root, name, config, plan):
        called.append(name)
        if name == "second":
            raise RuntimeError("injected failure")
        (root / name).mkdir()
        study.write_new(root / name / "complete.json", {"status": "complete"})
        return {"counters": {"decisions": 1, "rounds": 1, "matches": 1, "updates": 1}}

    monkeypatch.setattr(study, "train_arm", fake_arm)
    with pytest.raises(RuntimeError, match="injected"):
        study.run(root)
    assert called == ["first", "second"]
    assert (root / "first/complete.json").exists()
    assert (
        "injected"
        in json.loads((root / "failed.json").read_text(encoding="utf-8"))["error"]
    )
    assert not (root / "complete.json").exists()
    with pytest.raises(ValueError, match="already started"):
        study.run(root)
    assert called == ["first", "second"]


def test_input_drift_refused_before_start_receipt(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.write_bytes(b"original")
    root = tmp_path / "study"
    protocol(root, {str(source): study.sha(source)})
    source.write_bytes(b"changed")
    monkeypatch.setattr(study, "train_arm", lambda *a: pytest.fail("must not train"))
    with pytest.raises(ValueError, match="Changed frozen input"):
        study.run(root)
    assert not (root / "started.json").exists()


def test_prepare_preserves_existing_directory(tmp_path):
    root = tmp_path / "existing"
    root.mkdir()
    sentinel = root / "sentinel"
    sentinel.write_bytes(b"keep")
    with pytest.raises(ValueError, match="Preserve existing"):
        study.prepare(root)
    assert sentinel.read_bytes() == b"keep"
