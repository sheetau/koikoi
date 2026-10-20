"""Relocation changes artifact roots without altering numerical protocol."""

import json

import pytest

from scripts import finish_score_auxiliary_recovery as queue
from scripts.run_arbitration_study import sha


def test_relocation_binds_original_bytes_and_preserves_design(tmp_path, monkeypatch):
    artifact = tmp_path / "receipt.json"
    artifact.write_bytes(b'{"status":"complete"}\n')
    path = tmp_path / "protocol.json"
    before = {
        "pairs": 4096,
        "bootstrap": {"seed": 109310001, "replicates": 10000},
        "inputs": {str(artifact): sha(artifact)},
        "cells": {"fixed": "actor"},
    }
    path.write_text(json.dumps(before), encoding="utf-8")
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_bytes(b"# fixture\n")
    monkeypatch.setattr(
        queue, "relocation_bindings", lambda: {str(wrapper): sha(wrapper)}
    )
    queue.bind_relocation(path)
    after = json.loads(path.read_text(encoding="utf-8"))
    assert {k: v for k, v in after.items() if k != "inputs"} == {
        k: v for k, v in before.items() if k != "inputs"
    }
    assert after["inputs"] == before["inputs"] | {str(wrapper): sha(wrapper)}
    artifact.write_bytes(b"changed")
    with pytest.raises((AssertionError, ValueError)):
        queue.bind_relocation(path)


def test_jobs_keep_original_evaluation_and_audit():
    jobs = dict(queue.jobs())
    assert list(jobs) == ["training_audit", "prepare", "evaluate", "evaluation_audit"]
    assert jobs["training_audit"] == [
        "scripts.audit_score_auxiliary_training",
        "--root",
        str(queue.TRAINING),
    ]
    assert jobs["evaluate"] == [
        "scripts.evaluate_score_auxiliary",
        "run",
        "--protocol",
        str(queue.EVALUATION / "evaluation_protocol.json"),
    ]
    assert jobs["evaluation_audit"][0] == "scripts.audit_score_auxiliary_evaluation"
