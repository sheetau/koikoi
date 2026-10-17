"""Timing safety and independent accounting; no real benchmark is run here."""

import json

import numpy as np
import pytest

from koikoi.agents import EndgameAgent
from scripts import audit_final_latency as audit
from scripts import benchmark_final_latency as run


def corpus():
    states = np.zeros((1400, 208), dtype=np.int32)
    states[:, 99] = np.arange(1400) % 4
    sources = np.ones(1400, dtype=np.int64)
    sources[:40] = -1
    guards = np.full(1400, -1, dtype=np.int64)
    guards[-64:] = 48
    states[-64:, 99] = 3
    masks = np.zeros((1400, 50), dtype=bool)
    masks[:, 0] = True
    masks[-64:, 48] = True
    return {"states": states, "sources": sources, "guards": guards, "masks": masks}


def test_fixed_sampling_excludes_selected_diagnostic_and_hidden_variant_states():
    d = corpus()
    actual = run.sample_groups(d["states"], d["sources"], d["guards"])
    assert actual == audit.groups_from_records(d["states"], d["sources"], d["guards"])
    assert {name: len(rows) for name, rows in actual.items()} == {
        "mixed": 1024,
        **{f"unguarded_phase{i}": 64 for i in range(4)},
        "guarded": 32,
    }
    assert all(
        len(set(rows)) == len(rows) and min(rows) >= 40 for rows in actual.values()
    )


def test_percentiles_use_raw_call_times_in_milliseconds():
    expected = {
        "calls": 4,
        "mean_ms": 2.5,
        "median_ms": 2.5,
        "p95_ms": 3.85,
        "p99_ms": 3.97,
        "max_ms": 4,
    }
    values = np.array([4, 1, 3, 2]) * 1000000
    assert run.statistics(values) == pytest.approx(expected)
    assert audit.statistics(values) == pytest.approx(expected)


@pytest.mark.parametrize(
    "fault",
    [None, "time", "action_drift", "illegal", "guard", "indices", "missing_repeat"],
)
def test_independent_audit_rejects_corrupt_timing_or_action_records(fault):
    d = corpus()
    groups = run.sample_groups(d["states"], d["sources"], d["guards"])
    indices = sorted({i for rows in groups.values() for i in rows})
    p = {"groups": groups, "indices": indices}
    expected = np.maximum(d["guards"][indices], 0)[None].repeat(4, axis=0)
    raw = {
        "indices": np.array(indices),
        "expected_actions": expected,
        "actions": expected[:, None].repeat(8, axis=1),
        "elapsed_ns": np.full((4, 8, len(indices)), 1000, dtype=np.int64),
    }
    if fault == "time":
        raw["elapsed_ns"][0, 0, 0] = 0
    elif fault == "action_drift":
        raw["actions"][0, 1, 0] = 1
    elif fault == "illegal":
        raw["expected_actions"][0, 0] = raw["actions"][0, :, 0] = 1
    elif fault == "guard":
        raw["expected_actions"][0, -1] = raw["actions"][0, :, -1] = 0
    elif fault == "indices":
        raw["indices"][0] -= 1
    elif fault == "missing_repeat":
        raw["elapsed_ns"] = raw["elapsed_ns"][:, :7]
    if fault is None:
        assert audit.verify_arrays(p, d, raw) == indices
    else:
        with pytest.raises(AssertionError):
            audit.verify_arrays(p, d, raw)


def test_live_final_study_blocks_before_loading_agents_or_creating_timing_files(
    tmp_path, monkeypatch
):
    final = tmp_path / "final"
    final.mkdir()
    (final / "process.json").write_text(
        json.dumps({"status": "running", "jobs": {}}), encoding="utf-8"
    )
    monkeypatch.setattr(run.final, "BASE", final)
    monkeypatch.setattr(run, "BASE", tmp_path / "timing")
    monkeypatch.setattr(
        run,
        "make_agent",
        lambda *a, **k: pytest.fail("Loaded an agent before the completion gate"),
    )
    with pytest.raises(RuntimeError, match="original final evaluation"):
        run.run()
    assert not run.BASE.exists()


def test_completed_final_study_still_blocks_if_project_workload_is_live(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(run.final, "BASE", tmp_path)
    jobs = {str(i): {"status": "complete", "returncode": 0} for i in range(4)}
    (tmp_path / "process.json").write_text(
        json.dumps(
            {
                "status": "complete_awaiting_audit",
                "jobs": jobs,
                "protocol_sha256": run.FINAL_SHA,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    checked = {
        "status": "passed",
        "protocol_sha256": run.FINAL_SHA,
        "auditor_sha256": run.sha(run.final.AUDITOR),
        "summary_sha256": run.sha(tmp_path / "summary.json"),
    }
    (tmp_path / "independent_audit.json").write_text(
        json.dumps(checked), encoding="utf-8"
    )
    monkeypatch.setattr(
        run, "project_workers", lambda: [{"pid": 123, "command": "another worker"}]
    )
    with pytest.raises(RuntimeError, match="workloads still active"):
        run.terminal_gate()


def test_deployment_wrapper_matches_final_evaluator_guard_layers(monkeypatch):
    model = object()
    assisted = EndgameAgent(object())
    monkeypatch.setattr(
        run,
        "make_agent",
        lambda spec, *a, **k: assisted if spec.startswith("rl-") else model,
    )
    assert run.deployment_agent("checkpoint.pt").agent is model
    assert run.deployment_agent("rl-wp-assisted") is assisted
    assert run.deployment_agent("rl-point-assisted") is assisted
