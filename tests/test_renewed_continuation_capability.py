"""Capability execution guards and full saved-state equality checks."""

import copy

import numpy as np
import pytest
import torch

from scripts.check_renewed_continuation import (
    assert_same_endpoint,
    require_completed_study,
)
from scripts.run_arbitration_study import sha, write


def test_requires_finished_audited_unchanged_comparison(tmp_path):
    write(tmp_path / "process.json", {"status": "running"})
    with pytest.raises(ValueError, match="complete multi-round"):
        require_completed_study(tmp_path)
    write(tmp_path / "process.json", {"status": "complete"})
    with pytest.raises(ValueError, match="independent multi-round"):
        require_completed_study(tmp_path)
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("original\n", encoding="utf-8", newline="\n")
    write(
        tmp_path / "independent_audit.json",
        {"status": "passed", "inputs": {str(evidence): sha(evidence)}},
    )
    result = require_completed_study(tmp_path)
    assert result[str(evidence)] == sha(evidence)
    assert result[str(tmp_path / "independent_audit.json")] == sha(
        tmp_path / "independent_audit.json"
    )
    evidence.write_text("changed\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="Changed comparison input"):
        require_completed_study(tmp_path)


def test_saved_state_comparison_includes_optimizer_and_all_rngs():
    payload = {
        key: {"fixture": torch.arange(4)}
        for key in (
            "actor",
            "value",
            "optimizer",
            "bank",
            "root_sampler",
            "memory",
            "counters",
            "torch_rng",
            "cuda_rng",
            "python_rng",
            "prefix_actor",
            "opponents",
            "runtime",
        )
    }
    payload["numpy_rng"] = np.random.RandomState(7).get_state()
    duplicate = copy.deepcopy(payload)
    assert_same_endpoint(payload, duplicate)
    for key in (
        "optimizer",
        "root_sampler",
        "torch_rng",
        "cuda_rng",
        "python_rng",
        "prefix_actor",
        "opponents",
    ):
        damaged = copy.deepcopy(payload)
        damaged[key]["fixture"][0] += 1
        with pytest.raises(AssertionError):
            assert_same_endpoint(payload, damaged)
    damaged = copy.deepcopy(payload)
    damaged["numpy_rng"][1][0] ^= 1
    with pytest.raises(AssertionError):
        assert_same_endpoint(payload, damaged)
