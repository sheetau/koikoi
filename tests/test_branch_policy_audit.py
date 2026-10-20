import json
from dataclasses import asdict

import pytest
import torch

from koikoi.train import provenance
from scripts.audit_branch_policy_study import Replay, training
from scripts.run_arbitration_study import PARENTS
from scripts.train_branch_policy_gradient import Config, Trainer, sha, write


def test_reduced_actual_training_audit_and_corrupted_transition(tmp_path):
    config = Config("shared_actions", 110050001, matches=4, device="cpu")
    trainer = Trainer(PARENTS[0][0], config)
    metrics, data = trainer.update()
    directory = tmp_path / "case"
    directory.mkdir()
    torch.save(data, directory / "evidence_0001.pt")
    trainer.save(directory / "update_0128.pt")
    (directory / "metrics.jsonl").write_text(
        json.dumps(metrics) + "\n", encoding="utf-8", newline="\n"
    )
    write(
        tmp_path / "protocol.json",
        {
            "source": provenance(),
            "bindings": {},
            "updates_per_arm": 1,
            "parents": [{"path": PARENTS[0][0], "sha256": PARENTS[0][1]}],
            "cases": {"case": {"config": asdict(config)}},
        },
    )
    write(
        tmp_path / "complete.json",
        {
            "status": "complete",
            "protocol_sha256": sha(tmp_path / "protocol.json"),
            "cases": {
                "case": {
                    "endpoint_sha256": sha(directory / "update_0128.pt"),
                    "metrics_sha256": sha(directory / "metrics.jsonl"),
                    "evidence": {
                        "evidence_0001.pt": sha(directory / "evidence_0001.pt")
                    },
                    "counters": trainer.counters,
                }
            },
        },
    )
    training(tmp_path)
    receipt = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "passed"
    assert receipt["replayed_work"]["completions"] == (
        4 + data["branch"]["work"]["completions"]
    )
    with pytest.raises(ValueError, match="already"):
        training(tmp_path)
    corrupted = data["source"]
    corrupted["final"][0, 106] += 2
    with pytest.raises(AssertionError):
        Replay()(corrupted["initial"], corrupted, source=True)
