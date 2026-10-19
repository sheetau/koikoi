"""Fault injection against the actual first Point-study warmup artifact."""

import json
import shutil
from pathlib import Path

import pytest
import torch

from scripts.audit_point_initialization_checkpoint import audit

SOURCE = Path("runs/point_initialization_study")
SNAPSHOT = SOURCE / "seed1_wp_fixed/update_00000020.pt"
pytestmark = pytest.mark.skipif(
    not SNAPSHOT.exists(), reason="Actual Point-study artifact required"
)


@pytest.mark.parametrize(
    "corruption", ["adam", "episodes", "source", "metric", "temperature"]
)
def test_actual_warmup_corruption_rejected(tmp_path, corruption):
    torch.set_num_threads(2)
    shutil.copyfile(SOURCE / "protocol.json", tmp_path / "protocol.json")
    folder = tmp_path / "seed1_wp_fixed"
    folder.mkdir()
    shutil.copyfile(SOURCE / "seed1_wp_fixed/initial.pt", folder / "initial.pt")
    lines = (
        (SOURCE / "seed1_wp_fixed/metrics.jsonl")
        .read_bytes()
        .splitlines(keepends=True)[:20]
    )
    (folder / "metrics.jsonl").write_bytes(b"".join(lines))
    shutil.copyfile(SNAPSHOT, folder / SNAPSHOT.name)
    assert audit(tmp_path, "seed1_wp_fixed", 20)["status"] == "passed"
    payload = torch.load(folder / SNAPSHOT.name, map_location="cpu", weights_only=False)
    if corruption == "adam":
        next(iter(payload["optimizer"]["state"].values()))["step"] += 1
    elif corruption == "episodes":
        payload["env"][0, 111] += 1
    elif corruption == "source":
        payload["provenance"]["source_sha256"] = "0" * 64
    elif corruption == "temperature":
        payload["config"]["phase_temperatures"] = (0.2,) * 4
    else:
        row = json.loads(lines[-1])
        row["counters"]["decisions"] += 1
        lines[-1] = (json.dumps(row) + "\n").encode()
        (folder / "metrics.jsonl").write_bytes(b"".join(lines))
    torch.save(payload, folder / SNAPSHOT.name)
    with pytest.raises(AssertionError):
        audit(tmp_path, "seed1_wp_fixed", 20)
