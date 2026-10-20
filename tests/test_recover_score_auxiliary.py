"""Recovery preserves originals and rejects altered numerical prefixes."""

import json

import pytest
import torch

from scripts.recover_score_auxiliary import copy_completed, verify_repeated_prefix


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_copy_changes_only_receipt_endpoint(tmp_path):
    source, dest = tmp_path / "original", tmp_path / "recovery"
    write(
        source / "complete.json",
        {"status": "complete", "endpoint": str(source / "endpoint"), "fixed": 42},
    )
    (source / "endpoint").mkdir()
    (source / "endpoint" / "data.pt").write_bytes(b"\x00\xfffixture")
    before = {
        str(p.relative_to(source)): p.read_bytes()
        for p in source.rglob("*")
        if p.is_file()
    }
    result = copy_completed(source, dest)
    assert result["receipt"] == {
        "status": "complete",
        "endpoint": str(dest / "endpoint"),
        "fixed": 42,
    }
    assert (dest / "endpoint/data.pt").read_bytes() == b"\x00\xfffixture"
    assert before == {
        str(p.relative_to(source)): p.read_bytes()
        for p in source.rglob("*")
        if p.is_file()
    }
    with pytest.raises(FileExistsError):
        copy_completed(source, dest)


@pytest.mark.parametrize(
    "corruption", [None, "metric", "rng", "auxiliary", "collection"]
)
def test_prefix_rejects_corrupt_state(tmp_path, corruption):
    source, dest = tmp_path / "old", tmp_path / "new"
    for root in (source, dest):
        row = {
            "seconds": 1 if root == source else 99,
            "ppo": {"decisions": 128, "rounds": 9, "matches": 2, "loss": 0.25},
            "auxiliary_counters": {"matches": 2},
        }
        if root == dest and corruption == "metric":
            row["ppo"]["loss"] = 0.5
        write(root / "metrics.jsonl", row)
        write(root / "initial/manifest.json", {})
        torch.save(
            {
                "rng": torch.tensor([2 if root == dest and corruption == "rng" else 1]),
                "provenance": {"source_sha256": "same", "git": str(root)},
            },
            root / "initial/ppo.pt",
        )
        torch.save(
            {
                "moment": torch.tensor(
                    [2 if root == dest and corruption == "auxiliary" else 1]
                )
            },
            root / "initial/auxiliary.pt",
        )
        torch.save(
            {"actions": [2 if root == dest and corruption == "collection" else 1]},
            root / "collection_0001.pt",
        )
    if corruption:
        with pytest.raises(AssertionError):
            verify_repeated_prefix(source, dest)
    else:
        result = verify_repeated_prefix(source, dest)
        assert result["updates"] == result["collections"] == 1
        assert result["stages"] == ["initial"]
