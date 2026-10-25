"""CPU engineering checks; training here is not strength evidence."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from koikoi.public_stock_residual import load_residual
from scripts import export_public_stock_adaptation as exporter
from scripts.continue_public_stock_residual import initialise, train
from scripts.export_public_stock_results import sha

CHECKPOINT = Path("runs/public_stock_residual_v2/preview1_choice1_stakes0_parent0/policy_00128.pt")


@pytest.fixture
def args(tmp_path):
    if not CHECKPOINT.exists():
        pytest.skip("Retained research checkpoint is unavailable")
    torch.set_num_threads(1)
    return SimpleNamespace(
        checkpoint=str(CHECKPOINT), device="cpu", parent=0,
        profile="preview1_choice1_stakes0", out=str(tmp_path / "smoke"),
        seed=1107, deal_seed=1180000000, updates=2, envs=2, horizon=64,
        epochs=1, minibatch=128, save_every=2,
        lr=1e-4, kl=0.02, lam=0.99, entropy=0.01,
    )


def test_retained_weights_are_exact_and_optimizer_is_fresh(args):
    model, payload, optimizer, lineage = initialise(args)
    for name, value in model.correction_state().items():
        torch.testing.assert_close(value, payload["model"][name], rtol=0, atol=0)
    assert not optimizer.state
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in model.parameters() if p.requires_grad}
    assert all(not p.requires_grad and id(p) not in optimized for p in model.teacher.parameters())
    assert lineage["checkpoint_sha256"] == sha(CHECKPOINT)
    assert lineage["prior_counters"]["decisions"] == 4194304


@pytest.mark.parametrize("field,value", [("parent", 1), ("profile", "preview1_choice1_stakes1")])
def test_wrong_parent_or_rule_cannot_be_continued(args, field, value):
    setattr(args, field, value)
    with pytest.raises(ValueError, match="Wrong retained"):
        initialise(args)


def test_unbound_checkpoint_is_rejected_before_loading(args, tmp_path):
    bad = tmp_path / "unbound.pt"
    bad.write_bytes(b"not a retained checkpoint")
    args.checkpoint = str(bad)
    with pytest.raises(ValueError, match="retained V3"):
        initialise(args)


def test_cpu_training_preserves_lineage_and_separates_additional_work(args, monkeypatch):
    before = sha(CHECKPOINT)
    train(args)
    path = Path(args.out) / "policy_add_00002.pt"
    model, payload = load_residual(path, "cpu")
    initial = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    assert sha(CHECKPOINT) == before
    assert payload["training_method"] == "warm_start_fresh_adam_v1"
    assert payload["lineage"]["checkpoint_sha256"] == before
    assert payload["counters"]["decisions"] == 256
    assert payload["counters"]["updates"] == 2
    assert payload["cumulative_counters"]["decisions"] == 4194560
    assert payload["cumulative_counters"]["updates"] == 130
    assert payload["cumulative_counters"] == {
        k: initial["counters"][k] + v for k, v in payload["counters"].items()
    }
    assert not torch.equal(payload["model"]["actor.weight"], initial["model"]["actor.weight"])
    assert all(torch.isfinite(t).all() for t in model.correction_state().values())
    for name, digest in payload["bindings"].items():
        assert sha(name) == digest
    # Exercise the production audit on this explicitly smaller CPU engineering run.
    # The launch/qualification contracts are not relaxed for real studies.
    label = f"{args.profile}_parent0"
    study = {"systems": {label: {
        "profile": args.profile, "parent": 0,
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": before,
    }}, "sources": payload["bindings"]}
    monkeypatch.setattr(exporter, "training_config", lambda *unused: vars(args))
    audit = exporter.audit_training(Path(args.out).parent, study, label, {})
    assert audit["status"] == "passed" and audit["teacher_unchanged"]
    assert audit["additional"]["decisions"] == 256
    payload["cumulative_counters"]["decisions"] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="conflated"):
        exporter.audit_training(Path(args.out).parent, study, label, {})
    payload["cumulative_counters"]["decisions"] -= 1
    payload["parent"]["config"]["score_temperature"] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="frozen teacher"):
        exporter.audit_training(Path(args.out).parent, study, label, {})
    with pytest.raises(FileExistsError):
        train(args)
