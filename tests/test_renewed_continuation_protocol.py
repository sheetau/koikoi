"""Renewal protocol guards,independent statistics and saved-schedule audits."""

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import run_renewed_continuation_study as runner
from scripts.audit_final_round_training_study import tensor_equal
from scripts.audit_renewed_continuation_study import (
    audit,
    audit_sampler,
    independent_contrasts,
    independent_decisions,
    interval,
    verify_identity,
    verify_row,
)
from scripts.renewed_continuation_analysis import (
    analyse,
    bootstrap_intervals,
    decisions,
)
from scripts.renewed_root_sampler import RenewedRootSampler
from scripts.run_arbitration_study import sha, write


def test_readiness_guards_and_finite_allocations(tmp_path, monkeypatch):
    runner.validate_allocations()
    monkeypatch.setattr(runner, "FIXED", tmp_path)
    write(tmp_path / "process.json", {"status": "running"})
    with pytest.raises(ValueError, match="complete multi-round"):
        runner.make_plan()
    cap = tmp_path / "capability"
    with pytest.raises(ValueError, match="Complete renewed"):
        runner.require_capability(cap)
    write(cap / "process.json", {"status": "complete"})
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("fixture\n", encoding="utf-8")
    receipt = {
        "status": "passed",
        "matched_initial_and_warmup_exact": True,
        "logical_conditional_deal_constructions": 768,
        "replayed_conditional_deal_constructions": 256,
        "distinct_chance_keys_shared_between_arms": 384,
        "arms": {
            mode: {"exact_resume": True, "counters": {"fresh_conditional_deals": 384}}
            for mode in ("public", "masked")
        },
        "inputs": {str(evidence): sha(evidence)},
    }
    write(cap / "audit.json", receipt)
    result = runner.require_capability(cap)
    assert result[str(cap / "audit.json")] == sha(cap / "audit.json")
    damaged = copy.deepcopy(receipt)
    damaged["arms"]["masked"]["exact_resume"] = False
    write(cap / "audit.json", damaged)
    with pytest.raises(AssertionError):
        runner.require_capability(cap)
    write(cap / "audit.json", receipt)
    evidence.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Changed capability input"):
        runner.require_capability(cap)


def test_auditor_checks_real_fixed_rng_and_rejects_schedule_tamper():
    # Real saved fixed-bank RNG states;synthetic renewed metadata only. No fresh
    # deals,simulator transitions,policy inference or training occur in this test.
    folder = Path("runs/public_continuation_study/parent1_public")
    provenance = {
        key: sha(path)
        for key, path in (
            ("renewed_continuation_source_sha256", "scripts/renewed_continuation.py"),
            ("root_sampler_source_sha256", "scripts/renewed_root_sampler.py"),
            ("chance_constructor_source_sha256", "scripts/continuation_chance.py"),
        )
    }
    for name, updates in (("initial.pt", 0), ("update_00000020.pt", 20)):
        fixed = torch.load(folder / name, map_location="cpu", weights_only=False)
        sampler = RenewedRootSampler(
            *fixed["bank"],
            fixed["memory"],
            sampling_seed=92110002,
            chance_seed=94110001,
            max_draws=66560,
        )
        state = sampler.state_dict()
        state.update(
            cursor=updates * 128,
            batch_lengths=[128] * updates,
            rng=fixed["bank_rng"].clone(),
        )
        sampler.load_state_dict(state)
        payload = {
            "bank": fixed["bank"],
            "memory": fixed["memory"],
            "root_sampler": sampler.state_dict(),
            "counters": {
                "updates": updates,
                "fresh_conditional_deals": updates * 128,
                "conditional_tail_completions": updates * 128,
            },
            "provenance": provenance,
        }
        audit_sampler(payload, fixed, 1, updates)
        for key, value in (
            ("cursor", updates * 128 + 1),
            ("batch_lengths", [127] * (updates + 1)),
        ):
            bad = copy.deepcopy(payload)
            bad["root_sampler"][key] = value
            with pytest.raises(AssertionError):
                audit_sampler(bad, fixed, 1, updates)
        bad = copy.deepcopy(payload)
        bad["root_sampler"]["rng"] = torch.Generator().manual_seed(777).get_state()
        with pytest.raises(AssertionError):
            audit_sampler(bad, fixed, 1, updates)
        bad = copy.deepcopy(payload)
        bad["root_sampler"]["config"]["chance_seed"] += 1
        with pytest.raises(AssertionError):
            audit_sampler(bad, fixed, 1, updates)
        bad = copy.deepcopy(payload)
        bad["memory"][0] ^= 1
        with pytest.raises(AssertionError):
            audit_sampler(bad, fixed, 1, updates)


def test_fixed_and_renewed_initial_models_and_global_rng_match(tmp_path):
    from scripts.public_memory_continuation import (
        ContinuationConfig,
        ContinuationTrainer,
    )
    from scripts.renewed_continuation import (
        RenewedContinuationConfig,
        RenewedContinuationTrainer,
    )
    from scripts.run_arbitration_study import PARENTS

    bank = torch.load(
        "runs/public_memory_continuation_bank/parent1.pt",
        map_location="cpu",
        weights_only=False,
    )
    data = tuple(bank[key] for key in ("states", "seats", "opponents", "memory"))
    base_config = {"device": "cpu", "envs": 8, "critic_width": 256, "seed": 92110001}
    trainer = ContinuationTrainer.from_parent(
        ContinuationConfig(**base_config), PARENTS[0][0], *data
    )
    trainer.save(tmp_path / "fixed.pt")
    del trainer
    renewed = RenewedContinuationTrainer.from_parent(
        RenewedContinuationConfig(
            **base_config, chance_seed=94110001, chance_draws=66560
        ),
        PARENTS[0][0],
        *data,
    )
    renewed.save(tmp_path / "renewed.pt")
    left, right = [
        torch.load(tmp_path / name, map_location="cpu", weights_only=False)
        for name in ("fixed.pt", "renewed.pt")
    ]
    for key in (
        "actor",
        "value",
        "optimizer",
        "bank",
        "memory",
        "torch_rng",
        "cuda_rng",
        "python_rng",
        "prefix_actor",
        "opponents",
    ):
        tensor_equal(left[key], right[key])
    assert left["numpy_rng"][0] == right["numpy_rng"][0]
    np.testing.assert_array_equal(left["numpy_rng"][1], right["numpy_rng"][1])
    assert left["numpy_rng"][2:] == right["numpy_rng"][2:]
    assert torch.equal(left["bank_rng"], right["root_sampler"]["rng"])
    assert (
        right["root_sampler"]["cursor"]
        == right["counters"]["fresh_conditional_deals"]
        == 0
    )


def test_independent_statistics_and_claims_agree_on_synthetic_outcomes():
    points = np.zeros((2, 5, 2, 1024, 2))
    points[..., 0], points[..., 1] = 2, -2
    points[:, 3] = 4
    produced, reproduced = analyse(points), independent_contrasts(points)
    assert produced == reproduced and len(reproduced) == 66
    assert decisions(produced) == independent_decisions(reproduced)

    # Nonconstant small paired arrays exercise the independent10,000-draw
    # scalar bootstrap without generating or inspecting any evaluation deals.
    varied = np.random.default_rng(42).integers(-5, 6, size=(2, 5, 2, 5, 2))
    actual = independent_contrasts(varied)
    for parent in (1, 2):
        for metric in ("win", "point"):
            outcomes = varied[parent - 1].astype(float)
            if metric == "win":
                outcomes = (np.sign(outcomes) + 1) / 2
            paired = (outcomes[3] - outcomes[4] - outcomes[1] + outcomes[2]).mean(
                axis=-1
            )
            row = next(
                r
                for r in actual
                if (r["parent"], r["scope"], r["comparison"])
                == (parent, "mixture", "chance_by_memory")
            )
            assert row[metric]["change"] == float(paired.mean())
            np.testing.assert_allclose(
                row[metric]["interval95"],
                bootstrap_intervals(paired[None])[0],
                rtol=0,
                atol=1e-15,
            )
            # The actual protocol has1,024 paired deals per stratum. Integer/
            # half-integer outcomes then have exact binary seat/stratum means;
            # require exact equality at that production shape,including ties.
            full = np.tile(paired, (1, 205))[:, :1024]
            np.testing.assert_array_equal(
                interval(full), bootstrap_intervals(full[None])[0]
            )
    with pytest.raises(AssertionError):
        independent_decisions([])


def test_independent_match_identity_guards_without_consuming_fresh_deals():
    for policy_index, policy in enumerate(
        ("parent", "fixed_public", "fixed_masked", "renewed_public", "renewed_masked")
    ):
        row = {
            "parent": 2,
            "opponent": "point",
            "policy": policy,
            "replicate": 1023,
            "seat": 1,
            "seed": 95121024,
            "memory": None if policy_index == 0 else {},
        }
        assert verify_identity(row, 2, "point") == (policy_index, 1023, 1)
        for key, value in (
            ("seed", 93121024),
            ("replicate", True),
            ("seat", 2),
            ("parent", 1),
        ):
            with pytest.raises(AssertionError):
                verify_row({**row, key: value}, 2, "point")
        with pytest.raises(AssertionError):
            verify_identity(
                {**row, "memory": {} if policy_index == 0 else None}, 2, "point"
            )


def test_independent_audit_refuses_incomplete_producer(tmp_path):
    write(tmp_path / "process.json", {"status": "running"})
    write(tmp_path / "analysis.json", {"status": "complete"})
    write(tmp_path / "protocol.json", {})
    with pytest.raises(AssertionError):
        audit(tmp_path)
    assert not (tmp_path / "independent_audit.json").exists()


def test_worker_loads_all_five_policy_formats_on_old_correctness_deal(
    tmp_path, monkeypatch
):
    from koikoi.train import provenance
    from scripts.audit_public_continuation_study import replay_match
    from scripts.renewed_continuation import (
        RenewedContinuationConfig,
        RenewedContinuationTrainer,
    )
    from scripts.run_arbitration_study import PARENTS

    root, fixed_root = tmp_path / "renewed", tmp_path / "fixed"
    bank = torch.load(
        "runs/public_memory_continuation_bank/parent1.pt",
        map_location="cpu",
        weights_only=False,
    )
    data = tuple(bank[key] for key in ("states", "seats", "opponents", "memory"))
    inputs = {PARENTS[0][0]: PARENTS[0][1]}
    for mode in ("public", "masked"):
        fixed = fixed_root / f"parent1_{mode}/final.pt"
        fixed.parent.mkdir(parents=True)
        # Initial checkpoints only:do not inspect the ongoing study's trained
        # actions or consume any93M/95M/101M evaluation deals.
        shutil.copyfile(
            f"runs/public_continuation_study/parent1_{mode}/initial.pt", fixed
        )
        trainer = RenewedContinuationTrainer.from_parent(
            RenewedContinuationConfig(
                device="cpu",
                envs=8,
                critic_width=256,
                seed=92110001,
                public_memory=mode == "public",
                chance_seed=94110001,
                chance_draws=66560,
            ),
            PARENTS[0][0],
            *data,
        )
        path = root / f"parent1_{mode}/final.pt"
        trainer.save(path)
        inputs.update({str(p): sha(p) for p in (fixed, path)})
        del trainer
    write(
        root / "protocol.json",
        {"inputs": inputs, "core_source_sha256": provenance()["source_sha256"]},
    )
    monkeypatch.setattr(runner, "FIXED", fixed_root)
    monkeypatch.setattr(runner, "configure", lambda: torch.set_num_threads(2))
    monkeypatch.setattr(
        runner,
        "cases",
        lambda parent, opponent: [
            {
                "parent": parent,
                "opponent": opponent,
                "seat": 0,
                "replicate": 0,
                "seed": 88110001,
            }
        ],
    )
    path = runner.evaluate_cell(str(root), {"index": 1, "path": PARENTS[0][0]}, "wp")
    rows = [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]
    assert [r["policy"] for r in rows] == [
        "parent",
        "fixed_public",
        "fixed_masked",
        "renewed_public",
        "renewed_masked",
    ]
    assert all(row["actions"] == rows[0]["actions"] for row in rows)
    for row in rows:
        _, work, _ = replay_match(row)
        assert work["completed_matches"] == 1
    assert all(row["memory"] == rows[1]["memory"] for row in rows[1:])
