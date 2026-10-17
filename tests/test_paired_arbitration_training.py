import numpy as np
import pytest
import torch

from koikoi.reference import ReferenceEnv
from koikoi.search import sample_hidden_batch
from scripts import run_paired_arbitration_study as study
from scripts.analyze_arbitration_study import outcome, tensor_equal
from scripts.analyze_paired_arbitration_study import data_report, fit_report
from scripts.diagnose_paired_targets import diagnose
from scripts.paired_rollout_targets import match_split
from scripts.run_arbitration_study import PARENTS, sha, write
from scripts.train_paired_arbitration import (
    PairedRolloutAgent,
    collect,
    fit,
    prediction,
)


def test_public_dataset_fit_and_embedded_deployment(tmp_path, monkeypatch):
    dataset = tmp_path / "parent1" / "targets.pt"
    raw = collect(
        PARENTS[0][0], dataset, seed=18950001, roots=32, batch=16, device="cpu"
    )
    assert raw["public_observations"].shape == (32, 562)
    assert raw["branch_work"]["completed_branches"] == 64
    assert raw["branch_work"]["decisions"] > 64
    assert raw["source_work"]["decisions"] > 0
    assert set(raw) == {
        "format",
        "public_observations",
        "match_groups",
        "outcomes",
        "point_differences",
        "experts",
        "parent_value",
        "parent",
        "network_config",
        "source_work",
        "branch_work",
        "provenance",
        "script_hashes",
        "collection",
    }
    assert torch.equal(raw["outcomes"], raw["point_differences"].sign().float())
    split = match_split(raw["match_groups"])
    assert split.any() and (~split).any()
    audit_protocol = {
        "collection": {"roots_per_parent": 32, "seeds": [18950001]},
        "parents": [{"sha256": PARENTS[0][1]}],
        "source": raw["provenance"],
        "fit": {"epochs": 2},
    }
    _, audited_data = data_report(tmp_path, 1, audit_protocol)
    assert sum(audited_data["target_counts"].values()) == 32
    for context in ("full", "no_match", "phase"):
        case = "parent1_" + context
        target = tmp_path / case / "final.pt"
        payload = fit(
            dataset, target, context=context, seed=18940001, device="cpu", epochs=2
        )
        assert tensor_equal(raw["experts"], payload["experts"])
        assert (
            payload["training_config"]["algorithm"]
            == "paired complete-match advantage regression"
        )
        assert payload["counters"]["validation_roots"] == int(split.sum())
        assert all(np.isfinite(r["validation_mse"]) for r in payload["history"])
        write(
            target.parent / "training_complete.json",
            {
                "checkpoint_sha256": sha(target),
                "dataset_sha256": sha(dataset),
                "counters": payload["counters"],
                "history": payload["history"],
            },
        )
        inspected = fit_report(tmp_path, case, raw, audit_protocol)
        assert np.isfinite(inspected["validation_mse_recomputed_cpu"])
        agent = PairedRolloutAgent(target)
        env = ReferenceEnv(18930001, "paper8")
        action = agent.act(env)
        assert env.legal()[action]
        original = agent.bank.observe(agent.env)[0]
        worlds = sample_hidden_batch(env, np.random.default_rng(4), 1)
        agent.env.state.copy_(torch.tensor(worlds))
        changed = agent.bank.observe(agent.env)[0]
        torch.testing.assert_close(original, changed, rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(
                prediction(agent.model, original),
                prediction(agent.model, changed),
                rtol=0,
                atol=0,
            )
        with pytest.raises(ValueError):
            fit(dataset, target, device="cpu", epochs=2)
        if context == "full":
            diagnostic = tmp_path / "held_out_diagnostic.json"
            diagnose(dataset, [target], diagnostic)
            checked_diagnostic = study.read(diagnostic)
            sections = checked_diagnostic["models"][str(target)]["sections"]
            assert sections["all"]["roots"] == int(split.sum())
            assert sections["all"]["source_matches"] == len(
                torch.unique(raw["match_groups"][split], dim=0)
            )
            assert sections["all"]["validation_mse"] == pytest.approx(
                inspected["validation_mse_recomputed_cpu"], abs=1e-6
            )
            p = {
                "cases": {
                    "test": {
                        "path": str(target),
                        "kind": "regression",
                        "parent": {"path": PARENTS[0][0]},
                    }
                },
                "evaluation": {"first_seed": 18920001, "pairs": 2},
            }
            monkeypatch.setattr(study, "checked", lambda base, protocol=p: protocol)
            write(tmp_path / "endpoint_identities.json", {"test": sha(target)})
            study.evaluate(tmp_path, "test", "parent")
            report, _, _ = outcome(tmp_path, "test", "parent", p)
            assert report["matches"] == 4
    with pytest.raises(ValueError):
        collect(PARENTS[0][0], dataset, seed=18950001, roots=32, batch=16, device="cpu")
