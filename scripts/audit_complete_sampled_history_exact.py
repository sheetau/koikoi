"""Independently check all hand sets and scalar posterior/error arithmetic."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from koikoi.layout import DEALER, PLAYER
from koikoi.reference import ReferenceEnv
from scripts.enumerate_history_support import context
from scripts.run_arbitration_study import read, sha, write


def verify_row(row, original, batch):
    source_path = Path(
        f"runs/final_round_training_study/parent{row['parent']}/source.jsonl"
    )
    source = json.loads(
        source_path.read_text(encoding="utf-8").splitlines()[row["case_index"]]
    )
    assert all(
        row[k] == original[k]
        for k in ("parent", "opponent", "case_index", "seed", "decision")
    )
    assert row["seed"] == source["seed"] and original["seat"] == source["seat"]
    env = ReferenceEnv(source["seed"], "paper8")
    for action in source["actions"][: row["decision"]]:
        assert env.legal()[action]
        env.step(action)
    assert int(env.state[PLAYER]) == source["seat"]
    public_key = hashlib.sha256(
        env.observe().tobytes()
        + np.asarray(env.state[[PLAYER, DEALER]], dtype=np.int32).tobytes()
    ).hexdigest()
    assert (
        public_key
        == row["public_root_key"]
        == batch["root_key"]
        == original["proposals"]["root_key"]
    )
    unknown, size, played = context(env.state)
    valid, invalid = [], 0
    for hand in itertools.combinations(unknown, size):
        initial = played + list(hand)
        if any(sum(c // 4 == month for c in initial) == 4 for month in range(12)):
            invalid += 1
        else:
            valid.append(list(hand))
    assert batch["mode"] == "enumerated" and batch["model_names"] == ["point", "wp"]
    assert valid == batch["hands"] and invalid == batch["invalid_draws"]
    assert len(valid) + invalid == batch["candidate_hands"] == batch["proposed_draws"]
    assert len(batch["compatible"]) == len(valid)
    assert all(
        len(c) == 2 and all(type(b) is bool for b in c) for c in batch["compatible"]
    )
    truth = {c for c in range(48) if env.state[c] == 1 - source["seat"]}
    assert len(truth) == size
    scores = {}
    for name in ("uniform", "joint", "identified"):
        estimate, comparison = row["exact_beliefs"][name], row["comparison"][name]
        if name == "uniform":
            weights, model_names, compat = (
                [1] * len(valid),
                ["uniform"],
                [[True]] * len(valid),
            )
        elif name == "joint":
            weights = [sum(c) for c in batch["compatible"]]
            model_names, compat = batch["model_names"], batch["compatible"]
        else:
            # Use the audited prior public identification,not the true matchup label.
            model = original["public_identification"]["model"]
            index = batch["model_names"].index(model)
            weights = [int(c[index]) for c in batch["compatible"]]
            model_names, compat = [model], [[c[index]] for c in batch["compatible"]]
        total = sum(weights)
        assert total > 0 and estimate["status"] == comparison["status"] == "estimated"
        probabilities = [
            sum(w for h, w in zip(valid, weights) if c in h) / total for c in range(48)
        ]
        np.testing.assert_allclose(
            probabilities, estimate["card_probabilities"], rtol=0, atol=1e-12
        )
        assert abs(sum(probabilities) - size) < 1e-12 and estimate["exact_enumeration"]
        for index, model in enumerate(model_names):
            assert (
                abs(
                    estimate["model_probabilities"][model]
                    - sum(c[index] for c in compat) / total
                )
                < 1e-12
            )
        supported = sum(w > 0 for w in weights)
        assert (
            supported
            == estimate["distinct_supported_hands"]
            == estimate["supported_proposal_occurrences"]
            == comparison["exact_supported_hands"]
        )
        assert (
            abs(
                estimate["effective_proposals"] - total**2 / sum(w * w for w in weights)
            )
            < 1e-8
        )
        brier = sum((probabilities[c] - int(c in truth)) ** 2 for c in unknown) / len(
            unknown
        )
        assert abs(brier - comparison["exact_card_brier"]) < 1e-12
        sampled = original["beliefs"][name]
        assert sampled["status"] == comparison["sampled_status"]
        if sampled["status"] == "estimated":
            errors = [
                abs(probabilities[c] - sampled["card_probabilities"][c])
                for c in unknown
            ]
            assert (
                abs(
                    sum(errors) / len(errors)
                    - comparison["mean_absolute_sampling_error"]
                )
                < 1e-12
            )
            assert (
                abs(max(errors) - comparison["maximum_absolute_sampling_error"]) < 1e-12
            )
            assert sampled["card_brier"] == comparison["sampled_card_brier"]
            assert (
                sampled["effective_proposals"]
                == comparison["sampled_effective_proposals"]
            )
        else:
            assert "sampled_card_brier" not in comparison
        scores[name] = brier
    return scores, row["decision"], len(valid), invalid


def audit(root):
    output = root / "arithmetic_verification.json"
    if output.exists():
        raise ValueError("Refusing to overwrite the completed audit")
    assert read(root / "process.json")["status"] == "complete"
    report = read(root / "report.json")
    assert report["status"] == "complete"
    assert report["protocol_sha256"] == sha(root / "protocol.json")
    assert (
        report["new_evaluation_deals"]
        == report["future_rollouts"]
        == report["new_training_matches"]
        == 0
    )
    inputs = {
        **report["inputs"],
        str(root / "report.json"): sha(root / "report.json"),
        str(Path(__file__)): sha(__file__),
    }
    for path in (
        "tests/test_audit_complete_sampled_history_exact.py",
        "runs/audit_complete_sampled_history_exact_tests.xml",
        "runs/audit_complete_sampled_history_exact_tests_initial.xml",
    ):
        inputs[path] = sha(path)
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    records = [
        json.loads(s)
        for s in Path("runs/sampled_history_diagnostic/records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    eligible = {
        (r["parent"], r["case_index"]): r for r in records if r["status"] == "eligible"
    }
    assert len(records) == 32 and len(eligible) == len(report["results"]) == 25
    assert [(r["parent"], r["case_index"]) for r in report["results"]] == list(eligible)
    prior_keys = {
        read(Path("runs/sampled_history_exact_check") / r["support_file"])["root_key"]
        for r in read("runs/sampled_history_exact_check/report.json")["results"]
    }
    seen_keys, scores, work = set(prior_keys), [], {}
    source_steps = valid_total = invalid_total = newly_computed = 0
    for row in report["results"]:
        support = root / row["support_file"]
        assert sha(support) == row["support_sha256"]
        inputs[str(support)] = sha(support)
        batch = read(support)
        values, steps, valid, invalid = verify_row(
            row, eligible[row["parent"], row["case_index"]], batch
        )
        scores.append(
            {"parent": row["parent"], "opponent": row["opponent"], "errors": values}
        )
        source_steps += steps
        valid_total += valid
        invalid_total += invalid
        assert row["reused_exact_table"] == (row["public_root_key"] in seen_keys)
        if not row["reused_exact_table"]:
            newly_computed += 1
            for name, value in batch["counts"].items():
                work[name] = work.get(name, 0) + value
        seen_keys.add(row["public_root_key"])
    assert report["new_exact_tables"] == newly_computed
    assert report["cached_root_uses"] == 25 - newly_computed
    assert report["new_hypothetical_history_replay"] == work
    assert report["source_prefix_replay_decisions"] == source_steps
    assert report["distinct_public_roots"] == len(
        {r["public_root_key"] for r in report["results"]}
    )
    assert report["shared_source_deals"] == len({r["seed"] for r in report["results"]})
    for scope in ("all", "parent1", "parent2", "wp", "point"):
        rows = [
            r
            for r in scores
            if scope == "all"
            or scope == f"parent{r['parent']}"
            or scope == r["opponent"]
        ]
        assert report["aggregate"][scope]["roots"] == len(rows)
        for name in ("uniform", "joint", "identified"):
            metric = report["aggregate"][scope]["beliefs"][name]
            assert metric["exact_estimated"] == len(rows)
            assert (
                abs(
                    metric["exact_mean_card_brier"]
                    - sum(r["errors"][name] for r in rows) / len(rows)
                )
                < 1e-12
            )
            original_rows = [
                r
                for r in eligible.values()
                if scope == "all"
                or scope == f"parent{r['parent']}"
                or scope == r["opponent"]
            ]
            sampled = [
                r["beliefs"][name]["card_brier"]
                for r in original_rows
                if r["beliefs"][name]["status"] == "estimated"
            ]
            assert metric["sampled_estimated"] == len(sampled)
            if sampled:
                assert (
                    abs(metric["sampled_mean_card_brier"] - sum(sampled) / len(sampled))
                    < 1e-12
                )
            else:
                assert metric["sampled_mean_card_brier"] is None
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    write(
        output,
        {
            "status": "passed",
            "inputs": inputs,
            "eligible_roots": 25,
            "ineligible_attempts": 7,
            "candidate_hands_across_root_records": valid_total + invalid_total,
            "valid_hands_across_root_records": valid_total,
            "distinct_public_roots": report["distinct_public_roots"],
            "new_exact_tables": newly_computed,
            "additional_source_replay_decisions": source_steps,
            "additional_neural_calls": 0,
            "scope": "Complete candidate sets and independent scalar marginal/Brier/sampling-error checks; likelihood tables are not recomputed by this audit.",
        },
    )
    print(json.dumps({k: v for k, v in read(output).items() if k != "inputs"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
