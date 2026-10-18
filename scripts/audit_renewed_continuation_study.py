"""Independent saved-schedule and full-match checks for chance renewal."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.published_actor import PublishedActor
from scripts.audit_final_round_training_study import tensor_equal
from scripts.audit_public_continuation_study import audit_roots, replay_match
from scripts.final_round_training import bank_identity
from scripts.public_memory_tail import memory_identity
from scripts.run_arbitration_study import PARENTS, read, sha, write

FIXED = Path("runs/public_continuation_study")
POLICIES = (
    "parent",
    "fixed_public",
    "fixed_masked",
    "renewed_public",
    "renewed_masked",
)


def audit_sampler(payload, fixed, parent, updates):
    """Reconstruct only index RNG;never generate a deal or execute a policy."""
    sampler = payload["root_sampler"]
    assert sampler["format"] == "public_context_chance_sampler_v1"
    assert sampler["config"] == {
        "sampling_seed": 92110002 + 10000 * (parent - 1),
        "chance_seed": 94110001 + 100000 * (parent - 1),
        "max_draws": 66560,
    }
    assert sampler["identity"] == {
        "bank": bank_identity(*payload["bank"]),
        "memory": memory_identity(payload["memory"]),
        "sampler_source": sha("scripts/renewed_root_sampler.py"),
        "deal_source": sha("scripts/continuation_chance.py"),
    }
    assert sampler["cursor"] == updates * 128
    assert sampler["batch_lengths"] == [128] * updates
    assert payload["counters"]["updates"] == fixed["counters"]["updates"] == updates
    assert payload["counters"]["fresh_conditional_deals"] == updates * 128
    assert payload["counters"]["conditional_tail_completions"] == updates * 128
    tensor_equal(payload["bank"], fixed["bank"])
    tensor_equal(payload["memory"], fixed["memory"])
    generator = torch.Generator(device="cpu").manual_seed(
        92110002 + 10000 * (parent - 1)
    )
    for _ in range(updates):
        torch.randint(len(payload["bank"][0]), (128,), generator=generator)
    assert torch.equal(sampler["rng"], generator.get_state())
    assert torch.equal(sampler["rng"], fixed["bank_rng"])
    for key, source in {
        "renewed_continuation_source_sha256": "scripts/renewed_continuation.py",
        "root_sampler_source_sha256": "scripts/renewed_root_sampler.py",
        "chance_constructor_source_sha256": "scripts/continuation_chance.py",
    }.items():
        assert payload["provenance"][key] == sha(source)


def audit_saved_schedules(root, parent, mode):
    paths = []
    endpoints = (
        [("initial.pt", 0)]
        + [(f"update_{u:08d}.pt", u) for u in (20, 100, 200, 300, 400, 500, 520)]
        + [("final.pt", 520)]
    )
    for filename, update in endpoints:
        fresh_path = root / f"parent{parent}_{mode}" / filename
        fixed_path = FIXED / f"parent{parent}_{mode}" / filename
        fresh, fixed = [
            torch.load(p, map_location="cpu", weights_only=False)
            for p in (fresh_path, fixed_path)
        ]
        audit_sampler(fresh, fixed, parent, update)
        if filename == "update_00000520.pt":
            final_update = fresh
        if filename == "final.pt":
            for key in fresh:
                if key != "numpy_rng":
                    tensor_equal(fresh[key], final_update[key])
            assert fresh["numpy_rng"][0] == final_update["numpy_rng"][0]
            np.testing.assert_array_equal(
                fresh["numpy_rng"][1], final_update["numpy_rng"][1]
            )
            assert fresh["numpy_rng"][2:] == final_update["numpy_rng"][2:]
        if update == 0:
            for key in (
                "actor",
                "value",
                "optimizer",
                "torch_rng",
                "cuda_rng",
                "python_rng",
                "prefix_actor",
                "opponents",
            ):
                tensor_equal(fresh[key], fixed[key])
            assert fresh["numpy_rng"][0] == fixed["numpy_rng"][0]
            np.testing.assert_array_equal(fresh["numpy_rng"][1], fixed["numpy_rng"][1])
            assert fresh["numpy_rng"][2:] == fixed["numpy_rng"][2:]
        paths.extend((fresh_path, fixed_path))
    return {str(p): sha(p) for p in paths}


def audit_endpoint(root, parent, mode):
    folder = root / f"parent{parent}_{mode}"
    paths = [
        folder / n for n in ("initial.pt", "final.pt", "metrics.jsonl", "config.json")
    ]
    initial, final = [
        torch.load(p, map_location="cpu", weights_only=False) for p in paths[:2]
    ]
    path, digest = PARENTS[parent - 1]
    assert sha(path) == digest
    original = torch.load(path, map_location="cpu", weights_only=False)
    assert final["format"] == initial["format"] == "renewed_public_continuation_v1"
    cfg = final["config"]
    assert cfg == initial["config"] == read(folder / "config.json")
    assert cfg["seed"] == 92110001 + 10000 * (parent - 1)
    assert cfg["public_memory"] == (mode == "public")
    for key, expected in {
        "envs": 128,
        "horizon": 448,
        "epochs": 3,
        "minibatch": 512,
        "critic_warmup_updates": 20,
        "lr": 3e-6,
        "value_lr": 1e-4,
        "score_temperature": 0.05,
        "device": "cuda",
        "chance_seed": 94110001 + 100000 * (parent - 1),
        "chance_draws": 66560,
    }.items():
        assert cfg[key] == expected
    assert final["provenance"]["weight_initialization"]["sha256"] == digest
    for payload in (initial, final):
        tensor_equal(payload["prefix_actor"], original["actor"])
    tensor_equal(initial["value"], original["value"])
    for phase, weights in original["actor"]["current"]["weights"].items():
        for name, value in weights.items():
            expected = value.clone()
            if name == "encoder_block.f1.weight":
                expected[:, 2:4] = 0
            assert torch.equal(
                initial["actor"]["current"]["weights"][phase][name], expected
            )
    tensor_equal(final["opponents"], initial["opponents"])
    bank_path = Path(f"runs/public_memory_continuation_bank/parent{parent}.pt")
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    assert bank["format"] == "public_memory_continuation_bank_v1"
    for payload in (initial, final):
        tensor_equal(
            payload["bank"], tuple(bank[k] for k in ("states", "seats", "opponents"))
        )
        tensor_equal(payload["memory"], bank["memory"])
    c = final["counters"]
    assert (
        c["updates"] == 520 and c["policy_updates"] == 500 and c["warmup_updates"] == 20
    )
    assert c["conditional_tail_completions"] == c["fresh_conditional_deals"] == 66560
    assert 66560 <= c["conditional_round_endings"] <= 66560 * 7
    assert c["matches"] == c["rounds"] == 0
    assert c["decisions"] == c["learner_decisions"] + c["opponent_decisions"]
    actor = PublishedActor(final["actor"], "cpu")
    optimizer = final["optimizer"]
    names = list(actor.named_parameters())
    assert len(names) == len(optimizer["param_groups"][0]["params"])
    for (name, parameter), index in zip(names, optimizer["param_groups"][0]["params"]):
        phase = {"0": "discard", "1": "pick", "3": "koi"}[name.split(".")[1]]
        assert int(optimizer["state"][index]["step"]) == c["optimizer_steps_" + phase]
        if mode == "masked" and name.endswith("encoder_block.f1.weight"):
            assert not bool(parameter[:, 2:4].any())
    for index in optimizer["param_groups"][1]["params"]:
        assert int(optimizer["state"][index]["step"]) == c["optimizer_steps_value"]
    assert all(
        bool(torch.isfinite(v).all())
        for entry in optimizer["state"].values()
        for v in entry.values()
    )
    metrics = [json.loads(s) for s in paths[2].read_text(encoding="utf-8").splitlines()]
    assert [r["update"] for r in metrics] == list(range(1, 521))
    assert metrics[-1]["counters"] == c
    for index, row in enumerate(metrics, 1):
        assert row["metrics"]["critic_only"] == (index <= 20)
        assert row["counters"]["conditional_tail_completions"] == index * 128
        assert row["counters"]["fresh_conditional_deals"] == index * 128
    schedule_inputs = audit_saved_schedules(root, parent, mode)
    return {
        "status": "passed",
        "counters": c,
        "finite_adam_states": len(optimizer["state"]),
        "inputs": {
            **{str(p): sha(p) for p in paths + [bank_path, Path(path)]},
            **schedule_inputs,
        },
    }


def audit_matched_warmup(root, parent):
    for endpoint in ("initial.pt", "update_00000020.pt"):
        left, right = [
            torch.load(
                root / f"parent{parent}_{mode}" / endpoint,
                map_location="cpu",
                weights_only=False,
            )
            for mode in ("public", "masked")
        ]
        for key in (
            "actor",
            "value",
            "optimizer",
            "bank",
            "root_sampler",
            "prefix_actor",
            "opponents",
            "runtime",
            "memory",
            "counters",
            "torch_rng",
            "cuda_rng",
            "python_rng",
        ):
            tensor_equal(left[key], right[key])
        assert left["numpy_rng"][0] == right["numpy_rng"][0]
        np.testing.assert_array_equal(left["numpy_rng"][1], right["numpy_rng"][1])
        assert left["numpy_rng"][2:] == right["numpy_rng"][2:]


def interval(values):
    values = np.asarray(values, dtype=float)
    if np.all(np.ptp(values, axis=1) == 0):
        return [float(values.mean())] * 2
    rng = np.random.default_rng(95310001)
    results = []
    for start in range(0, 10000, 128):
        n = min(128, 10000 - start)
        selections = rng.integers(
            values.shape[1], size=(n, values.shape[0], values.shape[1])
        )
        strata = np.zeros((n, values.shape[0]))
        for group in range(values.shape[0]):
            strata[:, group] = values[group][selections[:, group]].mean(axis=1)
        results.extend(strata.mean(axis=1))
    return np.percentile(results, [2.5, 97.5]).tolist()


def verify_identity(row, parent, opponent):
    assert type(row["parent"]) is int and row["parent"] == parent
    assert row["opponent"] == opponent
    policy = POLICIES.index(row["policy"])
    rep, seat = row["replicate"], row["seat"]
    assert type(rep) is int and 0 <= rep < 1024
    assert type(seat) is int and seat in (0, 1)
    assert type(row["seed"]) is int
    assert row["seed"] == 95110001 + 10000 * (opponent == "point") + rep
    assert (row["memory"] is None) == (policy == 0)
    return policy, rep, seat


def verify_row(row, parent, opponent):
    key = verify_identity(row, parent, opponent)
    prefix, work, checks = replay_match(row)
    return key, prefix, work, checks


def audit_cell(path, parent, opponent):
    points = np.empty((5, 1024, 2))
    seen, prefixes, memories = set(), {}, {}
    work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    checks = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key, prefix, counts, nchecks = verify_row(row, parent, opponent)
            assert key not in seen
            seen.add(key)
            policy, rep, seat = key
            checks += nchecks
            for name, count in counts.items():
                work[name] += count
            if policy == 0:
                prefixes[rep, seat] = prefix
            else:
                assert prefix == prefixes[rep, seat][: len(prefix)]
                if row["memory"]["activation"] is None:
                    assert prefix == prefixes[rep, seat]
                evidence = {
                    name: row["memory"][name]
                    for name in (
                        "events",
                        "candidates",
                        "prediction_calls",
                        "first_identification",
                        "activation",
                    )
                }
                if policy == 1:
                    memories[rep, seat] = evidence
                else:
                    assert evidence == memories[rep, seat]
            points[policy, rep, seat] = (
                row["terminal_points"][seat] - row["terminal_points"][1 - seat]
            )
    assert len(seen) == 10240 and work["completed_matches"] == 10240
    return points, work, checks


def independent_contrasts(points):
    """Recompute scalar contrasts/strata without the producer analysis module."""
    pairs = (
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 4),
        (4, 0),
        (4, 1),
        (4, 2),
        (1, 0),
        (1, 2),
        (2, 0),
    )
    named = [(f"{POLICIES[a]}-minus-{POLICIES[b]}", a, b) for a, b in pairs]
    records = []
    for parent in (1, 2):
        for scope, groups in (("mixture", [0, 1]), ("wp", [0]), ("point", [1])):
            for name, candidate, baseline in named + [("chance_by_memory", None, None)]:
                record = {"parent": parent, "comparison": name, "scope": scope}
                for metric in ("win", "point"):
                    outcomes = points[parent - 1].copy()
                    if metric == "win":
                        outcomes = np.where(
                            outcomes > 0, 1.0, np.where(outcomes < 0, 0.0, 0.5)
                        )
                    if candidate is None:
                        change = (
                            outcomes[3, groups]
                            - outcomes[4, groups]
                            - outcomes[1, groups]
                            + outcomes[2, groups]
                        )
                    else:
                        change = (
                            outcomes[candidate, groups] - outcomes[baseline, groups]
                        )
                    paired = (change[:, :, 0] + change[:, :, 1]) / 2
                    record[metric] = {
                        "change": float(paired.mean()),
                        "interval95": interval(paired),
                    }
                records.append(record)
    return records


def independent_decisions(records):
    names = {
        "replicated_strength_gain": (
            "renewed_public-minus-parent",
            "renewed_public-minus-fixed_public",
        ),
        "replicated_public_memory_gain": ("renewed_public-minus-renewed_masked",),
        "replicated_positive_interaction": ("chance_by_memory",),
    }
    answer = {}
    for label, comparisons in names.items():
        selected = [
            r
            for r in records
            if r["scope"] == "mixture" and r["comparison"] in comparisons
        ]
        assert {(r["parent"], r["comparison"]) for r in selected} == {
            (p, c) for p in (1, 2) for c in comparisons
        }
        assert len(selected) == 2 * len(comparisons)
        answer[label] = all(r["win"]["interval95"][0] > 0 for r in selected)
    return answer


def audit(root):
    output = root / "independent_audit.json"
    if output.exists():
        raise ValueError("Refusing to overwrite independent audit")
    plan, produced = read(root / "protocol.json"), read(root / "analysis.json")
    assert read(root / "process.json")["status"] == produced["status"] == "complete"
    assert plan["kind"] == "renewed_continuation_study_v1"
    assert (
        plan["updates"],
        plan["pairs_per_opponent"],
        plan["bootstrap_seed"],
        plan["bootstrap_replicates"],
    ) == (520, 1024, 95310001, 10000)
    assert plan["fixed_root"] == str(FIXED)
    assert plan["parents"] == [
        {
            "index": p,
            "path": path,
            "sha256": digest,
            "seed": 92110001 + 10000 * (p - 1),
            "chance_seed": 94110001 + 100000 * (p - 1),
        }
        for p, (path, digest) in enumerate(PARENTS, 1)
    ]
    inputs = {
        **plan["inputs"],
        **produced["inputs"],
        str(root / "analysis.json"): sha(root / "analysis.json"),
    }
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    points = np.empty((2, 5, 2, 1024, 2))
    work = dict.fromkeys(("decisions", "round_endings", "completed_matches"), 0)
    root_work = dict.fromkeys(work, 0)
    training, checks = [], 0
    for parent in (1, 2):
        for key, value in audit_roots(parent).items():
            root_work[key] += value
        for mode in ("public", "masked"):
            report = audit_endpoint(root, parent, mode)
            assert report == read(root / f"parent{parent}_{mode}/training_audit.json")
            training.append(report["counters"])
        audit_matched_warmup(root, parent)
        for group, opponent in enumerate(("wp", "point")):
            values, counts, nchecks = audit_cell(
                root / f"eval_parent{parent}_{opponent}.jsonl", parent, opponent
            )
            points[parent - 1, :, group] = values
            checks += nchecks
            for key, value in counts.items():
                work[key] += value
    reproduced = independent_contrasts(points)
    assert len(reproduced) == 66 and reproduced == produced["contrasts"]
    verdicts = independent_decisions(reproduced)
    assert verdicts == produced["decisions"]
    assert work["completed_matches"] == 40960
    assert sum(c["fresh_conditional_deals"] for c in training) == 266240
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    write(
        output,
        {
            "status": "passed",
            "inputs": inputs,
            "contrasts": reproduced,
            "decisions": verdicts,
            "training_counters": training,
            "evaluation_replay_work": work,
            "root_replay_work": root_work,
            "strict_input_checks": checks,
            "distinct_training_chance_keys": 133120,
            "distinct_evaluation_deals": 2048,
            "new_neural_inferences": 0,
            "new_independent_matches": 0,
        },
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
