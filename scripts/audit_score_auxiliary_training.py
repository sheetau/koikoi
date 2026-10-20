"""Independent stored-state and auxiliary trajectory audit; no learner imports.

Checks completed artifacts, not gradients or neural action replay. Training-loss
summaries are checked for finiteness, not recomputed. No playing-strength claim.
"""

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from koikoi.upstream_batch import BatchPublishedAgent
from scripts.audit_pretrained_ppo_screen import tensor_digest, validate_accounting
from scripts.audit_score_auxiliary_collection import auxiliary_steps, check_collection

PROTOCOL_SHA = "9ba0bd7b1d187d59269af336537ceab890bf8884a969a9df073dcd626e686f46"
COUNTERS = (
    "cycles",
    "decisions",
    "rounds",
    "matches",
    "retained_rows",
    "example_presentations",
    "optimizer_steps",
    "optimizer_steps_discard",
    "optimizer_steps_pick",
    "optimizer_steps_koi",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def equal(left, right):
    """Exact recursive equality, including saved RNG and tensor state."""
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray) and left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal(a, b)
    else:
        assert left == right


@lru_cache(maxsize=1)
def identities():
    digest = hashlib.sha256()
    for path in sorted(Path("koikoi").rglob("*")):
        if path.suffix in (".py", ".h", ".cpp", ".cu"):
            digest.update(path.as_posix().encode())
            digest.update(path.read_bytes())
    original = BatchPublishedAgent("point", "cpu").checkpoint()
    metadata = {
        key: original[key]
        for key in ("variant", "upstream_revision", "model_hashes", "parameters_sha256")
    }
    shapes = {
        phase: {name: tuple(t.shape) for name, t in weights.items()}
        for phase, weights in original["weights"].items()
    }
    return digest.hexdigest(), metadata, shapes


def check_adam(optimizer, parameters, expected_steps, rates):
    """Check phase-specific call counts, parameter order, and moment validity."""
    groups, state = optimizer["param_groups"], optimizer["state"]
    assert len(groups) == len(parameters) == len(expected_steps) == len(rates) == 2
    all_ids = []
    for group, tensors, counts, rate in zip(
        groups, parameters, expected_steps, rates, strict=True
    ):
        ids = group["params"]
        assert len(ids) == len(tensors) == len(counts)
        all_ids += ids
        assert group["lr"] == rate and group["eps"] == 1e-5
        assert tuple(group["betas"]) == (0.9, 0.999) and group["weight_decay"] == 0
        for flag in (
            "amsgrad",
            "maximize",
            "capturable",
            "differentiable",
            "decoupled_weight_decay",
        ):
            assert group[flag] is False
        for index, tensor, count in zip(ids, tensors, counts, strict=True):
            assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
            if count == 0:
                assert index not in state
                continue
            entry = state[index]
            assert set(entry) == {"step", "exp_avg", "exp_avg_sq"}
            assert entry["step"].numel() == 1 and float(entry["step"]) == count
            for key in ("exp_avg", "exp_avg_sq"):
                moment = entry[key]
                assert moment.shape == tensor.shape and moment.dtype == torch.float32
                assert torch.isfinite(moment).all()
            assert (entry["exp_avg_sq"] >= 0).all()
    assert len(set(all_ids)) == len(all_ids) and set(state) <= set(all_ids)


def check_payloads(ppo, auxiliary, specification, rows, expected_auxiliary):
    pconfig, aconfig = specification["ppo"], specification["auxiliary"]
    assert ppo["format"] == "published_ppo_v1" and ppo["config"] == pconfig
    assert (
        auxiliary["format"] == "score_auxiliary_state_v1"
        and auxiliary["config"] == aconfig
    )
    assert auxiliary["counters"] == expected_auxiliary
    assert set(expected_auxiliary) == set(COUNTERS)
    assert all(type(v) is int and v >= 0 for v in expected_auxiliary.values())
    updates = len(rows)
    validate_accounting(ppo, rows, updates)
    assert (
        expected_auxiliary["cycles"]
        == max(0, updates - pconfig["critic_warmup_updates"]) // aconfig["interval"]
    )
    source, initial, shapes = identities()
    assert ppo["provenance"]["source_sha256"] == source
    assert ppo["provenance"]["published_initialization"] == initial
    actor = ppo["actor"]
    assert (
        actor["format"] == "published_actor_v1" and actor["initialization"] == initial
    )
    current = actor["current"]
    assert current["endgame"] is False
    for key in ("variant", "upstream_revision", "model_hashes"):
        assert current[key] == initial[key]
    weights = current["weights"]
    assert list(weights) == [0, 1, 3]
    for phase in weights:
        assert list(weights[phase]) == list(shapes[phase])
        assert all(
            tuple(t.shape) == shapes[phase][key] for key, t in weights[phase].items()
        )
    digest = tensor_digest(weights)
    assert digest == current["parameters_sha256"]
    assert (digest == initial["parameters_sha256"]) == (
        updates <= pconfig["critic_warmup_updates"]
    )
    runtime = ppo["runtime"]
    for key, value in {
        "torch": ppo["provenance"]["torch"],
        "device": "cuda:0",
        "cpu_threads": pconfig["cpu_threads"],
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }.items():
        assert runtime[key] == value
    assert pconfig["device"] == "cuda" and pconfig["precision"] == "float32"
    actor_parameters = [
        tensor for phase in (0, 1, 3) for tensor in weights[phase].values()
    ]
    assert len(actor_parameters) == 90
    value = ppo["value"]
    assert list(value) == [
        "0.weight",
        "0.bias",
        "2.weight",
        "2.bias",
        "4.weight",
        "4.bias",
    ]
    assert [tuple(t.shape) for t in value.values()] == [
        (256, 560),
        (256,),
        (256, 256),
        (256,),
        (1, 256),
        (1,),
    ]
    pcounts = ppo["counters"]
    counts = [
        pcounts["optimizer_steps_" + phase]
        for phase in ("discard", "pick", "koi")
        for _ in range(30)
    ]
    rates = [pconfig["lr"], pconfig["value_lr"]]
    check_adam(
        ppo["optimizer"],
        [actor_parameters, list(value.values())],
        [counts, [pcounts["optimizer_steps_value"]] * 6],
        rates,
    )
    heads = auxiliary["heads"]
    assert list(heads) == [
        f"{phase}.{name}" for phase in (0, 1, 3) for name in ("weight", "bias")
    ]
    assert [tuple(t.shape) for t in heads.values()] == [(65, 256), (65,)] * 3
    acounts = [
        expected_auxiliary["optimizer_steps_" + phase]
        for phase in ("discard", "pick", "koi")
    ]
    check_adam(
        auxiliary["optimizer"],
        [actor_parameters, list(heads.values())],
        [
            [count for count in acounts for _ in range(30)],
            [count for count in acounts for _ in range(2)],
        ],
        rates,
    )
    if not expected_auxiliary["optimizer_steps"]:
        bias = torch.full((65,), -math.log(32), dtype=torch.float32)
        bias[32] = 0
        for phase in (0, 1, 3):
            assert torch.count_nonzero(heads[f"{phase}.weight"]) == 0
            assert torch.equal(heads[f"{phase}.bias"], bias)
    if not expected_auxiliary["cycles"]:
        assert auxiliary["last_collection"] is auxiliary["last_metrics"] is None
    else:
        record = auxiliary["last_collection"]
        assert (
            isinstance(record["actor_sha256"], str)
            and len(record["actor_sha256"]) == 64
        )
        int(record["actor_sha256"], 16)
        if (
            aconfig["kind"] == "none"
            and rows[-1]["policy_updates"] % aconfig["interval"] == 0
        ):
            assert record["actor_sha256"] == digest
    return {
        "parameters_sha256": digest,
        "ppo_counters": pcounts,
        "auxiliary_counters": expected_auxiliary.copy(),
    }


def checkpoint(folder, specification, rows, counts, bindings):
    folder = Path(folder)
    manifest = read(folder / "manifest.json")
    assert manifest["format"] == "score_auxiliary_checkpoint_v1"
    assert manifest["config"] == specification["auxiliary"]
    assert set(manifest["files"]) == {"ppo.pt", "auxiliary.pt"}
    assert set(manifest["sources"]) == {
        f"scripts/{name}.py"
        for name in (
            "score_auxiliary",
            "collect_score_auxiliary",
            "score_auxiliary_trainer",
        )
    }
    for source, digest in manifest["sources"].items():
        assert sha(source) == digest
        bindings[source] = digest
    bindings[str(folder / "manifest.json")] = sha(folder / "manifest.json")
    for name, digest in manifest["files"].items():
        assert sha(folder / name) == digest
        bindings[str(folder / name)] = digest
    ppo, aux = load(folder / "ppo.pt"), load(folder / "auxiliary.pt")
    assert ppo["counters"] == manifest["ppo_counters"]
    assert aux["counters"] == manifest["auxiliary_counters"]
    result = check_payloads(ppo, aux, specification, rows, counts)
    result["manifest_sha256"] = bindings[str(folder / "manifest.json")]
    return aux, result


def check_phase(metrics, config, cycle, collection, steps):
    assert metrics["kind"] == config["kind"] and metrics["cycle"] == cycle
    assert metrics["retained_rows"] == len(collection["targets"])
    assert metrics["collection_work"] == collection["work"]
    assert metrics["example_presentations"] == steps["example_presentations"]
    losses = metrics["mean_training_losses"]
    if config["kind"] == "none":
        assert losses is None
    else:
        assert set(losses) == {"outcome_nll", "within_outcome_nll", "clone_kl"}
        assert all(math.isfinite(v) for v in losses.values())
        assert losses["outcome_nll"] >= 0 and losses["within_outcome_nll"] >= 0
        assert losses["clone_kl"] >= -1e-5


def audit_arm(root, name, protocol, bindings, *, through=520):
    """Audit a closed checkpoint prefix without binding a still-growing log."""
    folder = Path(root) / name
    specification = protocol["arms"][name]
    config = specification["auxiliary"]
    log_bytes = (folder / "metrics.jsonl").read_bytes()
    lines = log_bytes.splitlines(keepends=True)
    assert len(lines) >= through and all(
        line.endswith(b"\n") for line in lines[:through]
    )
    log_prefix = b"".join(lines[:through])
    rows = [json.loads(line) for line in lines[:through]]
    counts = dict.fromkeys(COUNTERS, 0)
    checkpoints = {}
    _, checkpoints["initial"] = checkpoint(
        folder / "initial", specification, [], counts, bindings
    )
    previous_seconds = -1
    collections = []
    last_collection = last_metrics = None
    for update, row in enumerate(rows, 1):
        ppo = row["ppo"]
        assert ppo["updates"] == update
        assert math.isfinite(row["seconds"]) and row["seconds"] >= previous_seconds
        previous_seconds = row["seconds"]
        cycle = max(0, update - protocol["warmup"]) // config["interval"]
        assert ppo["auxiliary_cycles"] == cycle
        if cycle != counts["cycles"]:
            assert cycle == counts["cycles"] + 1
            path = folder / f"collection_{cycle:04d}.pt"
            bindings[str(path)] = sha(path)
            last_collection = load(path)
            check = check_collection(last_collection, config, cycle)
            steps = auxiliary_steps(last_collection, config, cycle)
            last_metrics = row["auxiliary_phase"]
            check_phase(last_metrics, config, cycle, last_collection, steps)
            counts["cycles"] += 1
            counts["retained_rows"] += check["retained_rows"]
            for key, value in (check["work"] | steps).items():
                counts[key] += value
            collections.append(
                check
                | {"cycle": cycle, "sha256": bindings[str(path)], "optimization": steps}
            )
        else:
            assert row["auxiliary_phase"] is None
        assert row["auxiliary_counters"] == counts
        policy_updates = max(0, update - protocol["warmup"])
        if (
            update == 20
            or policy_updates == 5
            or (policy_updates and policy_updates % 25 == 0)
        ):
            stage = f"update_{update:08d}"
            aux, checkpoints[stage] = checkpoint(
                folder / stage,
                specification,
                [r["ppo"] for r in rows[:update]],
                counts,
                bindings,
            )
            equal(aux["last_collection"], last_collection)
            equal(aux["last_metrics"], last_metrics)
    endpoint = f"update_{through:08d}"
    assert endpoint in checkpoints, "Prefix must end at an immutable checkpoint"
    result = {
        "status": "passed",
        "through_update": through,
        "metrics_prefix_sha256": hashlib.sha256(log_prefix).hexdigest(),
        "ppo_work": rows[-1]["ppo"],
        "auxiliary_work": counts,
        "checkpoints": checkpoints,
        "collections": collections,
        "endpoint_manifest_sha256": checkpoints[endpoint]["manifest_sha256"],
    }
    if through == protocol["updates"]:
        assert len(lines) == through
        bindings[str(folder / "metrics.jsonl")] = sha(folder / "metrics.jsonl")
        receipt = read(folder / "complete.json")
        assert (
            receipt["status"] == "complete"
            and receipt["protocol_sha256"] == PROTOCOL_SHA
        )
        assert receipt["manifest_sha256"] == result["endpoint_manifest_sha256"]
        assert receipt["metrics_sha256"] == bindings[str(folder / "metrics.jsonl")]
        assert Path(receipt["endpoint"]) == folder / endpoint
        assert receipt["auxiliary_work"] == counts
        assert receipt["ppo_work"] == checkpoints[endpoint]["ppo_counters"]
        assert receipt["ppo_work"]["decisions"] == protocol["ppo_decisions_per_arm"]
        assert counts["matches"] == protocol["auxiliary_matches_per_arm"]
        bindings[str(folder / "complete.json")] = sha(folder / "complete.json")
    return result


def audit(root):
    root = Path(root)
    assert not (root / "training_audit.json").exists(), "Preserve completed audit"
    assert not (root / "failed.json").exists()
    assert sha(root / "protocol.json") == PROTOCOL_SHA
    protocol, complete = read(root / "protocol.json"), read(root / "complete.json")
    assert (
        complete["status"] == "complete" and complete["protocol_sha256"] == PROTOCOL_SHA
    )
    assert (
        complete["evaluation"] == "not_started"
        and complete["deployment_changed"] is False
    )
    names = [
        f"seed{seed}_{kind}" for seed in (1, 2) for kind in ("none", "win", "score")
    ]
    assert set(complete["arms"]) == set(protocol["arms"]) == set(names)
    suite = (
        ET.parse("runs/score_auxiliary_training_auditor_tests.xml")
        .getroot()
        .find("testsuite")
    )
    assert int(suite.attrib["tests"]) == 10
    assert all(int(suite.attrib[key]) == 0 for key in ("failures", "errors", "skipped"))
    bindings = dict(protocol["inputs"])
    for path, digest in bindings.items():
        assert sha(path) == digest, path
    for path in (
        root / "protocol.json",
        root / "complete.json",
        Path(__file__),
        Path("scripts/audit_score_auxiliary_collection.py"),
        Path("scripts/audit_pretrained_ppo_screen.py"),
        Path("scripts/audit_round_continuation_screen.py"),
        Path("tests/test_audit_score_auxiliary_training.py"),
        Path("runs/score_auxiliary_training_auditor_tests.xml"),
    ):
        bindings[str(path)] = sha(path)
    results = {}
    for name in names:
        results[name] = audit_arm(root, name, protocol, bindings)
        assert complete["arms"][name] == read(root / name / "complete.json")
        print(
            json.dumps(
                {"audited": name, "collections": len(results[name]["collections"])}
            ),
            flush=True,
        )
    for seed in (1, 2):
        for kind in ("win", "score"):
            for stage in ("initial", "update_00000020"):
                left = load(root / f"seed{seed}_none" / stage / "ppo.pt")
                right = load(root / f"seed{seed}_{kind}" / stage / "ppo.pt")
                # Git revision can advance between sequential arms; bound source cannot.
                equal(
                    {k: v for k, v in left.items() if k != "provenance"},
                    {k: v for k, v in right.items() if k != "provenance"},
                )
                assert (
                    left["provenance"]["source_sha256"]
                    == right["provenance"]["source_sha256"]
                )
            equal(
                load(root / f"seed{seed}_none/collection_0001.pt"),
                load(root / f"seed{seed}_{kind}/collection_0001.pt"),
            )
    totals = {
        group: {
            key: sum(r[group][key] for r in results.values())
            for key in ("decisions", "rounds", "matches")
        }
        for group in ("ppo_work", "auxiliary_work")
    }
    assert complete["totals"] == totals
    assert totals["ppo_work"]["decisions"] == 51118080
    assert totals["auxiliary_work"]["matches"] == 76800
    for path, digest in bindings.items():
        assert sha(path) == digest, path
    output = {
        "format": "score_auxiliary_training_audit_v1",
        "status": "passed",
        "protocol_sha256": PROTOCOL_SHA,
        "complete_sha256": sha(root / "complete.json"),
        "auditor_sha256": sha(__file__),
        "inputs": bindings,
        "arms": results,
        "totals": totals,
        "scope": "Stored parameter identities/Adam counts and moments; complete auxiliary native-action replay, reservoir rows, public features and terminal labels. No neural action replay, gradient replay or strength claim. Logged losses checked for finiteness only.",
    }
    with (root / "training_audit.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        stream.write(json.dumps(output, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/score_auxiliary_study"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    audit(args.root)
