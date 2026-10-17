"""Independent stored-score/gradient/statistics and original-weight audit."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.upstream import PublishedAgent

BASE = Path("runs/published_actor_validation")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, atol=1e-12):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=atol)


def main():
    torch.set_num_threads(2)
    report = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    protocol = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    for key in ("protocol", "provenance", "corpus_sha256", "script_sha256"):
        assert report[key] == protocol[key]
    assert sha("scripts/validate_published_actor.py") == report["script_sha256"]
    assert sha(BASE / "data.npz") == report["data_sha256"]
    assert (
        sha("runs/published_batch_validation.npz")
        == report["corpus_sha256"]
        == "e49004ce6795da3c23fbcb036c4d23712b6676a8d96cde1b43f6fca28d17d322"
    )
    assert (
        sha("runs/published_batch_validation.json")
        == "0b040cd31c1d42e945bd233ea94d4f57e06d8d2ffae4a63ffded5c68c8aed544"
    )
    source = hashlib.sha256()
    for path in sorted(Path("koikoi").rglob("*")):
        if path.suffix in (".py", ".h", ".cpp", ".cu"):
            source.update(path.as_posix().encode())
            source.update(path.read_bytes())
    assert report["provenance"]["source_sha256"] == source.hexdigest()
    original = PublishedAgent("wp")
    original_digest = hashlib.sha256()
    parameter_count = 0
    for phase in (0, 1, 3):
        model = original.models[phase]
        parameter_count += sum(p.numel() for p in model.parameters())
        for name, tensor in sorted(model.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            original_digest.update(
                f"{phase}:{name}:{value.dtype}:{tuple(value.shape)}".encode()
            )
            original_digest.update(value.numpy().tobytes())
    checked = {}
    with (
        np.load("runs/published_batch_validation.npz", allow_pickle=False) as corpus,
        np.load(BASE / "data.npz", allow_pickle=False) as raw,
    ):
        states, mask, oracle = corpus["states"], corpus["masks"], corpus["oracle_wp"]
        choices = mask.sum(axis=1) > 1
        indices = np.concatenate(
            [
                np.flatnonzero((states[:, 99] == phase) & choices)[:32]
                for phase in range(4)
            ]
        )
        np.testing.assert_array_equal(raw["gradient_indices"], indices)
        assert len(indices) == 128
        for device, case in report["cases"].items():
            scores, frozen = raw[device + "_scores"], raw[device + "_frozen_scores"]
            assert scores.shape == frozen.shape == oracle.shape == (13029, 50)
            assert np.isfinite(scores[mask]).all() and np.isneginf(scores[~mask]).all()
            assert (
                case["rows"] == len(states)
                and case["parameter_count"] == parameter_count
            )
            assert (
                case["parameters_unchanged_sha256"]
                == case["initialization"]["parameters_sha256"]
                == original_digest.hexdigest()
            )
            for comparison, other in (
                ("frozen_same_batch", frozen),
                ("original_cpu", oracle),
            ):
                error = np.max(np.abs(scores[mask].astype(float) - other[mask]))
                close(case["max_legal_error_vs_" + comparison], error)
                key = (
                    "action_disagreements_vs_frozen"
                    if comparison == "frozen_same_batch"
                    else "action_disagreements_vs_original_cpu"
                )
                assert (
                    case[key]
                    == np.count_nonzero(scores.argmax(1) != other.argmax(1))
                    == 0
                )
            for t, by_phase in case["temperatures"].items():
                for phase in range(4):
                    selected = np.flatnonzero((states[:, 99] == phase) & choices)
                    entropies, greedy = [], []
                    for index in selected:
                        z = scores[index, mask[index]].astype(float) / float(t)
                        weights = np.exp(z - z.max())
                        p = weights / weights.sum()
                        entropies.append(-np.sum(p * np.log(np.maximum(p, 1e-300))))
                        greedy.append(p.max())
                    stored = by_phase[str(phase)]
                    assert stored["rows"] == len(selected)
                    close(stored["mean_entropy"], np.average(entropies))
                    close(stored["mean_greedy_probability"], np.average(greedy))
            # Single phase-balanced probe loss, original greedy labels.
            losses = []
            for index in indices:
                values = scores[index, mask[index]].astype(float)
                maximum = values.max()
                losses.append(
                    maximum
                    + np.log(np.exp(values - maximum).sum())
                    - float(scores[index, oracle[index].argmax()])
                )
            close(case["probe_loss"], np.mean(losses), atol=3e-6)
            for phase, item in case["gradient_checks"].items():
                actual = raw[f"{device}_gradient_{phase}"].astype(float)
                expected = raw[f"{device}_original_gradient_{phase}"].astype(float)
                assert actual.shape == expected.shape == (item["parameters"],)
                assert np.isfinite(actual).all() and np.isfinite(expected).all()
                close(item["norm"], np.sqrt(np.dot(actual, actual)))
                close(item["maximum_error"], np.abs(actual - expected).max())
                np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-6)
            assert len(case["timings"]) == (3 if device == "cuda" else 0)
            for timing in case["timings"]:
                assert len(timing["seconds"]) == 5 and min(timing["seconds"]) > 0
                close(timing["median_seconds"], sorted(timing["seconds"])[2])
                assert (
                    0
                    < timing["baseline_allocated_bytes"]
                    <= timing["peak_allocated_bytes"]
                    <= timing["peak_reserved_bytes"]
                )
            checked[device] = {
                "states": len(states),
                "gradient_probe_rows": len(indices),
                "parameters": parameter_count,
            }
    output = {
        "status": "passed",
        "report_sha256": sha(BASE / "report.json"),
        "auditor_sha256": sha(__file__),
        "cases": checked,
        "scope": "Independent original-parameter digest and stored score/action/gradient-vector/temperature/timing arithmetic. Producer independently compares original-module backward and all native feature digests. Auditor does not rerun inference/backward/timing or certify universal numerical compatibility. No policy update or strength evidence.",
    }
    (BASE / "audit.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
