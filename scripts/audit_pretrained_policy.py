"""Independent scalar arithmetic/identity audit; does not rerun neural inference."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.audit_pretrained_ppo_screen import tensor_digest

BASE = Path("runs/pretrained_policy_diagnostic")
CORPUS_SHA = "e49004ce6795da3c23fbcb036c4d23712b6676a8d96cde1b43f6fca28d17d322"
SCREEN_AUDIT_SHA = "643432440c451836559ab398d45a095f21043c4f04b60e26ec470dfa09d8c982"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def recompute(scores, mask, original, temperature):
    maximums, original_probs, entropies, gaps, chosen = [], [], [], [], []
    for row, legal, reference in zip(scores, mask, original):
        ids = [i for i, valid in enumerate(legal) if valid]
        assert len(ids) > 1 and int(reference) in ids and temperature > 0
        values = [float(row[i]) for i in ids]
        assert all(math.isfinite(v) for v in values)
        maximum = max(values)
        unnormalized = [math.exp((v - maximum) / temperature) for v in values]
        total = math.fsum(unnormalized)
        probabilities = [v / total for v in unnormalized]
        chosen.append(ids[values.index(maximum)])
        maximums.append(max(probabilities))
        original_probs.append(probabilities[ids.index(int(reference))])
        entropies.append(-math.fsum(p * math.log(p) for p in probabilities if p))
        ordered = sorted(values)
        gaps.append(ordered[-1] - ordered[-2])
    n = len(chosen)
    ordered = sorted(maximums)
    quantiles = []
    for fraction in (0.1, 0.5, 0.9):
        index = (n - 1) * fraction
        lower, upper = math.floor(index), math.ceil(index)
        quantiles.append(
            ordered[lower] + (index - lower) * (ordered[upper] - ordered[lower])
        )
    return {
        "rows": n,
        "changed_greedy_actions": sum(int(i) != j for i, j in zip(original, chosen)),
        "mean_greedy_probability": math.fsum(maximums) / n,
        "greedy_probability_quantiles_10_50_90": quantiles,
        "mean_initial_greedy_probability": math.fsum(original_probs) / n,
        "mean_entropy": math.fsum(entropies) / n,
        "mean_top_two_raw_score_gap": math.fsum(gaps) / n,
        "initial_stop_to_continue": sum(
            int(i) == 48 and j == 49 for i, j in zip(original, chosen)
        ),
        "initial_continue_to_stop": sum(
            int(i) == 49 and j == 48 for i, j in zip(original, chosen)
        ),
    }


def main():
    protocol = read(BASE / "protocol.json")
    report = read(BASE / "report.json")
    corpus_path = Path("runs/published_batch_validation.npz")
    screen = Path("runs/pretrained_ppo_screen")
    assert sha(corpus_path) == protocol["corpus_sha256"] == CORPUS_SHA
    assert (
        sha(screen / "independent_audit.json")
        == protocol["screen_audit_sha256"]
        == SCREEN_AUDIT_SHA
    )
    assert protocol["temperatures"] == [0.05, 0.025, 0.01]
    assert protocol["devices"] == ["cpu", "cuda"] and protocol["batch_size"] == 128
    assert protocol["new_matches"] == protocol["policy_updates"] == 0
    assert protocol["single_row_probe_per_phase"] == 16
    assert protocol["script_sha256"] == sha("scripts/diagnose_pretrained_policy.py")
    assert report["protocol_sha256"] == sha(BASE / "protocol.json")
    assert report["scores_sha256"] == sha(BASE / "scores.npz")
    assert report["status"] == "complete"
    digest = hashlib.sha256()
    for path in sorted(Path("koikoi").rglob("*")):
        if path.suffix in (".py", ".h", ".cpp", ".cu"):
            digest.update(path.as_posix().encode())
            digest.update(path.read_bytes())
    assert (
        digest.hexdigest()
        == protocol["provenance"]["source_sha256"]
        == "4b4360ed783c1d12cb5258e255f4b85f2e30dfa2e111fd8a078cf3439ce612cb"
    )
    with np.load(corpus_path, allow_pickle=False) as archive:
        corpus = {k: archive[k] for k in archive.files}
    with np.load(BASE / "scores.npz", allow_pickle=False) as archive:
        raw = {k: archive[k] for k in archive.files}
    mask = corpus["masks"].copy()
    for i, guard in enumerate(corpus["guards"]):
        if guard >= 0:
            assert mask[i, guard]
            mask[i] = False
            mask[i, guard] = True
    np.testing.assert_array_equal(raw["mask"], mask)
    np.testing.assert_array_equal(raw["phases"], corpus["states"][:, 99])
    np.testing.assert_array_equal(raw["sources"], corpus["sources"])
    selected = (corpus["sources"] >= 0) & (mask.sum(1) > 1)
    probes = [
        i
        for phase in range(4)
        for i in [
            j for j in range(len(mask)) if selected[j] and raw["phases"][j] == phase
        ][:16]
    ]
    np.testing.assert_array_equal(raw["probe_indices"], probes)
    screen_audit = read(screen / "independent_audit.json")
    cases = ("initial", "seed6681001", "seed6691001")
    assert list(protocol["checkpoints"]) == list(cases)
    results = {}
    for name in cases:
        relative, expected_hash = protocol["checkpoints"][name]
        checkpoint_path = screen / relative
        expected_path = (
            "initial/initial.pt" if name == "initial" else f"{name}/update_00000520.pt"
        )
        assert relative == expected_path and sha(checkpoint_path) == expected_hash
        label = name if name == "initial" else name + ":520"
        assert expected_hash == screen_audit["checkpoints"][label]["sha256"]
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        parameters = tensor_digest(payload["actor"]["current"]["weights"])
        assert parameters == screen_audit["checkpoints"][label]["parameters_sha256"]
        for device in ("cpu", "cuda"):
            key = device + "_" + name
            scores = raw[key]
            assert scores.shape == (13029, 50)
            assert np.isfinite(scores[corpus["masks"]]).all()
            assert np.isneginf(scores[~corpus["masks"]]).all()
            assert report["cases"][key]["parameters_sha256"] == parameters
            if name == "initial":
                np.testing.assert_array_equal(
                    scores.argmax(1), corpus["oracle_wp"].argmax(1)
                )
            original = np.where(mask, raw[device + "_initial"], -np.inf).argmax(1)
            results[key] = {}
            for t in protocol["temperatures"]:
                results[key][str(t)] = {}
                for phase in range(4):
                    chosen = selected & (raw["phases"] == phase)
                    result = recompute(
                        scores[chosen], mask[chosen], original[chosen], t
                    )
                    stored = report["cases"][key]["temperatures"][str(t)][str(phase)]
                    assert result.keys() == stored.keys()
                    for field in result:
                        np.testing.assert_allclose(
                            result[field], stored[field], rtol=0, atol=1e-12
                        )
                    results[key][str(t)][str(phase)] = result
        cpu = np.where(mask, raw["cpu_" + name], -np.inf).argmax(1)
        cuda = np.where(mask, raw["cuda_" + name], -np.inf).argmax(1)
        single_scores = raw[name + "_single_scores"]
        assert single_scores.shape == (64, 50)
        single = np.where(mask[probes], single_scores, -np.inf).argmax(1)
        assert report["compatibility"][name] == {
            "cpu_cuda_greedy_disagreements": int((cpu != cuda).sum()),
            "cpu_single_probe_disagreements": int((cpu[probes] != single).sum()),
        }
    output = {
        "status": "passed",
        "scope": "actual file/tensor identities and stored score arithmetic; no inference rerun, no strength or causal claim",
        "report_sha256": sha(BASE / "report.json"),
        "data_sha256": sha(BASE / "scores.npz"),
        "auditor_sha256": sha(__file__),
        "tensor_helper_sha256": sha("scripts/audit_pretrained_ppo_screen.py"),
        "statistics_groups_checked": 72,
        "cases": results,
    }
    (BASE / "audit.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {"status": "passed", "groups": 72, "report_sha256": output["report_sha256"]}
        )
    )


if __name__ == "__main__":
    main()
