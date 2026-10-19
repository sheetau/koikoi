"""Independent scalar and population audit of the final-policy score diagnostic."""

import argparse
import math
import statistics
from pathlib import Path

import numpy as np
import torch

from koikoi.env import BatchEnv
from koikoi.layout import PHASE, TERMINAL
from scripts.run_arbitration_study import read, sha, write

ARMS = ("seed1_fixed", "seed1_calibrated", "seed2_fixed", "seed2_calibrated")
MODELS = ("original", "parent1", "parent2", *ARMS)


def scalar_statistics(scores, legal, temperature):
    """Scalar exp/log arithmetic, independently of the producer vector code."""
    if not math.isfinite(temperature) or temperature <= 0 or len(scores) < 1:
        raise ValueError("Expected positive temperature and nonempty choices")
    entropies, normalized, greedy = [], [], []
    for row, mask in zip(scores, legal, strict=True):
        values = [float(x) for x, allowed in zip(row, mask, strict=True) if allowed]
        if len(values) < 2 or not all(math.isfinite(x) for x in values):
            raise ValueError("Expected at least two finite legal values")
        largest = max(values)
        weights = [math.exp((v - largest) / temperature) for v in values]
        total = math.fsum(weights)
        probs = [w / total for w in weights]
        entropy = -math.fsum(p * math.log(p) for p in probs if p > 0)
        entropies.append(entropy)
        normalized.append(entropy / math.log(len(values)))
        greedy.append(max(probs))
    return {
        "rows": len(scores),
        "mean_entropy": statistics.mean(entropies),
        "mean_normalized_entropy": statistics.mean(normalized),
        "mean_greedy_probability": statistics.mean(greedy),
        "median_greedy_probability": statistics.median(greedy),
    }


def equal_scalars(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_scalars(a[key], b[key])
    elif isinstance(a, float):
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)
    else:
        assert a == b, (a, b)


def independent_phase_report(scores, mask, phases, actions, original, temperature):
    result = {}
    for phase in range(4):
        all_indices = [i for i, p in enumerate(phases) if p == phase]
        indices = [i for i in all_indices if np.count_nonzero(mask[i]) > 1]
        record = {"all_rows": len(all_indices), "unforced_rows": len(indices)}
        if indices:
            gaps = []
            for i in indices:
                sorted_scores = sorted(
                    float(x) for x, yes in zip(scores[i], mask[i], strict=True) if yes
                )
                gaps.append(sorted_scores[-1] - sorted_scores[-2])
            record.update(
                {
                    "mean_top_two_raw_score_gap": statistics.mean(gaps),
                    "changed_greedy_actions": sum(
                        int(actions[i] != original[i]) for i in indices
                    ),
                    "stop_to_continue": sum(
                        int(original[i] == 48 and actions[i] == 49) for i in indices
                    ),
                    "continue_to_stop": sum(
                        int(original[i] == 49 and actions[i] == 48) for i in indices
                    ),
                    "common_temperature": scalar_statistics(
                        scores[indices], mask[indices], 0.05
                    ),
                    "declared_training_temperature": scalar_statistics(
                        scores[indices],
                        mask[indices],
                        temperature if phase == 3 else 0.05,
                    ),
                }
            )
        result[str(phase)] = record
    return result


def audit(root):
    output = root / "independent_audit.json"
    if output.exists():
        raise ValueError("Preserve completed mechanism audit")
    plan, report = read(root / "protocol.json"), read(root / "report.json")
    assert report["status"] == "complete" and plan["models"] == list(MODELS)
    assert (
        sha(root / "protocol.json")
        == "a25292a03b0d343870fd2e3b3c829bd2834aa41a3cdf6f467bb6877fd571c941"
    )
    assert all(report["inputs"].get(k) == v for k, v in plan["inputs"].items())
    assert report["scope"] == plan["scope"]
    assert set(report["results"]) == set(MODELS)
    inputs = dict(report["inputs"])
    for p in (
        Path(__file__),
        Path("tests/test_audit_koi_temperature_mechanism.py"),
        Path("runs/koi_temperature_mechanism_audit_tests.xml"),
        root / "report.json",
        root / "scores.npz",
    ):
        inputs[str(p)] = sha(p)
    for name, digest in inputs.items():
        assert sha(name) == digest, name
    assert sha(root / "scores.npz") == report["scores_sha256"]
    study = Path(plan["study"])
    with np.load("runs/published_batch_validation.npz", allow_pickle=False) as old:
        selected = (old["sources"] >= 0) & (old["sources"] % 4 == 3)
        old_states, old_sources = old["states"][selected], old["sources"][selected]
        old_indices = np.flatnonzero(selected)
    payloads = {
        arm: torch.load(
            study / arm / "final.pt", map_location="cpu", weights_only=False
        )
        for arm in ARMS
    }
    states = np.concatenate(
        [old_states, *[payloads[arm]["env"].numpy() for arm in ARMS]]
    )
    assert len(old_states) == 3258 and len(states) == 3770
    env = BatchEnv(len(states), "cpu", "paper8")
    env.state.copy_(torch.as_tensor(states, dtype=torch.int32))
    _, native_mask = env.published_observe()
    guards = env.solved_actions().numpy()
    mask = native_mask.numpy() & (
        (guards[:, None] < 0) | (np.arange(50)[None] == guards[:, None])
    )
    assert np.array_equal(env.state.numpy(), states)
    groups = np.concatenate(
        [np.zeros(3258, dtype=int), np.repeat(np.arange(1, 5), 128)]
    )
    populations = [
        ("old_validation", groups == 0),
        ("pooled_final_snapshots", groups > 0),
    ]
    populations += [(arm, groups == i) for i, arm in enumerate(ARMS, 1)]
    with np.load(root / "scores.npz", allow_pickle=False) as raw:
        assert set(raw.files) == {
            "mask",
            "phases",
            "guards",
            "old_corpus_indices",
            "population",
            "source_ids",
            *(m + suffix for m in MODELS for suffix in ("_scores", "_actions")),
        }
        for name, expected in {
            "mask": mask,
            "phases": states[:, PHASE],
            "guards": guards,
            "old_corpus_indices": old_indices,
            "population": groups,
            "source_ids": np.concatenate([old_sources, np.full(512, -1)]),
        }.items():
            np.testing.assert_array_equal(raw[name], expected)
        eligible = mask.sum(1) > 1
        for model in MODELS:
            scores, actions = raw[model + "_scores"], raw[model + "_actions"]
            assert scores.shape == (len(states), 50) and scores.dtype == np.float32
            assert actions.shape == (len(states),)
            assert np.isfinite(scores[mask & eligible[:, None]]).all()
            assert np.isneginf(scores[~mask | ~eligible[:, None]]).all()
            expected_actions = np.full(len(states), -1, dtype=np.int64)
            expected_actions[eligible] = (
                (torch.from_numpy(scores[eligible]) / 0.05).argmax(1).numpy()
            )
            np.testing.assert_array_equal(actions, expected_actions)
            temperature = (
                payloads[model]["config"]["koi_temperature"] if model in ARMS else 0.05
            )
            result = report["results"][model]
            assert result["koi_training_temperature"] == temperature
            assert set(result["populations"]) == {name for name, _ in populations}
            for name, choose in populations:
                expected = {
                    "all_rows": int(choose.sum()),
                    "terminal_rows": int(
                        (choose & (states[:, PHASE] == TERMINAL)).sum()
                    ),
                    "forced_rows": int((choose & (mask.sum(1) == 1)).sum()),
                    "phases": independent_phase_report(
                        scores[choose],
                        mask[choose],
                        states[choose, PHASE],
                        actions[choose],
                        raw["original_actions"][choose],
                        temperature,
                    ),
                }
                equal_scalars(expected, result["populations"][name])
    expected_work = {
        "singleton_neural_calls": int(eligible.sum()) * 7,
        "new_matches": 0,
        "simulator_steps": 0,
        "training_updates": 0,
    }
    assert report["work"] == expected_work
    for name, digest in inputs.items():
        assert sha(name) == digest, name
    write(
        output,
        {
            "status": "passed",
            "inputs": inputs,
            "producer_work": expected_work,
            "models": 7,
            "populations_per_model": 6,
            "phases_per_population": 4,
            "new_neural_calls": 0,
            "new_simulator_steps": 0,
            "scope": "Independent population selection, native public masks/guards, recorded-score argmax, scalar entropy/gap/change statistics and accounting. Does not rerun neural score prediction or establish strength.",
        },
    )
    print({"status": "passed", "inputs": len(inputs), "producer_work": expected_work})


if __name__ == "__main__":
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
