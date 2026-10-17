"""Fixed zero-update pretrained-actor compatibility and gradient feasibility."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from koikoi.env import BatchEnv
from koikoi.layout import PHASE
from koikoi.published_actor import PublishedActor
from koikoi.train import provenance
from koikoi.upstream_batch import BatchPublishedAgent

BASE = Path("runs/published_actor_validation")
CORPUS = Path("runs/published_batch_validation.npz")
CORPUS_SHA = "e49004ce6795da3c23fbcb036c4d23712b6676a8d96cde1b43f6fca28d17d322"
PROTOCOL = {
    "variant": "wp",
    "devices": ["cpu", "cuda"],
    "comparison_batch": 128,
    "gradient_rows_per_public_phase": 32,
    "gradient_probe_temperature": 1.0,
    "temperature_descriptions": [1.0, 0.25, 0.1, 0.05],
    "benchmark_batch_sizes": [32, 64, 128],
    "benchmark_warmups": 3,
    "benchmark_repeats": 5,
    "new_matches": 0,
    "policy_updates": 0,
    "note": "Reused audited states. Cross entropy against original greedy actions is a gradient probe only. No optimizer or stochastic training policy selected. Dropout/TF32/autocast disabled.",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_rows(states, legal, per_phase):
    groups = [
        np.flatnonzero((states[:, PHASE] == p) & (legal.sum(1) > 1))[:per_phase]
        for p in range(4)
    ]
    assert all(len(g) == per_phase for g in groups)
    return np.concatenate(groups)


def description(scores, legal, phases, temperature):
    result = {}
    for phase in range(4):
        chosen = (phases == phase) & (legal.sum(1) > 1)
        z = scores[chosen].astype(np.float64) / temperature
        z -= z.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        p = np.exp(logp)
        entropy = -(p * np.where(np.isfinite(logp), logp, 0)).sum(1)
        result[str(phase)] = {
            "rows": int(chosen.sum()),
            "mean_greedy_probability": float(p.max(1).mean()),
            "mean_entropy": float(entropy.mean()),
        }
    return result


def gradient_vector(model):
    values = []
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        values.append(parameter.grad.detach().flatten())
    vector = torch.cat(values)
    assert vector.square().sum() > 0
    return vector


def backward(actor, features, phases, mask, target):
    actor.zero_grad(set_to_none=True)
    F.cross_entropy(actor(features, phases, mask), target).backward()


def main():
    assert sha(CORPUS) == CORPUS_SHA
    assert (
        sha(CORPUS.with_suffix(".json"))
        == "0b040cd31c1d42e945bd233ea94d4f57e06d8d2ffae4a63ffded5c68c8aed544"
    )
    corpus_report = json.loads(CORPUS.with_suffix(".json").read_text(encoding="utf-8"))
    BASE.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with np.load(CORPUS, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    states, legal = data["states"], data["masks"]
    report = {
        "protocol": PROTOCOL,
        "script_sha256": sha(__file__),
        "provenance": provenance(),
        "corpus_sha256": CORPUS_SHA,
        "cases": {},
    }
    (BASE / "protocol.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    raw = {}
    for device in PROTOCOL["devices"]:
        actor = PublishedActor.from_published("wp", device)
        original = BatchPublishedAgent("wp", device)
        initial = actor.checkpoint()
        assert (
            initial["initialization"]["model_hashes"]
            == corpus_report["teacher"]["wp"]["models"]
        )
        assert (
            initial["initialization"]["upstream_revision"]
            == corpus_report["teacher"]["wp"]["revision"]
        )
        assert (
            initial["current"]["parameters_sha256"]
            == original.checkpoint()["parameters_sha256"]
        )
        outputs, originals = [], []
        for start in range(0, len(states), 128):
            batch = states[start : start + 128]
            env = BatchEnv(len(batch), device)
            env.state.copy_(torch.tensor(batch, device=device))
            features, mask = env.published_observe()
            for i, feature in enumerate(features.cpu().numpy()):
                assert (
                    hashlib.sha256(feature.tobytes()).digest()
                    == data["oracle_features_sha256"][start + i].tobytes()
                )
            np.testing.assert_array_equal(mask.cpu(), legal[start : start + len(batch)])
            expected = original.logits(env)
            actual = actor(features, env.state[:, PHASE], mask)
            assert actual.requires_grad
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            outputs.append(actual.detach().cpu().numpy())
            originals.append(expected.cpu().numpy())
            np.testing.assert_array_equal(env.state.cpu(), batch)
        outputs, originals = np.concatenate(outputs), np.concatenate(originals)
        del env, features, mask, actual, expected
        raw[device + "_scores"], raw[device + "_frozen_scores"] = outputs, originals
        case = {
            "initialization": initial["initialization"],
            "parameter_count": sum(p.numel() for p in actor.parameters()),
            "rows": len(states),
            "max_legal_error_vs_frozen_same_batch": float(
                np.max(np.abs(outputs[legal] - originals[legal]))
            ),
            "action_disagreements_vs_frozen": int(
                np.count_nonzero(outputs.argmax(1) != originals.argmax(1))
            ),
            "max_legal_error_vs_original_cpu": float(
                np.max(np.abs(outputs[legal] - data["oracle_wp"][legal]))
            ),
            "action_disagreements_vs_original_cpu": int(
                np.count_nonzero(outputs.argmax(1) != data["oracle_wp"].argmax(1))
            ),
            "gradient_checks": {},
            "temperatures": {
                str(t): description(outputs, legal, states[:, PHASE], t)
                for t in PROTOCOL["temperature_descriptions"]
            },
        }
        pairs = data["hidden_pairs"]
        np.testing.assert_allclose(
            outputs[pairs[:, 0]], outputs[pairs[:, 1]], rtol=1e-5, atol=2e-5
        )
        terminal = BatchEnv(len(data["terminal_states"]), device)
        terminal.state.copy_(torch.tensor(data["terminal_states"], device=device))
        terminal_features, terminal_mask = terminal.published_observe()
        assert torch.isneginf(
            actor(terminal_features, terminal.state[:, PHASE], terminal_mask)
        ).all()
        indices = selected_rows(states, legal, 32)
        raw["gradient_indices"] = indices
        env = BatchEnv(len(indices), device)
        env.state.copy_(torch.tensor(states[indices], device=device))
        features, mask = env.published_observe()
        phases = env.state[:, PHASE]
        labels = torch.tensor(data["oracle_wp"][indices].argmax(1), device=device)
        actor.zero_grad(set_to_none=True)
        loss = F.cross_entropy(actor(features, phases, mask), labels)
        loss.backward()
        case["probe_loss"] = float(loss.detach())
        # Original modules, separate phase-specific losses and indexing: no actor helper.
        for phase, model in original.models.items():
            model.requires_grad_(True).zero_grad(set_to_none=True)
            selected = (phases == phase) | (
                (phases == 2)
                if phase == 1
                else torch.zeros_like(phases, dtype=torch.bool)
            )
            width = 50 if phase == 3 else 48
            logits = model(features[selected, :, :width].contiguous())
            phase_mask = mask[selected, 48:] if phase == 3 else mask[selected, :48]
            target = labels[selected] - (48 if phase == 3 else 0)
            logp = logits.masked_fill(~phase_mask, -torch.inf).log_softmax(-1)
            oracle_loss = -logp[
                torch.arange(len(target), device=device), target
            ].sum() / len(indices)
            oracle_loss.backward()
            actual, expected = (
                gradient_vector(actor.models[str(phase)]),
                gradient_vector(model),
            )
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)
            case["gradient_checks"][str(phase)] = {
                "parameters": actual.numel(),
                "norm": float(actual.double().norm()),
                "maximum_error": float((actual - expected).abs().max()),
            }
            (
                raw[f"{device}_gradient_{phase}"],
                raw[f"{device}_original_gradient_{phase}"],
            ) = actual.cpu().numpy(), expected.cpu().numpy()
            model.requires_grad_(False).zero_grad(set_to_none=True)
        actor.zero_grad(set_to_none=True)
        del actual, expected, loss, oracle_loss, logp, logits
        case["timings"] = []
        if device == "cuda":
            for size in PROTOCOL["benchmark_batch_sizes"]:
                selected = selected_rows(states, legal, size // 4)
                bench = BatchEnv(size, device)
                bench.state.copy_(torch.tensor(states[selected], device=device))
                x, mask_b = bench.published_observe()
                phase_b = bench.state[:, PHASE]
                target = torch.tensor(
                    data["oracle_wp"][selected].argmax(1), device=device
                )

                for _ in range(3):
                    backward(actor, x, phase_b, mask_b, target)
                torch.cuda.synchronize()
                actor.zero_grad(set_to_none=True)
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                seconds = []
                for _ in range(5):
                    start = time.perf_counter()
                    backward(actor, x, phase_b, mask_b, target)
                    torch.cuda.synchronize()
                    seconds.append(time.perf_counter() - start)
                case["timings"].append(
                    {
                        "batch": size,
                        "seconds": seconds,
                        "median_seconds": float(np.median(seconds)),
                        "baseline_allocated_bytes": baseline,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "scope": "Resident features: forward, cross entropy and backward; no collection, critic, Adam, transfers or policy updates.",
                    }
                )
                actor.zero_grad(set_to_none=True)
                del bench, x, mask_b, phase_b, target
        current = actor.checkpoint()["current"]["parameters_sha256"]
        assert (
            current
            == initial["current"]["parameters_sha256"]
            == original.checkpoint()["parameters_sha256"]
        )
        case["parameters_unchanged_sha256"] = current
        report["cases"][device] = case
        print(
            json.dumps(
                {
                    "device": device,
                    "rows": len(states),
                    "action_disagreements_vs_original_cpu": case[
                        "action_disagreements_vs_original_cpu"
                    ],
                    "gradient_checks": case["gradient_checks"],
                    "timings": case["timings"],
                }
            ),
            flush=True,
        )
        del (
            actor,
            original,
            initial,
            terminal,
            terminal_features,
            terminal_mask,
            env,
            features,
            mask,
        )
    np.savez_compressed(BASE / "data.npz", **raw)
    report.update(
        status="complete",
        data_sha256=sha(BASE / "data.npz"),
        limitations="Finite-corpus numerical compatibility and gradient/resource feasibility only. No simulation, optimizer step, critic, learning throughput, stochastic policy choice or playing-strength evidence. Original published weights/source remain a disclosed dependency.",
    )
    (BASE / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
