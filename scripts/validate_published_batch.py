"""Fixed correctness/numerics/latency diagnostic, never a strength evaluation.

Run from the repository root with python -m scripts.validate_published_batch.
Artifacts stay in runs/published_batch_validation.{json,npz}.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from koikoi.agents import solved_koi_action
from koikoi.env import BatchEnv
from koikoi.layout import KOI, PHASE, TERMINAL
from koikoi.reference import ReferenceEnv
from koikoi.search import sample_hidden_batch
from koikoi.train import provenance
from koikoi.upstream import PublishedAgent, reconstruct
from koikoi.upstream_batch import BatchPublishedAgent

ROOTS = Path("runs/rollout_opponent_roots.json")
ROOTS_SHA = "f7d2023685927b884adc9a54e674493b2c6614cffa26c0190e1af63d36fa9dc3"
OUTPUT = Path("runs/published_batch_validation")
PROTOCOL = {
    "purpose": "feature compatibility, numerical differences and latency, not strength",
    "matches": 100,
    "first_seed": 6200100,
    "action_rng": 62021,
    "continue_probability": 0.7,
    "hidden_rng": 62029,
    "hidden_roots_per_phase": 16,
    "worlds_per_hidden_root": 2,
    "batch_size": 256,
    "variants": ["wp", "point"],
    "devices": ["cpu", "cuda"],
    "cpu_threads": 2,
    "tf32": False,
    "feature_benchmark_repeats": 20,
    "policy_benchmark_repeats": 5,
    "benchmark_warmups": 3,
    "benchmark_sizes": [1, 32, 256],
    "extra_mixed_size": 1024,
    "strict_cpu_benchmark_rows_per_phase": 32,
    "strict_cpu_benchmark_repeats": 3,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(state):
    env = ReferenceEnv.__new__(ReferenceEnv)
    env.state = state.copy()
    return env


def digests(features):
    return np.stack(
        [
            np.frombuffer(hashlib.sha256(x.tobytes()).digest(), dtype=np.uint8)
            for x in features
        ]
    )


def corpus():
    rng = np.random.default_rng(PROTOCOL["action_rng"])
    states, sources, terminals, rounds = [], [], [], 0
    for seed in range(
        PROTOCOL["first_seed"], PROTOCOL["first_seed"] + PROTOCOL["matches"]
    ):
        env = ReferenceEnv(seed)
        while env.state[PHASE] != TERMINAL:
            states.append(env.state.copy())
            sources.append(seed)
            action = (
                48 + int(rng.random() < PROTOCOL["continue_probability"])
                if env.state[PHASE] == KOI
                else int(rng.choice(np.flatnonzero(env.legal())))
            )
            result = env.step(action)
            rounds += int(result[1])
        terminals.append(env.state.copy())
    trajectory_count = len(states)
    assert sha(ROOTS) == ROOTS_SHA
    for row in json.loads(ROOTS.read_text(encoding="utf-8"))["roots"]:
        states.append(np.array(row["state"], dtype=np.int32))
        sources.append(-1)
    rng = np.random.default_rng(PROTOCOL["hidden_rng"])
    hidden_pairs = []
    base = np.stack(states)
    for phase in range(4):
        indices = np.flatnonzero(base[:, PHASE] == phase)
        indices = indices[:: max(1, len(indices) // 16)][:16]
        for index in indices:
            for world in sample_hidden_batch(reference(base[index]), rng, 2):
                hidden_pairs.append([int(index), len(states)])
                states.append(world)
                sources.append(-2)
    return {
        "states": np.stack(states),
        "sources": np.array(sources),
        "terminal_states": np.stack(terminals),
        "hidden_pairs": np.array(hidden_pairs),
    }, {"trajectory_decisions": trajectory_count, "rounds": rounds}


def comparison(oracle, actual, masks, states, guards):
    a, b = oracle.argmax(1), actual.argmax(1)
    assisted_a = np.where(guards >= 0, guards, a)
    assisted_b = np.where(guards >= 0, guards, b)
    indices = np.flatnonzero(a != b)
    ordered = np.sort(oracle, axis=1)
    return {
        "states": len(a),
        "action_disagreements": len(indices),
        "assisted_action_disagreements": int(
            np.count_nonzero(assisted_a != assisted_b)
        ),
        "max_legal_logit_error": float(np.max(np.abs(actual[masks] - oracle[masks]))),
        "by_phase": {
            str(p): {
                "states": int(np.count_nonzero(states[:, PHASE] == p)),
                "action_disagreements": int(
                    np.count_nonzero((a != b) & (states[:, PHASE] == p))
                ),
            }
            for p in range(4)
        },
        "disagreement_indices": indices.tolist(),
        "oracle_margins_at_disagreements": (
            ordered[indices, -1] - ordered[indices, -2]
        ).tolist(),
    }


def timing(fn, device, repeats):
    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    for _ in range(PROTOCOL["benchmark_warmups"]):
        fn()
    sync()
    elapsed = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        elapsed.append(time.perf_counter() - start)
    return {"seconds": elapsed, "median_seconds": float(np.median(elapsed))}


def benchmark(states, agents, original):
    result = []
    for phase in [0, 1, 2, 3, "mixed"]:
        rows = states if phase == "mixed" else states[states[:, PHASE] == phase]
        sizes = PROTOCOL["benchmark_sizes"] + ([1024] if phase == "mixed" else [])
        for size in sizes:
            selected = rows[np.arange(size) % len(rows)]
            for device, agent in agents.items():
                env = BatchEnv(size, device)
                env.state.copy_(torch.tensor(selected, device=device))
                for name, fn, repeats in [
                    ("features", env.published_observe, 20),
                    ("policy", lambda agent=agent, env=env: agent.act(env), 5),
                ]:
                    row = {
                        "phase": phase,
                        "batch": size,
                        "device": device,
                        "operation": name,
                    }
                    row.update(timing(fn, device, repeats))
                    row["states_per_second"] = size / row["median_seconds"]
                    result.append(row)
        if phase != "mixed":
            references = [reference(s) for s in rows[:32]]
            row = {
                "phase": phase,
                "batch": len(references),
                "device": "cpu",
                "operation": "strict_policy_serial",
            }
            row.update(
                timing(
                    lambda references=references: [original.act(e) for e in references],
                    "cpu",
                    3,
                )
            )
            row["states_per_second"] = len(references) / row["median_seconds"]
            result.append(row)
        print(f"benchmark phase {phase} complete", flush=True)
    return result


@torch.no_grad()
def main():
    torch.set_num_threads(PROTOCOL["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available()
    assert not OUTPUT.with_suffix(".json").exists(), "Preserve the completed artifact"
    started = time.perf_counter()
    data, counters = corpus()
    states, n = data["states"], len(data["states"])
    data["masks"] = np.stack([reference(s).legal() for s in states]).astype(bool)
    data["guards"] = np.array(
        [-1 if (a := solved_koi_action(reference(s))) is None else a for s in states],
        dtype=np.int32,
    )
    originals = {v: PublishedAgent(v) for v in PROTOCOL["variants"]}
    for v in originals:
        data[f"oracle_{v}"] = np.full((n, 50), -np.inf, np.float32)
    data["oracle_features_sha256"] = np.empty((n, 32), np.uint8)
    feature_strides = {}
    for index, state in enumerate(states):
        g = reconstruct(reference(state))
        feature = g.feature_tensor
        phase = int(state[PHASE])
        feature_strides.setdefault(str(phase), list(feature.stride()))
        padded = np.zeros((300, 50), np.float32)
        padded[:, : feature.shape[1]] = feature.numpy()
        data["oracle_features_sha256"][index] = digests(padded[None])[0]
        for v, agent in originals.items():
            output = agent.models[phase](feature[None])[0].numpy()
            target = data[f"oracle_{v}"][index]
            if phase == KOI:
                target[48:] = output
            else:
                target[:48] = output
            target[~data["masks"][index]] = -np.inf
        if index % 2000 == 0:
            print(f"oracle {index}/{n}", flush=True)
    metadata, comparisons, wp_agents = {}, {}, {}
    for device in PROTOCOL["devices"]:
        agents = {v: BatchPublishedAgent(v, device) for v in originals}
        wp_agents[device] = agents["wp"]
        for v, agent in agents.items():
            metadata[v] = {
                "revision": agent.upstream_revision,
                "models": agent.model_hashes,
            }
            data[f"{device}_{v}"] = np.empty((n, 50), np.float32)
        hashes = np.empty((n, 32), np.uint8)
        for start in range(0, n, PROTOCOL["batch_size"]):
            stop = min(n, start + PROTOCOL["batch_size"])
            env = BatchEnv(stop - start, device)
            env.state.copy_(torch.tensor(states[start:stop], device=device))
            features, masks = env.published_observe()
            hashes[start:stop] = digests(features.cpu().numpy())
            np.testing.assert_array_equal(masks.cpu(), data["masks"][start:stop])
            np.testing.assert_array_equal(
                env.solved_actions().cpu(), data["guards"][start:stop]
            )
            for v, agent in agents.items():
                data[f"{device}_{v}"][start:stop] = agent.logits(env).cpu().numpy()
            np.testing.assert_array_equal(env.state.cpu(), states[start:stop])
        data[f"{device}_features_sha256"] = hashes
        np.testing.assert_array_equal(hashes, data["oracle_features_sha256"])
        env = BatchEnv(len(data["terminal_states"]), device)
        env.state.copy_(torch.tensor(data["terminal_states"], device=device))
        features, masks = env.published_observe()
        assert not features.any() and not masks.any()
        assert (agents["wp"].act(env) == -1).all()
        for v in agents:
            comparisons[f"{device}_{v}"] = comparison(
                data[f"oracle_{v}"],
                data[f"{device}_{v}"],
                data["masks"],
                states,
                data["guards"],
            )
        print(f"{device} compatibility complete: {comparisons}", flush=True)
    pairs = data["hidden_pairs"]
    np.testing.assert_array_equal(
        data["oracle_features_sha256"][pairs[:, 0]],
        data["oracle_features_sha256"][pairs[:, 1]],
    )
    np.savez_compressed(OUTPUT.with_suffix(".npz"), **data)
    timings = benchmark(states, wp_agents, originals["wp"])
    report = {
        "protocol": PROTOCOL,
        "provenance": provenance(),
        "script_sha256": sha(__file__),
        "cached_roots_sha256": sha(ROOTS),
        "raw_sha256": sha(OUTPUT.with_suffix(".npz")),
        "teacher": metadata,
        "runtime": {
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "threads": torch.get_num_threads(),
            "dtype": "float32",
            "original_feature_strides": feature_strides,
        },
        "counters": {
            **counters,
            "states": n,
            "cached_roots": 40,
            "hidden_variants": len(pairs),
            "terminal_checks_per_device": 100,
            "training_matches": 0,
            "strength_matches": 0,
        },
        "comparisons": comparisons,
        "timings_wp_unassisted": timings,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": "Finite diagnostic corpus; latency excludes construction/transfers but includes feature extraction, grouping and masks for policy calls. No action-identity certificate or strength claim. Strict comparator unchanged.",
    }
    OUTPUT.with_suffix(".json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {"output": str(OUTPUT), "states": n, "elapsed": report["elapsed_seconds"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
