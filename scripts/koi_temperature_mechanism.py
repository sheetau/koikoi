"""Final-policy score diagnostics on old public inputs and saved training snapshots."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from koikoi.env import BatchEnv
from koikoi.layout import PHASE, TERMINAL
from koikoi.published_actor import PublishedActor
from koikoi.published_opponent import runtime_identity
from scripts.phase_exploration_calibration import choice_statistics
from scripts.run_arbitration_study import PARENTS, read, sha, write

CORPUS = Path("runs/published_batch_validation.npz")
CORPUS_SHA = "e49004ce6795da3c23fbcb036c4d23712b6676a8d96cde1b43f6fca28d17d322"
ARMS = tuple(f"seed{p}_{mode}" for p in (1, 2) for mode in ("fixed", "calibrated"))


def checked(inputs):
    for name, digest in inputs.items():
        assert sha(name) == digest, name


def public_inputs(states):
    """Raw states are owned by the extractor; only public actor tensors leave it."""
    env = BatchEnv(len(states), "cpu", "paper8")
    env.state.copy_(torch.as_tensor(states, dtype=torch.int32))
    before = env.state.clone()
    features, raw_mask = env.published_observe()
    guards = env.solved_actions()
    mask = raw_mask & (
        (guards[:, None] < 0) | (torch.arange(50)[None] == guards[:, None])
    )
    assert torch.equal(env.state, before)
    return features, env.state[:, PHASE].clone(), mask, raw_mask, guards


@torch.no_grad()
def score_public(actor, features, phases, mask):
    """Use singleton CPU scores and the deployment's float32 scalar division."""
    assert features.device.type == phases.device.type == mask.device.type == "cpu"
    assert not actor.training and all(p.requires_grad for p in actor.parameters())
    scores = torch.full((len(features), 50), -torch.inf, dtype=torch.float32)
    actions = torch.full((len(features),), -1, dtype=torch.int64)
    eligible = torch.where(mask.sum(1) > 1)[0].tolist()
    rng = torch.get_rng_state().clone()
    digest = actor.checkpoint()["current"]["parameters_sha256"]
    for i in eligible:
        row = actor(features[i : i + 1], phases[i : i + 1], mask[i : i + 1])
        scores[i] = row[0]
        actions[i] = (row / 0.05).argmax(1)[0]
    assert torch.equal(rng, torch.get_rng_state())
    assert actor.checkpoint()["current"]["parameters_sha256"] == digest
    assert bool(torch.isfinite(scores[mask & (mask.sum(1) > 1)[:, None]]).all())
    return scores.numpy(), actions.numpy(), len(eligible)


def describe(scores, mask, phases, actions, original_actions, temperature):
    result = {}
    for phase in range(4):
        all_rows = phases == phase
        selected = all_rows & (mask.sum(1) > 1)
        record = {"all_rows": int(all_rows.sum()), "unforced_rows": int(selected.sum())}
        if selected.any():
            raw, legal = scores[selected], mask[selected]
            ordered = np.sort(np.where(legal, raw.astype(float), -np.inf), axis=1)
            current, original = actions[selected], original_actions[selected]
            assert (current >= 0).all() and legal[np.arange(len(raw)), current].all()
            record.update(
                {
                    "mean_top_two_raw_score_gap": float(
                        (ordered[:, -1] - ordered[:, -2]).mean()
                    ),
                    "changed_greedy_actions": int((current != original).sum()),
                    "stop_to_continue": int(((original == 48) & (current == 49)).sum()),
                    "continue_to_stop": int(((original == 49) & (current == 48)).sum()),
                    "common_temperature": choice_statistics(raw, legal, 0.05),
                    "declared_training_temperature": choice_statistics(
                        raw, legal, temperature if phase == 3 else 0.05
                    ),
                }
            )
        result[str(phase)] = record
    return result


def prepare(root, study):
    if root.exists():
        raise ValueError("Use a fresh diagnostic directory")
    assert sha(CORPUS) == CORPUS_SHA
    assert (
        sha(study / "protocol.json")
        == "b9075ac261ce6a3e681a64fa7845f031850ef20fba60f618435b3ad108a74d19"
    )
    inputs = dict(read(study / "protocol.json")["inputs"])
    paths = [
        Path(__file__),
        CORPUS,
        study / "protocol.json",
        Path("docs/KOI_TEMPERATURE_MECHANISM.rst"),
        Path("tests/test_koi_temperature_mechanism.py"),
        Path("runs/koi_temperature_mechanism_tests.xml"),
        Path("scripts/phase_exploration_calibration.py"),
    ]
    inputs.update({str(p): sha(p) for p in paths})
    checked(inputs)
    root.mkdir(parents=True, exist_ok=False)
    write(
        root / "protocol.json",
        {
            "study": str(study),
            "inputs": inputs,
            "arms": ARMS,
            "models": ["original", "parent1", "parent2", *ARMS],
            "scope": "Descriptive fixed-input score diagnostic; final endpoints only; no outcome labels, new trajectories, selection or training.",
            "corpus_selection": "nonnegative source ID modulo4 ==3; all rows retained for coverage, only unforced summarized",
            "snapshots": "all128 lanes at each final endpoint; common pooled population plus each source arm",
            "execution": "singleton CPU float32, two threads, eval, requires_grad unchanged, no_grad",
        },
    )


def run(root):
    if (root / "report.json").exists() or (root / "scores.npz").exists():
        raise ValueError("Preserve completed or partial diagnostics")
    plan = read(root / "protocol.json")
    checked(plan["inputs"])
    study = Path(plan["study"])
    paths = {
        "original": study / "seed1_fixed/initial.pt",
        **{f"parent{i}": Path(p) for i, (p, _) in enumerate(PARENTS, 1)},
        **{arm: study / arm / "final.pt" for arm in ARMS},
    }
    receipts = {arm: read(study / arm / "complete.json") for arm in ARMS}
    assert all(
        r["status"] == "complete" and r["counters"]["updates"] == 520
        for r in receipts.values()
    )
    assert all(sha(paths[arm]) == receipts[arm]["checkpoint_sha256"] for arm in ARMS)
    payloads = {
        name: torch.load(p, map_location="cpu", weights_only=False)
        for name, p in paths.items()
    }
    for arm in ARMS:
        assert payloads[arm]["counters"] == receipts[arm]["counters"]
        assert payloads[arm]["env"].shape == (128, 208)
    inputs = {**plan["inputs"], **{str(p): sha(p) for p in paths.values()}}
    inputs.update(
        {
            str(study / arm / "complete.json"): sha(study / arm / "complete.json")
            for arm in ARMS
        }
    )
    inputs[str(root / "protocol.json")] = sha(root / "protocol.json")
    with np.load(CORPUS, allow_pickle=False) as old:
        selected = (old["sources"] >= 0) & (old["sources"] % 4 == 3)
        indices = np.flatnonzero(selected)
        states = np.concatenate(
            [old["states"][selected], *[payloads[arm]["env"].numpy() for arm in ARMS]]
        )
        features, phases, mask, raw_mask, guards = public_inputs(states)
        np.testing.assert_array_equal(
            raw_mask[: len(indices)].numpy(), old["masks"][selected]
        )
        np.testing.assert_array_equal(
            guards[: len(indices)].numpy(), old["guards"][selected]
        )
        for i, expected in enumerate(old["oracle_features_sha256"][selected]):
            assert (
                hashlib.sha256(features[i].numpy().tobytes()).digest()
                == expected.tobytes()
            )
        source_ids = np.concatenate(
            [old["sources"][selected], np.full(512, -1, dtype=np.int64)]
        )
    groups = np.concatenate(
        [np.zeros(len(indices), dtype=np.int64), np.repeat(np.arange(1, 5), 128)]
    )
    raw = {
        "mask": mask.numpy(),
        "phases": phases.numpy(),
        "source_ids": source_ids,
        "population": groups,
        "old_corpus_indices": indices,
        "guards": guards.numpy(),
    }
    results, calls = {}, 0
    for name in plan["models"]:
        payload = payloads[name]
        actor = PublishedActor(payload["actor"], "cpu").eval()
        values, actions, count = score_public(actor, features, phases, mask)
        raw[name + "_scores"], raw[name + "_actions"] = values, actions
        calls += count
        temperature = payload["config"].get("koi_temperature", 0.05)
        result = {}
        populations = [
            ("old_validation", groups == 0),
            ("pooled_final_snapshots", groups > 0),
        ]
        populations += [(arm, groups == i) for i, arm in enumerate(ARMS, 1)]
        for population, choose in populations:
            result[population] = {
                "all_rows": int(choose.sum()),
                "terminal_rows": int((choose & (raw["phases"] == TERMINAL)).sum()),
                "forced_rows": int((choose & (raw["mask"].sum(1) == 1)).sum()),
                "phases": describe(
                    values[choose],
                    raw["mask"][choose],
                    raw["phases"][choose],
                    actions[choose],
                    raw["original_actions"][choose],
                    temperature,
                ),
            }
        results[name] = {"koi_training_temperature": temperature, "populations": result}
        print({"scored_model": name, "singleton_calls": count}, flush=True)
    checked(inputs)
    np.savez_compressed(root / "scores.npz", **raw)
    write(
        root / "report.json",
        {
            "status": "complete",
            "inputs": inputs,
            "results": results,
            "scores_sha256": sha(root / "scores.npz"),
            "runtime": runtime_identity(torch.device("cpu")),
            "work": {
                "singleton_neural_calls": calls,
                "new_matches": 0,
                "simulator_steps": 0,
                "training_updates": 0,
            },
            "scope": plan["scope"],
        },
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument(
        "--study", type=Path, default=Path("runs/koi_temperature_study")
    )
    args = parser.parse_args()
    prepare(args.output, args.study) if args.prepare else run(args.output)
