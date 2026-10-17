"""Fixed saved-state score-temperature and greedy-policy diagnostic, not strength."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.env import BatchEnv
from koikoi.published_actor import PublishedActor
from koikoi.published_opponent import runtime_identity
from koikoi.train import provenance

BASE = Path("runs/pretrained_policy_diagnostic")
CORPUS = Path("runs/published_batch_validation.npz")
CORPUS_SHA = "e49004ce6795da3c23fbcb036c4d23712b6676a8d96cde1b43f6fca28d17d322"
CASES = {
    "initial": (
        "initial/initial.pt",
        "290e301702428a1762ffec2004ec9e9fafbb35a7d68a79e849c8e47a1126893e",
    ),
    "seed6681001": (
        "seed6681001/update_00000520.pt",
        "9ae3911b436e67f219b016e25bf598cf1ad0734a0420332485b5b05d9ce166ff",
    ),
    "seed6691001": (
        "seed6691001/update_00000520.pt",
        "6d997a980848bb3c068f89a0c4219dba20958810a3a2a75e7033e22087b16d96",
    ),
}
TEMPERATURES = (0.05, 0.025, 0.01)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def describe(scores, legal, initial_actions, temperature):
    """Nonforced rows only; probabilities are a declared score transformation."""
    assert len(scores) and (legal.sum(1) > 1).all()
    assert np.isfinite(scores[legal]).all() and temperature > 0
    assert legal[np.arange(len(scores)), initial_actions].all()
    z = np.where(legal, scores.astype(np.float64) / temperature, -np.inf)
    z -= z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    probabilities = np.exp(logp)
    actions = z.argmax(1)
    greedy_probability = probabilities.max(1)
    ordered = np.sort(np.where(legal, scores, -np.inf), axis=1)
    return {
        "rows": len(scores),
        "changed_greedy_actions": int(np.count_nonzero(actions != initial_actions)),
        "mean_greedy_probability": float(greedy_probability.mean()),
        "greedy_probability_quantiles_10_50_90": np.quantile(
            greedy_probability, [0.1, 0.5, 0.9]
        ).tolist(),
        "mean_initial_greedy_probability": float(
            probabilities[np.arange(len(scores)), initial_actions].mean()
        ),
        "mean_entropy": float(
            -(probabilities * np.where(np.isfinite(logp), logp, 0)).sum(1).mean()
        ),
        "mean_top_two_raw_score_gap": float(
            (ordered[:, -1].astype(float) - ordered[:, -2]).mean()
        ),
        "initial_stop_to_continue": int(
            ((initial_actions == 48) & (actions == 49)).sum()
        ),
        "initial_continue_to_stop": int(
            ((initial_actions == 49) & (actions == 48)).sum()
        ),
    }


@torch.no_grad()
def main():
    assert sha(CORPUS) == CORPUS_SHA
    screen = Path("runs/pretrained_ppo_screen")
    audit = screen / "independent_audit.json"
    assert (
        sha(audit) == "643432440c451836559ab398d45a095f21043c4f04b60e26ec470dfa09d8c982"
    )
    assert (
        provenance()["source_sha256"]
        == "4b4360ed783c1d12cb5258e255f4b85f2e30dfa2e111fd8a078cf3439ce612cb"
    )
    payloads = {}
    for name, (path, digest) in CASES.items():
        assert sha(screen / path) == digest
        payloads[name] = torch.load(
            screen / path, map_location="cpu", weights_only=False
        )
    BASE.mkdir(exist_ok=False)
    protocol = {
        "corpus_sha256": CORPUS_SHA,
        "screen_audit_sha256": sha(audit),
        "checkpoints": CASES,
        "temperatures": TEMPERATURES,
        "devices": ["cpu", "cuda"],
        "batch_size": 128,
        "single_row_probe_per_phase": 16,
        "new_matches": 0,
        "policy_updates": 0,
        "population": "Primary summaries use source>=0: existing100 random-policy matches. Exclude selected search roots and hidden variants from summaries. All13029 rows remain in feature/action compatibility checks. Observations are correlated and not training/deployment visitation frequencies; no strength/causal or confidence-interval claim.",
        "provenance": provenance(),
        "script_sha256": sha(__file__),
    }
    write(BASE / "protocol.json", protocol)
    with np.load(CORPUS, allow_pickle=False) as archive:
        corpus = {k: archive[k] for k in archive.files}
    states, raw_mask = corpus["states"], corpus["masks"]
    phases, sources, guards = states[:, 99], corpus["sources"], corpus["guards"]
    mask = raw_mask & ((guards[:, None] < 0) | (np.arange(50)[None] == guards[:, None]))
    eligible = (sources >= 0) & (mask.sum(1) > 1)
    probe = np.concatenate(
        [np.flatnonzero(eligible & (phases == phase))[:16] for phase in range(4)]
    )
    assert len(probe) == 64
    raw = {"mask": mask, "phases": phases, "sources": sources, "probe_indices": probe}
    report = {
        "status": "complete",
        "protocol_sha256": sha(BASE / "protocol.json"),
        "cases": {},
        "runtime": {},
    }
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    for device in ("cpu", "cuda"):
        env = BatchEnv(len(states), device, "paper8")
        env.state.copy_(torch.tensor(states, device=device))
        features, legal = env.published_observe()
        np.testing.assert_array_equal(legal.cpu(), raw_mask)
        np.testing.assert_array_equal(env.solved_actions().cpu(), guards)
        for i, feature in enumerate(features.cpu().numpy()):
            assert (
                hashlib.sha256(feature.tobytes()).digest()
                == corpus["oracle_features_sha256"][i].tobytes()
            )
        report["runtime"][device] = runtime_identity(env.state.device)
        for name, payload in payloads.items():
            actor = PublishedActor(payload["actor"], device).requires_grad_(False)
            digest = actor.checkpoint()["current"]["parameters_sha256"]
            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
            scores = np.concatenate(
                [
                    actor(
                        features[start : start + 128],
                        env.state[start : start + 128, 99],
                        legal[start : start + 128],
                    )
                    .cpu()
                    .numpy()
                    for start in range(0, len(states), 128)
                ]
            )
            assert (
                np.isfinite(scores[raw_mask]).all()
                and np.isneginf(scores[~raw_mask]).all()
            )
            if name == "initial":
                np.testing.assert_array_equal(
                    scores.argmax(1), corpus["oracle_wp"].argmax(1)
                )
            if device == "cpu":
                raw[name + "_single_scores"] = np.concatenate(
                    [
                        actor(
                            features[i : i + 1],
                            env.state[i : i + 1, 99],
                            legal[i : i + 1],
                        )
                        .cpu()
                        .numpy()
                        for i in probe
                    ]
                )
            assert actor.checkpoint()["current"]["parameters_sha256"] == digest
            assert torch.equal(cpu_rng, torch.get_rng_state())
            if cuda_rng is not None:
                assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
            key = device + "_" + name
            raw[key] = scores
            reference = np.where(mask, raw[device + "_initial"], -np.inf).argmax(1)
            report["cases"][key] = {
                "parameters_sha256": digest,
                "temperatures": {
                    str(t): {
                        str(phase): describe(
                            scores[chosen], mask[chosen], reference[chosen], t
                        )
                        for phase in range(4)
                        for chosen in [eligible & (phases == phase)]
                    }
                    for t in TEMPERATURES
                },
            }
            print(json.dumps({"scored": key, "rows": len(states)}), flush=True)
            del actor
        del env, features, legal
    report["compatibility"] = {}
    for name in CASES:
        cpu = np.where(mask, raw["cpu_" + name], -np.inf).argmax(1)
        gpu = np.where(mask, raw["cuda_" + name], -np.inf).argmax(1)
        single = np.where(mask[probe], raw[name + "_single_scores"], -np.inf).argmax(1)
        report["compatibility"][name] = {
            "cpu_cuda_greedy_disagreements": int((cpu != gpu).sum()),
            "cpu_single_probe_disagreements": int((cpu[probe] != single).sum()),
        }
    np.savez_compressed(BASE / "scores.npz", **raw)
    report["scores_sha256"] = sha(BASE / "scores.npz")
    write(BASE / "report.json", report)
    print(
        json.dumps({"status": "complete", "compatibility": report["compatibility"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
