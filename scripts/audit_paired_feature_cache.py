"""Check paired public-feature identities, accounting and every cached head score."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from koikoi.arbitration import ExpertBank, read_checkpoint
from koikoi.layout import DISCARD, KOI, PICK_DRAW, PICK_HAND
from koikoi.train import provenance
from scripts.run_arbitration_study import sha


@torch.no_grad()
def inspect(dataset, cache_path):
    torch.set_num_threads(2)
    data, digest = read_checkpoint(dataset)
    cache, cache_digest = read_checkpoint(cache_path)
    if (
        data.get("format") != "paired_expert_targets_v1"
        or cache.get("format") != "paired_public_expert_features_v1"
    ):
        raise ValueError("Unexpected data/cache format")
    if cache["dataset_sha256"] != digest:
        raise ValueError("Dataset identity mismatch")
    source = provenance()["source_sha256"]
    if (
        data["provenance"]["source_sha256"] != source
        or cache["provenance"]["source_sha256"] != source
    ):
        raise ValueError("Production source mismatch")
    for path, expected in data["script_hashes"].items():
        if sha(path) != expected:
            raise ValueError(f"Collection helper changed: {path}")
    producer = (
        "scripts/replay_reusable_paired_features.py"
        if "workspace" in cache
        else "scripts/cache_paired_expert_features.py"
    )
    if sha(producer) != cache["script_sha256"]:
        raise ValueError("Feature producer changed")
    if "feature_helper_sha256" in cache and cache["feature_helper_sha256"] != sha(
        "scripts/cache_paired_expert_features.py"
    ):
        raise ValueError("Feature helper changed")
    count = data["collection"]["roots"]
    if (
        cache["verified_roots"] != count
        or cache["feature_batch"] != data["collection"]["batch"]
        or cache["device"] != data["network_config"]["device"]
    ):
        raise ValueError("Replay shape/device metadata mismatch")
    if (
        cache["new_source_replay_work"] != data["source_work"]
        or cache["reused_branch_work"] != data["branch_work"]
        or cache["new_branch_decisions"] != 0
    ):
        raise ValueError("Replay work mismatch")
    embeddings, scores, dealer = (
        cache["embeddings"],
        cache["scores"],
        cache["public_dealer"],
    )
    if (
        embeddings.shape != (count, 2, 2, 256)
        or scores.shape != (count, 2, 2)
        or dealer.shape != (count,)
    ):
        raise ValueError("Cache tensor shape mismatch")
    if (
        not torch.isfinite(embeddings).all()
        or not torch.isfinite(scores).all()
        or not ((dealer == 0) | (dealer == 1)).all()
    ):
        raise ValueError("Nonfinite features or invalid public dealer")
    phase_inputs = data["public_observations"][:, 528:532]
    if (
        not ((phase_inputs == 0) | (phase_inputs == 1)).all()
        or not (phase_inputs.sum(-1) == 1).all()
    ):
        raise ValueError("Invalid decision phases")
    phases = phase_inputs.argmax(-1)
    bank = ExpertBank(data["experts"], "cpu")
    maximum = 0.0
    for expert, actor in enumerate(bank.experts):
        for phase in (DISCARD, PICK_HAND, KOI):
            active = phases == phase
            if phase == PICK_HAND:
                active |= phases == PICK_DRAW
            head = actor.models[str(phase)].out
            for ids in torch.where(active)[0].split(4096):
                if not len(ids):
                    continue
                expected = F.linear(
                    embeddings[ids, :, expert], head.weight[:, :, 0], head.bias
                ).squeeze(-1)
                actual = scores[ids, :, expert]
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
                maximum = max(maximum, float((actual - expected).abs().max()))
    return {
        "cache_sha256": cache_digest,
        "dataset_sha256": digest,
        "roots": count,
        "cache_bytes": Path(cache_path).stat().st_size,
        "source_replay_work": cache["new_source_replay_work"],
        "new_branch_decisions": 0,
        "elapsed_seconds": cache["elapsed_seconds"],
        "max_cpu_head_reconstruction_error": maximum,
        "feature_producer": producer,
    }


def audit(dataset_base, feature_base):
    target = feature_base / "audit.json"
    if target.exists():
        raise ValueError("Use a fresh audit output")
    result = {
        "status": "passed",
        "scope": "Stored identities, replay accounting, finite public representations and all cached scores reconstructed from their vectors. No second source-trajectory replay or strength estimate.",
        "auditor_sha256": sha(__file__),
        "parents": {
            str(parent): inspect(
                dataset_base / f"parent{parent}" / "targets.pt",
                feature_base / f"parent{parent}.pt",
            )
            for parent in (1, 2)
        },
    }
    target.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-base", type=Path, required=True)
    parser.add_argument("--feature-base", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset_base, args.feature_base), indent=2))


if __name__ == "__main__":
    main()
