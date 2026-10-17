"""Certificate coverage on existing TRAINING roots; no new simulator work."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.paired_rollout_targets import match_split
from scripts.public_win_certificate import certify_turn
from scripts.run_arbitration_study import read, sha


def public_legal(obs):
    cards = obs[:528].reshape(48, 11)
    phase = int(obs[528:532].argmax())
    legal = np.zeros(50, dtype=bool)
    if phase == 0:
        legal[:48] = cards[:, 0] > 0
    elif phase in (1, 2):
        pending = np.flatnonzero(cards[:, 5])
        if len(pending) != 1:
            raise ValueError("Expected one public pending card")
        legal[:48] = (cards[:, 1] > 0) & (np.arange(48) // 4 == pending[0] // 4)
        assert legal.sum() == 2
    else:
        legal[48:] = True
    return legal


def run(output):
    if output.exists():
        raise ValueError("Use a fresh diagnostic output")
    torch.set_num_threads(2)
    started = time.perf_counter()
    base = Path("runs/opponent_target_study")
    result_path = base / "results.json"
    completed = read(result_path)
    inputs = {
        str(path): sha(path)
        for path in (
            Path(__file__),
            Path("scripts/public_win_certificate.py"),
            Path("scripts/paired_rollout_targets.py"),
            Path("koikoi/reference.py"),
            Path("koikoi/layout.py"),
            result_path,
        )
    }
    reports = {}
    for index in (1, 2):
        path = base / "data" / f"parent{index}" / "opponent_preserving/targets.pt"
        digest = sha(path)
        assert (
            digest
            == completed[str(index)]["collection"]["conditions"]["opponent_preserving"][
                "dataset_sha256"
            ]
        )
        inputs[str(path)] = digest
        data = torch.load(path, map_location="cpu", weights_only=False)
        assert data["format"] == "paired_expert_targets_v1"
        keep = ~match_split(data["match_groups"])
        roots = data["public_observations"][keep].numpy()
        outcomes = data["outcomes"][keep, 0].numpy()
        rows = []
        for phase in range(4):
            rows.append(
                {
                    "phase": phase,
                    "roots": 0,
                    "certified": 0,
                    "parent_not_certified": 0,
                    "point_not_certified": 0,
                    "neither_proposal_certified": 0,
                    "parent_continuation_nonwin": 0,
                    "draw_cases": 0,
                    "leaf_checks": 0,
                }
            )
        for public, parent_outcome in zip(roots, outcomes, strict=True):
            obs, proposals = public[:560], public[560:].astype(int)
            legal = public_legal(obs)
            assert legal[proposals].all()
            cert = certify_turn(obs, legal)
            row = rows[int(obs[528:532].argmax())]
            row["roots"] += 1
            row["draw_cases"] += cert.draw_cases
            row["leaf_checks"] += cert.leaf_checks
            if cert.actions:
                row["certified"] += 1
                parent_miss, point_miss = (
                    int(a) not in cert.actions for a in proposals
                )
                row["parent_not_certified"] += parent_miss
                row["point_not_certified"] += point_miss
                row["neither_proposal_certified"] += parent_miss and point_miss
                row["parent_continuation_nonwin"] += bool(parent_outcome < 1)
        reports[str(index)] = rows
        print(json.dumps({"parent": index, "phases": rows}), flush=True)
    report = {
        "inputs": inputs,
        "parents": reports,
        "elapsed_seconds": time.perf_counter() - started,
        "simulator_work": {
            "matches": 0,
            "rounds": 0,
            "decisions": 0,
            "rollout_decisions": 0,
        },
        "scope": "Existing training source-match split, expert-disagreement roots only. Certificates use only public observations and reconstructed legal masks. Stored parent branch outcomes describe frozen continuation in the sampled world, not new guarded matches. Roots are correlated and selected for expert disagreement; no prevalence, independent strength, or out-of-sample benefit estimate follows. Symbolic draw/leaf checks are counted separately from simulator work.",
    }
    assert all(sha(path) == digest for path, digest in inputs.items())
    output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"output": str(output), "sha256": sha(output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/win_certificate_diagnostic.json")
    )
    run(parser.parse_args().output)
