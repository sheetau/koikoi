"""Describe round coverage of two completed Transformer search diagnostics."""

import json
from pathlib import Path

import numpy as np
import torch

from koikoi.layout import PHASE, ROUND, TURN
from scripts.run_arbitration_study import sha, write


def read(path):
    return path, json.loads(path.read_text(encoding="utf-8"))


def run():
    output = Path("runs/search_round_coverage.json")
    if output.exists():
        raise ValueError("Preserve previous coverage report")
    action = Path("runs/published_action_search_capability")
    koi = Path("runs/published_koi_search_roots")
    records = [
        read(action / "protocol.json"),
        read(action / "audit.json"),
        read(koi / "audit.json"),
    ]
    protocol, action_audit, koi_audit = [r for _, r in records]
    assert action_audit["status"] == koi_audit["status"] == "passed"
    assert action_audit["protocol_sha256"] == sha(records[0][0])
    inputs = {str(p): sha(p) for p, _ in records}
    inputs[str(Path(__file__))] = sha(Path(__file__))
    for _, report in records:
        for p, digest in report["inputs"].items():
            assert sha(p) == digest, p
            inputs[p] = digest
    summaries = []
    for parent in (1, 2):
        card_path = Path(
            f"runs/observer_seat_information_v2/parent{parent}_states.npy"
        )
        indices = protocol["roots"][str(parent)]
        assert len(indices) == len(set(indices)) == 12
        cards = np.load(card_path, allow_pickle=False)[indices]
        assert np.isin(cards[:, PHASE], [0, 1, 2]).all()
        koi_path = koi / f"parent{parent}_roots.pt"
        assert sha(koi_path) == koi_audit["cases"][f"parent{parent}"]["roots_sha256"]
        inputs[str(koi_path)] = sha(koi_path)
        kois = torch.load(koi_path, map_location="cpu", weights_only=True)[
            "states"
        ].numpy()
        assert len(kois) == 32 and (kois[:, PHASE] == 3).all()
        for name, states in (("all_card", cards), ("koi", kois)):
            rounds = states[:, ROUND]
            assert np.isin(rounds, np.arange(8)).all()
            summaries.append(
                {
                    "study": name,
                    "parent": parent,
                    "roots": len(states),
                    "round_counts_zero_based": np.bincount(
                        rounds, minlength=8
                    ).tolist(),
                    "turn_range_zero_based": [
                        int(states[:, TURN].min()), int(states[:, TURN].max())
                    ],
                    "phase_counts": np.bincount(
                        states[:, PHASE], minlength=4
                    ).tolist(),
                }
            )
    assert all(sha(p) == h for p, h in inputs.items())
    result = {
        "status": "passed",
        "inputs": inputs,
        "summaries": summaries,
        "new_simulator_decisions": 0,
        "scope": (
            "Retrospective coverage of two fixed diagnostic corpora only; "
            "roots may share matches. No new strength evidence or prospective "
            "search protocol. Counts do not describe every past experiment."
        ),
    }
    write(output, result)
    print(json.dumps({"report_sha256": sha(output), "summaries": summaries}))


if __name__ == "__main__":
    run()
