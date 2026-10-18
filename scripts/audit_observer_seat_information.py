"""Check saved seat probes against the pinned original Python feature builder."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from koikoi.env import BatchEnv
from koikoi.layout import PHASE
from koikoi.reference import ReferenceEnv
from koikoi.upstream import UPSTREAM_PATH, UPSTREAM_REV, reconstruct
from scripts.diagnose_observer_seat_information import relabel, sha


def run(base, output):
    if output.exists():
        raise ValueError("Use a fresh audit output")
    torch.set_num_threads(2)
    audit = json.loads((base / "audit.json").read_text(encoding="utf-8"))
    protocol = json.loads(
        Path("runs/observer_selfplay_study/protocol.json").read_text(encoding="utf-8")
    )
    inputs = dict(audit["inputs"])
    for path in (
        Path(__file__),
        Path("koikoi/upstream.py"),
        UPSTREAM_PATH / "koikoigame.py",
        base / "audit.json",
    ):
        inputs[str(path)] = sha(path)
    assert all(sha(p) == h for p, h in inputs.items())
    assert all(sha(p) == h for p, h in protocol["inputs"].items())
    results = {}
    env = BatchEnv(1, "cpu", "paper8", seed=0)
    for name, expected in audit["parents"].items():
        rowpath, statepath = base / f"{name}.jsonl", base / f"{name}_states.npy"
        assert sha(rowpath) == expected["rows_sha256"]
        assert sha(statepath) == expected["states_sha256"]
        inputs[str(rowpath)], inputs[str(statepath)] = sha(rowpath), sha(statepath)
        rows = [json.loads(s) for s in rowpath.read_text(encoding="utf-8").splitlines()]
        states = np.load(statepath, allow_pickle=False)
        assert len(rows) == len(states) == 256
        assert (
            sum(r["original_action"] != r["relabeled_action"] for r in rows)
            == expected["greedy_changes"]
        )
        assert (
            float(np.mean([r["total_variation"] for r in rows]))
            == expected["mean_total_variation"]
        )
        assert (
            max(r["total_variation"] for r in rows) == expected["max_total_variation"]
        )
        for i, state in enumerate(states):
            assert rows[i]["root"] == i and rows[i]["phase"] == int(state[PHASE])
            original_features = []
            for probe in (state, relabel(state)):
                ref = ReferenceEnv.__new__(ReferenceEnv)
                ref.state = probe.copy()
                upstream = reconstruct(ref)
                env.state.copy_(torch.from_numpy(probe).unsqueeze(0))
                features, masks = env.published_observe()
                width = 50 if probe[PHASE] == 3 else 48
                torch.testing.assert_close(
                    features[0, :, :width], upstream.feature_tensor, rtol=0, atol=0
                )
                np.testing.assert_array_equal(masks[0], ref.legal())
                original_features.append(upstream.feature_tensor)
            changed = torch.where(
                (original_features[0] != original_features[1]).any(-1)
            )[0].tolist()
            assert changed == rows[i]["changed_published_rows"] == [50, 51]
        results[name] = {
            "roots": len(states),
            "upstream_reconstructions": 2 * len(states),
            "changed_rows": [50, 51],
        }
    failed = Path("runs/observer_seat_information")
    # The first attempt failed while serializing its completed parent1 counters.
    # Identical retained roots/actions establish that it executed the same source
    # trajectory; this audit does not replay a second set of source matches.
    for suffix in (".jsonl", "_states.npy"):
        old = failed / f"parent1{suffix}"
        assert sha(old) == sha(base / old.name)
        inputs[str(old)] = sha(old)
    assert all(sha(p) == h for p, h in inputs.items())
    assert all(sha(p) == h for p, h in protocol["inputs"].items())
    result = {
        "status": "passed",
        "inputs": inputs,
        "upstream_revision": UPSTREAM_REV,
        "parents": results,
        "failed_attempt_source_work": audit["parents"]["parent1"]["source_work"],
        "scope": "1024 upstream within-round history reconstructions, including forced transitions; no new source trajectories, fitted models, strength evaluations or alternate future continuations. Confirms feature parity and raw summaries, not policy-value bias or playing-strength improvement.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"status": "passed", "parents": results, "sha256": sha(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("runs/observer_seat_information_v2")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.output)
