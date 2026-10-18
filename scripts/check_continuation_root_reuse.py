"""Measure fixed training-root reuse without stepping or evaluating a policy."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from koikoi.layout import DEALER, EPISODE, POINTS, ROUND, SEED
from scripts.run_arbitration_study import read, sha, write


def describe(bank):
    states = bank["states"].cpu().numpy()
    seats = bank["seats"].cpu().numpy()
    memory = bank["memory"].cpu().numpy()
    keys = [tuple(map(int, row[[SEED, EPISODE, ROUND]])) for row in states]
    streams = {key[:2] for key in keys}
    contexts = {
        (
            int(s[ROUND]),
            int(s[DEALER]),
            int(s[POINTS]),
            int(s[POINTS + 1]),
            int(seat),
            int(code),
        )
        for s, seat, code in zip(states, seats, memory)
    }
    decks_by_key = {}
    for key, state in zip(keys, states):
        deck = tuple(map(int, state[48:96]))
        assert key not in decks_by_key or decks_by_key[key] == deck
        decks_by_key[key] = deck
    return {
        "roots": len(states),
        "distinct_packed_states": len({s.tobytes() for s in states}),
        "distinct_chance_streams": len(streams),
        "distinct_root_deal_keys": len(set(keys)),
        "distinct_root_permutations": len(set(decks_by_key.values())),
        "distinct_public_match_contexts_with_seat_and_memory": len(contexts),
        "root_round_counts": dict(
            sorted(Counter(map(int, states[:, ROUND] + 1)).items())
        ),
        "chance_key_multiplicity_histogram": dict(
            sorted(Counter(Counter(keys).values()).items())
        ),
    }


def main(output):
    if output.exists():
        raise ValueError("Use a fresh diagnostic output")
    root = Path("runs/public_continuation_study")
    assert (
        sha(root / "protocol.json")
        == "95ce0a01a76c790f5dbdeb0c78028199e61219053167fc7309543cfa4c4815a1"
    )
    inputs = dict(read(root / "protocol.json")["inputs"])
    paths = [
        Path(__file__),
        root / "protocol.json",
        root / "parent1_public/initial.pt",
        root / "parent1_public/final.pt",
    ]
    inputs.update({str(p): sha(p) for p in paths})
    assert (
        sha(paths[-1])
        == "e953c46a455ad96b1b518d10c3e704e9cbf651a9a7ff900658f68841b4723f0e"
    )
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    banks = [
        torch.load(
            f"runs/public_memory_continuation_bank/parent{p}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for p in (1, 2)
    ]
    initial, final = [
        torch.load(p, map_location="cpu", weights_only=False) for p in paths[-2:]
    ]
    generator = torch.Generator(device="cpu")
    generator.set_state(initial["bank_rng"])
    draws = torch.cat(
        [
            torch.randint(len(banks[0]["states"]), (128,), generator=generator)
            for _ in range(520)
        ]
    )
    assert torch.equal(generator.get_state(), final["bank_rng"])
    counts = torch.bincount(draws, minlength=len(banks[0]["states"]))
    independent = np.bincount(draws.numpy(), minlength=len(counts))
    np.testing.assert_array_equal(counts.numpy(), independent)
    assert (
        int(counts.sum()) == final["counters"]["conditional_tail_completions"] == 66560
    )
    report = {
        "status": "passed",
        "inputs": inputs,
        "banks": {str(p): describe(bank) for p, bank in zip((1, 2), banks)},
        "completed_parent1_public_sampling": {
            "draws": len(draws),
            "distinct_roots_sampled": int((counts > 0).sum()),
            "minimum_uses_per_root": int(counts.min()),
            "maximum_uses_per_root": int(counts.max()),
            "mean_uses_per_root": float(counts.double().mean()),
            "sampling_rng_matches_completed_endpoint": True,
            "uses_histogram": dict(sorted(Counter(map(int, counts)).items())),
        },
        "new_simulator_decisions": 0,
        "new_neural_predictions": 0,
        "new_independent_deals": 0,
        "scope": "Training-only diversity diagnostic. Reusing a fixed seed/episode/round reuses its shuffled deck; action and score trajectories can differ. This is not an overfitting or strength test, and changes no running study.",
    }
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    write(output, report)
    print({k: v for k, v in report.items() if k != "inputs"})


if __name__ == "__main__":
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
