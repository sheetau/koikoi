"""Verify public-context chance renewal on every audited continuation root."""

import argparse
from pathlib import Path

import numpy as np
import torch

from koikoi.layout import EPISODE, SEED
from koikoi.reference import valid_deal
from scripts.continuation_chance import PublicContinuationContext, renew_deal
from scripts.run_arbitration_study import read, sha, write


def main(output):
    if output.exists():
        raise ValueError("Use a fresh capability output")
    plan = Path("runs/public_continuation_study/protocol.json")
    assert (
        sha(plan) == "95ce0a01a76c790f5dbdeb0c78028199e61219053167fc7309543cfa4c4815a1"
    )
    inputs = dict(read(plan)["inputs"])
    for path in [
        Path(__file__),
        plan,
        Path("scripts/continuation_chance.py"),
        Path("tests/test_continuation_chance.py"),
        Path("runs/continuation_chance_cpu_final_tests.xml"),
    ]:
        inputs[str(path)] = sha(path)
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    groups = {}
    for parent in (1, 2):
        bank = torch.load(
            f"runs/public_memory_continuation_bank/parent{parent}.pt",
            map_location="cpu",
            weights_only=False,
        )
        changed = 0
        permutations = set()
        for index, state in enumerate(bank["states"].numpy()):
            before = state.copy()
            context = PublicContinuationContext.from_root(state)
            replayed = renew_deal(context, int(state[SEED]), int(state[EPISODE]))
            np.testing.assert_array_equal(replayed, state)
            # Capability-only fresh keys, not a proposed training/evaluation grid.
            fresh = renew_deal(context, 92900001 + (parent - 1) * 10000 + index)
            assert PublicContinuationContext.from_root(fresh) == context
            assert valid_deal(fresh[48:96])
            np.testing.assert_array_equal(np.sort(fresh[48:96]), np.arange(48))
            np.testing.assert_array_equal(state, before)
            changed += int(not np.array_equal(fresh[48:96], state[48:96]))
            permutations.add(fresh[48:96].tobytes())
        assert changed == len(permutations) == len(bank["states"])
        groups[str(parent)] = {
            "roots_checked": len(bank["states"]),
            "same_key_exact_reconstructions": len(bank["states"]),
            "fresh_conditional_deals": changed,
            "distinct_fresh_permutations": len(permutations),
        }
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    report = {
        "status": "passed",
        "inputs": inputs,
        "groups": groups,
        "new_simulator_decisions": 0,
        "new_policy_predictions": 0,
        "full_matches": 0,
        "scope": "CPU deal-construction capability only. No CUDA/trainer integration or strength result;all current-study inputs remain unchanged.",
    }
    write(output, report)
    print({k: v for k, v in report.items() if k != "inputs"})


if __name__ == "__main__":
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
