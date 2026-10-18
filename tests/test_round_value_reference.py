from dataclasses import asdict

import numpy as np
import pytest
import torch

from koikoi.agents import NeuralAgent
from koikoi.layout import DEALER, PHASE, PLAYER, POINTS, ROUND, TERMINAL
from koikoi.published_actor import PublishedActor
from koikoi.reference import ReferenceEnv
from scripts.diagnose_history_support import reference
from scripts.history_conditioned_search import MixtureContinuation
from scripts.round_value_reference import FreshDealBank, TerminalReference
from scripts.run_arbitration_study import PARENTS


def independent_start(base_seed, index, r, p, dealer):
    ref = ReferenceEnv(base_seed + 1000000 * int(r) + int(index), "paper8")
    ref.state[ROUND] = r
    ref.state[POINTS : POINTS + 2] = (p, 60 - p)
    ref.state[DEALER] = dealer
    ref.deal()
    return ref


def test_deal_bank_matches_reference_for_all_rounds_and_public_axes():
    bank = FreshDealBank(40110001, 3, "cpu")
    contexts = [[r, p, d] for r in range(8) for p in (1, 30, 59) for d in (0, 1)]
    indices = np.arange(len(contexts)) % 3
    actual = bank.starts(contexts, indices).numpy()
    for row, index, (r, p, d) in zip(actual, indices, contexts):
        np.testing.assert_array_equal(
            row, independent_start(bank.base_seed, index, r, p, d).state
        )
    for contexts, indices in (
        ([[8, 30, 0]], [0]),
        ([[0, 0, 0]], [0]),
        ([[0, 30, 2]], [0]),
        ([[0, 30, 0]], [3]),
        ([[0.0, 30, 0]], [0]),
    ):
        with pytest.raises(ValueError):
            bank.starts(contexts, indices)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("seat", [0, 1])
def test_terminal_reference_both_chance_modes_match_stepwise_python(device, seat):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parent = NeuralAgent(PARENTS[0][0], deterministic=True)
    continuation = MixtureContinuation(
        PublishedActor(parent.model.actor.checkpoint(), device), device
    )
    bank = FreshDealBank(40120001, 8, device)
    contexts = [
        [0, 30, 0],
        [3, 30, 1],
        [6, 1, 0],
        [6, 59, 1],
        [7, 30, 0],
        [7, 30, 1],
        [7, 1, 1],
        [7, 59, 0],
    ]
    indices = torch.arange(8, device=device)
    original = bank.starts(contexts, indices)
    saved = original.cpu().numpy().copy()
    outcomes = []
    for mode in ("native", "independent"):
        continuation.prepare(seat, [0, 1, 0, 1])
        refs = [reference(s) for s in saved]
        counts = dict.fromkeys(
            ("decisions", "round_endings", "terminal_paths", "reseeded_rounds"), 0
        )
        routes = set()
        frozen = set()

        def checked(
            env, refs=refs, counts=counts, frozen=frozen, routes=routes, mode=mode
        ):
            np.testing.assert_array_equal(
                env.state.cpu().numpy(), np.stack([r.state for r in refs])
            )
            features, masks = env.published_observe()
            actions = continuation(env)
            for i, (ref, action) in enumerate(zip(refs, actions.tolist())):
                if ref.state[PHASE] == TERMINAL:
                    assert action == -1 and not features[i].any() and not masks[i].any()
                    frozen.add(i)
                    continue
                routes.add(
                    2 if ref.state[PLAYER] == seat else int(continuation.labels[i])
                )
                result = ref.step(action)
                counts["decisions"] += 1
                counts["round_endings"] += int(result[1])
                counts["terminal_paths"] += int(result[2])
                if mode == "independent" and result[1] and not result[2]:
                    refs[i] = independent_start(
                        bank.base_seed,
                        i,
                        ref.state[ROUND],
                        ref.state[POINTS],
                        ref.state[DEALER],
                    )
                    counts["reseeded_rounds"] += 1
            return actions

        workspace = TerminalReference(8, device)
        endpoints, used = workspace.run(
            original, checked, indices, bank if mode == "independent" else None
        )
        np.testing.assert_array_equal(
            endpoints.cpu().numpy(), np.stack([r.state for r in refs])
        )
        np.testing.assert_array_equal(original.cpu().numpy(), saved)
        assert asdict(used) == counts
        assert used.terminal_paths == 8 and used.round_endings > 8
        assert routes == {0, 1, 2} and frozen
        assert used.reseeded_rounds == (
            used.round_endings - 8 if mode == "independent" else 0
        )
        outcomes.append(endpoints.cpu().numpy())
    # Final-round starts must have identical complete trajectories in both modes.
    np.testing.assert_array_equal(outcomes[0][4:], outcomes[1][4:])
    # At least the early surviving path was truly reseeded, not merely re-labelled.
    assert not np.array_equal(outcomes[0][0], outcomes[1][0])
