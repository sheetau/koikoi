import numpy as np
import pytest
import torch

from koikoi.env import BatchEnv
from koikoi.layout import PHASE, TERMINAL
from scripts.fixed_microbatch_continuation import (
    FixedMicrobatchContinuation,
    fixed_scores,
)
from scripts.history_conditioned_search import HistoryConditionedSearch
from scripts.run_arbitration_study import PARENTS, sha


def setup(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    path = "runs/midround_batch_divergence/divergence.npz"
    assert (
        sha(path) == "c8ddf9210b4ac5ae06a61adf762b2731f46da05a616e32214826b260d89557ae"
    )
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    old = HistoryConditionedSearch(
        PARENTS[1][0], particles=512, confirmation_particles=0, device=device
    )
    fixed = FixedMicrobatchContinuation(old.continuation.actors[2], device, 64)
    fixed.prepare(0, arrays["labels"])
    return arrays, fixed


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_saved_same_baselines_have_same_actions_across_alternative_batches(device):
    arrays, fixed = setup(device)
    env = BatchEnv(1024, device, "paper8")
    outputs = []
    for key in ("first_states", "other_states"):
        env.state.copy_(torch.from_numpy(arrays[key]).to(device))
        actions = fixed(env)
        legal = env.observe()[1]
        active = env.state[:, PHASE] != TERMINAL
        assert legal[active, actions[active]].all()
        assert (actions[~active] == -1).all()
        outputs.append(actions)
    np.testing.assert_array_equal(
        arrays["first_states"][::2], arrays["other_states"][::2]
    )
    torch.testing.assert_close(outputs[0][::2], outputs[1][::2], rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_all_heads_order_padding_and_fixed_shape(device):
    arrays, fixed = setup(device)
    env = BatchEnv(1024, device, "paper8")
    env.state.copy_(torch.from_numpy(arrays["first_states"]).to(device))
    features, legal = env.published_observe()
    phases = env.state[:, PHASE]
    # Include the reproduced near-tie lane and examples from every live phase.
    selected = [458]
    for phase in range(4):
        indices = torch.where(phases == phase)[0][:35].tolist()
        assert indices
        selected.extend(indices)
    features, legal, phases = features[selected], legal[selected], phases[selected]
    order = torch.arange(len(selected) - 1, -1, -1, device=device)
    for actor in fixed.actors:
        calls = []
        hooks = [
            m.register_forward_pre_hook(
                lambda m, args, calls=calls: calls.append(tuple(args[0].shape))
            )
            for m in actor.models.values()
        ]
        original = fixed_scores(actor, features, phases, legal)
        changed = fixed_scores(actor, features[order], phases[order], legal[order])
        torch.testing.assert_close(original, changed[order], rtol=0, atol=0)
        for count in (1, 63, 64, 65, 129):
            indices = torch.arange(count, device=device).remainder(len(selected))
            result = fixed_scores(
                actor, features[indices], phases[indices], legal[indices]
            )
            torch.testing.assert_close(result, original[indices], rtol=0, atol=0)
        assert calls and all(
            shape[0] == 64 and shape[1] == 300 and shape[2] in (48, 50)
            for shape in calls
        )
        for hook in hooks:
            hook.remove()
    with pytest.raises(ValueError, match="positive integer"):
        fixed_scores(fixed.actors[0], features, phases, legal, microbatch=0)
