"""Exact restart,independent draw schedule and atomic exhaustion/tamper checks."""

import copy

import numpy as np
import pytest
import torch

from koikoi.layout import EPISODE, SEED
from scripts.continuation_chance import PublicContinuationContext, renew_deal
from scripts.renewed_root_sampler import RenewedRootSampler


def bank():
    contexts = [
        PublicContinuationContext(1, 0, (30, 30)),
        PublicContinuationContext(4, 1, (17, 43)),
        PublicContinuationContext(7, 1, (5, 55)),
    ]
    states = torch.from_numpy(
        np.stack([renew_deal(c, 401 + i) for i, c in enumerate(contexts)])
    )
    return (
        states,
        torch.tensor([0, 1, 0]),
        torch.tensor([0, 1, 1]),
        torch.tensor([1, 0, 1]),
    )


def sampler(data=None):
    return RenewedRootSampler(
        *(bank() if data is None else data),
        sampling_seed=37,
        chance_seed=92920001,
        max_draws=24,
    )


def same(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], torch.Tensor):
            assert torch.equal(left[key], right[key]), key
        else:
            assert left[key] == right[key], key


def test_sampler_restart_and_metadata_match_independent_schedule(tmp_path):
    data = bank()
    source = sampler(data)
    generator = torch.Generator(device="cpu").manual_seed(37)
    first = source.sample(8)
    expected = torch.randint(3, (8,), generator=generator)
    assert torch.equal(first["indices"], expected)
    assert len(set(expected.tolist())) < 8
    assert len({row[48:96].numpy().tobytes() for row in first["states"]}) == 8
    for name, values in zip(("seats", "opponents", "memory"), data[1:]):
        assert torch.equal(first[name], values[expected])
    assert bool((first["memory"] != first["opponents"]).any())
    assert torch.equal(first["states"][:, SEED].long(), first["chance_seeds"])
    assert bool((first["states"][:, EPISODE] == 0).all())
    checkpoint = tmp_path / "sampler.pt"
    torch.save(source.state_dict(), checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    second = source.sample(7)
    assert torch.equal(second["indices"], torch.randint(3, (7,), generator=generator))
    resumed = sampler(data)
    resumed.load_state_dict(saved)
    same(second, resumed.sample(7))
    same(source.state_dict(), resumed.state_dict())
    paired = sampler(data)
    same(first, paired.sample(8))
    same(second, paired.sample(7))
    assert int(first["chance_seeds"].max()) < int(second["chance_seeds"].min())
    # Caller-side metadata mutation cannot change the sampler's frozen mapping.
    data[3].zero_()
    assert source.memory.tolist() == [1, 0, 1]


def test_sampler_refuses_overrun_and_inconsistent_restore_atomically():
    source = sampler()
    source.sample(8)
    saved = source.state_dict()
    for count in (0, 17, -1):
        with pytest.raises(ValueError):
            source.sample(count)
        same(saved, source.state_dict())
    bad_states = []
    changed = copy.deepcopy(saved)
    changed["cursor"] += 1
    bad_states.append(changed)
    changed = copy.deepcopy(saved)
    changed["rng"][0] ^= 1
    bad_states.append(changed)
    changed = copy.deepcopy(saved)
    changed["identity"]["memory"] = "changed"
    bad_states.append(changed)
    changed = copy.deepcopy(saved)
    changed["config"]["chance_seed"] += 1
    bad_states.append(changed)
    for changed in bad_states:
        with pytest.raises(ValueError):
            source.load_state_dict(changed)
        same(saved, source.state_dict())
    source.sample(16)
    assert source.cursor == 24
    with pytest.raises(ValueError):
        source.sample(1)
