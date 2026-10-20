import copy

import numpy as np
import pytest
import torch

from koikoi.agents import solved_koi_action
from koikoi.layout import KOI, PHASE, PLAYER, POINTS, TERMINAL
from koikoi.published_actor import PublishedActor
from koikoi.reference import ReferenceEnv
from koikoi.upstream import UPSTREAM_PATH, reconstruct
from scripts.collect_score_auxiliary import (
    IncompleteScoreCollection,
    collect_outcomes,
    compact_evidence,
    tensor_sha,
)

pytestmark = [
    pytest.mark.upstream,
    pytest.mark.skipif(not UPSTREAM_PATH.exists(), reason="Upstream not fetched"),
]


def verify_reference(batch):
    refs = [
        ReferenceEnv(batch["deal_seed"] + lane, "paper8")
        for lane in range(batch["count"])
    ]
    row_lookup = {
        (int(step), int(identity[0])): row
        for row, (step, identity) in enumerate(
            zip(batch["selected_steps"], batch["identities"], strict=True)
        )
    }
    visits = torch.zeros_like(batch["visits"])
    counters = dict.fromkeys(("decisions", "rounds", "matches"), 0)
    for step, actions in enumerate(batch["actions"].tolist()):
        for lane, (ref, action) in enumerate(zip(refs, actions, strict=True)):
            if ref.state[PHASE] == TERMINAL:
                assert action == -1
                continue
            seat, phase = int(ref.state[PLAYER]), int(ref.state[PHASE])
            visits[lane, seat, phase] += 1
            assert action in np.where(ref.legal())[0]
            if (step, lane) in row_lookup:
                row = row_lookup[step, lane]
                assert batch["identities"][row].tolist() == [lane, seat, phase]
                original = reconstruct(ref)
                width = 50 if phase == KOI else 48
                torch.testing.assert_close(
                    batch["features"][row, :, :width],
                    original.feature_tensor,
                    rtol=0,
                    atol=0,
                )
                legal = torch.from_numpy(ref.legal()).bool()
                solution = solved_koi_action(ref)
                if solution is not None:
                    legal &= torch.arange(50) == solution
                assert torch.equal(legal, batch["legal"][row])
            result = ref.step(action)
            counters["decisions"] += 1
            counters["rounds"] += int(result[1])
            counters["matches"] += int(result[2])
    assert counters == batch["work"]
    assert torch.equal(visits, batch["visits"])
    assert len(batch["identities"]) == int((visits > 0).sum())
    for row, (lane, seat, phase) in enumerate(batch["identities"].tolist()):
        assert refs[lane].state[PHASE] == TERMINAL
        points = refs[lane].state[POINTS : POINTS + 2]
        assert batch["final_points"][lane].tolist() == points.tolist()
        assert int(batch["margins"][row]) == int(points[seat] - points[1 - seat])
        assert int((batch["targets"][row] - 32).sign()) == np.sign(
            points[seat] - points[1 - seat]
        )
        assert float(batch["inclusion_probabilities"][row]) == 1 / int(
            visits[lane, seat, phase]
        )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_complete_reference_labels_reservoir_and_rng_isolation(device, record_property):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    actor = PublishedActor.from_published("point", device)
    weights = copy.deepcopy(actor.state_dict())
    kwargs = {
        "count": 2,
        "deal_seed": 105020001,
        "action_seed": 105030001,
        "reservoir_seed": 105040001,
    }
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all()
    batch = collect_outcomes(actor, **kwargs)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(
        torch.equal(a, b)
        for a, b in zip(torch.cuda.get_rng_state_all(), cuda_rng, strict=True)
    )
    assert all(torch.equal(weights[k], v) for k, v in actor.state_dict().items())
    verify_reference(batch)
    assert set(batch["phases"].tolist()) == {0, 1, 2, 3}
    torch.rand(5, device=device)  # Global RNG must not determine collection.
    repeated = collect_outcomes(actor, **kwargs)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, repeated[key]), key
        else:
            assert value == repeated[key]
    compact = compact_evidence(batch)
    assert "features" not in compact
    assert compact["features_sha256"] == tensor_sha(batch["features"])
    for key, value in batch["work"].items():
        record_property(key, 2 * value)
    record_property("unique_deals", 2)
    record_property("auxiliary_rows_per_collection", len(batch["targets"]))


def test_bound_failure_never_yields_partial_training_labels(record_property):
    torch.set_num_threads(2)
    actor = PublishedActor.from_published("point", "cpu")
    with pytest.raises(IncompleteScoreCollection) as error:
        collect_outcomes(
            actor,
            count=2,
            deal_seed=105020001,
            action_seed=105030001,
            reservoir_seed=105040001,
            max_steps=1,
        )
    evidence = error.value.evidence
    assert evidence["work"] == {"decisions": 2, "rounds": 0, "matches": 0}
    assert "targets" not in evidence and "features" not in evidence
    record_property("decisions", 2)
