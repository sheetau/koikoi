"""Independent native replay of complete auxiliary collections, without actors."""

import hashlib

import torch

from koikoi.env import BatchEnv
from koikoi.layout import PHASE, PLAYER, POINTS, TERMINAL


def check_collection(record, config, cycle, *, return_features=False):
    """Reconstruct selected public inputs, reservoir decisions and terminal labels.

    cycle is one-based. Stored probabilities are checked for support/normalization,
    not recomputed from the policy network. Replay adds no independent deals.
    """
    assert record["format"] == "score_auxiliary_collection_v1"
    assert type(cycle) is int and cycle >= 1
    count = config["matches"]
    assert record["count"] == count
    assert record["deal_seed"] == config["deal_seed"] + (cycle - 1) * count
    assert record["action_seed"] == config["action_seed"] + cycle - 1
    assert record["reservoir_seed"] == config["reservoir_seed"] + cycle - 1
    assert record["max_steps"] == config["max_steps"]
    assert record["temperature"] == 0.05
    trace = record["actions"]
    assert trace.ndim == 2 and trace.shape[1] == count and trace.dtype == torch.int32
    assert 1 <= len(trace) <= config["max_steps"]
    identities = record["identities"]
    n = len(identities)
    assert identities.shape == (n, 3) and identities.dtype == torch.int64
    keys = [tuple(row) for row in identities.tolist()]
    assert keys == sorted(set(keys)) and 1 <= n <= 8 * count
    assert all(
        0 <= lane < count and seat in (0, 1) and 0 <= phase < 4
        for lane, seat, phase in keys
    )
    positions = record["selected_steps"]
    assert positions.shape == (n,) and positions.dtype == torch.int64
    assert ((positions >= 0) & (positions < len(trace))).all()
    assert record["phases"].dtype == torch.int64
    assert torch.equal(record["phases"], identities[:, 2])
    assert record["legal"].shape == (n, 50) and record["legal"].dtype == torch.bool
    selected = {}
    for row, ((lane, _, _), step) in enumerate(
        zip(keys, positions.tolist(), strict=True)
    ):
        assert (step, lane) not in selected
        selected[step, lane] = row
    features = torch.empty(n, 300, 50, dtype=torch.float32)
    observed_rows = set()
    counts = torch.zeros(count, 2, 4, dtype=torch.int64)
    retained = {}
    selection_rng = torch.Generator(device="cpu").manual_seed(record["reservoir_seed"])
    env = BatchEnv(count, "cpu", "paper8", record["deal_seed"])
    work = {"decisions": 0, "rounds": 0, "matches": 0}
    for step, actions in enumerate(trace):
        active = env.state[:, PHASE] != TERMINAL
        ids = torch.where(active)[0]
        assert len(ids), "Recorded a redundant post-terminal step"
        assert (actions[~active] == -1).all()
        assert ((actions[active] >= 0) & (actions[active] < 50)).all()
        _, legal = env.observe()
        solved = env.solved_actions()
        legal &= (solved[:, None] < 0) | (torch.arange(50)[None] == solved[:, None])
        assert legal[ids, actions[ids].long()].all()
        chosen_rows, chosen_lanes = [], []
        for lane in ids.tolist():
            key = (lane, int(env.state[lane, PLAYER]), int(env.state[lane, PHASE]))
            counts[key] += 1
            if int(torch.randint(int(counts[key]), (), generator=selection_rng)) == 0:
                retained[key] = step
            row = selected.get((step, lane))
            if row is not None:
                assert key == keys[row]
                assert torch.equal(record["legal"][row], legal[lane])
                chosen_rows.append(row)
                chosen_lanes.append(lane)
                observed_rows.add(row)
        if chosen_rows:
            # Only the chosen rows need their large feature tensor rebuilt.
            sample = BatchEnv(len(chosen_rows), "cpu", "paper8", 1)
            sample.state.copy_(env.state[chosen_lanes])
            public, _ = sample.published_observe()
            features[chosen_rows] = public
        result = env.step(actions.clamp_min(0), freeze_done=True)
        assert (result[ids, 0] != 99).all()
        work["decisions"] += len(ids)
        work["rounds"] += int(result[ids, 1].sum())
        work["matches"] += int(result[ids, 2].sum())
    assert (env.state[:, PHASE] == TERMINAL).all() and work["matches"] == count
    assert work == record["work"]
    assert observed_rows == set(range(n))
    assert keys == sorted(retained)
    assert positions.tolist() == [retained[key] for key in keys]
    assert torch.equal(counts, record["visits"])
    assert record["visits"].dtype == torch.int64
    assert (
        hashlib.sha256(features.numpy().tobytes()).hexdigest()
        == record["features_sha256"]
    )
    final_points = env.state[:, POINTS : POINTS + 2].long()
    assert record["final_points"].dtype == torch.int64
    assert torch.equal(final_points, record["final_points"])
    expected_margins = torch.tensor(
        [
            int(final_points[lane, seat] - final_points[lane, 1 - seat])
            for lane, seat, _ in keys
        ],
        dtype=torch.float32,
    )
    assert record["margins"].dtype == torch.float32
    assert torch.equal(record["margins"], expected_margins)
    labels = [
        int((max(-64, min(64, margin)) + 64) / 2)
        for margin in expected_margins.tolist()
    ]
    assert (
        record["targets"].dtype == torch.int64 and record["targets"].tolist() == labels
    )
    assert all(int(margin) % 2 == 0 for margin in expected_margins.tolist())
    probability = torch.tensor(
        [1 / int(counts[key]) for key in keys], dtype=torch.float64
    )
    assert record["inclusion_probabilities"].dtype == torch.float64
    assert torch.equal(record["inclusion_probabilities"], probability)
    log_probs = record["reference_log_probs"]
    mask = record["legal"]
    assert log_probs.shape == (n, 50) and log_probs.dtype == torch.float32
    assert (
        torch.isfinite(log_probs[mask]).all() and torch.isneginf(log_probs[~mask]).all()
    )
    assert torch.allclose(log_probs.exp().sum(-1), torch.ones(n), atol=1e-5, rtol=0)
    result = {
        "status": "passed",
        "work": work,
        "retained_rows": n,
        "phase_rows": [int((record["phases"] == phase).sum()) for phase in range(4)],
        "features_sha256": record["features_sha256"],
    }
    if return_features:
        result["features"] = features
    return result


def auxiliary_steps(record, config, cycle):
    """Reconstruct phase-specific Adam call counts from shuffled retained rows."""
    result = dict.fromkeys(
        (
            "optimizer_steps",
            "optimizer_steps_discard",
            "optimizer_steps_pick",
            "optimizer_steps_koi",
            "example_presentations",
        ),
        0,
    )
    if config["kind"] == "none":
        return result
    assert config["kind"] in ("win", "score")
    rng = torch.Generator(device="cpu").manual_seed(config["order_seed"] + cycle - 1)
    phases = record["phases"]
    for _ in range(config["epochs"]):
        order = torch.randperm(len(phases), generator=rng)
        for start in range(0, len(order), config["minibatch"]):
            group = phases[order[start : start + config["minibatch"]]].tolist()
            result["optimizer_steps"] += 1
            result["optimizer_steps_discard"] += int(0 in group)
            result["optimizer_steps_pick"] += int(1 in group or 2 in group)
            result["optimizer_steps_koi"] += int(3 in group)
            result["example_presentations"] += len(group)
    return result
