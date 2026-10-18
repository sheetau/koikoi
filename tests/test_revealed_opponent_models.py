import numpy as np
import pytest

from koikoi.agents import EndgameAgent
from koikoi.layout import PHASE, PLAYER, TERMINAL
from koikoi.reference import ReferenceEnv
from scripts.revealed_opponent_inputs import public_snapshot
from scripts.revealed_opponent_models import RevealedModelSet
from tests.test_final_round_actual_match import ContinueFirst


def test_strict_published_prediction_matches_original_assisted_cpu_actions():
    import torch

    from koikoi.upstream import PublishedAgent
    from scripts.revealed_opponent_inputs import reconstruct_revealed
    from scripts.revealed_opponent_models import StrictPublishedPrediction
    from tests.test_revealed_opponent_inputs import fixtures

    torch.set_num_threads(2)
    probes = {}
    for decision, state, reveal in fixtures():
        probes.setdefault(
            (int(state[PLAYER]), int(state[PHASE])), (decision, state, reveal)
        )
    assert len(probes) == 8
    for variant in ("wp", "point"):
        original = PublishedAgent(variant)
        prediction = StrictPublishedPrediction(original)
        assisted_original = EndgameAgent(original)
        for decision, state, reveal in probes.values():
            ref = ReferenceEnv.__new__(ReferenceEnv)
            ref.state = state
            snapshot = public_snapshot(
                ref.observe(),
                int(state[PLAYER]),
                decision,
                match_key=reveal["match_key"],
            )
            features, _, assisted = reconstruct_revealed(
                snapshot, reveal, reveal["available_after_decision"]
            )
            assert prediction(
                int(state[PHASE]), features, assisted
            ) == assisted_original.act(ref)


def continue_policy(phase, features, legal):
    return 49 if legal[49] else int(np.flatnonzero(legal)[0])


def stop_policy(phase, features, legal):
    return 48 if legal[48] else int(np.flatnonzero(legal)[0])


def transcript(seed, viewer, models):
    env = ReferenceEnv(seed, "paper8")
    actor = EndgameAgent(ContinueFirst())
    tracker = RevealedModelSet(f"toy-{seed}", viewer, models)
    statuses, snapshots = [], []
    decision = 0
    while env.state[PHASE] != TERMINAL:
        other = int(env.state[PLAYER]) != viewer
        if other:
            snapshot = public_snapshot(
                env.observe(),
                int(env.state[PLAYER]),
                decision,
                match_key=tracker.match_key,
            )
        action = actor.act(env)
        env.step(action)
        if other:
            before = tracker.candidates.copy()
            event = tracker.observe(snapshot, int(action))
            assert tracker.candidates <= before
            if not tracker.events:
                assert sum(tracker.prediction_calls.values()) == 0
            if event:
                assert event["available_after_decision"] == decision + 1
                assert len(event["initial_opponent_hand"]) == 8
                assert all(c["decision"] <= decision for c in event["checks"])
            statuses.append(tracker.status)
            snapshots.append(snapshot)
        decision += 1
    return tracker, statuses, snapshots


def test_candidates_persist_and_correct_toy_model_is_never_eliminated():
    identified = later_round_events = 0
    for viewer in (0, 1):
        for seed in range(31010001, 31010007):
            tracker, statuses, _ = transcript(
                seed, viewer, {"continue": continue_policy, "stop": stop_policy}
            )
            assert "continue" in tracker.candidates
            assert "inconsistent" not in statuses
            if tracker.first_identification:
                identified += 1
                assert tracker.candidates == {"continue"}
                first = tracker.first_identification["round"]
                later_round_events += sum(e["round"] > first for e in tracker.events)
                assert all(
                    e["after"] == ["continue"]
                    for e in tracker.events
                    if e["round"] >= first
                )
    assert identified > 0 and later_round_events > 0


def test_ambiguous_and_empty_sets_do_not_guess_and_match_context_is_checked():
    ambiguous, _, _ = transcript(
        31010001, 0, {"one": continue_policy, "two": continue_policy}
    )
    assert ambiguous.status == "ambiguous" and ambiguous.first_identification is None
    mismatches = 0
    for seed in range(31010001, 31010007):
        tracker, statuses, _ = transcript(
            seed, 0, {"one": stop_policy, "two": stop_policy}
        )
        if "inconsistent" in statuses:
            mismatches += 1
            assert not tracker.candidates and tracker.status == "inconsistent"
            assert tracker.first_identification is None
    assert mismatches > 0
    tracker, _, snapshots = transcript(31010001, 0, {"one": continue_policy})
    with pytest.raises(ValueError, match="increase"):
        tracker.observe(snapshots[0], 0)
    another = RevealedModelSet("different-match", 0, {"one": continue_policy})
    with pytest.raises(ValueError, match="different match"):
        another.observe(snapshots[0], 0)
