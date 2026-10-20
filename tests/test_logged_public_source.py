from copy import deepcopy

import numpy as np
import pytest

from koikoi.agents import EndgameAgent
from koikoi.env import BatchEnv
from koikoi.layout import DEALER, PLAYER, ROUND
from koikoi.reference import ReferenceEnv
from scripts.logged_public_source import audit, collect
from scripts.uniform_deal_source import DeckSample, LoggedDealSource


class Minimum:
    def __init__(self, keep_playing=False):
        self.keep_playing = keep_playing
        self.trace = []

    def act(self, env):
        assert type(env) is ReferenceEnv
        assert not hasattr(env, "deals") and not hasattr(env, "_draw")
        self.trace.append(env.state.copy())
        legal = np.flatnonzero(env.legal())
        return 49 if self.keep_playing and 49 in legal else int(legal[0])


def predict(phase, features, legal):
    return int(np.flatnonzero(legal)[0])


def fixture_source():
    decks = iter(tuple(map(int, ReferenceEnv(71 + i).state[48:96])) for i in range(8))
    return LoggedDealSource(53, "paper8", lambda: DeckSample(next(decks), 1, 47))


@pytest.mark.parametrize("seat", [0, 1])
def test_real_public_collection_native_parity_and_complete_logged_replay(seat):
    source, actor = fixture_source(), EndgameAgent(Minimum())
    initial_dealer = int(source.env.state[DEALER])
    record = collect(
        source,
        "fixture-public-id",
        seat,
        actor,
        actor,
        {"a": predict, "b": predict},
        BatchEnv(1, debug=True),
    )
    assert len(record["deals"]) == 8
    assert initial_dealer == ReferenceEnv(53).state[DEALER]
    assert record["selected_root"] is None  # Identical models stay ambiguous.
    assert record["work"] == record["native_work"]
    assert record["sampling_work"] == {
        "accepted_deals": 8,
        "attempts": 8,
        "bounded_draws": 376,
    }
    checked = audit(record, {"a": predict, "b": predict}, actor, actor)
    assert checked["replay_work"] == record["work"]
    assert checked["fresh_external_chance_draws"] == 0
    assert (
        sum(checked["policy_verification_calls"].values())
        == record["work"]["decisions"]
    )
    changed = deepcopy(record)
    changed["actions"][0] = int(np.flatnonzero(fixture_source().env.legal())[1])
    with pytest.raises(ValueError, match="policy"):
        audit(changed, {"a": predict, "b": predict}, actor, actor)
    for key in ("work", "native_work"):
        changed = deepcopy(record)
        changed[key]["decisions"] += 1
        with pytest.raises(ValueError, match=key):
            audit(changed, {"a": predict, "b": predict}, actor, actor)
    changed = deepcopy(record)
    changed["selected_root"] = {"decision": 0}
    with pytest.raises(ValueError, match="selected_root"):
        audit(changed, {"a": predict, "b": predict}, actor, actor)
    changed = deepcopy(record)
    changed["deals"].append(changed["deals"][-1])
    with pytest.raises(ValueError, match="exactly one"):
        audit(changed, {"a": predict, "b": predict}, actor, actor)


def test_real_delayed_revelation_identifies_a_model_from_logged_deals():
    source = fixture_source()
    seat = 1 - int(source.env.state[DEALER])
    actor = EndgameAgent(Minimum(keep_playing=True))

    def keep(phase, features, legal):
        return 49 if legal[49] else int(np.flatnonzero(legal)[0])

    def largest(phase, features, legal):
        return int(np.flatnonzero(legal)[-1])

    record = collect(
        source,
        "revelation-fixture",
        seat,
        actor,
        actor,
        {"minimum": keep, "maximum": largest},
        BatchEnv(1, debug=True),
    )
    assert record["identification_events"]
    assert record["identification_events"][0]["after"] == ["minimum"]
    assert record["identification_calls"]["minimum"] > 0
    assert (
        audit(record, {"minimum": keep, "maximum": largest}, actor, actor)[
            "replay_work"
        ]
        == record["work"]
    )
    changed = deepcopy(record)
    changed["identification_events"][0]["after"] = ["maximum"]
    with pytest.raises(ValueError, match="identification_events"):
        audit(changed, {"minimum": keep, "maximum": largest}, actor, actor)


@pytest.mark.parametrize("seat", [0, 1])
def test_first_eligible_gate_and_root_state_with_synthetic_identification(
    monkeypatch, seat
):
    # Isolate gate wiring with a declared synthetic public inference fixture.
    # The previous test exercises the real delayed-revelation implementation.
    class Identity:
        def __init__(self, source_key, viewer, predictors):
            self.match_key = source_key
            self.first_identification = None
            self.status = "ambiguous"
            self.events, self.prediction_calls = [], {}

        def observe(self, snapshot, action):
            assert not snapshot.features[:528].reshape(48, 11)[:, [0, 4]].any()
            if self.first_identification is None:
                self.first_identification = {
                    "round": 0,
                    "available_after_decision": snapshot.decision + 1,
                    "model": "fixture",
                }
                self.status = "identified"

    monkeypatch.setattr("scripts.logged_public_source.RevealedModelSet", Identity)
    # No endgame wrapper: one trace entry per decision makes an independent
    # first-final-round learner-decision address available to this gate test.
    actor = Minimum()
    record = collect(
        fixture_source(),
        "gate-fixture",
        seat,
        actor,
        actor,
        {"unused": predict},
        BatchEnv(1, debug=True),
        max_draws=16,
    )
    first = next(
        i
        for i, state in enumerate(actor.trace)
        if state[ROUND] == 7 and state[PLAYER] == seat
    )
    assert record["selected_root"]["decision"] == first
    assert record["selected_root"]["state"] == actor.trace[first].tolist()
    assert (
        audit(record, {"unused": predict}, actor, actor)["replay_work"]
        == record["work"]
    )
    changed = deepcopy(record)
    changed["selected_root"]["decision"] += 1
    with pytest.raises(ValueError, match="selected_root"):
        audit(changed, {"unused": predict}, actor, actor)


def test_invalid_source_and_public_cap_refuse_before_actor_calls():
    source, actor = fixture_source(), Minimum()
    native = BatchEnv(1, debug=True)
    with pytest.raises(ValueError, match="draw cap"):
        collect(
            source, "fixture", 0, actor, actor, {"a": predict}, native, max_draws=-1
        )
    assert not actor.trace
    source.step(int(np.flatnonzero(source.env.legal())[0]))
    with pytest.raises(ValueError, match="fresh"):
        collect(source, "fixture", 0, actor, actor, {"a": predict}, native)
    assert not actor.trace
