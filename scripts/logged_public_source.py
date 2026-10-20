"""Collect public-identification prefixes from externally logged round deals.

This is a collector/auditor component, not an evaluation protocol. The caller
must specify the sampling law and bind the actors before collecting a study.
"""

import numpy as np
import torch

from koikoi.layout import (
    BANKRUPT,
    KOI,
    PHASE,
    PICK_DRAW,
    PLAYER,
    ROUND,
    ROUNDS,
    SEED,
    TERMINAL,
    TURN,
)
from scripts.check_belief_tail import add_step, empty_work
from scripts.diagnose_history_support import reference
from scripts.revealed_opponent_inputs import public_snapshot
from scripts.revealed_opponent_models import RevealedModelSet
from scripts.uniform_deal_source import DeckSample, LoggedDealSource


def _identity(source, source_key, seat, predictors, max_draws):
    if type(seat) is not int or seat not in (0, 1):
        raise ValueError("Invalid learner seat")
    if type(max_draws) is not int or not 0 <= max_draws <= 16:
        raise ValueError("Invalid public draw cap")
    state = source.env.state
    if (
        source.failed
        or source.actions
        or state[ROUND] != 0
        or state[TURN] != 0
        or state[PHASE] != 0
        or state[ROUNDS] != 8
        or not state[BANKRUPT]
        or len(source.deals) != 1
    ):
        raise ValueError("Expected a fresh, successful paper8 source")
    return RevealedModelSet(source_key, seat, predictors)


def _eligible(env, identity, seat, decision, max_draws):
    known = identity.first_identification
    draws = 16 - int(env.state[TURN]) - int(env.state[PHASE] in (PICK_DRAW, KOI))
    return (
        env.state[PLAYER] == seat
        and env.state[ROUND] == 7
        and draws <= max_draws
        and identity.status == "identified"
        and known is not None
        and known["round"] < 7
        and known["available_after_decision"] <= decision
    )


def _root(env, identity, decision):
    # The full state is evaluator evidence. Planning must reconstruct its own
    # hidden-world support; actors never receive this record or the deal log.
    return {
        "decision": decision,
        "state": env.state.tolist(),
        "public_identification": dict(identity.first_identification),
    }


def _snapshot(env, identity, seat, decision):
    actor = int(env.state[PLAYER])
    return (
        public_snapshot(env.observe(), actor, decision, match_key=identity.match_key)
        if actor != seat
        else None
    )


def collect(
    source, source_key, seat, parent, opponent, predictors, native, max_draws=5
):
    """Play one complete parent match, checking every transition against native.

    Fresh source identity is public and need not encode opponent/model labels.
    Gate selection uses only the public reconstruction and public draw count.
    """
    identity = _identity(source, source_key, seat, predictors, max_draws)
    env = source.env
    native.state.copy_(torch.from_numpy(env.state[None].copy()))
    native_work, selected = empty_work(), None
    while env.state[PHASE] != TERMINAL:
        decision = len(source.actions)
        if decision >= 1024:
            raise ValueError("Source exceeded the complete-match decision bound")
        if selected is None and _eligible(env, identity, seat, decision, max_draws):
            selected = _root(env, identity, decision)
        snapshot = _snapshot(env, identity, seat, decision)
        actor = parent if env.state[PLAYER] == seat else opponent
        action = int(actor.act(env))
        before_round = int(env.state[ROUND])
        result = source.step(action)
        checked = native.step(torch.tensor([action]), freeze_done=True)[0].numpy()
        np.testing.assert_array_equal(checked, result)
        if env.state[PHASE] != TERMINAL and env.state[ROUND] != before_round:
            # Both engines auto-deal from their seed. The harness replaced the
            # Python deal already; validate metadata before replacing native's.
            view = reference(native.state[0].numpy())
            np.testing.assert_array_equal(view.state[96:128], env.state[96:128])
            view.deal(source.deals[-1]["deck"])
            native.state.copy_(torch.from_numpy(view.state[None].copy()))
        np.testing.assert_array_equal(native.state[0].numpy(), env.state)
        add_step(native_work, checked)
        if snapshot is not None:
            identity.observe(snapshot, action)
    assert source.work["completions"] == 1 and native_work == source.work
    return {
        "source_key": source_key,
        "seat": seat,
        "seed": int(env.state[SEED]),
        "profile": "paper8",
        "deal_source": "logged_round_decks",
        "max_draws": max_draws,
        "actions": list(source.actions),
        "deals": list(source.deals),
        "work": dict(source.work),
        "native_work": native_work,
        "sampling_work": {
            "accepted_deals": len(source.deals),
            "attempts": sum(d["attempts"] for d in source.deals),
            "bounded_draws": sum(d["bounded_draws"] for d in source.deals),
        },
        "final": env.state.tolist(),
        "selected_root": selected,
        "identification_events": identity.events,
        "identification_calls": identity.prediction_calls,
    }


def audit(record, predictors, parent, opponent):
    """Replay logged actions/deals and public inference, without fresh chance.

    This checks trajectory, gate, reconstruction and actor actions. A study must
    bind the actor implementations supplied here to its declared policies.
    """
    if record["profile"] != "paper8" or record["deal_source"] != "logged_round_decks":
        raise ValueError("Unsupported recorded source")
    if not 1 <= len(record["actions"]) <= 1024:
        raise ValueError("Invalid complete-source action count")
    position = 0

    def draw():
        nonlocal position
        if position >= len(record["deals"]):
            raise ValueError("Missing logged deal")
        entry = record["deals"][position]
        if entry["round"] != position:
            raise ValueError("Out-of-order logged deal")
        position += 1
        return DeckSample(
            tuple(entry["deck"]), entry["attempts"], entry["bounded_draws"]
        )

    source = LoggedDealSource(record["seed"], record["profile"], draw)
    seat, cap = record["seat"], record["max_draws"]
    identity = _identity(source, record["source_key"], seat, predictors, cap)
    selected = None
    policy_calls = {"parent": 0, "opponent": 0}
    for decision, action in enumerate(record["actions"]):
        if source.env.state[PHASE] == TERMINAL:
            raise ValueError("Actions after match completion")
        if selected is None and _eligible(source.env, identity, seat, decision, cap):
            selected = _root(source.env, identity, decision)
        snapshot = _snapshot(source.env, identity, seat, decision)
        label = "parent" if source.env.state[PLAYER] == seat else "opponent"
        actor = parent if label == "parent" else opponent
        policy_calls[label] += 1
        if int(actor.act(source.env)) != action:
            raise ValueError(f"Recorded action disagrees with {label} policy")
        source.step(action)
        if snapshot is not None:
            identity.observe(snapshot, action)
    expected = {
        "work": source.work,
        "native_work": source.work,
        "sampling_work": {
            "accepted_deals": len(source.deals),
            "attempts": sum(d["attempts"] for d in source.deals),
            "bounded_draws": sum(d["bounded_draws"] for d in source.deals),
        },
        "final": source.env.state.tolist(),
        "selected_root": selected,
        "identification_events": identity.events,
        "identification_calls": identity.prediction_calls,
    }
    if source.env.state[PHASE] != TERMINAL or position != len(record["deals"]):
        raise ValueError("Expected exactly one complete match and its used deals")
    for key, value in expected.items():
        if record[key] != value:
            raise ValueError(f"Replay mismatch: {key}")
    return {
        "replay_work": source.work,
        "identification_calls": identity.prediction_calls,
        "policy_verification_calls": policy_calls,
        "fresh_external_chance_draws": 0,
    }
