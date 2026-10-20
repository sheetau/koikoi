"""Fixed composition fixtures; no new source-law strength samples."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
import torch

from koikoi.agents import EndgameAgent, NeuralAgent
from koikoi.env import BatchEnv
from koikoi.layout import HISTORY, KOI, PHASE, PLAYER, ROUND, TERMINAL, TURN
from koikoi.published_train import packed_observation
from koikoi.reference import ReferenceEnv
from koikoi.upstream import UPSTREAM_PATH, PublishedAgent
from scripts.belief_tail import visible_keys
from scripts.check_belief_tail import add_step, empty_work
from scripts.deferred_belief_tail import deferred_worlds
from scripts.diagnose_history_support import reference
from scripts.enumerate_history_support import canonical_world, context
from scripts.history_conditioned_worlds import enumerate_support
from scripts.logged_public_source import audit, collect
from scripts.revealed_opponent_models import StrictPublishedPrediction
from scripts.run_arbitration_study import PARENTS
from scripts.uniform_deal_source import DeckSample, LoggedDealSource

# First eligible record in each parent/opponent/seat cell of the closed cohort.
# Root availability alone selects these fixtures; no gain/outcome is consulted.
INDICES = (1, 3, 4, 5, 7, 8, 10, 46)
PROTOCOL = "700e11608e3ced60919ca7ef31da460785ff9391459ac83c7e8a16f4a1a0c170"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def pinned():
    torch.set_num_threads(2)
    base = Path("runs/prospective_incumbent")
    protocol, summary = read(base / "protocol.json"), read(base / "summary.json")
    assert sha(base / "protocol.json") == summary["protocol_sha256"] == PROTOCOL
    assert summary["status"] == "complete"
    for path, digest in PARENTS:
        assert sha(path) == digest == protocol["inputs"][path]
    for kind in ("wp", "point"):
        for phase in ("discard", "pick", "koikoi"):
            path = str(UPSTREAM_PATH / f"model_agent/{phase}_rl_{kind}.pt")
            key = Path(path).as_posix()
            assert sha(path) == protocol["inputs"][key]
    parents = [NeuralAgent(path, "cpu", deterministic=True) for path, _ in PARENTS]
    published = {name: PublishedAgent(name) for name in ("wp", "point")}
    opponents = {name: EndgameAgent(actor) for name, actor in published.items()}
    predictors = {
        name: StrictPublishedPrediction(actor) for name, actor in published.items()
    }
    metrics = defaultdict(Counter)
    rows = []
    yield base, protocol, summary, parents, opponents, predictors, metrics, rows
    print(
        json.dumps(
            {
                "fixed_fixture_indices": INDICES,
                "completed_fixtures": len(rows),
                "work": dict(metrics),
                "fixtures": rows,
            },
            sort_keys=True,
        )
    )


def recover_decks(source):
    """Independent rule replay recovers old seeded decks solely as test tapes."""
    env = ReferenceEnv(source["seed"], "paper8")
    decks = [tuple(map(int, env.state[48:96]))]
    work = empty_work()
    for action in source["actions"]:
        before = int(env.state[ROUND])
        add_step(work, env.step(action))
        if env.state[PHASE] != TERMINAL and env.state[ROUND] != before:
            decks.append(tuple(map(int, env.state[48:96])))
    assert env.state.tolist() == source["final"] and work == source["work"]
    return decks, work


def inputs(native, state):
    native.state.copy_(torch.from_numpy(state[None].copy()))
    obs, mask = packed_observation(native)
    return obs.clone(), mask.clone()


def compare_own_history_inputs(root, support, native):
    """Hidden-hand alternatives cannot alter the learner's past packed inputs."""
    viewer = int(root.state[PLAYER])
    actual = tuple(np.flatnonzero(root.state[:48] == 1 - viewer))
    hand = next(h for h in support.hands if h != actual)
    targets = [root.state, canonical_world(root.state, hand)]
    envs = [reference(state) for state in targets]
    for env in envs:
        env.deal(env.state[48:96].copy())
    work, checks = empty_work(), 0
    while (envs[0].state[TURN], envs[0].state[PHASE]) != (
        root.state[TURN],
        root.state[PHASE],
    ):
        left, right = envs
        assert work["decisions"] < 128
        assert (
            visible_keys(np.stack([left.state, right.state]), viewer)[0]
            == visible_keys(np.stack([left.state, right.state]), viewer)[1]
        )
        if left.state[PLAYER] == viewer:
            a, b = inputs(native, left.state), inputs(native, right.state)
            assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            checks += 1
        phase, turn = int(left.state[PHASE]), int(left.state[TURN])
        encoded = int(root.state[HISTORY + 5 * turn + (0, 1, 3, 4)[phase]])
        action = 48 + encoded - 1 if phase == KOI else encoded - 1
        for env in envs:
            result = env.step(action)
            assert not result[1]
            add_step(work, result)
    for env, target in zip(envs, targets, strict=True):
        np.testing.assert_array_equal(env.state, target)
    assert checks > 0
    return work, checks


@pytest.mark.parametrize("index", INDICES)
def test_pinned_actors_logged_source_and_complete_hand_posterior(pinned, index):
    base, protocol, summary, parents, opponents, predictors, metrics, rows = pinned
    folder = base / f"chunk_{index // 16:03}"
    assert (
        sha(folder / "complete.json")
        == summary["chunk_bindings"][str(folder / "complete.json")]
    )
    complete = read(folder / "complete.json")
    assert sha(folder / "records.json") == complete["records_sha256"]
    old = read(folder / "records.json")[index % 16]["source"]
    case = protocol["cases"][index]
    parent, opponent = parents[case["parent"]], opponents[case["opponent"]]
    decks, recovered = recover_decks(old)
    iterator = iter(decks)
    # Deterministic tape for wiring verification; it makes no IID sampling claim.
    source = LoggedDealSource(
        case["seed"], "paper8", lambda: DeckSample(next(iterator), 1, 47)
    )
    native = BatchEnv(1, debug=True)
    record = collect(
        source,
        json.dumps(case, sort_keys=True),
        case["seat"],
        parent,
        opponent,
        predictors,
        native,
        max_draws=5,
    )
    for field in (
        "actions",
        "work",
        "final",
        "selected_root",
        "identification_events",
        "identification_calls",
    ):
        assert record[field] == old[field], field
    assert len(record["deals"]) == len(decks)
    verified = audit(record, predictors, parent, opponent)
    assert verified["fresh_external_chance_draws"] == 0
    root = reference(record["selected_root"]["state"])
    identified = record["selected_root"]["public_identification"]["model"]
    assert identified == case["opponent"]  # Auditor-only assertion, not a gate input.
    support = enumerate_support(root, {identified: opponent}, max_hands=1000)
    assert support.status == "complete"
    true_hand = tuple(np.flatnonzero(root.state[:48] == 1 - root.state[PLAYER]))
    assert support.compatible[support.hands.index(true_hand)] == (True,)
    worlds = deferred_worlds(root, support, max_draws=5, max_hands=1000)
    compatible = [
        h for h, yes in zip(support.hands, support.compatible, strict=True) if yes[0]
    ]
    assert len(worlds.states) == len(compatible)
    n = len(context(root.state)[0]) - len(true_hand)
    assert n - worlds.draws == 8
    assert set(map(int, worlds.weights)) == {math.perm(n, worlds.draws)}
    assert set(map(int, worlds.models)) == {0}
    assert worlds.model_names == (identified,) and worlds.viewer == case["seat"]
    expected_key = visible_keys(root.state[None], worlds.viewer)[0]
    assert set(visible_keys(worlds.states, worlds.viewer)) == {expected_key}
    for state, hand in zip(worlds.states, compatible, strict=True):
        assert tuple(np.flatnonzero(state[:48] == 1 - worlds.viewer)) == hand
        np.testing.assert_array_equal(reference(state).observe(), root.observe())
    # Initial slot order cannot influence the actual packed neural inputs.
    changed = root.state.copy()
    for start in (48, 56, 64):
        changed[start : start + 8] = changed[start : start + 8][::-1]
    original_inputs, permuted_inputs = (
        inputs(native, root.state),
        inputs(native, changed),
    )
    assert torch.equal(original_inputs[0], permuted_inputs[0])
    assert torch.equal(original_inputs[1], permuted_inputs[1])
    history_work, checks = compare_own_history_inputs(root, support, native)
    for label, work in (
        ("seeded_fixture_replay", recovered),
        ("logged_fixture_source", record["work"]),
        ("native_verification", record["native_work"]),
        ("logged_source_audit", verified["replay_work"]),
        ("counterfactual_history_replay", history_work),
        ("history_support", support.counts),
        ("source_actor_calls", verified["policy_verification_calls"]),
        ("audit_actor_calls", verified["policy_verification_calls"]),
        ("source_identification_calls", record["identification_calls"]),
        ("audit_identification_calls", verified["identification_calls"]),
    ):
        metrics[label].update(work)
    rows.append(
        {
            "index": index,
            "candidates": support.candidate_hands,
            "valid_hands": support.valid_hands,
            "compatible_hands": len(compatible),
            "own_history_input_checks": checks,
        }
    )


def test_uniform_deal_conditioning_by_exhaustive_small_deck_counting():
    # A ten-card analogue: two cards per month and per initial hand/field.
    # Condition on own hand {0,2}, field {4,6}, opponent discard8 and draw7.
    # Enumerate all remaining slot permutations, including each group order.
    before, after, prefixes = Counter(), Counter(), Counter()
    for own in permutations((0, 2)):
        for field in permutations((4, 6)):
            for other in permutations((1, 3, 5, 7, 8, 9)):
                deck = own + other[:2] + field + other[2:]
                hand = deck[2:4]
                if 8 not in hand or deck[-1] != 7 or hand[0] // 2 == hand[1] // 2:
                    continue
                remaining = next(c for c in hand if c != 8)
                before[remaining] += 1
                # Fixed observation-only opponent; its observed discard is8.
                action = max(hand) if sum(hand) % 4 == 1 else min(hand)
                if action == 8:
                    after[remaining] += 1
                    prefixes[remaining, deck[-2]] += 1
    assert before == {1: 48, 3: 48, 5: 48}  # {8,9} is an invalid initial hand.
    assert after == {1: 48, 5: 48}  # Observed play removes a valid hand.
    assert {Fraction(n, sum(after.values())) for n in after.values()} == {
        Fraction(1, 2)
    }
    assert len(prefixes) == 6 and set(prefixes.values()) == {16}
    for hand in after:
        assert {card for h, card in prefixes if h == hand} == {1, 3, 5, 9} - {hand}
