"""Game integration, privacy and settlement checks; not strength evaluation."""

import json
import random
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
import torch

from koikoi.agents import GreedyAgent
from koikoi.game import GameEnv, GameSession, choose_dealer, yaku_rows
from koikoi.game_opponent import GameOpponent
from koikoi.game_server import GameService, make_server
from koikoi.layout import DEALER, PHASE, PLAYER, ROUND, SEED, TERMINAL
from koikoi.play import ASSETS, CHECKPOINT
from koikoi.reference import ReferenceEnv, score
from koikoi.rule_variants import Variant, VariantEnv
from koikoi.variant_native import VariantBackend, VariantBatch


@pytest.mark.parametrize("flags", [0, 8, 16, 24])
def test_complete_twelve_rounds_equal_existing_transitions(flags):
    rng = random.Random(728)
    phases, kinds = set(), set()
    for seed in [71, 93, 115]:
        session = GameSession(GreedyAgent(), seed, flags)
        reference = VariantEnv(seed)
        reference.variant = Variant("test12", flags, rounds=12, bankrupt=False)
        reference.state[:] = session.env.state
        rounds = 0
        for _ in range(1500):
            if session.env.state[PHASE] == TERMINAL:
                break
            if session.paused:
                session.next_round()
            player = int(session.env.state[PLAYER])
            phase = int(session.env.state[PHASE])
            phases.add(phase)
            action = (
                GreedyAgent().act(reference)
                if player
                else rng.choice(np.flatnonzero(reference.legal()).tolist())
            )
            expected = reference.step(action)
            session.act(None if player else action)
            np.testing.assert_array_equal(session.env.state, reference.state)
            kinds.update(e["kind"] for e in session.events)
            if expected[1]:
                rounds += 1
                assert session.paused
                view = session.response()["view"]
                assert view["round"] == rounds
                assert view["legal"] == []
                assert sum(view["points"]) == 60
                # The result table is preserved instead of leaking the next deal.
                assert session.end["view"]["captured"] == view["captured"]
        assert rounds == 12
        assert session.env.state[PHASE] == TERMINAL
        with pytest.raises(ValueError):
            session.next_round()
    assert phases == {0, 1, 2, 3}
    assert {"play", "draw", "capture", "place", "koi", "round_end"} <= kinds


def test_base_game_equal_research12_reference():
    game = GameEnv(77, 0)
    reference = ReferenceEnv(77, "research12")
    reference.state[DEALER] = reference.state[PLAYER] = 0
    for _ in range(1000):
        if game.state[PHASE] == TERMINAL:
            break
        action = GreedyAgent().act(game)
        a, b = game.step(action), reference.step(action)
        np.testing.assert_array_equal(a, b)
        # VariantEnv additionally counts completed rounds in its reserved slot.
        np.testing.assert_array_equal(game.state[:112], reference.state[:112])
        np.testing.assert_array_equal(game.state[128:], reference.state[128:])
    assert game.state[ROUND] == 11


def test_view_has_no_hidden_card_identities_or_future_deal():
    game = GameEnv(197, 0)
    original = game.view()
    s = game.state
    ai = np.flatnonzero(s[:48] == 1).tolist()
    stock = np.flatnonzero(s[:48] == -1).tolist()
    a, b = ai[0], stock[0]
    s[a], s[b] = s[b], s[a]
    s[56:64] = [b if c == a else c for c in s[56:64]]
    s[72:96] = [a if c == b else c for c in s[72:96]]
    s[72:96] = s[72:96][::-1]
    s[SEED] = 499
    assert original == game.view()
    assert set(original) == {
        "hand",
        "opponent_slots",
        "field",
        "captured",
        "yaku",
        "yaku_points",
        "koi",
        "points",
        "dealer",
        "player",
        "round",
        "phase",
        "pending",
        "pending_slot",
        "stock_count",
        "legal",
    }


def test_public_pending_hand_slot_tracks_the_revealed_card():
    game = GameEnv(197, 0)
    seats = set()
    for _ in range(1000):
        view = game.view()
        if view["phase"] == 1:
            seat = view["player"]
            slots = game.state[48 + seat * 8 : 56 + seat * 8].tolist()
            assert slots[view["pending_slot"]] == view["pending"]
            assert view["pending"] not in view["hand"]
            seats.add(seat)
        else:
            assert view["pending_slot"] == -1
        if seats == {0, 1} or game.state[PHASE] == TERMINAL:
            break
        game.step(int(np.flatnonzero(game.legal())[0]))
    assert seats == {0, 1}


def test_yaku_breakdown_matches_scoring_including_overlap():
    rng = random.Random(9)
    for _ in range(1000):
        cards = rng.sample(range(48), rng.randrange(49))
        koi = rng.randrange(9)
        assert sum(r["points"] for r in yaku_rows(cards, koi)) == score(cards, koi)


def test_light_upgrades_are_exclusive_and_other_overlaps_remain_additive():
    for cards, expected in [
        ([0, 8, 28], "three"), ([0, 8, 28, 44], "four"),
        ([0, 8, 28, 40], "rain_four"), ([0, 8, 28, 40, 44], "five"),
    ]:
        rows = yaku_rows(cards)
        assert [r["key"] for r in rows] == [expected]
        assert sum(r["points"] for r in rows) == score(cards)
    assert yaku_rows([0, 8, 40]) == []  # Rain does not make three lights.
    rows = yaku_rows([0, 8, 28, 32])
    assert {r["key"] for r in rows} == {"three", "flower", "moon"}
    assert sum(r["points"] for r in rows) == score([0, 8, 28, 32]) == 7
    rows = yaku_rows([1, 5, 9, 21, 33, 37])
    assert {r["key"] for r in rows} == {"red", "blue", "red_blue", "ribbons"}
    assert sum(r["points"] for r in rows) == score([1, 5, 9, 21, 33, 37]) == 22


@pytest.mark.parametrize("rounds", [1, 3, 6, 12])
def test_selected_match_length_ends_at_exact_round(rounds):
    session = GameSession(GreedyAgent(), seed=71, rounds=rounds)
    ended = 0
    for _ in range(1500):
        if session.env.state[PHASE] == TERMINAL:
            break
        if session.paused:
            session.next_round()
        action = None if session.env.state[PLAYER] else int(GreedyAgent().act(session.env))
        session.act(action)
        if session.paused:
            ended += 1
            assert session.response()["view"]["rounds"] == rounds
            assert session.response()["view"]["round"] == ended
    assert ended == rounds and session.env.state[PHASE] == TERMINAL


def test_restart_applies_new_settings_without_accepting_invalid_lengths():
    service = GameService(lambda _: GreedyAgent())
    service.request("/start", {"revision": 0, "rounds": 1})
    previous = service.session
    for invalid in [True, 0, 2, 24, "6", 6.0]:
        with pytest.raises(ValueError):
            service.request("/start", {"revision": 1, "rounds": invalid})
        assert service.session is previous and service.revision == 1
    result = service.request("/start", {"revision": 1, "rounds": 6, "flags": 24})
    assert service.session is not previous
    assert result["view"]["round"] == 1 and result["view"]["rounds"] == 6
    assert result["view"]["points"] == [30, 30]
    assert result["events"][0]["kind"] == "opening"


def test_dealer_month_value_and_tie():
    class Fixed:
        pairs = iter([(2, 3), (0, 1)])

        def sample(self, *_args):
            return next(self.pairs)

    winner, reveals = choose_dealer(Fixed())
    assert winner == 0
    assert reveals[0]["winner"] == -1
    assert len(reveals) == 2


def test_illegal_wrong_seat_and_revisions_do_not_mutate():
    service = GameService(lambda _: GreedyAgent())
    result = service.request("/start", {"revision": 0})
    before = service.session.env.state.copy()
    with pytest.raises(ValueError):
        service.request("/start", {"revision": 0})
    with pytest.raises(ValueError):
        service.request("/action", {"revision": 1, "action": 99})
    with pytest.raises(ValueError):
        service.request("/action", {"revision": 1, "action": True})
    np.testing.assert_array_equal(before, service.session.env.state)
    assert service.revision == result["revision"] == 1


def test_authenticated_http_bridge():
    server = make_server(GameService(lambda _: GreedyAgent()), "test-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/start"
    data = json.dumps({"revision": 0, "opponent": "greedy"}).encode()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(urllib.request.Request(url, data=data))
        assert exc.value.code == 403
        request = urllib.request.Request(
            url, data=data, headers={"Authorization": "Bearer test-token"}
        )
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
        assert result["revision"] == 1
        assert result["events"][0]["kind"] == "opening"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="Trained checkpoint not restored")
@pytest.mark.parametrize("flags", [0, 8, 16, 24])
def test_existing_trained_ai_completes_twelve_round_product_match(flags):
    session = GameSession(GameOpponent("trained"), seed=7200041, flags=flags)
    for _ in range(1000):
        if session.env.state[PHASE] == TERMINAL:
            break
        if session.paused:
            session.next_round()
        session.act(
            None if session.env.state[PLAYER] == 1 else GreedyAgent().act(session.env)
        )
    assert session.env.state[PHASE] == TERMINAL
    assert session.response()["view"]["round"] == 12


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_game_configuration_matches_shared_native_transitions(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    refs = [GameEnv(68001 + i, i % 2, f) for i, f in enumerate([0, 8, 16, 24])]
    batch = VariantBatch(
        [68001 + i for i in range(4)], "paper_control", device, VariantBackend(device)
    )
    batch.state.copy_(torch.tensor(np.asarray([e.state for e in refs]), device=device))
    rng = random.Random(818)
    for _ in range(1500):
        if all(e.state[PHASE] == TERMINAL for e in refs):
            break
        actions, expected = [], []
        for env in refs:
            if env.state[PHASE] == TERMINAL:
                actions.append(0)
                expected.append(np.zeros(4, dtype=np.int32))
            else:
                action = rng.choice(np.flatnonzero(env.legal()).tolist())
                actions.append(action)
                expected.append(env.step(action))
        np.testing.assert_array_equal(batch.step(actions).cpu().numpy(), expected)
        np.testing.assert_array_equal(
            batch.state.cpu().numpy(), [e.state for e in refs]
        )
    assert all(e.state[PHASE] == TERMINAL for e in refs)


@pytest.mark.skipif(not ASSETS.is_dir(), reason="Pinned source not restored")
@pytest.mark.parametrize("name", ["rl-wp", "rl-point"])
def test_optional_published_opponents_use_legal_twelve_round_actions(name):
    agent = GameOpponent(name)
    env = GameEnv(23, 1)
    env.state[ROUND] = 11
    before = env.state.copy()
    action = agent.act(env)
    assert env.legal()[action]
    np.testing.assert_array_equal(env.state, before)
