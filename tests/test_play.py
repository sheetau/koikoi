"""Human-control boundaries, information privacy and complete playable matches."""

import tkinter as tk

import numpy as np
import pytest

from koikoi.agents import GreedyAgent
from koikoi.layout import CONTINUE, KOI, PHASE, PLAYER, POINTS, ROUND, TERMINAL
from koikoi.play import ASSETS, CHECKPOINT, PlaySession, PlayWindow, load_opponent
from koikoi.reference import ReferenceEnv


def test_complete_matches_match_reference_and_pause_at_boundaries():
    rng = np.random.default_rng(42)
    phases = set()
    continues = rounds = 0
    for seed in range(20):
        opponent = GreedyAgent()
        game = PlaySession(opponent, seed)
        reference = ReferenceEnv(seed, "paper8")
        for _ in range(1000):
            if game.finished:
                break
            np.testing.assert_array_equal(game.env.state, reference.state)
            phases.add(int(reference.state[PHASE]))
            if game.human_turn:
                action = int(rng.choice(np.flatnonzero(reference.legal())))
                continues += action == CONTINUE
                game.human_action(action)
            else:
                action = opponent.act(reference)
                game.ai_action()
            result = reference.step(action)
            np.testing.assert_array_equal(game.env.state, reference.state)
            if result[1]:
                rounds += 1
                assert game.paused and game.result
                before = game.env.state.copy()
                with pytest.raises(ValueError):
                    game.human_action(0)
                with pytest.raises(ValueError):
                    game.ai_action()
                np.testing.assert_array_equal(before, game.env.state)
                if not game.finished:
                    game.next_round()
        assert game.finished and reference.state[PHASE] == TERMINAL
        assert sum(game.view()["points"]) == 60
        with pytest.raises(ValueError):
            game.next_round()
    assert phases == {0, 1, 2, 3}
    assert continues > 0 and rounds >= 20


def test_wrong_turn_and_illegal_actions_do_not_change_game():
    game = PlaySession(GreedyAgent(), 1)
    before = game.env.state.copy()
    wrong_side = game.ai_action if game.human_turn else lambda: game.human_action(0)
    with pytest.raises(ValueError):
        wrong_side()
    np.testing.assert_array_equal(before, game.env.state)
    game.env.state[PLAYER] = 0
    before = game.env.state.copy()
    with pytest.raises(ValueError):
        game.human_action(48)
    np.testing.assert_array_equal(before, game.env.state)
    assert game.moves == 0


def test_human_view_invariant_to_hidden_hand_and_stock():
    game = PlaySession(GreedyAgent(), 10)
    before = game.view()
    s = game.env.state
    hidden_hand = np.flatnonzero(s[:48] == 1)
    hidden_stock = np.flatnonzero(s[:48] == -1)
    a, b = int(hidden_hand[0]), int(hidden_stock[0])
    s[a], s[b] = s[b], s[a]
    s[48:96] = s[48:96][::-1]
    assert game.view() == before
    assert set(before) == {
        "hand",
        "field",
        "captured",
        "opponent_count",
        "pending",
        "points",
        "koi",
        "yaku",
        "dealer",
        "round",
        "phase",
        "legal",
    }


@pytest.fixture
def window():
    if not ASSETS.is_dir():
        pytest.skip("Pinned upstream card images are not restored")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    app = PlayWindow(root, "greedy", seed=1, agent=GreedyAgent())
    app.cancel_timer()
    yield app
    if root.winfo_exists():
        app.close()


def test_window_koi_buttons_round_pause_and_restart(window):
    game = window.session
    # A concrete final-round stop: the human's three dry lights score five.
    game.env.state[:48] = -1
    game.env.state[[0, 8, 28]] = 3
    game.env.state[PLAYER] = 0
    game.env.state[PHASE] = KOI
    game.env.state[ROUND] = 7
    window.render()
    assert str(window.stop_button["state"]) == "normal"
    assert str(window.koi_button["state"]) == "normal"
    window.stop_button.invoke()
    assert game.finished and game.paused
    assert game.env.state[POINTS : POINTS + 2].tolist() == [35, 25]
    assert "あなたの勝ち" in game.result
    assert str(window.stop_button["state"]) == "disabled"
    assert str(window.next_button["state"]) == "disabled"
    assert window.timer is None
    window.restart()
    assert window.session is not game and not window.session.finished
    assert window.session.moves == 0
    window.cancel_timer()


def test_window_cancels_old_ai_callback_and_renders_dense_field(window):
    window.session.env.state[PLAYER] = 1
    window.schedule_ai()
    old_timer = window.timer
    assert old_timer is not None
    window.start("greedy", seed=2)
    assert old_timer not in window.root.tk.call("after", "info")
    window.cancel_timer()
    # Large field: eight original cards in three months, one in each other month.
    field = [0, 1, 2, 4, 5, 6, 8, 9] + list(range(12, 48, 4))
    window.session.env.state[:48] = -1
    window.session.env.state[field] = 2
    hand = sorted(set(range(48)) - set(field))[:8]
    window.session.env.state[hand] = 0
    window.root.attributes("-alpha", 0)
    window.root.geometry("1000x750")
    window.root.deiconify()
    window.root.update_idletasks()
    window.render()
    bottom = max(window.board.bbox(f"card-{c}")[3] for c in field)
    assert bottom < window.board.winfo_height() - 165  # Above the hand heading.
    assert window.side.bbox("all")[3] <= window.side.winfo_height()


@pytest.mark.upstream
def test_canonical_model_plays_complete_human_session():
    if not CHECKPOINT.is_file():
        pytest.skip("Canonical trained checkpoint is not restored")
    game = PlaySession(load_opponent("trained"), 7200017)
    human = GreedyAgent()
    for _ in range(1000):
        if game.finished:
            break
        if game.paused:
            game.next_round()
        if game.human_turn:
            game.human_action(human.act(game.env))
        else:
            game.ai_action()
    assert game.finished and game.moves > 20
    assert sum(game.view()["points"]) == 60


@pytest.mark.upstream
@pytest.mark.parametrize("name", ["rl-wp", "rl-point"])
def test_original_opponents_load_and_choose_legal_action(name):
    if not ASSETS.is_dir():
        pytest.skip("Pinned upstream checkout is not restored")
    game = PlaySession(load_opponent(name), 7200018)
    game.env.state[PLAYER] = 1
    game.ai_action()
    assert game.moves == 1
