import pytest

from scripts.check_public_hand_revelation import revealed_hands


def test_revelation_uses_only_completed_public_discards():
    events = []
    for card in range(7):
        events += [(0, 0, 0, 16 + card), (0, 0, 1, card), (0, 1, 1, 24 + card)]
    assert revealed_hands(events, 0) == []
    # Capture choices and the viewer's own cards are not opponent hand plays.
    events += [(0, 2, 1, 7), (0, 0, 0, 7)]
    assert revealed_hands(events, 0) == []
    events.append((0, 0, 1, 7))
    assert revealed_hands(events, 0) == [
        {
            "round": 0,
            "available_after_decision": len(events),
            "initial_opponent_hand": list(range(8)),
        }
    ]
    previous = revealed_hands(events, 0)
    events += [(1, 0, 1, c) for c in range(7)]
    assert revealed_hands(events, 0) == previous
    events.append((1, 0, 1, 7))
    assert len(revealed_hands(events, 0)) == 2


def test_rejects_duplicate_hand_discard_and_invalid_viewer():
    with pytest.raises(ValueError, match="duplicate"):
        revealed_hands([(0, 0, 1, 3), (0, 0, 1, 3)], 0)
    with pytest.raises(ValueError, match="viewer"):
        revealed_hands([], 2)
