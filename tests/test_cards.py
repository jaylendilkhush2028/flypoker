from flypoker.cards import (_score_five, best_of_seven, hand_strength, Deck,
                            STRAIGHT_FLUSH, QUADS, FULL_HOUSE, FLUSH, STRAIGHT,
                            TRIPS, TWO_PAIR, PAIR, HIGH_CARD)


def h(*cs):
    return list(cs)


def test_category_ordering_is_strict():
    hands = [
        h((14, 0), (13, 0), (12, 0), (11, 0), (10, 0)),   # straight flush
        h((9, 0), (9, 1), (9, 2), (9, 3), (2, 0)),        # quads
        h((8, 0), (8, 1), (8, 2), (3, 0), (3, 1)),        # full house
        h((14, 0), (11, 0), (8, 0), (5, 0), (2, 0)),      # flush
        h((14, 0), (2, 1), (3, 2), (4, 3), (5, 0)),       # wheel straight
        h((7, 0), (7, 1), (7, 2), (4, 0), (2, 0)),        # trips
        h((10, 0), (10, 1), (6, 0), (6, 1), (3, 0)),      # two pair
        h((5, 0), (5, 1), (9, 0), (4, 0), (2, 0)),        # pair
        h((14, 0), (11, 1), (8, 0), (5, 0), (3, 0)),      # high card
    ]
    scores = [_score_five(x) for x in hands]
    cats = [s[0] for s in scores]
    assert cats == [STRAIGHT_FLUSH, QUADS, FULL_HOUSE, FLUSH, STRAIGHT, TRIPS, TWO_PAIR, PAIR, HIGH_CARD]
    assert all(scores[i] > scores[i + 1] for i in range(len(scores) - 1))


def test_wheel_straight_recognized():
    s = _score_five(h((14, 0), (2, 1), (3, 2), (4, 3), (5, 0)))
    assert s[0] == STRAIGHT and s[1] == 5      # 5-high straight


def test_flush_beats_straight():
    flush = _score_five(h((14, 0), (10, 0), (7, 0), (4, 0), (2, 0)))
    straight = _score_five(h((9, 0), (8, 1), (7, 2), (6, 3), (5, 0)))
    assert flush > straight


def test_best_of_seven_picks_the_flush():
    seven = h((14, 0), (11, 0), (8, 0), (5, 0), (2, 0), (14, 1), (14, 2))  # flush available
    assert best_of_seven(seven)[0] == FLUSH


def test_hand_strength_monotone_and_bounded():
    weak = hand_strength(_score_five(h((7, 0), (5, 1), (4, 2), (3, 3), (2, 0))))
    strong = hand_strength(_score_five(h((14, 0), (14, 1), (14, 2), (14, 3), (13, 0))))
    assert 0.0 <= weak < strong <= 1.0


def test_deck_deals_distinct_cards():
    d = Deck().shuffle()
    cards = d.draw(20)
    assert len(set(cards)) == 20
