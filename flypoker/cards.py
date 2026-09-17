"""A 52-card deck and a correct 7-card Texas Hold'em hand evaluator.

Cards are (rank, suit): rank 2..14 (14 = Ace), suit 0..3. ``best_of_seven``
returns a comparable score for the best 5-of-7 hand (bigger = better), so two
players are compared just by ``score_a > score_b``. ``hand_strength`` maps a
score to a monotone 0..1 value used as the fly's perceived hand quality.
"""

from __future__ import annotations

import itertools
import random

RANKS = list(range(2, 15))          # 2..14 (J=11, Q=12, K=13, A=14)
SUITS = list(range(4))
DECK = [(r, s) for r in RANKS for s in SUITS]

RANK_CHAR = {11: "J", 12: "Q", 13: "K", 14: "A"}
SUIT_CHAR = "♣♦♥♠"

# hand categories (higher = stronger)
HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
CATEGORY_NAME = {
    HIGH_CARD: "high card", PAIR: "pair", TWO_PAIR: "two pair", TRIPS: "trips",
    STRAIGHT: "straight", FLUSH: "flush", FULL_HOUSE: "full house", QUADS: "quads",
    STRAIGHT_FLUSH: "straight flush",
}


def card_str(card) -> str:
    r, s = card
    return f"{RANK_CHAR.get(r, str(r))}{SUIT_CHAR[s]}"


def _straight_high(ranks_set) -> int:
    """Return the high card of a 5-in-a-row within a set of ranks, else 0.

    Handles the wheel A-2-3-4-5 (Ace low), where the high card is 5.
    """
    rs = set(ranks_set)
    if {14, 2, 3, 4, 5} <= rs:
        best = 5
    else:
        best = 0
    for high in range(14, 5, -1):
        if all(high - i in rs for i in range(5)):
            return high
    return best


def _score_five(cards) -> tuple:
    """Score exactly 5 cards -> (category, tiebreakers...) comparable tuple."""
    ranks = sorted((c[0] for c in cards), reverse=True)
    suits = [c[1] for c in cards]
    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    # order ranks by (count, rank) descending -> kickers
    by_count = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    kickers = tuple(r for r, c in by_count)
    count_shape = tuple(c for r, c in by_count)

    is_flush = len(set(suits)) == 1
    sh = _straight_high(counts.keys()) if len(counts) == 5 else 0

    if is_flush and sh:
        return (STRAIGHT_FLUSH, sh)
    if count_shape[0] == 4:
        return (QUADS, kickers)              # quad rank, then kicker
    if count_shape[:2] == (3, 2):
        return (FULL_HOUSE, kickers)
    if is_flush:
        return (FLUSH, tuple(ranks))
    if sh:
        return (STRAIGHT, sh)
    if count_shape[0] == 3:
        return (TRIPS, kickers)
    if count_shape[:2] == (2, 2):
        return (TWO_PAIR, kickers)
    if count_shape[0] == 2:
        return (PAIR, kickers)
    return (HIGH_CARD, tuple(ranks))


def best_of_seven(seven) -> tuple:
    """Best 5-of-7 score. ``seven`` is a list of 7 (rank, suit) cards."""
    return max(_score_five(combo) for combo in itertools.combinations(seven, 5))


# --- normalisation: pack a score to a single int, then to 0..1 -----------
def _pack(score) -> int:
    cat = score[0]
    tb = score[1] if isinstance(score[1], tuple) else (score[1],)
    tb = (tb + (0, 0, 0, 0, 0))[:5]
    v = cat
    for k in tb:
        v = v * 15 + k
    return v


_MAX_PACK = _pack((STRAIGHT_FLUSH, 14))
_MIN_PACK = _pack((HIGH_CARD, (7, 5, 4, 3, 2)))


def hand_strength(score) -> float:
    """Monotone 0..1 quality of a hand score (rough, for the fly's perception)."""
    v = _pack(score)
    x = (v - _MIN_PACK) / (_MAX_PACK - _MIN_PACK)
    return max(0.0, min(1.0, x))


class Deck:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.cards = list(DECK)

    def shuffle(self):
        self.rng.shuffle(self.cards)
        self._i = 0
        return self

    def draw(self, n):
        out = self.cards[self._i:self._i + n]
        self._i += n
        return out
