"""The flies should learn poker: fold weak hands, play strong ones, and some die."""

from collections import Counter

from flypoker.table import Table, TableConfig
from flypoker.agent import ACTIONS


def _fold_rate(agent, hs, n=400):
    ctx = {"hand_strength": hs, "pot": 6.0, "to_call": 4.0,
           "n_active": 6, "n_seats": 6, "start_chips": 100.0}
    c = Counter(ACTIONS[agent.act(ctx)[0]] for _ in range(n))
    return c["fold"] / n


def test_flies_learn_to_fold_weak_and_play_strong():
    t = Table(TableConfig(n_seats=6, start_chips=100, ante=1, bet=4, seed=1))
    t.run(4000)
    fly = max(t.seats, key=lambda s: s.hands)     # a well-trained survivor
    fold_weak = _fold_rate(fly, 0.08)
    fold_strong = _fold_rate(fly, 0.95)
    assert fold_weak > fold_strong + 0.25          # clearly folds weak more than strong
    assert fold_strong < 0.15                      # almost always plays strong hands


def test_flies_die_and_get_replaced():
    t = Table(TableConfig(n_seats=6, start_chips=60, ante=1, bet=6, seed=2))
    t.run(3000)
    assert t.deaths > 0                            # the ecology turns over
