"""A poker table of fruit flies: deal, bet, showdown, pay out, live and die.

A **simplified, tractable poker**: every active fly antes, is dealt 2 hole cards
plus 5 shared community cards, then makes ONE commitment decision --
**fold / call / raise** (commit 0 / B / 2B more) based on its perceived hand
strength. Non-folders go to showdown; the best real 5-of-7 hand takes the whole
pot. Winning fires dopamine, losing fires pain, and the result rewrites the
fly's KC->action weights. A fly that can't cover the ante **dies** and a fresh
fly buys into its seat.

(One simultaneous decision per hand keeps the huge sequential-betting tree of
real No-Limit Hold'em out of scope while keeping real cards, real hand rankings,
folding, pot odds, and bluffing.)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .cards import Deck, best_of_seven, hand_strength, card_str
from .agent import FlyAgent, ACTIONS

FLY_NAMES = ["Buzz", "Zip", "Dart", "Gnat", "Midge", "Vex", "Pip", "Skit", "Blip", "Fizz"]


@dataclass
class TableConfig:
    n_seats: int = 6
    start_chips: float = 100.0
    ante: float = 1.0
    bet: float = 4.0            # B: a "call" commits B, a "raise" commits 2B
    seed: int = 0


class Table:
    def __init__(
        self,
        cfg: TableConfig | None = None,
        agent_kwargs: dict | None = None,
        agent_factory=None,
    ):
        """``agent_factory(seed, name)`` builds a seat's brain; defaults to the
        fast behavioural :class:`FlyAgent`. Pass a factory that returns a
        :class:`flypoker.connectome_agent.ConnectomeFlyAgent` to seat the real
        FlyWire mushroom body instead (see ``scripts/run_connectome_table.py``).
        Any agent with the same ``act`` / ``learn`` API and life-stat fields works.
        """
        self.cfg = cfg or TableConfig()
        self.agent_kwargs = agent_kwargs or {}
        self.agent_factory = agent_factory or (
            lambda seed, name: FlyAgent(seed=seed, name=name, **self.agent_kwargs))
        self.rng = random.Random(self.cfg.seed)
        self._next_id = 0
        self.hand_no = 0
        self.deaths = 0
        self.seats = [self._new_fly(i) for i in range(self.cfg.n_seats)]

    def _new_fly(self, seat: int):
        fid = self._next_id
        self._next_id += 1
        a = self.agent_factory(seed=self.cfg.seed * 1000 + fid,
                               name=f"{FLY_NAMES[seat % len(FLY_NAMES)]}#{fid}")
        a.chips = self.cfg.start_chips
        a.alive = True
        a.born_hand = self.hand_no
        a.seat = seat
        return a

    def play_hand(self) -> dict:
        cfg = self.cfg
        self.hand_no += 1

        # replace any fly that can't cover the ante (it busted last hand)
        newborn = []
        for i, fly in enumerate(self.seats):
            if fly.chips < cfg.ante:
                self.deaths += 1
                self.seats[i] = self._new_fly(i)
                newborn.append(i)

        active = list(range(cfg.n_seats))
        deck = Deck(random.Random(self.rng.random())).shuffle()
        community = deck.draw(5)
        holes = {i: deck.draw(2) for i in active}

        pot = 0.0
        committed = {}
        for i in active:
            self.seats[i].chips -= cfg.ante
            committed[i] = cfg.ante
            pot += cfg.ante

        # each fly perceives its hand + context and commits
        decisions = {}
        contexts = {}
        for i in active:
            fly = self.seats[i]
            score = best_of_seven(holes[i] + community)
            hs = hand_strength(score)
            ctx = {"hand_strength": hs, "pot": pot, "to_call": cfg.bet,
                   "n_active": len(active), "n_seats": cfg.n_seats, "start_chips": cfg.start_chips}
            a_idx, learn_ctx = fly.act(ctx)
            want = (0.0, cfg.bet, 2 * cfg.bet)[a_idx]
            put = min(want, fly.chips)            # all-in if short
            fly.chips -= put
            committed[i] += put
            pot += put
            decisions[i] = (a_idx, put)
            contexts[i] = (learn_ctx, score, hs)

        contenders = [i for i in active if decisions[i][0] != 0 and committed[i] > cfg.ante]
        winners = []
        if contenders:
            best = max(best_of_seven(holes[i] + community) for i in contenders)
            winners = [i for i in contenders if best_of_seven(holes[i] + community) == best]
        elif active:
            winners = active[:]                   # everyone folded -> antes chopped back

        share = pot / len(winners) if winners else 0.0
        for i in winners:
            self.seats[i].chips += share

        # learning + per-seat report
        seats_report = []
        for i in active:
            fly = self.seats[i]
            a_idx, put = decisions[i]
            learn_ctx, score, hs = contexts[i]
            delta = (share if i in winners else 0.0) - committed[i]
            fb = fly.learn(learn_ctx, a_idx, delta / cfg.bet)
            fly.wins += 1 if (i in winners and a_idx != 0) else 0
            seats_report.append({
                "seat": i, "name": fly.name, "action": ACTIONS[a_idx], "hand_strength": round(hs, 3),
                "delta": round(delta, 2), "chips": round(fly.chips, 2), "won": i in winners,
                "dopamine": round(fb["dopamine"], 3), "pain": round(fb["pain"], 3),
                "mood": round(fb["mood"], 3), "newborn": i in newborn,
            })

        return {
            "hand": self.hand_no, "pot": round(pot, 2),
            "community": [card_str(c) for c in community],
            "winners": [self.seats[i].name for i in winners],
            "deaths_total": self.deaths, "newborn_seats": newborn,
            "seats": seats_report,
        }

    def run(self, n_hands: int, on_hand=None):
        for _ in range(n_hands):
            rec = self.play_hand()
            if on_hand is not None:
                on_hand(rec)
